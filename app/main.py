"""HTTP API. Every route except /health requires the X-API-Key header."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.errors import AppError
from app.graph import ask
from app.ingest import chunk_text, extract_text
from app.observability import flush_langfuse, observation, update_observation
from app.store import add_document, list_documents

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("migx")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    with observation(
        "api-startup",
        {
            "llm_configured": bool(settings.openrouter_api_key.strip()),
            "embeddings_configured": bool(settings.cohere_api_key.strip()),
            "langfuse_configured": settings.langfuse_enabled,
            "model": settings.openrouter_model,
            "embedding_model": settings.cohere_embed_model,
        },
    ) as span:
        update_observation(span, output={"status": "started"})

    try:
        yield
    finally:
        with observation("api-shutdown", {"status": "stopping"}) as span:
            update_observation(span, output={"status": "stopped"})
        flush_langfuse()


app = FastAPI(title="Medical Document Assistant", version="0.1.0", lifespan=lifespan)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
MAX_BYTES = 10 * 1024 * 1024


def require_api_key(x_api_key: str | None = Security(api_key_header)) -> None:
    if not x_api_key or x_api_key != get_settings().api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: str | None = None


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_configured": bool(settings.openrouter_api_key.strip()),
        "embeddings_configured": bool(settings.cohere_api_key.strip()),
        "langfuse_configured": settings.langfuse_enabled,
        "model": settings.openrouter_model,
        "embed_model": settings.cohere_embed_model,
    }


@app.get("/documents", dependencies=[Depends(require_api_key)])
def documents() -> dict:
    return {"documents": list_documents()}


@app.post("/documents", dependencies=[Depends(require_api_key)])
async def upload(file: UploadFile = File(...)) -> dict:
    filename = Path(file.filename or "document.txt").name
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="File is larger than 10 MB.")
    try:
        return await run_in_threadpool(_index_document, filename, data)
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


def _index_document(filename: str, data: bytes) -> dict:
    settings = get_settings()
    started = time.perf_counter()
    with observation(
        "document-indexing",
        {
            "file_type": Path(filename).suffix.lower(),
            "file_bytes": len(data),
        },
    ) as span:
        text = extract_text(filename, data)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise AppError("No text could be extracted from this file.")
        record = add_document(filename, text, chunks)
        update_observation(
            span,
            output={
                "status": "indexed",
                "chunk_count": len(chunks),
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return record


@app.post("/ask", dependencies=[Depends(require_api_key)])
def ask_route(body: AskRequest) -> dict:
    try:
        return ask(body.question.strip(), body.document_id)
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(status_code=502, detail="The assistant could not complete this request.") from exc
