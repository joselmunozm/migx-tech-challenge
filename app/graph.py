"""LangGraph: retrieve, generate, check citations, and rewrite once if needed."""

import logging
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.errors import AppError
from app.grounding import check_citations
from app.observability import observation, update_observation
from app.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    render_repair_prompt,
    render_user_prompt,
)

logger = logging.getLogger("migx")


class GraphState(TypedDict):
    question: str
    document_id: str | None
    sources: list[dict]
    answer: str
    citation_check: dict
    attempts: int


def build_graph(retrieve_fn, model, min_similarity: float = 0.3):
    """retrieve_fn(question, document_id) -> hits. model.invoke(messages) -> response."""

    def retrieve(state: GraphState) -> dict:
        hits = [
            hit
            for hit in retrieve_fn(state["question"], state.get("document_id"))
            if hit.score >= min_similarity
        ]
        sources = [
            {
                "ref": f"S{index}",
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "filename": hit.filename,
                "chunk_index": hit.chunk_index,
                "score": hit.score,
                "text": hit.text,
            }
            for index, hit in enumerate(hits, start=1)
        ]
        return {"sources": sources}

    def generate(state: GraphState) -> dict:
        if state["attempts"] >= 1:
            user_prompt = render_repair_prompt(
                state["question"],
                state["sources"],
                state["answer"],
                state["citation_check"],
            )
        else:
            user_prompt = render_user_prompt(state["question"], state["sources"])
        try:
            response = model.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
        except AppError:
            raise
        except Exception as exc:
            raise AppError(f"The language model failed: {exc}", 502) from exc
        return {"answer": _message_text(response).strip(), "attempts": state["attempts"] + 1}

    def check(state: GraphState) -> dict:
        allowed = {source["ref"] for source in state["sources"]}
        return {"citation_check": check_citations(state["answer"], allowed)}

    def abstain(state: GraphState) -> dict:
        from app.prompts import ABSTAIN_RESPONSE

        answer = ABSTAIN_RESPONSE
        return {
            "answer": answer,
            "attempts": 0,
            "citation_check": check_citations(answer, set()),
        }

    def after_retrieve(state: GraphState) -> str:
        return "generate" if state["sources"] else "abstain"

    def after_check(state: GraphState) -> str:
        if state["citation_check"]["ok"] or state["attempts"] >= 2:
            return "end"
        return "generate"

    builder = StateGraph(GraphState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("check", check)
    builder.add_node("abstain", abstain)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {"generate": "generate", "abstain": "abstain"},
    )
    builder.add_edge("generate", "check")
    builder.add_conditional_edges("check", after_check, {"generate": "generate", "end": END})
    builder.add_edge("abstain", END)
    return builder.compile()


def ask(question: str, document_id: str | None = None) -> dict:
    settings = get_settings()
    metadata = {
        "prompt_version": PROMPT_VERSION,
        "document_id": document_id or "",
    }
    with observation("medical-document-ask", metadata) as span:
        result = _ask(question, document_id, settings)
        update_observation(
            span,
            output={
                "model": settings.openrouter_model,
                "prompt_version": PROMPT_VERSION,
                "attempts": result["attempts"],
                "source_count": len(result["citations"]),
                "citation_check": result["citation_check"],
            },
        )
        return result


def _ask(question: str, document_id: str | None, settings: Settings) -> dict:
    _require_credentials(settings)
    from app.mcp_client import search_documents

    model = _build_chat_model(settings)
    graph = build_graph(
        lambda query, selected_document_id: search_documents(
            query,
            selected_document_id,
            settings.top_k,
        ),
        model,
        settings.min_similarity,
    )
    state: GraphState = {
        "question": question,
        "document_id": document_id,
        "sources": [],
        "answer": "",
        "citation_check": {},
        "attempts": 0,
    }
    result = _invoke(graph, state)
    logger.info(
        "ask attempts=%s citations_ok=%s sources=%s",
        result["attempts"],
        result["citation_check"].get("ok"),
        len(result["sources"]),
    )
    return {
        "answer": result["answer"],
        "model": settings.openrouter_model,
        "prompt_version": PROMPT_VERSION,
        "attempts": result["attempts"],
        "citations": result["sources"],
        "citation_check": result["citation_check"],
    }


def _build_chat_model(settings: Settings):
    from langchain_openrouter import ChatOpenRouter

    models = list(
        dict.fromkeys(
            model_name.strip()
            for model_name in (settings.openrouter_model, settings.openrouter_fallback_model)
            if model_name.strip()
        )
    )
    return ChatOpenRouter(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        temperature=0,
        max_tokens=500,
        timeout=30_000,
        max_retries=0,
        model_kwargs={"models": models},
    )


def _invoke(graph, state: GraphState) -> dict:
    config: dict = {
        "run_name": "medical-document-ask",
        "metadata": {
            "prompt_version": PROMPT_VERSION,
            "document_id": state.get("document_id") or "",
        },
    }
    return graph.invoke(state, config=config)


def _require_credentials(settings: Settings) -> None:
    if not settings.cohere_api_key.strip():
        raise AppError("COHERE_API_KEY is not set.", 503)
    if not settings.openrouter_api_key.strip():
        raise AppError("OPENROUTER_API_KEY is not set.", 503)


def _message_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", block)))
        return "".join(parts)
    return str(content)
