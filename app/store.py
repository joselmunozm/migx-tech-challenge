"""Chroma collection plus a small JSON catalog of uploaded documents."""

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import chromadb

from app.config import Settings, get_settings
from app.embeddings import embed_documents, embed_query

_lock = threading.Lock()
_store: "VectorStore | None" = None


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.data_dir / "chroma"))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        marker = self.settings.data_dir / "embedder.txt"
        signature = f"{self.settings.cohere_embed_model}:{self.settings.embed_dimension}"
        previous = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
        if previous and previous != signature:
            self._client.delete_collection("documents")
            registry = self._registry_path()
            if registry.exists():
                registry.unlink()
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        marker.write_text(signature, encoding="utf-8")

    def add_document(self, filename: str, text: str, chunks: list[str]) -> dict:
        document_id = uuid.uuid4().hex[:12]
        ids = [f"{document_id}:{index}" for index in range(len(chunks))]
        metadatas = [
            {"document_id": document_id, "filename": filename, "chunk_index": index}
            for index in range(len(chunks))
        ]
        embeddings = embed_documents(chunks)
        record = {
            "document_id": document_id,
            "filename": filename,
            "chunk_count": len(chunks),
            "char_count": len(text),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        with _lock:
            catalog = self._read_catalog()
            catalog = self._deduplicate_catalog(catalog)
            replaced_ids = [
                row["document_id"] for row in catalog if row["filename"] == filename
            ]
            self._collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            for replaced_id in replaced_ids:
                self._collection.delete(where={"document_id": replaced_id})
            catalog = [row for row in catalog if row["filename"] != filename]
            catalog.append(record)
            self._write_catalog(catalog)
        return record

    def list_documents(self) -> list[dict]:
        with _lock:
            return self._deduplicate_catalog(self._read_catalog())

    def search(
        self,
        query: str,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> list[Hit]:
        limit = top_k or self.settings.top_k
        available = self._available(document_id)
        if available == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embed_query(query)],
            n_results=min(limit, available),
            where={"document_id": document_id} if document_id else None,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    document_id=str(meta["document_id"]),
                    filename=str(meta["filename"]),
                    chunk_index=int(meta["chunk_index"]),
                    text=text,
                    score=round(1.0 - float(distance), 4),
                )
            )
        return hits

    def _available(self, document_id: str | None) -> int:
        if not document_id:
            return self._collection.count()
        found = self._collection.get(where={"document_id": document_id}, include=[])
        return len(found.get("ids") or [])

    def _registry_path(self):
        return self.settings.data_dir / "registry.json"

    def _read_catalog(self) -> list[dict]:
        path = self._registry_path()
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _deduplicate_catalog(self, rows: list[dict]) -> list[dict]:
        latest_by_filename: dict[str, dict] = {}
        for row in rows:
            latest_by_filename.pop(row["filename"], None)
            latest_by_filename[row["filename"]] = row
        unique_rows = list(latest_by_filename.values())
        retained_ids = {row["document_id"] for row in unique_rows}
        for row in rows:
            if row["document_id"] not in retained_ids:
                self._collection.delete(where={"document_id": row["document_id"]})
        if len(unique_rows) != len(rows):
            self._write_catalog(unique_rows)
        return unique_rows

    def _write_catalog(self, rows: list[dict]) -> None:
        path = self._registry_path()
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def get_store() -> VectorStore:
    global _store
    settings = get_settings()
    if _store is None or _store.settings.data_dir != settings.data_dir:
        _store = VectorStore(settings)
    return _store


def add_document(filename: str, text: str, chunks: list[str]) -> dict:
    return get_store().add_document(filename, text, chunks)


def list_documents() -> list[dict]:
    return get_store().list_documents()


def search(query: str, document_id: str | None = None, top_k: int | None = None) -> list[Hit]:
    return get_store().search(query, document_id, top_k)
