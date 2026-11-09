from app.config import Settings
from app.store import VectorStore


def test_reindexing_same_filename_replaces_catalog_and_chunks(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "app.store.embed_documents",
        lambda chunks: [[float(index + 1), 0.0] for index, _ in enumerate(chunks)],
    )
    store = VectorStore(
        Settings(
            data_dir=tmp_path,
            cohere_embed_model="test-model",
            embed_dimension=2,
        )
    )

    first = store.add_document("report.pdf", "old content", ["old chunk"])
    second = store.add_document(
        "report.pdf",
        "new content",
        ["new chunk 1", "new chunk 2"],
    )

    stored = store._collection.get(include=["documents", "metadatas"])
    assert first["document_id"] != second["document_id"]
    assert store.list_documents() == [second]
    assert stored["documents"] == ["new chunk 1", "new chunk 2"]
    assert {metadata["document_id"] for metadata in stored["metadatas"]} == {
        second["document_id"]
    }


def test_listing_documents_removes_legacy_same_filename_duplicates(tmp_path) -> None:
    store = VectorStore(
        Settings(
            data_dir=tmp_path,
            cohere_embed_model="test-model",
            embed_dimension=2,
        )
    )
    records = [
        {
            "document_id": "older",
            "filename": "report.pdf",
            "uploaded_at": "2026-01-01",
        },
        {
            "document_id": "newer",
            "filename": "report.pdf",
            "uploaded_at": "2026-01-02",
        },
    ]
    store._collection.add(
        ids=["older:0", "newer:0"],
        documents=["old chunk", "new chunk"],
        metadatas=[
            {"document_id": row["document_id"], "filename": row["filename"], "chunk_index": 0}
            for row in records
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    store._write_catalog(records)

    assert store.list_documents() == [records[1]]
    stored = store._collection.get(include=["documents"])
    assert stored["ids"] == ["newer:0"]
    assert stored["documents"] == ["new chunk"]
