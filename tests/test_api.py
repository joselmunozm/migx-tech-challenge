from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import _index_document, app


@pytest.fixture
def without_langfuse(monkeypatch):
    @contextmanager
    def no_observation(name, metadata):
        yield None

    monkeypatch.setattr("app.main.observation", no_observation)
    monkeypatch.setattr("app.main.update_observation", lambda span, **values: None)
    monkeypatch.setattr("app.main.flush_langfuse", lambda: None)


def test_health_is_public_and_api_routes_require_key(without_langfuse) -> None:
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/documents").status_code == 401
    assert client.post("/ask", json={"question": "What is the endpoint?"}).status_code == 401


def test_upload_offloads_indexing_to_threadpool(monkeypatch, without_langfuse) -> None:
    client = TestClient(app)
    captured = {}

    async def fake_run_in_threadpool(function, *args):
        captured["function"] = function
        captured["args"] = args
        return {"document_id": "test-document", "chunk_count": 1}

    monkeypatch.setattr("app.main.run_in_threadpool", fake_run_in_threadpool)
    response = client.post(
        "/documents",
        headers={"X-API-Key": get_settings().api_key},
        files={"file": ("note.txt", b"document", "text/plain")},
    )

    assert response.status_code == 200
    assert captured["function"] is _index_document
    assert captured["args"] == ("note.txt", b"document")