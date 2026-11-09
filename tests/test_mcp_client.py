from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from mcp.types import TextContent

from app.errors import AppError
from app.mcp_client import search_documents


class FakeSession:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def initialize(self):
        return None

    async def call_tool(self, name, arguments, read_timeout_seconds):
        self.calls.append((name, arguments))
        return self.result


def install_fake_mcp(monkeypatch, result):
    session = FakeSession(result)

    @asynccontextmanager
    async def fake_stdio_client(parameters):
        yield object(), object()

    monkeypatch.setattr("app.mcp_client.stdio_client", fake_stdio_client)
    monkeypatch.setattr(
        "app.mcp_client.ClientSession",
        lambda *args, **kwargs: session,
    )
    return session


def test_search_documents_uses_mcp_tool_and_maps_hits(monkeypatch) -> None:
    result = SimpleNamespace(
        isError=False,
        content=[
            TextContent(
                type="text",
                text='[{"chunk_id":"doc-1:0","document_id":"doc-1","filename":"report.pdf",'
                '"chunk_index":0,"score":0.72,"text":"Relevant finding."}]',
            )
        ],
    )
    session = install_fake_mcp(monkeypatch, result)

    hits = search_documents("What was found?", "doc-1", 3)

    assert session.calls == [
        (
            "search_documents",
            {"query": "What was found?", "top_k": 3, "document_id": "doc-1"},
        )
    ]
    assert len(hits) == 1
    assert hits[0].filename == "report.pdf"
    assert hits[0].score == 0.72
    assert hits[0].text == "Relevant finding."


def test_search_documents_maps_unknown_document_error(monkeypatch) -> None:
    result = SimpleNamespace(
        isError=False,
        content=[
            TextContent(
                type="text",
                text='{"error":"Unknown document_id.","status_code":404}',
            )
        ],
    )
    install_fake_mcp(monkeypatch, result)

    with pytest.raises(AppError, match="Unknown document_id") as error:
        search_documents("Question", "missing", 3)

    assert error.value.status_code == 404


def test_search_documents_surfaces_mcp_tool_errors(monkeypatch) -> None:
    result = SimpleNamespace(
        isError=True,
        content=[TextContent(type="text", text="embedding provider failed")],
    )
    install_fake_mcp(monkeypatch, result)

    with pytest.raises(AppError, match="embedding provider failed") as error:
        search_documents("Question", None, 3)

    assert error.value.status_code == 502
