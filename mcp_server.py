"""MCP server exposing the local document catalog and retrieval tool.

Run it on stdio for an external agent host:

    uv run python mcp_server.py

The API starts it as a subprocess for document retrieval. It can also be
launched separately by an external MCP-compatible host.
"""

import json

from mcp.server.fastmcp import FastMCP

from app.prompts import SYSTEM_PROMPT
from app.store import list_documents as catalog
from app.store import search

mcp = FastMCP("medical-document-assistant")


@mcp.tool()
def list_documents() -> str:
    """List documents that have been ingested into the local collection."""
    return json.dumps(catalog(), ensure_ascii=False)


@mcp.tool()
def search_documents(query: str, top_k: int = 3, document_id: str = "") -> str:
    """Search ingested documents and return the closest source excerpts."""
    if document_id and document_id not in {
        row["document_id"] for row in catalog()
    }:
        return json.dumps(
            {"error": "Unknown document_id.", "status_code": 404},
            ensure_ascii=False,
        )
    hits = search(query, document_id or None, top_k)
    payload = [
        {
            "chunk_id": hit.chunk_id,
            "document_id": hit.document_id,
            "filename": hit.filename,
            "chunk_index": hit.chunk_index,
            "score": hit.score,
            "text": hit.text,
        }
        for hit in hits
    ]
    return json.dumps(payload, ensure_ascii=False)


@mcp.resource("policy://citation")
def citation_policy() -> str:
    """Static instructions the assistant gives the model together with retrieved excerpts."""
    return SYSTEM_PROMPT


if __name__ == "__main__":
    mcp.run()
