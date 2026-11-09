"""MCP client used by the LangGraph retrieval node."""

import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import TextContent

from app.config import PROJECT_ROOT
from app.errors import AppError
from app.store import Hit

_SERVER_TIMEOUT = timedelta(seconds=30)


def search_documents(
    query: str,
    document_id: str | None,
    top_k: int,
) -> list[Hit]:
    """Call the document search tool through a local MCP stdio server."""
    try:
        return asyncio.run(_search_documents(query, document_id, top_k))
    except AppError:
        raise
    except (McpError, OSError, TimeoutError) as error:
        raise AppError(f"MCP document search failed: {error}", 502) from error


async def _search_documents(
    query: str,
    document_id: str | None,
    top_k: int,
) -> list[Hit]:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server"],
        cwd=Path(PROJECT_ROOT),
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=_SERVER_TIMEOUT,
        ) as session:
            await session.initialize()
            result = await session.call_tool(
                "search_documents",
                arguments={
                    "query": query,
                    "top_k": top_k,
                    "document_id": document_id or "",
                },
                read_timeout_seconds=_SERVER_TIMEOUT,
            )

    if result.isError:
        details = "; ".join(
            item.text for item in result.content if isinstance(item, TextContent)
        )
        raise AppError(f"MCP document search failed: {details or 'tool returned an error.'}", 502)
    text_results = [item.text for item in result.content if isinstance(item, TextContent)]
    if len(text_results) != 1:
        raise AppError("MCP document search returned an invalid response.", 502)
    try:
        payload: Any = json.loads(text_results[0])
    except json.JSONDecodeError as error:
        raise AppError("MCP document search returned invalid JSON.", 502) from error
    if not isinstance(payload, list):
        if (
            isinstance(payload, dict)
            and payload.get("error") == "Unknown document_id."
            and payload.get("status_code") == 404
        ):
            raise AppError("Unknown document_id.", 404)
        raise AppError("MCP document search returned an invalid result list.", 502)

    hits: list[Hit] = []
    for row in payload:
        if not isinstance(row, dict):
            raise AppError("MCP document search returned an invalid result.", 502)
        try:
            hits.append(
                Hit(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    filename=str(row["filename"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    score=float(row["score"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AppError("MCP document search returned a malformed result.", 502) from error
    return hits
