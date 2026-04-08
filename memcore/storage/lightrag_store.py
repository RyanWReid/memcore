"""LightRAG storage client — calls the LightRAG MCP server's tools via MCP protocol."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.sse import sse_client

from memcore.config import LIGHTRAG_URL
from memcore.models import MemEntry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _get_session():
    """Connect to LightRAG MCP via SSE."""
    sse_url = f"{LIGHTRAG_URL}/mcp/sse"
    async with sse_client(sse_url, timeout=30, sse_read_timeout=120) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _discover_tools():
    """List available tools to discover the correct insert/query tool names."""
    try:
        async with _get_session() as session:
            tools = await session.list_tools()
            return [t.name for t in tools.tools]
    except Exception as e:
        logger.error("LightRAG tool discovery failed: %s", e)
        return []


async def store(entry: MemEntry) -> dict:
    """Store a MemEntry in LightRAG via MCP insert tool."""
    try:
        async with _get_session() as session:
            # Try common LightRAG MCP tool names
            result = await session.call_tool(
                "lightrag_insert",
                arguments={"content": entry.content},
            )
            text = result.content[0].text if result.content else "no response"
            logger.info("LightRAG stored document for entry %s", entry.id[:8])
            return {"status": "stored", "layer": "lightrag", "response": text}
    except Exception as e:
        logger.error("LightRAG store failed: %s", e)
        return {"status": "error", "layer": "lightrag", "error": str(e)}


async def search(query: str, mode: str = "hybrid", limit: int = 10) -> list[dict]:
    """Search LightRAG via MCP query tool."""
    try:
        async with _get_session() as session:
            result = await session.call_tool(
                "lightrag_query",
                arguments={"query": query, "mode": mode},
            )
            if result.content:
                text = result.content[0].text
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return [{"text": text}]
            return []
    except Exception as e:
        logger.error("LightRAG search failed: %s", e)
        return []
