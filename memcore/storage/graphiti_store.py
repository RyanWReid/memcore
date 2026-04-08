"""Graphiti storage client — calls the Graphiti MCP server's tools via MCP protocol.

Graphiti MCP tool names (as of April 2026):
  add_memory, search_memory_nodes, search_memory_facts,
  get_episodes, delete_episode, delete_entity_edge, get_entity_edge, clear_graph
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client

from memcore.config import GRAPHITI_URL, GRAPHITI_RECALL_TIMEOUT
from memcore.models import MemEntry

logger = logging.getLogger(__name__)

# Graphiti search involves LLM calls (DeepSeek via LiteLLM) — can take 60s+
_GRAPHITI_TIMEOUT = 120.0
# Fast timeout for recall path — don't block fast postgres results
_SEARCH_TIMEOUT = GRAPHITI_RECALL_TIMEOUT


@asynccontextmanager
async def _get_session():
    """Connect to Graphiti MCP via SSE with extended timeout."""
    sse_url = f"{GRAPHITI_URL}/sse"
    async with sse_client(sse_url, timeout=30, sse_read_timeout=_GRAPHITI_TIMEOUT) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def store(entry: MemEntry) -> dict:
    """Store a MemEntry as a Graphiti episode via MCP add_episode tool."""
    episode_body = entry.content
    if entry.temporal.referenced_at:
        date_str = entry.temporal.referenced_at.strftime("%Y-%m-%d")
        episode_body = f"[{date_str}] {episode_body}"

    source_desc = (
        f"MemCore {entry.memory_type.value} "
        f"(agent: {entry.source_agent}, score: {entry.epistemic_score:.2f})"
    )
    ref_time = (
        entry.temporal.referenced_at or entry.temporal.ingested_at
    ).isoformat()

    try:
        async with _get_session() as session:
            result = await session.call_tool(
                "add_memory",
                arguments={
                    "name": f"memcore_{entry.memory_type.value}_{entry.id[:8]}",
                    "episode_body": episode_body,
                    "source_description": source_desc,
                    "reference_time": ref_time,
                    "group_id": entry.group_id,
                },
            )
            text = result.content[0].text if result.content else "no response"
            logger.info("Graphiti stored episode for entry %s", entry.id[:8])
            return {"status": "stored", "layer": "graphiti", "response": text}
    except Exception as e:
        logger.error("Graphiti store failed: %s", e)
        return {"status": "error", "layer": "graphiti", "error": str(e)}


async def search(query: str, group_id: str = "homelab", limit: int = 10) -> list[dict]:
    """Search Graphiti via MCP search tool. Wrapped in timeout so recall doesn't hang."""
    try:
        return await asyncio.wait_for(
            _search_impl(query, group_id, limit),
            timeout=_SEARCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Graphiti search timed out after %ss for query: %s", _SEARCH_TIMEOUT, query[:80])
        return [{"warning": f"Graphiti search timed out after {_SEARCH_TIMEOUT}s"}]
    except Exception as e:
        logger.error("Graphiti search failed: %s", e)
        return []


async def _search_impl(query: str, group_id: str, limit: int) -> list[dict]:
    async with _get_session() as session:
        # Search both nodes and facts in parallel
        nodes_result = await session.call_tool(
            "search_memory_nodes",
            arguments={
                "query": query,
                "group_ids": [group_id],
                "max_nodes": limit,
            },
        )
        results = []
        if nodes_result.content:
            text = nodes_result.content[0].text
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    results.extend(parsed)
                else:
                    results.append(parsed)
            except json.JSONDecodeError:
                results.append({"text": text})
        return results
