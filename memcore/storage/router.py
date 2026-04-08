"""Storage router — dispatches MemEntries to the correct backend.

Every entry gets logged to PostgreSQL (audit trail).
The primary storage layer is determined by memory_type → layer mapping.
"""

from __future__ import annotations

import asyncio
import logging

from memcore.models import MemEntry, MemStatus, StorageLayer

from . import graphiti_store, lightrag_store, postgres_store

logger = logging.getLogger(__name__)


async def store(entry: MemEntry) -> dict:
    """Route a MemEntry to the appropriate storage layer.

    Always stores metadata in PostgreSQL. If the primary layer is Graphiti
    or LightRAG, also stores there. Checks for contradictions first.
    """
    results = {}

    # Duplicate check: find similar existing entries and block if too close
    similar = await postgres_store.find_similar(entry.content, group_id=entry.group_id)
    if similar:
        results["similar_existing"] = [
            {"id": s["id"], "content": s["content"][:100], "score": s["epistemic_score"]}
            for s in similar
        ]
        # Block storage if a high-overlap match exists (3+ keyword matches)
        top = similar[0]
        top_overlap = top.get("keyword_overlap", 0)
        if top_overlap >= 3:
            logger.info(
                "Blocked duplicate: new memory %s overlaps %d keywords with %s",
                entry.id[:8], top_overlap, top["id"][:8],
            )
            results["blocked"] = True
            results["reason"] = f"duplicate: {top_overlap} keyword overlap with {top['id'][:8]}"
            return results
        logger.info(
            "Found %d similar entries for new memory %s (max overlap: %d, allowing)",
            len(similar), entry.id[:8], top_overlap,
        )

    # Always log to PostgreSQL (audit trail + fact/goal/trajectory primary store)
    pg_result = await postgres_store.store(entry)
    results["postgres"] = pg_result

    # Route to primary layer if not already PostgreSQL
    if entry.layer == StorageLayer.GRAPHITI:
        graphiti_result = await graphiti_store.store(entry)
        results["graphiti"] = graphiti_result
    elif entry.layer == StorageLayer.LIGHTRAG:
        lightrag_result = await lightrag_store.store(entry)
        results["lightrag"] = lightrag_result

    return results


async def recall(
    query: str,
    group_id: str = "homelab",
    layers: list[str] | None = None,
    limit: int = 10,
) -> dict:
    """Parallel retrieval from specified layers.

    Default: postgres only (fast, ~0.1s). Graphiti available on request
    but uses a fast timeout (configurable, default 10s) to avoid blocking.
    """
    if layers is None:
        layers = ["postgres"]

    tasks = {}
    if "graphiti" in layers:
        tasks["graphiti"] = graphiti_store.search(query, group_id=group_id, limit=limit)
    # LightRAG MCP has broken message endpoint (404) — disabled until fixed
    # if "lightrag" in layers:
    #     tasks["lightrag"] = lightrag_store.search(query, limit=limit)
    if "postgres" in layers:
        tasks["postgres"] = postgres_store.search(query, group_id=group_id, limit=limit)

    results = {}
    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for key, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            logger.error("Recall from %s failed: %s", key, result)
            results[key] = []
        else:
            results[key] = result

    return results


async def forget(entry_id: str) -> dict:
    """Mark an entry as deleted in PostgreSQL. Graphiti/LightRAG entries persist
    but won't be returned by MemCore recall (filtered by status)."""
    success = await postgres_store.update_status(entry_id, MemStatus.DELETED)
    return {"status": "deleted" if success else "not_found", "id": entry_id}


async def get_entry(entry_id: str) -> dict | None:
    """Get a single entry from PostgreSQL."""
    return await postgres_store.get_entry(entry_id)
