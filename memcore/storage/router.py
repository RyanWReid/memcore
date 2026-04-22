"""Storage router — dispatches MemEntries to the correct backend.

Every entry gets logged to PostgreSQL (audit trail).
The primary storage layer is determined by memory_type → layer mapping.
Recall fuses results from postgres + graphiti into a single ranked list.
Bio-inspired metamemory: returns recall confidence signal alongside results.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from statistics import mean

from memcore.models import MemEntry, MemStatus, StorageLayer

from . import graphiti_store, postgres_store

logger = logging.getLogger(__name__)


def compute_recall_confidence(query: str, results: list[dict]) -> dict:
    """Metamemory — estimate confidence in recall results.

    Inspired by human 'feeling of knowing' (FOK): the brain can sense
    whether it has the answer before fully retrieving it. Returns a
    confidence signal the LLM can use to calibrate trust.

    Addresses the RAG paradox: more context increases hallucination on
    unanswerable questions. Low confidence = return fewer results.
    """
    if not results:
        return {
            "level": "no_memory",
            "signal": "No memories found for this topic",
            "score": 0.0,
        }

    # Score distribution analysis
    scores = [float(r.get("final_score", 0) or r.get("blended_score", 0) or r.get("rrf_score", 0)) for r in results]
    top_score = scores[0] if scores else 0
    score_gap = (scores[0] - scores[1]) if len(scores) > 1 else scores[0]

    # Recency analysis
    now = datetime.now(timezone.utc)
    recency_days = []
    for r in results[:5]:
        la = r.get("last_accessed_at") or r.get("created_at")
        if la:
            if isinstance(la, str):
                try:
                    la = datetime.fromisoformat(la.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
            if la.tzinfo is None:
                la = la.replace(tzinfo=timezone.utc)
            recency_days.append((now - la).total_seconds() / 86400)
    avg_recency = mean(recency_days) if recency_days else 999

    # Access count analysis — have these memories been validated by usage?
    access_counts = [int(r.get("access_count", 0)) for r in results[:5]]
    avg_access = mean(access_counts) if access_counts else 0

    # Confidence levels
    if top_score > 0.85 and score_gap > 0.15 and avg_access >= 2:
        return {
            "level": "high",
            "signal": "Strong match, well-accessed, clear winner",
            "score": min(0.95, top_score),
        }
    elif top_score > 0.7 and len(results) >= 3:
        if avg_recency > 30:
            return {
                "level": "stale",
                "signal": f"Good matches but avg {avg_recency:.0f} days old — may be outdated",
                "score": 0.5,
            }
        return {
            "level": "moderate",
            "signal": "Multiple relevant matches, moderate confidence",
            "score": 0.7,
        }
    elif top_score > 0.4:
        return {
            "level": "weak",
            "signal": "Partial matches only — answer may be incomplete",
            "score": 0.35,
        }
    else:
        return {
            "level": "very_weak",
            "signal": "Vague associations only — high risk of hallucination if used",
            "score": 0.1,
        }


def _normalize_graphiti_nodes(raw: list[dict]) -> list[dict]:
    """Convert Graphiti node results to the same format as postgres results."""
    normalized = []
    # Graphiti returns {"message": ..., "nodes": [...]} or a flat list
    nodes = raw
    if len(raw) == 1 and isinstance(raw[0], dict) and "nodes" in raw[0]:
        nodes = raw[0]["nodes"]

    for node in nodes:
        if not isinstance(node, dict):
            continue
        # Skip warning/error entries
        if "warning" in node or "error" in node:
            continue
        summary = node.get("summary", "")
        name = node.get("name", "")
        if not summary and not name:
            continue
        content = f"{name}: {summary}" if summary else name
        normalized.append({
            "id": node.get("uuid", hashlib.md5(content.encode()).hexdigest()),
            "content": content,
            "memory_type": "graphiti_entity",
            "epistemic_score": 0.8,
            "created_at": node.get("created_at", ""),
            "rrf_score": 0.0,  # no RRF from graphiti — cross-encoder will score it
            "source": "graphiti",
        })
    return normalized


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

    # Route to Graphiti for decisions/events (entity graph)
    if entry.layer == StorageLayer.GRAPHITI:
        graphiti_result = await graphiti_store.store(entry)
        results["graphiti"] = graphiti_result

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


async def recall_fused(
    query: str,
    group_id: str = "homelab",
    limit: int = 10,
) -> list[dict]:
    """Fused recall: postgres + graphiti results merged into a single ranked list.

    1. Run postgres hybrid search and graphiti node search in parallel
    2. Normalize graphiti results to common format
    3. Deduplicate by content similarity
    4. Re-rank combined set with cross-encoder
    5. Return top-k flat list
    """
    # Run both searches in parallel — graphiti has its own timeout
    pg_task = postgres_store.search(query, group_id=group_id, limit=limit)
    gr_task = graphiti_store.search(query, group_id=group_id, limit=limit)
    results = await asyncio.gather(pg_task, gr_task, return_exceptions=True)

    pg_results = results[0] if not isinstance(results[0], Exception) else []
    gr_raw = results[1] if not isinstance(results[1], Exception) else []

    if isinstance(results[0], Exception):
        logger.error("Fused recall — postgres failed: %s", results[0])
    if isinstance(results[1], Exception):
        logger.error("Fused recall — graphiti failed: %s", results[1])

    # Normalize graphiti results
    gr_results = _normalize_graphiti_nodes(gr_raw) if gr_raw else []

    # Tag postgres results with source
    for r in pg_results:
        r.setdefault("source", "postgres")

    # Combine and deduplicate (skip graphiti entries whose content is a substring
    # of an existing postgres result, or vice versa)
    combined = list(pg_results)
    pg_contents = {r.get("content", "")[:100].lower() for r in pg_results}
    for gr in gr_results:
        gr_snippet = gr.get("content", "")[:100].lower()
        # Simple overlap check — skip if first 100 chars match any postgres entry
        if not any(gr_snippet[:50] in pc or pc[:50] in gr_snippet for pc in pg_contents):
            combined.append(gr)

    if not combined:
        return []

    # Only re-rank with cross-encoder when graphiti actually contributed new results.
    # postgres_store.search() already does its own cross-encoder reranking —
    # running it again would double-rerank and distort scores.
    graphiti_contributed = len(combined) > len(pg_results)
    if graphiti_contributed and len(combined) > 1:
        try:
            from memcore.retrieval.reranker import rerank
            combined = await rerank(query, combined, top_k=limit)
        except Exception as e:
            logger.warning("Fused reranking failed: %s", e)
            combined = combined[:limit]
    else:
        combined = combined[:limit]

    # Compute metamemory confidence signal
    confidence = compute_recall_confidence(query, combined)

    # Check prospective memory — surface intents whose triggers match this query
    try:
        from memcore.lifecycle.prospective import check_intents
        intents = await check_intents(query, group_id=group_id)
        if intents:
            # Prepend triggered intents to results (highest priority)
            combined = intents + combined
            confidence["intents_triggered"] = len(intents)
    except Exception as e:
        logger.debug("Prospective memory check failed: %s", e)

    logger.info(
        "Fused recall: %d postgres + %d graphiti → %d combined (limit %d, confidence=%s)",
        len(pg_results), len(gr_results), len(combined), limit, confidence["level"],
    )

    # Fire-and-forget reconsolidation for top memories (skip intents)
    from memcore.config import RECONSOLIDATION_ENABLED
    if RECONSOLIDATION_ENABLED and combined:
        from memcore.lifecycle.reconsolidation import trigger_reconsolidation
        recon_candidates = [m for m in combined[:5] if m.get("memory_type") != "intent"]
        if recon_candidates:
            asyncio.create_task(trigger_reconsolidation(recon_candidates[:3], query, confidence))

    return combined, confidence


async def forget(entry_id: str) -> dict:
    """Mark an entry as deleted in PostgreSQL. Graphiti/LightRAG entries persist
    but won't be returned by MemCore recall (filtered by status)."""
    success = await postgres_store.update_status(entry_id, MemStatus.DELETED)
    return {"status": "deleted" if success else "not_found", "id": entry_id}


async def get_entry(entry_id: str) -> dict | None:
    """Get a single entry from PostgreSQL."""
    return await postgres_store.get_entry(entry_id)
