"""Memory reconsolidation — enrich memories with new context on recall.

In the brain, recall makes memories temporarily unstable. New context gets
woven in before re-storage. MemCore does the same: when a well-accessed memory
is recalled in a new context, an LLM merges additive information into it.

Bio-inspired prediction-error gating (from D-MEM / Nemori):
Reconsolidation ONLY triggers when there's a genuine prediction error —
the recall context surprises the memory (related but novel). This prevents
wasteful LLM calls when queries just re-read existing info.

Gating ensures we only reconsolidate when it's worth it:
- Memory must be proven valuable (access_count >= threshold)
- Not over-enriched (reconsolidation_count < max)
- Cooldown period since last reconsolidation
- Prediction error: query must be in the "surprise window" (0.3-0.7 cosine)
- LLM confirms the new context is genuinely additive
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from memcore.config import (
    RECONSOLIDATION_ENABLED,
    RECONSOLIDATION_MIN_ACCESS,
    RECONSOLIDATION_MAX_COUNT,
    RECONSOLIDATION_COOLDOWN_HOURS,
)
from memcore.gate.llm_client import llm_json_call
from memcore.storage.postgres_store import generate_embedding, get_pool

logger = logging.getLogger(__name__)

def _compute_surprise(query_embedding: list[float], content_embedding: list[float]) -> float:
    """Compute prediction error as 1 - cosine_similarity.

    Returns 0.0 (identical, no surprise) to 1.0 (orthogonal, max surprise).
    The "surprise window" for reconsolidation is 0.3-0.7:
    - <0.3: query re-reads existing info (too similar, no new context)
    - 0.3-0.7: related but novel (genuine prediction error, worth enriching)
    - >0.7: unrelated (tangential recall, not worth enriching this memory)
    """
    import math
    dot = sum(a * b for a, b in zip(query_embedding, content_embedding))
    norm_q = math.sqrt(sum(a * a for a in query_embedding))
    norm_c = math.sqrt(sum(b * b for b in content_embedding))
    if norm_q == 0 or norm_c == 0:
        return 0.5  # Default to middle of window if degenerate
    cosine_sim = dot / (norm_q * norm_c)
    return 1.0 - max(0.0, min(1.0, cosine_sim))


ENRICHMENT_SYSTEM = """You are a memory reconsolidation engine. Your job is to enrich an existing memory with new context, without losing any original information.

Rules:
- Keep ALL original facts intact — never remove or contradict existing content
- Add new context ONLY if it's genuinely additive (new relationships, details, or corrections)
- Keep the enriched version concise — don't increase length by more than 50%
- Preserve the original tone and format
- If the query adds nothing new to the memory, return {"enriched": false}

Return JSON:
{"enriched": true, "content": "the enriched memory text"}
OR
{"enriched": false}"""

ENRICHMENT_USER = """Existing memory:
{content}

This memory was just recalled because of this query:
{query}

Should this memory be enriched with context from the query? If the query reveals new information not already in the memory (e.g., new relationships, updated status, additional details), produce an enriched version. If the query is just re-reading existing info, return enriched=false."""


async def maybe_reconsolidate(
    memory: dict,
    query: str,
    confidence_level: str,
) -> bool:
    """Attempt to reconsolidate a single memory with new context from a query.

    Returns True if reconsolidation occurred, False otherwise.
    """
    if not RECONSOLIDATION_ENABLED:
        return False

    memory_id = memory.get("id")
    if not memory_id:
        return False

    # Gate 1: Must be well-accessed
    access_count = memory.get("access_count", 0)
    if access_count < RECONSOLIDATION_MIN_ACCESS:
        return False

    # Gate 2: Not over-enriched
    recon_count = memory.get("reconsolidation_count", 0)
    if recon_count >= RECONSOLIDATION_MAX_COUNT:
        return False

    # Gate 3: Confidence must be high or moderate
    if confidence_level not in ("high", "moderate"):
        return False

    # Gate 4: Cooldown — check updated_at
    updated_at = memory.get("updated_at")
    if updated_at:
        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                updated_at = None
        if updated_at:
            cooldown = timedelta(hours=RECONSOLIDATION_COOLDOWN_HOURS)
            if datetime.now(timezone.utc) - updated_at < cooldown:
                return False

    # Gate 5: Prediction-error gate — only reconsolidate when surprised
    # Inspired by D-MEM's Reward Prediction Error and Nemori's Free Energy Principle.
    # High similarity (>0.7) = query re-reads existing info, no surprise.
    # Low similarity (<0.3) = query is unrelated, recall was tangential.
    # Surprise window (0.3-0.7) = related but novel context, worth enriching.
    content = memory.get("content", "")
    query_embedding = await generate_embedding(query)
    if query_embedding is not None:
        content_embedding = await generate_embedding(content[:500])
        if content_embedding is not None:
            surprise = _compute_surprise(query_embedding, content_embedding)
            if surprise < 0.3 or surprise > 0.7:
                logger.debug(
                    "Reconsolidation skipped for %s: surprise=%.2f (outside window)",
                    memory_id, surprise,
                )
                return False

    # Gate 6: LLM confirms new context is genuinely additive
    try:
        result = await llm_json_call(
            system_prompt=ENRICHMENT_SYSTEM,
            user_prompt=ENRICHMENT_USER.format(content=content[:1000], query=query[:300]),
        )
    except Exception as e:
        logger.warning("Reconsolidation LLM call failed for %s: %s", memory_id, e)
        return False

    if not result.get("enriched"):
        return False

    new_content = result.get("content", "")
    if not new_content or len(new_content) < len(content) * 0.8:
        # Sanity: enriched version shouldn't be shorter than 80% of original
        logger.warning("Reconsolidation produced shorter content for %s, skipping", memory_id)
        return False

    # Generate new embedding
    new_embedding = await generate_embedding(new_content)
    if new_embedding is None:
        logger.warning("Reconsolidation embedding failed for %s", memory_id)
        return False

    # Compute content drift from original (imagination inflation defense)
    drift = _compute_surprise(
        await generate_embedding(content[:500]) or [],
        new_embedding,
    ) if recon_count > 0 else 0.0

    if drift > 0.4:
        logger.warning(
            "Reconsolidation drift too high for %s (%.2f), skipping to prevent corruption",
            memory_id, drift,
        )
        return False

    # Persist the enriched memory
    await _persist_reconsolidation(memory_id, new_content, new_embedding, recon_count, content, drift)
    logger.info(
        "Reconsolidated memory %s (access=%d, recon=%d→%d, drift=%.2f): +%d chars",
        memory_id, access_count, recon_count, recon_count + 1, drift,
        len(new_content) - len(content),
    )
    return True


async def _persist_reconsolidation(
    memory_id: str,
    new_content: str,
    new_embedding: list[float],
    current_recon_count: int,
    original_content: str,
    content_drift: float,
):
    """Update the memory in PostgreSQL with enriched content.

    Uses optimistic locking via reconsolidation_count to prevent double-writes.
    Freezes original_content on first reconsolidation (imagination inflation defense).
    Tracks content_drift from original.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        if current_recon_count == 0:
            # First reconsolidation: freeze the original content
            result = await conn.execute("""
                UPDATE mem_entries
                SET content = $1,
                    embedding = $2::vector,
                    reconsolidation_count = $3,
                    original_content = COALESCE(original_content, content),
                    content_drift = $6,
                    updated_at = NOW()
                WHERE id = $4 AND reconsolidation_count = $5
            """, new_content, str(new_embedding), current_recon_count + 1,
                 memory_id, current_recon_count, content_drift)
        else:
            result = await conn.execute("""
                UPDATE mem_entries
                SET content = $1,
                    embedding = $2::vector,
                    reconsolidation_count = $3,
                    content_drift = $6,
                    updated_at = NOW()
                WHERE id = $4 AND reconsolidation_count = $5
            """, new_content, str(new_embedding), current_recon_count + 1,
                 memory_id, current_recon_count, content_drift)
        if result == "UPDATE 0":
            logger.debug("Reconsolidation skipped (concurrent write) for %s", memory_id)
            return False
    return True


async def trigger_reconsolidation(
    memories: list[dict],
    query: str,
    confidence: dict,
):
    """Fire-and-forget reconsolidation attempt for top recalled memories.

    Called from router.recall_fused() as an asyncio.create_task().
    Only processes the top N memories (typically 3) to limit LLM cost.
    """
    confidence_level = confidence.get("level", "unknown")

    for memory in memories:
        try:
            await maybe_reconsolidate(memory, query, confidence_level)
        except Exception as e:
            logger.warning("Reconsolidation error for %s: %s", memory.get("id", "?"), e)
