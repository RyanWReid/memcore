"""Prospective memory — remember to do something in the future.

The brain has two types of prospective memory:
- Time-based: "Remember to check DNS at 3pm"
- Event-based: "Remember to verify backups after the next reboot"

No other AI memory system implements this. MemCore is the first.

Intents are stored as a special memory type with trigger conditions.
On each recall, we check if any active intents match the current context.
Matched intents surface with a "prospective" flag so the agent knows
this is something it was asked to remember FOR THE FUTURE.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from memcore.storage.postgres_store import get_pool, generate_embedding

logger = logging.getLogger(__name__)


async def store_intent(
    content: str,
    trigger_condition: str,
    group_id: str = "homelab",
    trigger_time: str | None = None,
    expiry_hours: int = 168,  # 7 days default
) -> dict:
    """Store a prospective memory intent.

    Args:
        content: What to remember/surface when triggered
        trigger_condition: Natural language condition (e.g., "when DNS is rebooted")
        group_id: Namespace
        trigger_time: Optional ISO datetime for time-based triggers
        expiry_hours: Auto-expire if never triggered (default 7 days)

    Returns:
        Dict with intent ID and status
    """
    embedding = await generate_embedding(trigger_condition)

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO mem_intents (
                content, trigger_condition, trigger_embedding,
                group_id, trigger_time, expiry_hours, status
            ) VALUES ($1, $2, $3::vector, $4, $5, $6, 'active')
            RETURNING id, created_at
        """, content, trigger_condition,
             str(embedding) if embedding else None,
             group_id,
             trigger_time,
             expiry_hours)

    logger.info("Stored prospective intent: %s (trigger: %s)", row["id"][:8], trigger_condition[:50])
    return {
        "id": str(row["id"]),
        "status": "active",
        "trigger_condition": trigger_condition,
        "content": content,
        "created_at": str(row["created_at"]),
    }


async def check_intents(
    query: str,
    group_id: str = "homelab",
    similarity_threshold: float = 0.55,
) -> list[dict]:
    """Check if any active intents match the current query context.

    Called during recall to surface prospective memories whose trigger
    conditions match what's currently happening.

    Returns list of triggered intents (may be empty).
    """
    query_embedding = await generate_embedding(query)
    if query_embedding is None:
        return []

    pool = await get_pool()
    async with pool.acquire() as conn:
        # Find intents whose trigger_condition is semantically similar to the query
        # Also check time-based triggers that have matured
        rows = await conn.fetch("""
            SELECT id, content, trigger_condition, trigger_time, created_at,
                   1 - (trigger_embedding <=> $1::vector) as similarity
            FROM mem_intents
            WHERE status = 'active'
              AND group_id = $2
              AND trigger_embedding IS NOT NULL
              AND (
                -- Event-based: semantic match
                1 - (trigger_embedding <=> $1::vector) > $3
                -- Time-based: trigger_time has passed
                OR (trigger_time IS NOT NULL AND trigger_time <= NOW())
              )
            ORDER BY similarity DESC
            LIMIT 5
        """, str(query_embedding), group_id, similarity_threshold)

    if not rows:
        return []

    triggered = []
    for row in rows:
        triggered.append({
            "id": str(row["id"]),
            "content": row["content"],
            "trigger_condition": row["trigger_condition"],
            "similarity": float(row["similarity"]) if row["similarity"] else 0,
            "prospective": True,
            "memory_type": "intent",
        })

    if triggered:
        logger.info(
            "Prospective memory triggered: %d intents matched query '%s'",
            len(triggered), query[:50],
        )

    return triggered


async def complete_intent(intent_id: str) -> dict:
    """Mark an intent as completed (triggered and acknowledged)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE mem_intents
            SET status = 'completed', completed_at = NOW()
            WHERE id = $1 AND status = 'active'
        """, intent_id)

    if result == "UPDATE 0":
        return {"status": "not_found", "id": intent_id}

    logger.info("Completed prospective intent: %s", intent_id[:8])
    return {"status": "completed", "id": intent_id}


async def expire_stale_intents():
    """Expire intents that have passed their expiry window. Run periodically."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE mem_intents
            SET status = 'expired'
            WHERE status = 'active'
              AND created_at + (expiry_hours || ' hours')::interval < NOW()
        """)
    count = int(result.split()[-1]) if result else 0
    if count > 0:
        logger.info("Expired %d stale intents", count)
    return count


async def ensure_table():
    """Create the mem_intents table if it doesn't exist."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS mem_intents (
                id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
                content TEXT NOT NULL,
                trigger_condition TEXT NOT NULL,
                trigger_embedding vector(384),
                group_id TEXT NOT NULL DEFAULT 'homelab',
                trigger_time TIMESTAMPTZ,
                expiry_hours INTEGER DEFAULT 168,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                completed_at TIMESTAMPTZ
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_intents_active
            ON mem_intents (group_id, status)
            WHERE status = 'active'
        """)
