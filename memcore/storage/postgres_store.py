"""PostgreSQL storage — facts, preferences, goals, trajectories.

Uses asyncpg for async access. pgvector for semantic search.
Hybrid retrieval: tsvector full-text + pgvector cosine similarity + RRF fusion.
Bio-inspired: Ebbinghaus retention scoring, access tracking, stability growth.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

import asyncpg
import httpx

from memcore.config import (
    POSTGRES_URL, EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_DIM,
    LITELLM_BASE_URL, LITELLM_API_KEY, GATE_MODEL,
)
from memcore.models import MemEntry, MemStatus

logger = logging.getLogger(__name__)

# Query expansion cache to avoid repeated LLM calls for identical queries
_expansion_cache: dict[str, str] = {}
_EXPANSION_CACHE_MAX = 200

_pool: asyncpg.Pool | None = None
_embedding_client: httpx.AsyncClient | None = None

SCHEMA_SQL = """\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS mem_entries (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    layer TEXT NOT NULL,
    group_id TEXT NOT NULL DEFAULT 'homelab',
    epistemic_score REAL NOT NULL DEFAULT 0.0,
    quality_checks JSONB,
    epistemic_scores JSONB,
    temporal JSONB,
    source_agent TEXT NOT NULL DEFAULT 'unknown',
    session_id TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    gate_passed BOOLEAN NOT NULL DEFAULT false,
    gate_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding vector(""" + str(EMBEDDING_DIM) + """),
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS idx_mem_group_id ON mem_entries(group_id);
CREATE INDEX IF NOT EXISTS idx_mem_type ON mem_entries(memory_type);
CREATE INDEX IF NOT EXISTS idx_mem_status ON mem_entries(status);
CREATE INDEX IF NOT EXISTS idx_mem_score ON mem_entries(epistemic_score);
CREATE INDEX IF NOT EXISTS idx_mem_search_vector ON mem_entries USING GIN(search_vector);

-- Observability: log every recall event for analysis
CREATE TABLE IF NOT EXISTS recall_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    query_length INTEGER NOT NULL,
    group_id TEXT NOT NULL,
    confidence_level TEXT,
    confidence_score REAL,
    top_score REAL,
    memory_ids TEXT[],
    source TEXT,
    session_id TEXT,
    source_agent TEXT,
    used_memory_ids TEXT[],
    feedback_received_at TIMESTAMPTZ,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_recall_events_created_at ON recall_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recall_events_group_id ON recall_events(group_id);
CREATE INDEX IF NOT EXISTS idx_recall_events_confidence ON recall_events(confidence_level);
CREATE INDEX IF NOT EXISTS idx_recall_events_session ON recall_events(session_id);
"""


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(POSTGRES_URL, min_size=2, max_size=20)
        async with _pool.acquire() as conn:
            # Register the vector type so asyncpg can handle it
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            await conn.execute(SCHEMA_SQL)
            # Add columns if upgrading from old schema
            for col_sql in [
                f"ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIM})",
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED",
                # Bio-inspired: access tracking + Ebbinghaus stability
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS access_count INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ",
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS stability REAL NOT NULL DEFAULT 1.0",
                # v6 prep: reconsolidation + arousal
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS reconsolidation_count INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS arousal_score REAL",
                # v6: original content freeze (imagination inflation defense)
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS original_content TEXT",
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS content_drift REAL NOT NULL DEFAULT 0.0",
                # v6: Memory Worth counters (success/failure tracking)
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS mw_success INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS mw_total INTEGER NOT NULL DEFAULT 0",
                # v6: retrieval-induced suppression
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS suppression_count INTEGER NOT NULL DEFAULT 0",
                # v7: fuzzy-trace dual storage — gist alongside verbatim
                "ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS gist TEXT",
                f"ALTER TABLE mem_entries ADD COLUMN IF NOT EXISTS gist_embedding vector({EMBEDDING_DIM})",
            ]:
                try:
                    await conn.execute(col_sql)
                except Exception:
                    pass  # Column already exists or generated column can't be added via ALTER
            # v6: Prospective memory (intents) table
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
                ON mem_intents (group_id, status) WHERE status = 'active'
            """)
        logger.info("PostgreSQL pool created, schema initialized with pgvector + access tracking")
    return _pool


# --- Embedding generation ---

def _get_embedding_client() -> httpx.AsyncClient:
    """Reuse a single httpx client for embedding calls."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = httpx.AsyncClient(timeout=10)
    return _embedding_client


async def generate_embedding(text: str) -> list[float] | None:
    """Generate embedding via the local embedding server."""
    try:
        client = _get_embedding_client()
        resp = await client.post(
            EMBEDDING_URL,
            json={"input": text[:8000], "model": EMBEDDING_MODEL},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        return None


def _embedding_to_pgvector(embedding: list[float]) -> str:
    """Convert embedding list to pgvector string format."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


# --- Storage ---

async def store(entry: MemEntry) -> dict:
    """Store a MemEntry in PostgreSQL with verbatim + gist embeddings.

    When FUZZY_TRACE_ENABLED, an LLM-generated one-sentence gist is stored
    in parallel with the verbatim content. Each gets its own embedding so
    search can fuse a gist-match leg (better for conceptual queries) with
    a verbatim-match leg (better for detail queries).
    """
    pool = await get_pool()

    # Generate verbatim embedding and (optionally) gist + gist embedding in parallel.
    from memcore.config import FUZZY_TRACE_ENABLED
    if FUZZY_TRACE_ENABLED:
        from memcore.lifecycle.gist import generate_gist
        gist = await generate_gist(entry.content)
        emb_task = asyncio.create_task(generate_embedding(entry.content))
        gist_emb_task = asyncio.create_task(generate_embedding(gist)) if gist else None
        embedding = await emb_task
        gist_embedding = await gist_emb_task if gist_emb_task is not None else None
    else:
        gist = None
        embedding = await generate_embedding(entry.content)
        gist_embedding = None

    embedding_str = _embedding_to_pgvector(embedding) if embedding else None
    gist_embedding_str = _embedding_to_pgvector(gist_embedding) if gist_embedding else None

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO mem_entries
                    (id, content, memory_type, layer, group_id, epistemic_score,
                     quality_checks, epistemic_scores, temporal,
                     source_agent, session_id, status, gate_passed, gate_reason,
                     embedding, gist, gist_embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                        $15::vector, $16, $17::vector)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    epistemic_score = EXCLUDED.epistemic_score,
                    status = EXCLUDED.status,
                    embedding = EXCLUDED.embedding,
                    gist = EXCLUDED.gist,
                    gist_embedding = EXCLUDED.gist_embedding,
                    updated_at = NOW()
                """,
                entry.id,
                entry.content,
                entry.memory_type.value,
                entry.layer.value,
                entry.group_id,
                entry.epistemic_score,
                json.dumps(entry.quality_checks.model_dump()) if entry.quality_checks else None,
                json.dumps(entry.epistemic_scores.model_dump()) if entry.epistemic_scores else None,
                json.dumps(entry.temporal.model_dump(), default=str) if entry.temporal else None,
                entry.source_agent,
                entry.session_id,
                entry.status.value,
                entry.gate_passed,
                entry.gate_reason,
                embedding_str,
                gist,
                gist_embedding_str,
            )
        logger.info(
            "PostgreSQL stored entry %s (emb=%s gist=%s)",
            entry.id[:8], "y" if embedding else "n", "y" if gist else "n",
        )
        return {"status": "stored", "layer": "postgres", "id": entry.id}
    except Exception as e:
        logger.error("PostgreSQL store failed: %s", e)
        return {"status": "error", "layer": "postgres", "error": str(e)}


# --- Hybrid Search ---

async def expand_query(query: str) -> str:
    """Expand query with synonyms/rephrasings via LLM for better recall.

    Keeps the original query intact and appends alternative phrasings.
    """
    if query in _expansion_cache:
        return _expansion_cache[query]

    if len(query.split()) < 3:
        return query

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.post(
                f"{LITELLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                json={
                    "model": GATE_MODEL,
                    "messages": [
                        {"role": "system", "content": (
                            "Rewrite this search query by adding 3-5 synonym phrases. "
                            "Keep the original query first, then append alternatives. "
                            "Output ONLY the expanded query, no explanation. Keep it under 60 words."
                        )},
                        {"role": "user", "content": query},
                    ],
                    "temperature": 0,
                    "max_tokens": 100,
                },
            )
            resp.raise_for_status()
            expanded = resp.json()["choices"][0]["message"]["content"].strip()

            if len(_expansion_cache) >= _EXPANSION_CACHE_MAX:
                _expansion_cache.clear()
            _expansion_cache[query] = expanded

            logger.info("Query expanded: '%s' → '%s'", query[:50], expanded[:80])
            return expanded
    except Exception as e:
        logger.warning("Query expansion failed (using original): %s", e)
        return query


async def search(
    query: str,
    group_id: str = "homelab",
    memory_type: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """Hybrid search: tsvector full-text + pgvector cosine similarity, fused with RRF.

    Query is expanded with synonyms via LLM before searching.
    Falls back to keyword-only search if embedding generation fails.
    Cross-encoder reranking applied when available (70/30 blend with RRF).
    """
    pool = await get_pool()

    # Expand query with synonyms for better vocabulary coverage
    expanded = await expand_query(query)

    # Generate embedding from expanded query for vector search
    query_embedding = await generate_embedding(expanded)

    # Retrieve wider for reranking — fetch 3x limit, rerank down to limit
    retrieval_limit = limit * 3

    results = []
    if query_embedding:
        results = await _hybrid_search(pool, expanded, query_embedding, group_id, memory_type, retrieval_limit)
        if not results:
            logger.info("Hybrid search returned empty, falling back to keyword search")

    if not results:
        if not query_embedding:
            logger.warning("Embedding generation failed, falling back to keyword search")
        results = await _keyword_search(pool, expanded, group_id, memory_type, retrieval_limit)

    # Cross-encoder reranking — rerank the wider result set, return top `limit`
    if results and len(results) > 1:
        try:
            from memcore.retrieval.reranker import rerank
            results = await rerank(query, results, top_k=limit)
        except Exception as e:
            logger.warning("Reranking skipped: %s", e)
            results = results[:limit]
    else:
        results = results[:limit]

    # Apply Ebbinghaus retention scoring as a post-rerank adjustment
    results = _apply_retention_scoring(results)

    # Track access for returned results (fire-and-forget)
    if results:
        asyncio.get_event_loop().create_task(_record_access(results))
        asyncio.get_event_loop().create_task(_apply_suppression(results))

    return results


# --- Ebbinghaus retention scoring ---

# Per-type decay lambdas — decisions are permanent, goals decay fast
_TYPE_LAMBDAS = {
    "decision": 0.0,       # permanent — decisions don't decay
    "relationship": 0.0,   # permanent
    "fact": 0.01,          # ~70 day half-life at stability=1
    "event": 0.02,         # ~35 day half-life at stability=1
    "document": 0.005,     # ~140 day half-life at stability=1
    "goal": 0.05,          # ~14 day half-life at stability=1
    "tool_hint": 0.0,      # permanent — tool catalog doesn't decay
}


def _apply_retention_scoring(results: list[dict]) -> list[dict]:
    """Apply Ebbinghaus retention as a multiplier on existing scores.

    R = e^(-lambda * days_since_access / stability)
    - Each access increases stability by 1.3x
    - Decisions are exempt from decay (lambda=0)
    - Floor of 0.3 so old memories don't vanish entirely
    """
    now = datetime.now(timezone.utc)

    for r in results:
        # Get the memory's stability and last access time
        stability = float(r.get("stability", 1.0)) or 1.0
        last_accessed = r.get("last_accessed_at")
        created_at = r.get("created_at")
        memory_type = r.get("memory_type", "fact")

        # Use last_accessed_at if available, otherwise created_at
        reference_time = last_accessed or created_at
        if reference_time is None:
            r["retention"] = 1.0
            continue

        # Handle string dates
        if isinstance(reference_time, str):
            try:
                reference_time = datetime.fromisoformat(reference_time.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                r["retention"] = 1.0
                continue

        # Make timezone-aware if needed
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        days_since = max((now - reference_time).total_seconds() / 86400, 0)
        lam = _TYPE_LAMBDAS.get(memory_type, 0.01)

        # Memories less than 1 day old: retention = 1.0 (no effect).
        # Prevents micro-reordering noise on freshly-ingested data where
        # seconds-apart timestamps create meaningless retention differences.
        if lam == 0 or days_since < 1.0:
            retention = 1.0
        else:
            retention = max(math.exp(-lam * days_since / stability), 0.3)

        r["retention"] = round(retention, 4)

        # Adjust the score — blend retention into existing score
        # 85% retrieval relevance + 15% retention (don't let decay dominate)
        base_score = float(r.get("blended_score", 0) or r.get("rrf_score", 0))
        if base_score > 0:
            r["final_score"] = round(0.85 * base_score + 0.15 * retention * base_score, 4)
        else:
            r["final_score"] = base_score

    # Re-sort by final_score
    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return results


def _stability_growth_factor(blended_score: float, rank: int) -> float:
    """Bjork's Testing Effect: harder retrievals grow stability MORE.

    Easy retrieval (top result, high score) = 1.1x growth.
    Hard retrieval (low rank, low score) = 1.8x growth.
    This creates a self-correcting system: weak-but-useful memories
    gain MORE durability, preventing the rich-get-richer feedback loop.
    """
    difficulty = 1.0 - min(blended_score, 1.0)
    rank_penalty = min(rank / 10.0, 0.5)
    effective_difficulty = (difficulty + rank_penalty) / 2.0
    return 1.1 + 0.7 * effective_difficulty


async def _record_access(results: list[dict]):
    """Increment access_count and stability for recalled memories.

    Stability growth is difficulty-weighted (Bjork's Testing Effect):
    hard retrievals grow stability more than easy ones.
    Capped at 200 to prevent overflow.

    Note: Memory Worth (mw_total, mw_success) is bumped separately, in
    record_recall_feedback(), so the ratio reflects ground-truth usage
    rather than raw retrieval traffic. See record_recall_feedback below.
    """
    if not results:
        return
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            for rank, r in enumerate(results):
                mid = r.get("id")
                if not mid:
                    continue
                score = float(r.get("blended_score", 0) or r.get("rrf_score", 0))
                growth = _stability_growth_factor(score, rank)
                await conn.execute("""
                    UPDATE mem_entries
                    SET access_count = access_count + 1,
                        last_accessed_at = NOW(),
                        stability = LEAST(stability * $1, 200.0)
                    WHERE id = $2
                """, growth, mid)
    except Exception as e:
        logger.warning("Failed to record access: %s", e)


async def log_recall_event(
    query: str,
    group_id: str,
    results: list[dict],
    confidence: dict | None = None,
    source: str = "unknown",
    session_id: str | None = None,
    source_agent: str | None = None,
    latency_ms: int | None = None,
) -> str | None:
    """Log a recall event for observability.

    Returns the event UUID so the caller can update it later with feedback
    (which memories were actually used by the LLM).
    """
    try:
        pool = await get_pool()
        memory_ids = [r.get("id") for r in results if r.get("id")]
        top_score = 0.0
        if results:
            top_score = float(
                results[0].get("final_score", 0)
                or results[0].get("blended_score", 0)
                or results[0].get("rrf_score", 0)
            )
        conf = confidence or {}

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO recall_events
                    (query, query_length, group_id, confidence_level, confidence_score,
                     top_score, memory_ids, source, session_id, source_agent, latency_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                query[:500],
                len(query),
                group_id,
                conf.get("level"),
                float(conf.get("score", 0) or 0),
                top_score,
                memory_ids,
                source,
                session_id,
                source_agent,
                latency_ms,
            )
            return str(row["id"]) if row else None
    except Exception as e:
        logger.warning("Failed to log recall event: %s", e)
        return None


async def record_recall_feedback(event_id: str, used_memory_ids: list[str]) -> bool:
    """Record which memories from a recall were actually used by the LLM,
    and bump the Memory Worth counters (mw_total for all recalled IDs,
    mw_success for the used subset).

    Memory Worth is bumped here — not at retrieval time — so the signal
    reflects ground-truth usefulness in recalls that actually had a
    judged response, rather than every raw retrieval (which includes
    recalls the hook discards due to weak confidence, secondary namespace
    lookups, and tool-hint lookups that never reach the prompt).

    Idempotent on the event: only bumps on the first feedback for a given
    event (feedback_received_at IS NULL). Re-posting feedback for the
    same event won't double-count.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    UPDATE recall_events
                    SET used_memory_ids = $1, feedback_received_at = NOW()
                    WHERE id = $2::uuid AND feedback_received_at IS NULL
                    RETURNING memory_ids
                    """,
                    used_memory_ids,
                    event_id,
                )
                if row is None:
                    return False

                recalled_ids = [m for m in (row["memory_ids"] or []) if m]
                if recalled_ids:
                    await conn.execute(
                        "UPDATE mem_entries SET mw_total = mw_total + 1 "
                        "WHERE id = ANY($1::text[])",
                        recalled_ids,
                    )
                used = [m for m in used_memory_ids if m]
                if used:
                    await conn.execute(
                        "UPDATE mem_entries SET mw_success = mw_success + 1 "
                        "WHERE id = ANY($1::text[])",
                        used,
                    )
                return True
    except Exception as e:
        logger.warning("Failed to record recall feedback: %s", e)
        return False


async def _apply_suppression(results: list[dict]):
    """Retrieval-induced suppression: the winner suppresses competitors.

    When the top memory wins, similar but lower-ranked memories get
    a stability penalty (0.9x). This creates natural canonical-version
    selection — richer memories suppress older, incomplete ones.

    Based on Wimber et al. 2015 (Nature Neuroscience).
    """
    if len(results) < 2:
        return

    winner = results[0]
    winner_content = winner.get("content", "")[:200].lower()
    suppressed_ids = []

    for r in results[1:]:
        # Only suppress memories that are similar to the winner
        competitor_content = r.get("content", "")[:200].lower()
        # Quick similarity check: shared word overlap
        winner_words = set(winner_content.split())
        competitor_words = set(competitor_content.split())
        if len(winner_words) < 3 or len(competitor_words) < 3:
            continue
        overlap = len(winner_words & competitor_words) / min(len(winner_words), len(competitor_words))
        if overlap > 0.5:
            suppressed_ids.append(r.get("id"))

    if not suppressed_ids:
        return

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE mem_entries
                SET stability = GREATEST(stability * 0.9, 0.5),
                    suppression_count = suppression_count + 1
                WHERE id = ANY($1::text[])
            """, suppressed_ids)
        logger.debug("Suppressed %d competitors of winner %s", len(suppressed_ids), winner.get("id", "?")[:8])
    except Exception as e:
        logger.warning("Suppression failed: %s", e)


async def _hybrid_search(
    pool: asyncpg.Pool,
    query: str,
    query_embedding: list[float],
    group_id: str,
    memory_type: Optional[str],
    limit: int,
) -> list[dict]:
    """Reciprocal Rank Fusion of full-text search + vector search."""
    K = 60  # RRF constant

    type_filter = ""
    params_extra = []
    if memory_type:
        type_filter = "AND memory_type = $4"
        params_extra = [memory_type]

    embedding_str = _embedding_to_pgvector(query_embedding)

    # Three-leg RRF: full-text + verbatim-vector + gist-vector.
    # The gist leg helps conceptual queries match memories whose verbatim
    # embedding is dominated by specific identifiers (fuzzy-trace theory).
    # A missing gist_embedding just means that memory doesn't contribute a
    # gist-rank, not that it's excluded from search.
    sql = f"""
    WITH fts AS (
        SELECT id, content, memory_type, epistemic_score, created_at, updated_at,
               access_count, last_accessed_at, stability, reconsolidation_count,
               ROW_NUMBER() OVER (ORDER BY ts_rank(search_vector, websearch_to_tsquery('english', $1)) DESC) as rank_pos
        FROM mem_entries
        WHERE status = 'active' AND group_id = $2
          AND search_vector @@ websearch_to_tsquery('english', $1)
          {type_filter}
        ORDER BY ts_rank(search_vector, websearch_to_tsquery('english', $1)) DESC
        LIMIT 30
    ),
    vec AS (
        SELECT id, content, memory_type, epistemic_score, created_at, updated_at,
               access_count, last_accessed_at, stability, reconsolidation_count,
               ROW_NUMBER() OVER (ORDER BY embedding <=> $3::vector) as rank_pos
        FROM mem_entries
        WHERE status = 'active' AND group_id = $2
          AND embedding IS NOT NULL
          {type_filter}
        ORDER BY embedding <=> $3::vector
        LIMIT 30
    ),
    vec_gist AS (
        SELECT id, content, memory_type, epistemic_score, created_at, updated_at,
               access_count, last_accessed_at, stability, reconsolidation_count,
               ROW_NUMBER() OVER (ORDER BY gist_embedding <=> $3::vector) as rank_pos
        FROM mem_entries
        WHERE status = 'active' AND group_id = $2
          AND gist_embedding IS NOT NULL
          {type_filter}
        ORDER BY gist_embedding <=> $3::vector
        LIMIT 30
    ),
    combined AS (
        SELECT
            COALESCE(fts.id, vec.id, vec_gist.id) as id,
            COALESCE(fts.content, vec.content, vec_gist.content) as content,
            COALESCE(fts.memory_type, vec.memory_type, vec_gist.memory_type) as memory_type,
            COALESCE(fts.epistemic_score, vec.epistemic_score, vec_gist.epistemic_score) as epistemic_score,
            COALESCE(fts.created_at, vec.created_at, vec_gist.created_at) as created_at,
            COALESCE(fts.updated_at, vec.updated_at, vec_gist.updated_at) as updated_at,
            COALESCE(fts.access_count, vec.access_count, vec_gist.access_count) as access_count,
            COALESCE(fts.last_accessed_at, vec.last_accessed_at, vec_gist.last_accessed_at) as last_accessed_at,
            COALESCE(fts.stability, vec.stability, vec_gist.stability) as stability,
            COALESCE(fts.reconsolidation_count, vec.reconsolidation_count, vec_gist.reconsolidation_count) as reconsolidation_count,
            -- RRF score: 1/(K+rank) for each leg, summed across fts, verbatim vec, gist vec
            COALESCE(1.0 / ({K} + fts.rank_pos), 0) +
            COALESCE(1.0 / ({K} + vec.rank_pos), 0) +
            COALESCE(1.0 / ({K} + vec_gist.rank_pos), 0) as rrf_score
        FROM fts
        FULL OUTER JOIN vec ON fts.id = vec.id
        FULL OUTER JOIN vec_gist ON COALESCE(fts.id, vec.id) = vec_gist.id
    )
    SELECT id, content, memory_type, epistemic_score, created_at, updated_at,
           access_count, last_accessed_at, stability, reconsolidation_count, rrf_score
    FROM combined
    ORDER BY rrf_score DESC
    LIMIT ${ '4' if memory_type else '3' }
    """

    limit_param_idx = 4 if memory_type else 3
    params = [query, group_id, embedding_str] + params_extra + [limit]

    # Fix: the LIMIT placeholder needs the right index
    sql = sql.replace(f"LIMIT ${ '4' if memory_type else '3' }", f"LIMIT ${len(params)}")

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error("Hybrid search failed: %s", e)
        return await _keyword_search(pool, query, group_id, memory_type, limit)


async def _keyword_search(
    pool: asyncpg.Pool,
    query: str,
    group_id: str,
    memory_type: Optional[str],
    limit: int,
) -> list[dict]:
    """Fallback keyword search using ILIKE (original method)."""
    conditions = ["status = 'active'", "group_id = $1"]
    params: list = [group_id]
    idx = 2

    if query:
        keywords = [w.strip() for w in query.split() if len(w.strip()) > 2]
        if keywords:
            keyword_conditions = []
            for kw in keywords:
                keyword_conditions.append(f"content ILIKE ${idx}")
                params.append(f"%{kw}%")
                idx += 1
            conditions.append(f"({' OR '.join(keyword_conditions)})")
        else:
            conditions.append(f"content ILIKE ${idx}")
            params.append(f"%{query}%")
            idx += 1

    if memory_type:
        conditions.append(f"memory_type = ${idx}")
        params.append(memory_type)
        idx += 1

    params.append(limit)

    sql = f"""
        SELECT id, content, memory_type, epistemic_score, created_at, updated_at,
               access_count, last_accessed_at, stability, reconsolidation_count
        FROM mem_entries
        WHERE {' AND '.join(conditions)}
        ORDER BY epistemic_score DESC, created_at DESC
        LIMIT ${idx}
    """

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error("Keyword search failed: %s", e)
        return []


# --- Dedup ---

async def find_similar(
    content: str,
    group_id: str = "homelab",
    threshold_words: int = 3,
) -> list[dict]:
    """Find existing entries with overlapping content.

    Uses keyword extraction for cheap similarity detection.
    Also uses vector similarity if embedding is available.
    """
    pool = await get_pool()

    # Try vector similarity first — more accurate
    embedding = await generate_embedding(content)
    if embedding:
        try:
            embedding_str = _embedding_to_pgvector(embedding)
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, content, memory_type, epistemic_score, created_at,
                           1 - (embedding <=> $2::vector) as cosine_similarity
                    FROM mem_entries
                    WHERE status = 'active' AND group_id = $1
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> $2::vector
                    LIMIT 5
                    """,
                    group_id, embedding_str,
                )
                # Filter to high similarity (>0.85 cosine)
                similar = [
                    {**dict(r), "keyword_overlap": 5}  # treat as high overlap for blocking
                    for r in rows
                    if r["cosine_similarity"] > 0.85
                ]
                if similar:
                    return similar
        except Exception as e:
            logger.error("Vector find_similar failed: %s", e)

    # Fallback: keyword-based similarity
    _stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "have", "been",
        "was", "were", "are", "will", "would", "could", "should", "into",
        "about", "which", "when", "where", "what", "there", "their", "some",
        "also", "than", "then", "only", "just", "more", "very", "using",
    }
    words = [
        w.lower().strip(".,;:!?()[]{}\"'")
        for w in content.split()
        if len(w) > 4 and w.lower() not in _stopwords
    ]
    if len(words) < threshold_words:
        return []

    keywords = sorted(set(words), key=len, reverse=True)[:5]

    try:
        async with pool.acquire() as conn:
            overlap_parts = [
                f"CASE WHEN content ILIKE ${i+2} THEN 1 ELSE 0 END"
                for i in range(len(keywords))
            ]
            overlap_expr = " + ".join(overlap_parts)
            params = [group_id] + [f"%{kw}%" for kw in keywords]
            rows = await conn.fetch(
                f"""
                SELECT id, content, memory_type, epistemic_score, created_at,
                       ({overlap_expr}) AS keyword_overlap
                FROM mem_entries
                WHERE status = 'active' AND group_id = $1
                  AND ({overlap_expr}) >= 2
                ORDER BY ({overlap_expr}) DESC, created_at DESC
                LIMIT 5
                """,
                *params,
            )
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error("PostgreSQL find_similar failed: %s", e)
        return []


# --- Lifecycle ---

async def update_status(entry_id: str, status: MemStatus) -> bool:
    """Update an entry's status (for forget/archive)."""
    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE mem_entries SET status = $1, updated_at = NOW() WHERE id = $2",
                status.value,
                entry_id,
            )
            return result == "UPDATE 1"
    except Exception as e:
        logger.error("PostgreSQL update failed: %s", e)
        return False


async def get_entry(entry_id: str) -> Optional[dict]:
    """Get a single entry by ID."""
    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM mem_entries WHERE id = $1", entry_id
            )
            return dict(row) if row else None
    except Exception as e:
        logger.error("PostgreSQL get failed: %s", e)
        return None
