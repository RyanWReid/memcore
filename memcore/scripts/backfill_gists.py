"""Backfill gist + gist_embedding for existing memories.

Fuzzy-trace dual storage was added after the store was already populated,
so existing memories have NULL in the gist and gist_embedding columns.
This script walks all active memories missing a gist and fills both fields.

Usage:
    python -m memcore.scripts.backfill_gists [--limit N] [--dry-run]

Safe to run multiple times — it only touches rows where gist IS NULL.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from memcore.lifecycle.gist import generate_gist
from memcore.storage.postgres_store import (
    _embedding_to_pgvector,
    generate_embedding,
    get_pool,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("backfill_gists")

CONCURRENCY = 4


async def process_one(row, dry_run: bool) -> tuple[str, bool, str]:
    mid = row["id"]
    content = row["content"]
    gist = await generate_gist(content)
    if not gist:
        return (mid, False, "no-gist")
    gist_emb = await generate_embedding(gist)
    if not gist_emb:
        return (mid, False, "no-embedding")
    if dry_run:
        return (mid, True, "dry-run")

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE mem_entries SET gist = $1, gist_embedding = $2::vector "
            "WHERE id = $3 AND gist IS NULL",
            gist,
            _embedding_to_pgvector(gist_emb),
            mid,
        )
    return (mid, True, "ok")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max rows this run (0 = all).")
    ap.add_argument("--dry-run", action="store_true", help="Skip writing updates.")
    args = ap.parse_args()

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, content FROM mem_entries "
            "WHERE status = 'active' AND gist IS NULL AND length(content) >= 40 "
            "ORDER BY created_at DESC "
            + (f"LIMIT {int(args.limit)}" if args.limit else "")
        )

    total = len(rows)
    if total == 0:
        logger.info("No memories need backfill. Done.")
        return 0
    logger.info("Backfilling gist for %d memories (concurrency=%d, dry_run=%s)",
                total, CONCURRENCY, args.dry_run)

    sem = asyncio.Semaphore(CONCURRENCY)
    results = {"ok": 0, "no-gist": 0, "no-embedding": 0, "dry-run": 0, "error": 0}

    async def bounded(row):
        async with sem:
            try:
                _, _, status = await process_one(row, args.dry_run)
                results[status] = results.get(status, 0) + 1
            except Exception as e:
                logger.warning("Row %s failed: %s", row["id"][:8], e)
                results["error"] += 1

    await asyncio.gather(*[bounded(r) for r in rows])

    logger.info("Backfill complete: %s", results)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
