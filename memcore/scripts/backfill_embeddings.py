#!/usr/bin/env python3
"""Backfill embeddings for all existing mem_entries that lack them.

Run inside the memcore container or anywhere with access to both PostgreSQL and the embedding server.
Usage: python3 -m memcore.scripts.backfill_embeddings
"""

import asyncio
import sys
import os

import asyncpg
import httpx

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://memcore:memcore@memcore-db:5432/memcore")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://192.168.8.141:8100/v1/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
BATCH_SIZE = 20  # Embed 20 at a time


async def generate_embeddings_batch(texts: list[str]) -> list[list[float] | None]:
    """Generate embeddings for a batch of texts."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                EMBEDDING_URL,
                json={"input": [t[:8000] for t in texts], "model": EMBEDDING_MODEL},
            )
            resp.raise_for_status()
            data = resp.json()
            # Sort by index to maintain order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [d["embedding"] for d in sorted_data]
    except Exception as e:
        print(f"  Embedding batch failed: {e}")
        return [None] * len(texts)


def embedding_to_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(str(v) for v in embedding) + "]"


async def main():
    pool = await asyncpg.create_pool(POSTGRES_URL, min_size=1, max_size=3)

    # Count entries needing embeddings
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT count(*) FROM mem_entries WHERE embedding IS NULL AND status = 'active'"
        )
        print(f"Entries needing embeddings: {total}")

        if total == 0:
            print("Nothing to backfill!")
            return

        # Process in batches
        offset = 0
        processed = 0
        errors = 0

        while offset < total:
            rows = await conn.fetch(
                "SELECT id, content FROM mem_entries WHERE embedding IS NULL AND status = 'active' "
                "ORDER BY created_at LIMIT $1 OFFSET $2",
                BATCH_SIZE, offset,
            )

            if not rows:
                break

            texts = [r["content"] for r in rows]
            ids = [r["id"] for r in rows]

            embeddings = await generate_embeddings_batch(texts)

            for entry_id, embedding in zip(ids, embeddings):
                if embedding is None:
                    errors += 1
                    continue
                try:
                    emb_str = embedding_to_pgvector(embedding)
                    await conn.execute(
                        "UPDATE mem_entries SET embedding = $1::vector WHERE id = $2",
                        emb_str, entry_id,
                    )
                    processed += 1
                except Exception as e:
                    print(f"  Failed to update {entry_id[:8]}: {e}")
                    errors += 1

            print(f"  Batch {offset//BATCH_SIZE + 1}: {processed}/{total} done, {errors} errors")
            offset += BATCH_SIZE

        print(f"\nBackfill complete: {processed} embedded, {errors} errors, {total} total")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
