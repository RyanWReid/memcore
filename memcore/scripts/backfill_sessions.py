#!/usr/bin/env python3
"""Backfill memories from historical Claude Code transcripts.

Reads all past session transcripts, extracts memorable content via LLM,
and stores through MemCore's remember API (which runs the full write gate).

This gives MemCore context from weeks of past work that happened before
the retain hook was active.

Usage:
    python -m memcore.scripts.backfill_sessions                    # All sessions
    python -m memcore.scripts.backfill_sessions --limit 10         # First 10 sessions
    python -m memcore.scripts.backfill_sessions --dry-run          # Extract but don't store
    python -m memcore.scripts.backfill_sessions --max-age-days 14  # Last 2 weeks only
"""

import argparse
import asyncio
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx

MEMCORE_URL = os.getenv("MEMCORE_URL", "http://localhost:8020")
LITELLM_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_API_KEY", "")
MODEL = os.getenv("BENCHMARK_MODEL", "deepseek-chat")

TRANSCRIPT_DIRS = {
    "homelab": os.path.expanduser("~/.claude/projects/-Users-ryanreid-Development-Projects-homelab"),
    "personal": os.path.expanduser("~/.claude/projects"),
}

# Same extraction prompt as the retain hook — proven to work well
EXTRACTION_PROMPT = """You are a memory retention agent. Analyze this conversation excerpt and decide what to remember.

RETAIN MISSION for this namespace ({namespace}):
{mission}

For each memory worth keeping, return a JSON object in the array.
Return an EMPTY array [] if nothing is worth remembering.

Output format:
{{
  "memories": [
    {{
      "content": "Self-contained description. Include specific details (names, IPs, versions). Must be understandable without context.",
      "importance": 0.0-1.0,
      "action": "ADD"
    }}
  ]
}}

Rules:
- Maximum 5 memories per conversation chunk
- importance >= 0.6 to be worth storing
- Each memory MUST be self-contained — no "it", "the service", "this thing"
- Include WHY, not just WHAT
- Return {{"memories": []}} if nothing is memorable
- NEVER create memories about the memory system itself
- At the end of each memory, append a parenthetical with 3-5 search keywords/synonyms"""

RETAIN_MISSIONS = {
    "homelab": (
        "Extract ONLY high-signal infrastructure memories: "
        "architectural decisions with rationale, service deployments with IPs/ports, "
        "incidents with root cause and fix, cross-system relationships discovered, "
        "configuration changes that affect other services. "
        "IGNORE: debugging steps, 'let me check' narration, code explanations, "
        "status updates, task progress, tool call descriptions."
    ),
    "personal": (
        "Extract personal preferences, goals, habits, project decisions. "
        "IGNORE: code details, debugging, routine conversations."
    ),
}


def extract_session_text(fpath: str, chunk_size: int = 8000) -> list[str]:
    """Extract conversation chunks from a transcript file.

    Returns list of text chunks, each containing several user+assistant exchanges.
    Chunks are sized to fit in LLM context without being too small to extract from.
    """
    try:
        with open(fpath, "rb") as f:
            lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
    except Exception:
        return []

    # Extract user and assistant text
    texts = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type")
        if msg_type == "user":
            content = obj.get("message", {}).get("content", "")
            if isinstance(content, str) and len(content) > 10:
                texts.append(f"User: {content[:300]}")
        elif msg_type == "assistant":
            content = obj.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(f"Claude: {block['text'][:400]}")
                        break

    # Chunk into groups
    chunks = []
    current = []
    current_len = 0
    for t in texts:
        current.append(t)
        current_len += len(t)
        if current_len >= chunk_size:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
    if current:
        chunks.append("\n\n".join(current))

    return chunks


async def extract_memories(
    client: httpx.AsyncClient,
    text: str,
    namespace: str,
) -> list[dict]:
    """Call LLM to extract memorable content from a conversation chunk."""
    mission = RETAIN_MISSIONS.get(namespace, RETAIN_MISSIONS["personal"])

    try:
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": EXTRACTION_PROMPT.format(
                        namespace=namespace, mission=mission,
                    )},
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": 600,
            },
            timeout=20,
        )
        resp.raise_for_status()
        parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
        memories = parsed.get("memories", [])
        return [m for m in memories if m.get("importance", 0) >= 0.6]
    except Exception as e:
        print(f"    LLM error: {e}", file=sys.stderr)
        return []


async def store_memory(client: httpx.AsyncClient, content: str, namespace: str) -> bool:
    """Store through MemCore's remember API (full write gate)."""
    try:
        resp = await client.post(
            f"{MEMCORE_URL}/api/remember",
            json={
                "content": content,
                "group_id": namespace,
                "source_agent": "backfill-sessions",
            },
            timeout=15,
        )
        data = resp.json()
        return data.get("stored", False)
    except Exception:
        return False


async def check_duplicate(client: httpx.AsyncClient, content: str, namespace: str) -> bool:
    """Check if a very similar memory already exists."""
    try:
        resp = await client.post(
            f"{MEMCORE_URL}/api/recall",
            json={"query": content[:200], "group_id": namespace, "limit": 1},
            timeout=10,
        )
        data = resp.json()
        results = data.get("results", [])
        if results:
            score = results[0].get("blended_score", 0) or results[0].get("final_score", 0)
            if score > 0.95:
                return True
    except Exception:
        pass
    return False


async def process_session(
    client: httpx.AsyncClient,
    fpath: str,
    namespace: str,
    dry_run: bool = False,
) -> dict:
    """Process one session file. Returns stats."""
    fname = os.path.basename(fpath)
    chunks = extract_session_text(fpath)

    if not chunks:
        return {"file": fname, "chunks": 0, "extracted": 0, "stored": 0, "duplicates": 0}

    extracted = 0
    stored = 0
    duplicates = 0

    for chunk in chunks:
        if len(chunk) < 100:
            continue

        memories = await extract_memories(client, chunk, namespace)
        extracted += len(memories)

        for mem in memories:
            content = mem.get("content", "")
            if len(content) < 30:
                continue

            # Dedup check
            is_dup = await check_duplicate(client, content, namespace)
            if is_dup:
                duplicates += 1
                continue

            if dry_run:
                print(f"    [DRY] {content[:100]}...")
                stored += 1
            else:
                ok = await store_memory(client, content, namespace)
                if ok:
                    stored += 1

        # Rate limit between chunks
        await asyncio.sleep(0.5)

    return {
        "file": fname,
        "chunks": len(chunks),
        "extracted": extracted,
        "stored": stored,
        "duplicates": duplicates,
    }


async def run_backfill(
    limit: int | None = None,
    max_age_days: int | None = None,
    dry_run: bool = False,
    namespace: str = "homelab",
):
    transcript_dir = TRANSCRIPT_DIRS.get(namespace, TRANSCRIPT_DIRS["homelab"])
    files = sorted(
        glob.glob(os.path.join(transcript_dir, "*.jsonl")),
        key=os.path.getmtime,
        reverse=True,
    )

    # Filter by age
    if max_age_days:
        cutoff = time.time() - max_age_days * 86400
        files = [f for f in files if os.path.getmtime(f) > cutoff]

    if limit:
        files = files[:limit]

    print(f"Backfilling from {len(files)} sessions ({namespace})")
    if dry_run:
        print("DRY RUN — extracting only, not storing")
    print("=" * 70)

    total_extracted = 0
    total_stored = 0
    total_duplicates = 0

    async with httpx.AsyncClient(timeout=30) as client:
        for i, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            size = os.path.getsize(fpath)
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d")

            print(f"\n[{i+1}/{len(files)}] {fname[:20]}... ({size//1024}KB, {mtime})")

            stats = await process_session(client, fpath, namespace, dry_run)
            total_extracted += stats["extracted"]
            total_stored += stats["stored"]
            total_duplicates += stats["duplicates"]

            print(f"  {stats['chunks']} chunks → {stats['extracted']} extracted → {stats['stored']} stored, {stats['duplicates']} duplicates")

    print(f"\n{'=' * 70}")
    print(f"BACKFILL COMPLETE")
    print(f"  Sessions processed: {len(files)}")
    print(f"  Memories extracted: {total_extracted}")
    print(f"  Memories stored:    {total_stored}")
    print(f"  Duplicates skipped: {total_duplicates}")


def main():
    parser = argparse.ArgumentParser(description="Backfill memories from past Claude Code sessions")
    parser.add_argument("--limit", type=int, help="Max sessions to process")
    parser.add_argument("--max-age-days", type=int, help="Only process sessions from last N days")
    parser.add_argument("--dry-run", action="store_true", help="Extract but don't store")
    parser.add_argument("--namespace", default="homelab", help="Namespace to store under")
    args = parser.parse_args()

    asyncio.run(run_backfill(
        limit=args.limit,
        max_age_days=args.max_age_days,
        dry_run=args.dry_run,
        namespace=args.namespace,
    ))


if __name__ == "__main__":
    main()
