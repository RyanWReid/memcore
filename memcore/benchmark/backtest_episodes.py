#!/usr/bin/env python3
"""Backtest episode segmentation against historical transcripts.

1. Process all historical transcripts through the episode segmenter
2. Store episode memories in a test namespace
3. Re-run real prompts against the enriched memory store
4. Compare recall coverage with vs without episodes

Usage:
    # Full pipeline: segment → store → evaluate
    python -m memcore.benchmark.backtest_episodes --limit 50

    # Just segment (dry run — show episodes without storing)
    python -m memcore.benchmark.backtest_episodes --segment-only --limit 10

    # Just evaluate (assumes episodes already stored)
    python -m memcore.benchmark.backtest_episodes --eval-only --limit 50
"""

import argparse
import asyncio
import glob
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import httpx

MEMCORE_URL = os.getenv("MEMCORE_URL", "http://localhost:8020")
LITELLM_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_API_KEY", "")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8100/v1/embeddings")
MODEL = os.getenv("BENCHMARK_MODEL", "deepseek-chat")

TRANSCRIPT_DIR = os.path.expanduser(
    "~/.claude/projects/-Users-ryanreid-Development-Projects-homelab"
)

WINDOW_K = 3
MIN_EPISODE = 5
MAX_EPISODE = 20

EPISODE_PROMPT = """Summarize this conversation segment into a single memory.

Conversation:
{transcript}

Output a JSON object:
{{
  "topic": "one-line topic",
  "actions": "what was done (code changes, deployments, research)",
  "outcomes": "what resulted (decisions, artifacts, numbers)",
  "decisions": "any choices made and why (or 'none')",
  "open_items": "anything unresolved (or 'none')"
}}

Rules:
- Be specific: include file names, IPs, tool names, numbers
- Self-contained: no pronouns — name everything
- Under 200 words total
- Append (3-5 search keywords/synonyms) at the end"""


# ---------------------------------------------------------------------------
# Embedding + similarity
# ---------------------------------------------------------------------------

async def get_embedding(client: httpx.AsyncClient, text: str) -> list[float] | None:
    try:
        resp = await client.post(
            EMBEDDING_URL,
            json={"input": text[:2000], "model": "nomic-embed-text"},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception:
        return None


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.5
    return dot / (na * nb)


def mean_embedding(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    dim = len(embeddings[0])
    result = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            result[i] += v
    return [v / len(embeddings) for v in result]


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def parse_transcript(fpath: str) -> list[dict]:
    """Parse a transcript into ordered messages with text content."""
    try:
        with open(fpath, "rb") as f:
            lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
    except Exception:
        return []

    messages = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type")
        if msg_type == "user":
            content = obj.get("message", {}).get("content", "")
            if isinstance(content, str) and len(content) > 5:
                messages.append({
                    "role": "User",
                    "text": content[:500],
                    "timestamp": obj.get("timestamp", ""),
                    "session_id": obj.get("sessionId", ""),
                })
        elif msg_type == "assistant":
            content = obj.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        messages.append({
                            "role": "Claude",
                            "text": block["text"][:500],
                            "timestamp": obj.get("timestamp", ""),
                            "session_id": obj.get("sessionId", ""),
                        })
                        break

    return messages


# ---------------------------------------------------------------------------
# Episode segmentation
# ---------------------------------------------------------------------------

async def segment_transcript(
    client: httpx.AsyncClient,
    messages: list[dict],
) -> list[list[dict]]:
    """Segment a transcript into episodes using TextTiling with embeddings."""
    episodes = []
    buffer = []
    embeddings = []
    similarities = []

    for msg in messages:
        buffer.append(msg)

        # Only embed user messages (they drive topic)
        if msg["role"] == "User":
            emb = await get_embedding(client, msg["text"])
            if emb:
                embeddings.append(emb)

        # Need enough data for boundary detection
        if len(embeddings) < WINDOW_K * 2:
            continue

        # Compute window similarity
        left = mean_embedding(embeddings[-(WINDOW_K * 2):-WINDOW_K])
        right = mean_embedding(embeddings[-WINDOW_K:])
        sim = cosine_sim(left, right)
        similarities.append(sim)

        if len(similarities) < 3:
            continue

        # Adaptive threshold
        mean_s = sum(similarities) / len(similarities)
        stdev_s = math.sqrt(sum((s - mean_s) ** 2 for s in similarities) / len(similarities))
        threshold = mean_s - stdev_s

        is_boundary = sim < threshold and len(buffer) >= MIN_EPISODE
        force_flush = len(buffer) >= MAX_EPISODE

        if is_boundary or force_flush:
            episodes.append(buffer[:])
            # Overlap
            buffer = buffer[-2:] if len(buffer) > 2 else []
            embeddings = embeddings[-2:] if len(embeddings) > 2 else []
            similarities = []

    # Final buffer
    if len(buffer) >= MIN_EPISODE:
        episodes.append(buffer)

    return episodes


async def summarize_episode(client: httpx.AsyncClient, episode: list[dict]) -> str | None:
    """LLM summarizes an episode into a storable memory."""
    # Cap per-message length based on episode size to stay under ~3500 chars
    per_msg = min(300, 3500 // max(len(episode), 1))
    transcript = "\n".join(f"{m['role']}: {m['text'][:per_msg]}" for m in episode)
    transcript = transcript[:3500]

    try:
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": EPISODE_PROMPT.format(
                    transcript=transcript,
                )}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": 300,
            },
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)

        parts = []
        if parsed.get("topic"):
            parts.append(f"Episode: {parsed['topic']}")
        if parsed.get("actions") and parsed["actions"] != "none":
            parts.append(f"Actions: {parsed['actions']}")
        if parsed.get("outcomes") and parsed["outcomes"] != "none":
            parts.append(f"Outcomes: {parsed['outcomes']}")
        if parsed.get("decisions") and parsed["decisions"] != "none":
            parts.append(f"Decisions: {parsed['decisions']}")
        if parsed.get("open_items") and parsed["open_items"] != "none":
            parts.append(f"Open: {parsed['open_items']}")

        return "\n".join(parts) if parts else None
    except Exception as e:
        print(f"    Summarize error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def evaluate_prompt(
    client: httpx.AsyncClient,
    prompt: str,
    response: str,
    group_id: str = "homelab",
) -> dict:
    """Recall against MemCore and check relevance."""
    try:
        resp = await client.post(
            f"{MEMCORE_URL}/api/recall",
            json={"query": prompt[:300], "group_id": group_id, "limit": 5},
            timeout=10,
        )
        data = resp.json()
        results = data.get("results", [])
        confidence = data.get("confidence", {})

        # Count episode-type memories in results
        episode_count = sum(1 for r in results if "Episode:" in r.get("content", ""))

        return {
            "confidence": confidence.get("level", "unknown"),
            "n_results": len(results),
            "episode_count": episode_count,
            "top_score": results[0].get("final_score", results[0].get("blended_score", 0)) if results else 0,
        }
    except Exception:
        return {"confidence": "error", "n_results": 0, "episode_count": 0, "top_score": 0}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def summarize_and_store(
    client: httpx.AsyncClient,
    episode: list[dict],
    sem: asyncio.Semaphore,
    segment_only: bool = False,
) -> tuple[str | None, bool]:
    """Summarize one episode and store it. Runs under semaphore for concurrency control."""
    async with sem:
        content = await summarize_episode(client, episode)
        if not content:
            return None, False

        if segment_only:
            return content, False

        try:
            resp = await client.post(
                f"{MEMCORE_URL}/api/remember",
                json={
                    "content": content,
                    "group_id": "homelab",
                    "source_agent": "episode-backtest",
                },
                timeout=15,
            )
            stored = resp.json().get("stored", False)
            return content, stored
        except Exception:
            return content, False


async def run_pipeline(
    limit: int = 50,
    segment_only: bool = False,
    eval_only: bool = False,
    max_sessions: int | None = None,
    split_date: str = "2026-04-20",
):
    all_files = sorted(
        glob.glob(os.path.join(TRANSCRIPT_DIR, "*.jsonl")),
        key=os.path.getmtime,
    )

    # Split: older sessions for training, recent for evaluation
    split_ts = datetime.fromisoformat(f"{split_date}T00:00:00").timestamp()
    train_files = [f for f in all_files if os.path.getmtime(f) < split_ts]
    eval_files = [f for f in all_files if os.path.getmtime(f) >= split_ts]

    if max_sessions:
        train_files = train_files[-max_sessions:]  # most recent N older sessions

    print(f"Train sessions (episodes): {len(train_files)} (before {split_date})")
    print(f"Eval sessions (prompts):   {len(eval_files)} (from {split_date}+)")

    # Concurrency limiter for LLM calls
    sem = asyncio.Semaphore(5)

    async with httpx.AsyncClient(timeout=30) as client:

        # --- Phase 1: Segment and store episodes from TRAIN sessions ---
        if not eval_only:
            total_episodes = 0
            total_stored = 0

            print(f"\n{'='*70}")
            print("PHASE 1: Episode Segmentation (train sessions)")
            print(f"{'='*70}")

            for i, fpath in enumerate(train_files):
                fname = os.path.basename(fpath)[:20]
                messages = parse_transcript(fpath)
                if len(messages) < MIN_EPISODE:
                    continue

                episodes = await segment_transcript(client, messages)
                print(f"\n[{i+1}/{len(train_files)}] {fname}... — {len(messages)} msgs → {len(episodes)} episodes")

                # Parallel summarize + store
                tasks = [
                    summarize_and_store(client, ep, sem, segment_only)
                    for ep in episodes
                ]
                results = await asyncio.gather(*tasks)

                for j, (content, stored) in enumerate(results):
                    if content:
                        total_episodes += 1
                        if stored:
                            total_stored += 1
                        print(f"    Ep {j+1}: {'STORED' if stored else 'ok' if content else 'FAIL':6s} {content[:90]}...")

            print(f"\nSegmentation complete: {total_episodes} episodes, {total_stored} stored")

            if segment_only:
                return

        # --- Phase 2: Evaluate recall on EVAL sessions (held out) ---
        print(f"\n{'='*70}")
        print("PHASE 2: Recall Evaluation (eval sessions — held out)")
        print(f"{'='*70}")

        all_prompts = []
        for fpath in eval_files:
            messages = parse_transcript(fpath)
            for k, msg in enumerate(messages):
                if msg["role"] != "User" or len(msg["text"]) < 25:
                    continue
                lower = msg["text"].strip().lower()
                if lower in ("yes", "no", "ok", "sure", "do it", "thanks", "continue", "proceed",
                             "now", "check", "perfect", "sounds good", "lets go", "go ahead"):
                    continue
                if msg["text"].startswith("/") or msg["text"].startswith("!"):
                    continue

                response = ""
                for m in messages[k+1:k+5]:
                    if m["role"] == "Claude":
                        response = m["text"][:300]
                        break

                all_prompts.append({"prompt": msg["text"], "response": response})

        # Sample evenly
        if limit and len(all_prompts) > limit:
            step = len(all_prompts) // limit
            all_prompts = all_prompts[::step][:limit]

        print(f"Evaluating {len(all_prompts)} prompts from held-out sessions")

        conf_dist = defaultdict(int)
        episode_hits = 0
        total_eval = 0

        # Parallel eval
        eval_sem = asyncio.Semaphore(10)

        async def _eval(entry):
            async with eval_sem:
                return await evaluate_prompt(client, entry["prompt"], entry["response"])

        eval_tasks = [_eval(e) for e in all_prompts]
        eval_results = await asyncio.gather(*eval_tasks)

        for i, result in enumerate(eval_results):
            conf = result["confidence"]
            conf_dist[conf] += 1
            if result["episode_count"] > 0:
                episode_hits += 1
            total_eval += 1

        total = total_eval
        coverage = (conf_dist.get("high", 0) + conf_dist.get("moderate", 0)) / max(total, 1)

        print(f"\n{'='*70}")
        print(f"RESULTS — {total} prompts from held-out eval sessions")
        print(f"{'='*70}")
        print(f"  Coverage (moderate+): {coverage*100:.0f}%")
        print(f"  Episode memories in results: {episode_hits}/{total} ({episode_hits/max(total,1)*100:.0f}%)")
        print(f"\n  Confidence distribution:")
        for level in ["high", "moderate", "stale", "weak", "very_weak", "no_memory"]:
            cnt = conf_dist.get(level, 0)
            if cnt:
                bar = "#" * int(cnt / max(total, 1) * 40)
                print(f"    {level:12s}: {cnt:3d} ({cnt/max(total,1)*100:4.0f}%) {bar}")

        output = os.path.join(os.path.dirname(__file__), "episode_backtest_results.json")
        with open(output, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "split_date": split_date,
                "train_sessions": len(train_files),
                "eval_sessions": len(eval_files),
                "total_prompts": total,
                "coverage": coverage,
                "episode_hits": episode_hits,
                "confidence_distribution": dict(conf_dist),
            }, f, indent=2)
        print(f"\nSaved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Backtest episode segmentation")
    parser.add_argument("--limit", type=int, default=50, help="Prompts to evaluate")
    parser.add_argument("--max-sessions", type=int, help="Max session files to process")
    parser.add_argument("--segment-only", action="store_true", help="Only segment, don't store or evaluate")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate (episodes already stored)")
    args = parser.parse_args()

    asyncio.run(run_pipeline(
        limit=args.limit,
        segment_only=args.segment_only,
        eval_only=args.eval_only,
        max_sessions=args.max_sessions,
    ))


if __name__ == "__main__":
    main()
