#!/usr/bin/env python3
"""Backtest dual-query strategy: baseline first, retry with enrichment only if weak.

Compares three conditions:
  1. Baseline: prompt alone
  2. Enriched: buffer + prompt (always)
  3. Dual: baseline first, retry with enrichment only if weak/very_weak/no_memory

Expected: Dual captures most of enriched's gains without the 20% regression rate.
"""

import asyncio
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import httpx

MEMCORE_URL = os.getenv("MEMCORE_URL", "http://localhost:8020")
TRANSCRIPT_DIR = os.path.expanduser(
    "~/.claude/projects/-Users-ryanreid-Development-Projects-homelab"
)
SPLIT_DATE = "2026-04-20"


def parse_transcript(fpath: str) -> list[dict]:
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
        t = obj.get("type")
        if t == "user":
            content = obj.get("message", {}).get("content", "")
            if isinstance(content, str) and len(content) > 5:
                messages.append({"role": "User", "text": content[:500]})
        elif t == "assistant":
            content = obj.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        messages.append({"role": "Claude", "text": block["text"][:500]})
                        break
    return messages


def build_buffer_context(messages: list[dict], idx: int, window: int = 6) -> str:
    start = max(0, idx - window)
    recent = messages[start:idx]
    parts = [m["text"][:100] for m in recent]
    return " ".join(parts)[:250]


async def recall(client: httpx.AsyncClient, query: str) -> dict:
    try:
        resp = await client.post(
            f"{MEMCORE_URL}/api/recall",
            json={"query": query[:400], "group_id": "homelab", "limit": 5},
            timeout=10,
        )
        data = resp.json()
        return {
            "confidence": data.get("confidence", {}).get("level", "unknown"),
            "n_results": len(data.get("results", [])),
        }
    except Exception:
        return {"confidence": "error", "n_results": 0}


async def dual_query(client: httpx.AsyncClient, prompt: str, context: str) -> dict:
    """Baseline first, retry with enrichment only if weak."""
    baseline = await recall(client, prompt)
    weak_levels = ("weak", "very_weak", "no_memory", "error")
    if baseline["confidence"] in weak_levels and context:
        enriched = await recall(client, f"{context} {prompt}")
        # Use enriched if it's better
        rank = {"high": 4, "moderate": 3, "stale": 2, "weak": 1, "very_weak": 0, "no_memory": 0, "error": -1}
        if rank.get(enriched["confidence"], -1) > rank.get(baseline["confidence"], -1):
            return enriched
    return baseline


async def run():
    split_ts = datetime.fromisoformat(f"{SPLIT_DATE}T00:00:00").timestamp()
    all_files = sorted(
        glob.glob(os.path.join(TRANSCRIPT_DIR, "*.jsonl")),
        key=os.path.getmtime,
    )
    eval_files = [f for f in all_files if os.path.getmtime(f) >= split_ts]

    prompts_with_context = []
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
            prompts_with_context.append({
                "prompt": msg["text"],
                "context": build_buffer_context(messages, k, window=6),
            })

    if len(prompts_with_context) > 50:
        step = len(prompts_with_context) // 50
        prompts_with_context = prompts_with_context[::step][:50]

    print(f"Testing {len(prompts_with_context)} prompts in 3 conditions\n")

    baseline_dist = defaultdict(int)
    enriched_dist = defaultdict(int)
    dual_dist = defaultdict(int)
    extra_calls = 0

    async with httpx.AsyncClient(timeout=15) as client:
        sem = asyncio.Semaphore(5)

        async def test_one(entry):
            async with sem:
                b = await recall(client, entry["prompt"])
                e = await recall(client, f"{entry['context']} {entry['prompt']}")
                d = await dual_query(client, entry["prompt"], entry["context"])
                # Track if dual actually made a second call
                used_enrichment = b["confidence"] != d["confidence"]
                return b, e, d, used_enrichment

        tasks = [test_one(e) for e in prompts_with_context]
        results = await asyncio.gather(*tasks)

    for b, e, d, used_enrich in results:
        baseline_dist[b["confidence"]] += 1
        enriched_dist[e["confidence"]] += 1
        dual_dist[d["confidence"]] += 1
        if used_enrich:
            extra_calls += 1

    total = len(results)

    def coverage(dist):
        return (dist.get("high", 0) + dist.get("moderate", 0)) / max(total, 1)

    print("=" * 70)
    print(f"RESULTS — {total} prompts")
    print("=" * 70)

    print(f"\n## Coverage (moderate+ confidence)")
    print(f"  Baseline (prompt alone):          {coverage(baseline_dist)*100:5.1f}%")
    print(f"  Enriched (always buffer+prompt):  {coverage(enriched_dist)*100:5.1f}%")
    print(f"  Dual-query (fallback only):       {coverage(dual_dist)*100:5.1f}%")

    print(f"\n## Distribution")
    print(f"  {'Level':<12s} {'Baseline':>10s} {'Enriched':>10s} {'Dual':>10s}")
    for level in ["high", "moderate", "stale", "weak", "very_weak", "no_memory", "error"]:
        b = baseline_dist.get(level, 0)
        e = enriched_dist.get(level, 0)
        d = dual_dist.get(level, 0)
        if b or e or d:
            print(f"  {level:<12s} {b:>10d} {e:>10d} {d:>10d}")

    print(f"\n## Efficiency")
    print(f"  Dual required 2nd call:  {extra_calls}/{total} ({extra_calls/total*100:.0f}%)")
    print(f"  Avg extra calls per prompt: {extra_calls/total:.2f}")

    output = os.path.join(os.path.dirname(__file__), "dual_query_backtest.json")
    with open(output, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "baseline_coverage": coverage(baseline_dist),
            "enriched_coverage": coverage(enriched_dist),
            "dual_coverage": coverage(dual_dist),
            "dual_extra_calls": extra_calls,
            "baseline_distribution": dict(baseline_dist),
            "enriched_distribution": dict(enriched_dist),
            "dual_distribution": dict(dual_dist),
        }, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    asyncio.run(run())
