#!/usr/bin/env python3
"""Backtest episode buffer enrichment vs baseline.

For each eval prompt, reconstruct the episode buffer (6 messages preceding
it in the transcript) and run TWO recalls:
  1. Baseline: prompt alone (matches current 28% measurement)
  2. Enriched: buffer + prompt (simulates the hook change)

Compare coverage and relevance between the two conditions.
This isolates the impact of query enrichment from everything else.
"""

import asyncio
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import httpx

MEMCORE_URL = os.getenv("MEMCORE_URL", "http://localhost:8020")
TRANSCRIPT_DIR = os.path.expanduser(
    "~/.claude/projects/-Users-ryanreid-Development-Projects-homelab"
)
SPLIT_DATE = "2026-04-20"  # eval sessions from this date+


def parse_transcript(fpath: str) -> list[dict]:
    """Parse transcript into ordered user/assistant text messages."""
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
    """Reconstruct what the episode buffer would look like at message index `idx`.

    Returns the last `window` messages before idx, concatenated.
    """
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
            "top_score": data.get("results", [{}])[0].get("final_score",
                         data.get("results", [{}])[0].get("blended_score", 0)) if data.get("results") else 0,
        }
    except Exception:
        return {"confidence": "error", "n_results": 0, "top_score": 0}


async def run():
    # Get eval sessions (Apr 20+)
    split_ts = datetime.fromisoformat(f"{SPLIT_DATE}T00:00:00").timestamp()
    all_files = sorted(
        glob.glob(os.path.join(TRANSCRIPT_DIR, "*.jsonl")),
        key=os.path.getmtime,
    )
    eval_files = [f for f in all_files if os.path.getmtime(f) >= split_ts]
    print(f"Eval sessions: {len(eval_files)}")

    # Extract prompts with their context
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

            context = build_buffer_context(messages, k, window=6)
            prompts_with_context.append({
                "prompt": msg["text"],
                "context": context,
            })

    # Sample 50 evenly
    if len(prompts_with_context) > 50:
        step = len(prompts_with_context) // 50
        prompts_with_context = prompts_with_context[::step][:50]

    print(f"Testing {len(prompts_with_context)} prompts in both conditions\n")

    baseline_dist = defaultdict(int)
    enriched_dist = defaultdict(int)
    improvements = []  # prompts that went from weak→moderate/high
    regressions = []
    same = 0

    async with httpx.AsyncClient(timeout=15) as client:
        sem = asyncio.Semaphore(5)

        async def test_prompt(entry):
            async with sem:
                # Baseline: prompt alone
                baseline = await recall(client, entry["prompt"])
                # Enriched: context + prompt
                enriched = await recall(client, f"{entry['context']} {entry['prompt']}")
                return entry, baseline, enriched

        tasks = [test_prompt(e) for e in prompts_with_context]
        results = await asyncio.gather(*tasks)

    rank = {"high": 4, "moderate": 3, "stale": 2, "weak": 1, "very_weak": 0, "no_memory": 0, "error": -1}

    for entry, baseline, enriched in results:
        baseline_dist[baseline["confidence"]] += 1
        enriched_dist[enriched["confidence"]] += 1

        b_rank = rank.get(baseline["confidence"], 0)
        e_rank = rank.get(enriched["confidence"], 0)

        if e_rank > b_rank:
            improvements.append({
                "prompt": entry["prompt"][:80],
                "from": baseline["confidence"],
                "to": enriched["confidence"],
            })
        elif e_rank < b_rank:
            regressions.append({
                "prompt": entry["prompt"][:80],
                "from": baseline["confidence"],
                "to": enriched["confidence"],
            })
        else:
            same += 1

    total = len(results)

    def coverage(dist):
        return (dist.get("high", 0) + dist.get("moderate", 0)) / max(total, 1)

    print("=" * 70)
    print(f"RESULTS — {total} prompts, baseline vs enriched")
    print("=" * 70)

    print(f"\n## Coverage (moderate+ confidence)")
    print(f"  Baseline (prompt alone):       {coverage(baseline_dist)*100:5.1f}%")
    print(f"  Enriched (buffer + prompt):    {coverage(enriched_dist)*100:5.1f}%")
    print(f"  Delta:                         {(coverage(enriched_dist) - coverage(baseline_dist))*100:+5.1f}pp")

    print(f"\n## Confidence distribution")
    print(f"  {'Level':<12s} {'Baseline':>10s} {'Enriched':>10s}")
    for level in ["high", "moderate", "stale", "weak", "very_weak", "no_memory", "error"]:
        b = baseline_dist.get(level, 0)
        e = enriched_dist.get(level, 0)
        if b or e:
            arrow = " ^" if e > b else " v" if e < b else "  "
            print(f"  {level:<12s} {b:>10d} {e:>10d}{arrow}")

    print(f"\n## Per-prompt changes")
    print(f"  Improved:   {len(improvements)} ({len(improvements)/total*100:.0f}%)")
    print(f"  Same:       {same} ({same/total*100:.0f}%)")
    print(f"  Regressed:  {len(regressions)} ({len(regressions)/total*100:.0f}%)")

    if improvements:
        print(f"\n## Sample improvements")
        for imp in improvements[:8]:
            print(f"  {imp['from']:10s} → {imp['to']:10s}  {imp['prompt']}")

    if regressions:
        print(f"\n## Sample regressions")
        for reg in regressions[:5]:
            print(f"  {reg['from']:10s} → {reg['to']:10s}  {reg['prompt']}")

    # Save
    output = os.path.join(os.path.dirname(__file__), "enrichment_backtest.json")
    with open(output, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "baseline_coverage": coverage(baseline_dist),
            "enriched_coverage": coverage(enriched_dist),
            "baseline_distribution": dict(baseline_dist),
            "enriched_distribution": dict(enriched_dist),
            "improvements": len(improvements),
            "regressions": len(regressions),
            "same": same,
        }, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    asyncio.run(run())
