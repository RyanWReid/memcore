#!/usr/bin/env python3
"""MemCore Production Evaluation Framework

Beyond cold-start benchmarks: measure whether memory actually helps over time.

Three evaluation modes:

1. RETROSPECTIVE — Run against historical transcripts
   "Would MemCore have helped in past sessions?"
   Scores: recall coverage, relevance, precision, missing gaps

2. LIVE — Hooks into Claude Code session lifecycle
   "Is MemCore helping right now?"
   Scores: per-prompt relevance, MW validation, latency

3. LONGITUDINAL — Weekly trend analysis
   "Is MemCore getting better?"
   Scores: MW convergence, reconsolidation effectiveness, stability distributions

This is NOT a retrieval benchmark. It measures whether the memory system
makes the AI assistant produce better responses — the actual goal.

Metrics:
  - Would-Help Rate:     % of prompts where recalled memories improve the response
  - Precision@5:         % of recalled memories that are relevant (vs noise)
  - Coverage:            % of prompts where MemCore has something useful
  - Missing Memory Rate: % of prompts where useful context exists but wasn't stored
  - Lifecycle Lift:      Do high-MW / reconsolidated memories correlate with relevance?
  - Signal-to-Noise:     Ratio of relevant to noise memories per recall

Usage:
    # Retrospective: test 100 real prompts from past sessions
    python -m memcore.benchmark.production_eval retro --limit 100

    # Longitudinal: compare this week vs last week
    python -m memcore.benchmark.production_eval trend

    # Quick health check: 20 prompts, no LLM judge
    python -m memcore.benchmark.production_eval health
"""

import argparse
import asyncio
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import httpx

MEMCORE_URL = os.getenv("MEMCORE_URL", "http://localhost:8020")
LITELLM_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_API_KEY", "")
MODEL = os.getenv("BENCHMARK_MODEL", "deepseek-chat")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "eval_results")

TRANSCRIPT_DIR = os.path.expanduser(
    os.getenv("TRANSCRIPT_DIR", "~/.claude/projects/-Users-ryanreid-Development-Projects-homelab")
)

# ---------------------------------------------------------------------------
# Transcript extraction
# ---------------------------------------------------------------------------

SKIP_PATTERNS = re.compile(
    r"^(yes|no|ok|sure|do it|go ahead|thanks|good|great|continue|proceed|next|"
    r"done|stop|wait|nevermind|compact|now|check|lets go|perfect|sounds good|"
    r"whats next|anything else)$",
    re.IGNORECASE,
)


def extract_prompts(
    transcript_dir: str = TRANSCRIPT_DIR,
    min_length: int = 25,
    max_age_days: int | None = None,
    session_id: str | None = None,
) -> list[dict]:
    """Extract user prompt + assistant response pairs from transcripts.

    Returns chronologically ordered list of:
    {
        "prompt": str,
        "response": str,         # first 500 chars of assistant reply
        "timestamp": datetime,
        "session_id": str,
        "has_tool_use": bool,    # did Claude use tools in the response?
    }
    """
    if session_id:
        files = [os.path.join(transcript_dir, f"{session_id}.jsonl")]
    else:
        files = sorted(
            glob.glob(os.path.join(transcript_dir, "*.jsonl")),
            key=os.path.getmtime,
            reverse=True,
        )

    cutoff = None
    if max_age_days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    entries = []

    for fpath in files:
        if not fpath.endswith(".jsonl"):
            continue

        try:
            with open(fpath, "rb") as f:
                lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
        except Exception:
            continue

        messages = []
        for line in lines:
            try:
                obj = json.loads(line)
                if obj.get("type") in ("user", "assistant"):
                    messages.append(obj)
            except json.JSONDecodeError:
                continue

        for i, msg in enumerate(messages):
            if msg.get("type") != "user":
                continue

            prompt = msg.get("message", {}).get("content", "")
            if not isinstance(prompt, str) or len(prompt) < min_length:
                continue
            if SKIP_PATTERNS.match(prompt.strip()):
                continue
            if prompt.startswith("/") or prompt.startswith("!"):
                continue

            # Parse timestamp
            ts_str = msg.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)

            if cutoff and ts < cutoff:
                continue

            # Find assistant response
            response = ""
            has_tool_use = False
            for j in range(i + 1, min(i + 10, len(messages))):
                if messages[j].get("type") == "assistant":
                    content = messages[j].get("message", {}).get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text" and not response:
                                    response = block["text"][:500]
                                if block.get("type") == "tool_use":
                                    has_tool_use = True
                    break

            entries.append({
                "prompt": prompt,
                "response": response,
                "timestamp": ts,
                "session_id": msg.get("sessionId", os.path.basename(fpath).replace(".jsonl", "")),
                "has_tool_use": has_tool_use,
            })

    # Sort chronologically
    entries.sort(key=lambda e: e["timestamp"])
    return entries


# ---------------------------------------------------------------------------
# MemCore recall
# ---------------------------------------------------------------------------

async def recall(client: httpx.AsyncClient, query: str, group_id: str = "homelab") -> dict:
    """Call MemCore recall. Returns full response including lifecycle signals."""
    try:
        resp = await client.post(
            f"{MEMCORE_URL}/api/recall",
            json={"query": query[:300], "group_id": group_id, "limit": 5},
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        return {"results": [], "confidence": {"level": "error"}, "error": str(e)}


# ---------------------------------------------------------------------------
# Relevance judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """Rate whether these retrieved memories help with the user's prompt.

Prompt: {prompt}
Response excerpt: {response}

Memories:
{memories}

For EACH memory (by index), output one of:
- ESSENTIAL: directly answers or informs the response
- HELPFUL: related context that improves the response
- TANGENTIAL: on-topic but not needed
- NOISE: irrelevant

Also answer:
- would_help: Would having these memories make the response better? (true/false)
- missing: What information SHOULD have been recalled but wasn't? (or "none")
- best_idx: Index of most useful memory (-1 if all noise)

JSON output:
{{"ratings": ["ESSENTIAL", "HELPFUL", ...], "would_help": true, "missing": "none", "best_idx": 0}}"""


async def judge_relevance(
    client: httpx.AsyncClient,
    prompt: str,
    response: str,
    memories: list[dict],
) -> dict:
    """LLM judges memory relevance. Returns structured evaluation."""
    if not memories:
        return {
            "ratings": [],
            "would_help": False,
            "missing": "no memories recalled",
            "best_idx": -1,
        }

    mem_lines = []
    for i, m in enumerate(memories):
        score = m.get("final_score", m.get("blended_score", 0))
        mem_lines.append(
            f"[{i}] (type={m.get('memory_type', '?')}, score={score:.3f}, "
            f"access={m.get('access_count', 0)}, stability={m.get('stability', 1):.1f}, "
            f"recon={m.get('reconsolidation_count', 0)}, "
            f"mw={m.get('mw_success', 0)}/{m.get('mw_total', 0)}) "
            f"{m.get('content', '')[:200]}"
        )

    try:
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                    prompt=prompt[:300],
                    response=response[:400],
                    memories="\n".join(mem_lines),
                )}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": 200,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_lifecycle_correlation(evaluations: list[dict]) -> dict:
    """Compute whether lifecycle signals predict relevance.

    The key question: do high-MW, reconsolidated, high-stability memories
    tend to be rated ESSENTIAL/HELPFUL more often than others?
    """
    lifecycle_bins = {
        "high_stability": {"relevant": 0, "noise": 0},  # stability > 3
        "low_stability": {"relevant": 0, "noise": 0},
        "reconsolidated": {"relevant": 0, "noise": 0},
        "not_reconsolidated": {"relevant": 0, "noise": 0},
        "high_access": {"relevant": 0, "noise": 0},  # access_count > 5
        "low_access": {"relevant": 0, "noise": 0},
        "high_mw": {"relevant": 0, "noise": 0},  # mw_success/mw_total > 0.5
        "low_mw": {"relevant": 0, "noise": 0},
    }

    for ev in evaluations:
        ratings = ev.get("ratings", [])
        memories = ev.get("memories", [])

        for rating, mem in zip(ratings, memories):
            is_relevant = rating in ("ESSENTIAL", "HELPFUL")
            bucket = "relevant" if is_relevant else "noise"

            stability = mem.get("stability", 1)
            if stability > 3:
                lifecycle_bins["high_stability"][bucket] += 1
            else:
                lifecycle_bins["low_stability"][bucket] += 1

            if mem.get("reconsolidation_count", 0) > 0:
                lifecycle_bins["reconsolidated"][bucket] += 1
            else:
                lifecycle_bins["not_reconsolidated"][bucket] += 1

            if mem.get("access_count", 0) > 5:
                lifecycle_bins["high_access"][bucket] += 1
            else:
                lifecycle_bins["low_access"][bucket] += 1

            mw_total = mem.get("mw_total", 0)
            mw_success = mem.get("mw_success", 0)
            if mw_total > 3 and mw_success / mw_total > 0.5:
                lifecycle_bins["high_mw"][bucket] += 1
            elif mw_total > 3:
                lifecycle_bins["low_mw"][bucket] += 1

    # Compute relevance rates per bin
    result = {}
    for label, counts in lifecycle_bins.items():
        total = counts["relevant"] + counts["noise"]
        if total > 0:
            result[label] = {
                "relevance_rate": counts["relevant"] / total,
                "total": total,
                "relevant": counts["relevant"],
            }

    return result


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

async def run_retrospective(limit: int = 50, max_age_days: int | None = None):
    """Test MemCore against historical transcripts."""
    entries = extract_prompts(max_age_days=max_age_days)
    if limit:
        # Sample evenly across sessions rather than just taking first N
        sessions = defaultdict(list)
        for e in entries:
            sessions[e["session_id"]].append(e)
        sampled = []
        per_session = max(1, limit // len(sessions)) if sessions else limit
        for sid, session_entries in sessions.items():
            sampled.extend(session_entries[:per_session])
        entries = sorted(sampled, key=lambda e: e["timestamp"])[:limit]

    print(f"Retrospective eval: {len(entries)} prompts from {len(set(e['session_id'] for e in entries))} sessions")
    print("=" * 70)

    evaluations = []
    conf_dist = defaultdict(int)
    t_start = time.time()

    async with httpx.AsyncClient(timeout=30) as client:
        for i, entry in enumerate(entries):
            # Recall
            result = await recall(client, entry["prompt"])
            memories = result.get("results", [])
            conf = result.get("confidence", {}).get("level", "unknown")
            conf_dist[conf] += 1

            # Judge
            judgment = await judge_relevance(
                client, entry["prompt"], entry["response"], memories,
            )

            # Attach memories to judgment for lifecycle analysis
            judgment["memories"] = memories
            judgment["confidence"] = conf
            judgment["prompt"] = entry["prompt"][:100]
            evaluations.append(judgment)

            # Progress
            ratings_str = ",".join(judgment.get("ratings", [])[:5])
            would_help = judgment.get("would_help", "?")
            print(f"  [{i+1}/{len(entries)}] {conf:10s} | {ratings_str:40s} | help={would_help} | {entry['prompt'][:50]}...")

    elapsed = time.time() - t_start

    # Compute metrics
    total = len(evaluations)
    valid = [e for e in evaluations if "ratings" in e]

    would_help_count = sum(1 for e in valid if e.get("would_help"))
    all_ratings = [r for e in valid for r in e.get("ratings", [])]
    essential = all_ratings.count("ESSENTIAL")
    helpful = all_ratings.count("HELPFUL")
    tangential = all_ratings.count("TANGENTIAL")
    noise = all_ratings.count("NOISE")
    total_ratings = len(all_ratings)

    missing = [e["missing"] for e in valid if e.get("missing") and e["missing"] != "none"]

    lifecycle = compute_lifecycle_correlation(valid)

    # Print report
    print(f"\n{'=' * 70}")
    print(f"RETROSPECTIVE EVALUATION — {total} prompts, {elapsed:.0f}s")
    print(f"{'=' * 70}")

    print(f"\n## Core Metrics")
    print(f"  Would-Help Rate:  {would_help_count}/{len(valid)} ({would_help_count/max(len(valid),1)*100:.0f}%)")
    if total_ratings:
        print(f"  Precision@5:      {(essential+helpful)/total_ratings*100:.0f}% relevant ({essential} essential + {helpful} helpful)")
        print(f"  Signal-to-Noise:  {(essential+helpful)/max(noise,1):.1f}:1")
    print(f"  Coverage:         {conf_dist.get('high',0)+conf_dist.get('moderate',0)}/{total} ({(conf_dist.get('high',0)+conf_dist.get('moderate',0))/max(total,1)*100:.0f}%) moderate+ confidence")

    print(f"\n## Confidence Distribution")
    for level in ["high", "moderate", "stale", "weak", "very_weak", "no_memory"]:
        cnt = conf_dist.get(level, 0)
        if cnt > 0:
            bar = "#" * int(cnt / max(total, 1) * 40)
            print(f"    {level:12s}: {cnt:3d} ({cnt/max(total,1)*100:4.0f}%) {bar}")

    if total_ratings:
        print(f"\n## Rating Breakdown ({total_ratings} total)")
        for label, count in [("ESSENTIAL", essential), ("HELPFUL", helpful),
                             ("TANGENTIAL", tangential), ("NOISE", noise)]:
            bar = "#" * int(count / max(total_ratings, 1) * 40)
            print(f"    {label:12s}: {count:3d} ({count/total_ratings*100:4.0f}%) {bar}")

    if lifecycle:
        print(f"\n## Lifecycle Signal Correlation")
        print(f"  {'Signal':<25s} {'Relevance Rate':>15s} {'Sample':>8s}")
        for label, data in sorted(lifecycle.items(), key=lambda x: -x[1]["relevance_rate"]):
            rate = data["relevance_rate"]
            n = data["total"]
            bar = "#" * int(rate * 20)
            print(f"    {label:<23s} {rate:>14.0%} {n:>7d}  {bar}")

    if missing:
        print(f"\n## Memory Gaps ({len(missing)} prompts missing useful context)")
        for m in missing[:10]:
            print(f"    - {m[:120]}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = os.path.join(RESULTS_DIR, f"retro_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(output, "w") as f:
        json.dump({
            "mode": "retrospective",
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "would_help_rate": would_help_count / max(len(valid), 1),
            "precision": (essential + helpful) / max(total_ratings, 1),
            "coverage": (conf_dist.get("high", 0) + conf_dist.get("moderate", 0)) / max(total, 1),
            "confidence_distribution": dict(conf_dist),
            "rating_breakdown": {"essential": essential, "helpful": helpful, "tangential": tangential, "noise": noise},
            "lifecycle_correlation": lifecycle,
            "missing_memories": missing[:20],
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"\nSaved to {output}")


async def run_health(limit: int = 20):
    """Quick health check — recall only, no LLM judge."""
    entries = extract_prompts(max_age_days=7)[:limit]
    print(f"Health check: {len(entries)} recent prompts (no LLM judge)")
    print("=" * 70)

    conf_dist = defaultdict(int)
    latencies = []

    async with httpx.AsyncClient(timeout=15) as client:
        for i, entry in enumerate(entries):
            t0 = time.time()
            result = await recall(client, entry["prompt"])
            latency = (time.time() - t0) * 1000

            conf = result.get("confidence", {}).get("level", "unknown")
            conf_dist[conf] += 1
            latencies.append(latency)
            n_results = len(result.get("results", []))

            print(f"  [{i+1}/{len(entries)}] {conf:10s} {latency:5.0f}ms {n_results} results | {entry['prompt'][:60]}...")

    total = len(entries)
    coverage = (conf_dist.get("high", 0) + conf_dist.get("moderate", 0)) / max(total, 1)
    avg_latency = sum(latencies) / max(len(latencies), 1)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    print(f"\n{'=' * 70}")
    print(f"HEALTH CHECK")
    print(f"  Coverage (moderate+): {coverage*100:.0f}%")
    print(f"  Avg latency:         {avg_latency:.0f}ms")
    print(f"  P95 latency:         {p95_latency:.0f}ms")
    for level in ["high", "moderate", "stale", "weak", "very_weak", "no_memory"]:
        cnt = conf_dist.get(level, 0)
        if cnt > 0:
            print(f"    {level:12s}: {cnt}")


async def run_trend():
    """Compare evaluation results over time."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "retro_*.json")))

    if len(result_files) < 2:
        print("Need at least 2 retrospective runs to show trends.")
        print(f"Run: python -m memcore.benchmark.production_eval retro --limit 50")
        print("Then run again in a few days to compare.")
        return

    print(f"Found {len(result_files)} evaluation snapshots")
    print("=" * 70)
    print(f"  {'Date':<20s} {'Would-Help':>12s} {'Precision':>12s} {'Coverage':>12s} {'S/N Ratio':>12s}")

    for fpath in result_files:
        with open(fpath) as f:
            data = json.load(f)
        ts = data.get("timestamp", "?")[:16]
        wh = data.get("would_help_rate", 0)
        prec = data.get("precision", 0)
        cov = data.get("coverage", 0)
        ratings = data.get("rating_breakdown", {})
        relevant = ratings.get("essential", 0) + ratings.get("helpful", 0)
        noise_count = ratings.get("noise", 0)
        sn = relevant / max(noise_count, 1)
        print(f"  {ts:<20s} {wh:>11.0%} {prec:>11.0%} {cov:>11.0%} {sn:>10.1f}:1")

    # Compare latest vs earliest
    with open(result_files[0]) as f:
        first = json.load(f)
    with open(result_files[-1]) as f:
        latest = json.load(f)

    print(f"\n## Trend (oldest → newest)")
    for metric in ["would_help_rate", "precision", "coverage"]:
        old = first.get(metric, 0)
        new = latest.get(metric, 0)
        delta = new - old
        arrow = "^" if delta > 0 else "v" if delta < 0 else "="
        print(f"  {metric:<20s}: {old:.0%} → {new:.0%} ({arrow} {abs(delta):.0%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MemCore Production Evaluation")
    sub = parser.add_subparsers(dest="mode", required=True)

    retro = sub.add_parser("retro", help="Retrospective eval against transcripts")
    retro.add_argument("--limit", type=int, default=50)
    retro.add_argument("--max-age-days", type=int, default=None)

    health = sub.add_parser("health", help="Quick health check (no LLM judge)")
    health.add_argument("--limit", type=int, default=20)

    sub.add_parser("trend", help="Compare evaluation results over time")

    args = parser.parse_args()

    if args.mode == "retro":
        asyncio.run(run_retrospective(limit=args.limit, max_age_days=args.max_age_days))
    elif args.mode == "health":
        asyncio.run(run_health(limit=args.limit))
    elif args.mode == "trend":
        asyncio.run(run_trend())


if __name__ == "__main__":
    main()
