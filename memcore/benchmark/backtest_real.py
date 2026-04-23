#!/usr/bin/env python3
"""Backtest MemCore against real Claude Code conversation transcripts.

Instead of synthetic benchmarks, this tests whether MemCore's stored memories
actually help with real prompts from real sessions.

For each user prompt:
1. Recall memories from MemCore (same as the recall hook would)
2. Check if recalled memories are relevant to what Claude actually discussed
3. Score lifecycle signals: MW, stability, reconsolidation, suppression

This answers: "Is MemCore making Claude better?" with real data.

Usage:
    python -m memcore.benchmark.backtest_real                     # All transcripts
    python -m memcore.benchmark.backtest_real --limit 50          # First 50 prompts
    python -m memcore.benchmark.backtest_real --session <id>      # Specific session
    python -m memcore.benchmark.backtest_real --min-length 30     # Skip short prompts
"""

import argparse
import asyncio
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

import httpx

MEMCORE_URL = os.getenv("MEMCORE_URL", "http://localhost:8020")
LITELLM_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_API_KEY", "")
MODEL = os.getenv("BENCHMARK_MODEL", "deepseek-chat")

TRANSCRIPT_DIR = os.path.expanduser(
    "~/.claude/projects/-Users-ryanreid-Development-Projects-homelab"
)

# Prompts that are just conversational noise — skip them
SKIP_PATTERNS = [
    r"^(yes|no|ok|sure|do it|go ahead|thanks|good|great|continue|proceed|next|done|stop|wait|nevermind|compact|now|check|lets go|perfect|sounds good)$",
    r"^/",  # slash commands
    r"^!",  # shell commands
]


def load_transcripts(session_id: str | None = None) -> list[dict]:
    """Load user prompts + following assistant responses from transcripts.

    Returns list of:
    {
        "prompt": str,
        "response": str (first 500 chars of assistant reply),
        "timestamp": str,
        "session_id": str,
        "file": str,
    }
    """
    if session_id:
        pattern = os.path.join(TRANSCRIPT_DIR, f"{session_id}.jsonl")
    else:
        pattern = os.path.join(TRANSCRIPT_DIR, "*.jsonl")

    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    entries = []

    for fpath in files:
        fname = os.path.basename(fpath)
        if fname == "MEMORY.md" or not fname.endswith(".jsonl"):
            continue

        try:
            with open(fpath, "rb") as f:
                lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
        except Exception:
            continue

        # Parse into ordered messages
        messages = []
        for line in lines:
            try:
                obj = json.loads(line)
                msg_type = obj.get("type")
                if msg_type in ("user", "assistant"):
                    messages.append(obj)
            except json.JSONDecodeError:
                continue

        # Pair user prompts with following assistant responses
        for i, msg in enumerate(messages):
            if msg.get("type") != "user":
                continue

            prompt = msg.get("message", {}).get("content", "")
            if not isinstance(prompt, str):
                continue

            # Find following assistant response
            response = ""
            for j in range(i + 1, min(i + 5, len(messages))):
                if messages[j].get("type") == "assistant":
                    content = messages[j].get("message", {}).get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                response = block["text"][:500]
                                break
                    break

            entries.append({
                "prompt": prompt,
                "response": response,
                "timestamp": msg.get("timestamp", ""),
                "session_id": msg.get("sessionId", fname.replace(".jsonl", "")),
                "file": fname,
            })

    return entries


def filter_prompts(entries: list[dict], min_length: int = 20) -> list[dict]:
    """Filter out noise: short prompts, commands, chatter."""
    filtered = []
    for e in entries:
        prompt = e["prompt"].strip()

        # Length filter
        if len(prompt) < min_length:
            continue

        # Pattern filter
        lower = prompt.lower().strip()
        skip = False
        for pat in SKIP_PATTERNS:
            if re.match(pat, lower):
                skip = True
                break
        if skip:
            continue

        filtered.append(e)

    return filtered


RELEVANCE_PROMPT = """You are evaluating whether retrieved memories are relevant to a user's prompt and the assistant's response.

User prompt: {prompt}

Assistant response (first 500 chars): {response}

Retrieved memories:
{memories}

For each memory, rate relevance:
- HIGH: Memory directly helps answer the prompt or contains information the assistant used
- MEDIUM: Memory is related to the topic but wasn't directly needed
- LOW: Memory is not relevant to this prompt
- NOISE: Memory is completely unrelated

Output JSON:
{{
  "relevant_count": <number of HIGH + MEDIUM>,
  "high_count": <number of HIGH>,
  "noise_count": <number of LOW + NOISE>,
  "best_memory_idx": <0-indexed, which memory was most useful, or -1>,
  "would_help": <true if having these memories would improve the response>,
  "missing": "<what memory SHOULD have been recalled but wasn't, or 'none'>"
}}"""


async def evaluate_recall(
    client: httpx.AsyncClient,
    prompt: str,
    response: str,
    memories: list[dict],
) -> dict:
    """LLM judges whether recalled memories are relevant to the prompt+response."""
    if not memories:
        return {
            "relevant_count": 0,
            "high_count": 0,
            "noise_count": 0,
            "best_memory_idx": -1,
            "would_help": False,
            "missing": "no memories recalled",
        }

    mem_text = "\n".join(
        f"[{i}] ({m.get('memory_type', '?')}, score={m.get('final_score', m.get('blended_score', 0)):.3f}, "
        f"access={m.get('access_count', 0)}, stability={m.get('stability', 1):.1f}, "
        f"mw={m.get('mw_success', 0)}/{m.get('mw_total', 0)}) "
        f"{m.get('content', '')[:200]}"
        for i, m in enumerate(memories)
    )

    try:
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": RELEVANCE_PROMPT.format(
                        prompt=prompt[:300],
                        response=response[:500],
                        memories=mem_text,
                    )},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": 200,
            },
            timeout=15,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


async def recall_memories(client: httpx.AsyncClient, query: str, group_id: str = "homelab") -> tuple[list[dict], dict]:
    """Call MemCore recall API. Returns (results, confidence)."""
    try:
        resp = await client.post(
            f"{MEMCORE_URL}/api/recall",
            json={"query": query[:300], "group_id": group_id, "limit": 5},
            timeout=10,
        )
        data = resp.json()
        results = data.get("results", [])
        confidence = data.get("confidence", {})
        return results, confidence
    except Exception:
        return [], {"level": "error"}


async def run_backtest(
    limit: int | None = None,
    session_id: str | None = None,
    min_length: int = 20,
    skip_eval: bool = False,
):
    """Run backtest against real transcripts."""
    print("Loading transcripts...")
    entries = load_transcripts(session_id)
    print(f"Found {len(entries)} total messages across transcripts")

    entries = filter_prompts(entries, min_length)
    print(f"After filtering: {len(entries)} substantive prompts")

    if limit:
        entries = entries[:limit]

    print(f"Testing {len(entries)} prompts against MemCore")
    print("=" * 70)

    # Aggregates
    total = 0
    recall_hits = 0  # confidence > weak
    recall_misses = 0  # confidence <= weak
    confidence_counts = {}
    relevance_scores = []
    lifecycle_stats = {
        "high_mw_relevant": 0,
        "low_mw_relevant": 0,
        "reconsolidated_relevant": 0,
        "suppressed_relevant": 0,
        "high_stability_relevant": 0,
    }
    missing_memories = []
    noise_prompts = []

    async with httpx.AsyncClient(timeout=30) as client:
        for i, entry in enumerate(entries):
            prompt = entry["prompt"]
            response = entry["response"]

            # Recall
            memories, confidence = await recall_memories(client, prompt)
            conf_level = confidence.get("level", "unknown")
            confidence_counts[conf_level] = confidence_counts.get(conf_level, 0) + 1

            if conf_level in ("very_weak", "no_memory", "weak"):
                recall_misses += 1
                status = "MISS"
            else:
                recall_hits += 1
                status = f"OK({conf_level})"

            # Evaluate relevance (LLM judge)
            eval_result = {}
            if not skip_eval and memories:
                eval_result = await evaluate_recall(client, prompt, response, memories)
                if "relevant_count" in eval_result:
                    relevance_scores.append(eval_result)

                    # Track lifecycle signal effectiveness
                    if eval_result.get("best_memory_idx", -1) >= 0:
                        best = memories[eval_result["best_memory_idx"]]
                        mw_total = best.get("mw_total", 0)
                        mw_success = best.get("mw_success", 0)
                        if mw_total > 0 and mw_success / mw_total > 0.5:
                            lifecycle_stats["high_mw_relevant"] += 1
                        if mw_total > 0 and mw_success / mw_total <= 0.5:
                            lifecycle_stats["low_mw_relevant"] += 1
                        if best.get("reconsolidation_count", 0) > 0:
                            lifecycle_stats["reconsolidated_relevant"] += 1
                        if best.get("stability", 1) > 3:
                            lifecycle_stats["high_stability_relevant"] += 1

                    missing = eval_result.get("missing", "none")
                    if missing and missing != "none":
                        missing_memories.append({
                            "prompt": prompt[:100],
                            "missing": missing,
                        })

                    if eval_result.get("noise_count", 0) > eval_result.get("relevant_count", 0):
                        noise_prompts.append(prompt[:80])

            # Print progress
            rel_str = ""
            if eval_result.get("relevant_count") is not None:
                rel_str = f" | relevant={eval_result['relevant_count']}/5, high={eval_result.get('high_count', 0)}"
            print(f"  [{i+1}/{len(entries)}] {status:12s}{rel_str} | {prompt[:60]}...")

            total += 1

    # Summary
    print(f"\n{'=' * 70}")
    print(f"BACKTEST RESULTS — {total} real prompts")
    print(f"{'=' * 70}")

    print(f"\n## Recall Coverage")
    print(f"  Hits (moderate+ confidence): {recall_hits}/{total} ({recall_hits/max(total,1)*100:.0f}%)")
    print(f"  Misses (weak/no_memory):     {recall_misses}/{total} ({recall_misses/max(total,1)*100:.0f}%)")
    print(f"\n  Confidence distribution:")
    for level in ["high", "moderate", "stale", "weak", "very_weak", "no_memory"]:
        cnt = confidence_counts.get(level, 0)
        if cnt > 0:
            bar = "#" * int(cnt / max(total, 1) * 40)
            print(f"    {level:12s}: {cnt:3d} ({cnt/max(total,1)*100:5.1f}%) {bar}")

    if relevance_scores:
        avg_relevant = sum(s.get("relevant_count", 0) for s in relevance_scores) / len(relevance_scores)
        avg_high = sum(s.get("high_count", 0) for s in relevance_scores) / len(relevance_scores)
        avg_noise = sum(s.get("noise_count", 0) for s in relevance_scores) / len(relevance_scores)
        would_help_pct = sum(1 for s in relevance_scores if s.get("would_help")) / len(relevance_scores) * 100

        print(f"\n## Relevance Quality (LLM-judged)")
        print(f"  Avg relevant memories per recall: {avg_relevant:.1f}/5")
        print(f"  Avg HIGH relevance per recall:    {avg_high:.1f}/5")
        print(f"  Avg noise per recall:             {avg_noise:.1f}/5")
        print(f"  Would help response:              {would_help_pct:.0f}%")

    if any(v > 0 for v in lifecycle_stats.values()):
        print(f"\n## Lifecycle Signal Effectiveness")
        print(f"  Best memory had high MW (>0.5):     {lifecycle_stats['high_mw_relevant']}")
        print(f"  Best memory had low MW (<=0.5):     {lifecycle_stats['low_mw_relevant']}")
        print(f"  Best memory was reconsolidated:     {lifecycle_stats['reconsolidated_relevant']}")
        print(f"  Best memory had high stability (>3): {lifecycle_stats['high_stability_relevant']}")

    if missing_memories:
        print(f"\n## Missing Memories ({len(missing_memories)} gaps found)")
        for m in missing_memories[:10]:
            print(f"  - {m['prompt']}")
            print(f"    MISSING: {m['missing']}")

    if noise_prompts:
        print(f"\n## Noisy Recalls ({len(noise_prompts)} prompts with more noise than signal)")
        for p in noise_prompts[:5]:
            print(f"  - {p}")

    # Write results to file
    output_file = os.path.join(os.path.dirname(__file__), "backtest_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "total": total,
            "recall_hits": recall_hits,
            "recall_misses": recall_misses,
            "confidence_distribution": confidence_counts,
            "avg_relevant": avg_relevant if relevance_scores else 0,
            "avg_high": avg_high if relevance_scores else 0,
            "would_help_pct": would_help_pct if relevance_scores else 0,
            "lifecycle_stats": lifecycle_stats,
            "missing_memories": missing_memories[:20],
            "noise_prompts": noise_prompts[:10],
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Backtest MemCore against real Claude Code transcripts")
    parser.add_argument("--limit", type=int, help="Only test first N prompts")
    parser.add_argument("--session", type=str, help="Specific session ID to test")
    parser.add_argument("--min-length", type=int, default=20, help="Min prompt length to include")
    parser.add_argument("--skip-eval", action="store_true", help="Skip LLM relevance evaluation (faster)")
    args = parser.parse_args()

    asyncio.run(run_backtest(
        limit=args.limit,
        session_id=args.session,
        min_length=args.min_length,
        skip_eval=args.skip_eval,
    ))


if __name__ == "__main__":
    main()
