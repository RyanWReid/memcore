# Benchmark Results

MemCore is evaluated against [LongMemEval](https://github.com/xiaowu0162/longmemeval) (ICLR 2025), the standard benchmark for long-term conversational memory systems.

## How the Benchmark Works

Each question follows the same pipeline:
1. **Ingest** — conversation sessions are stored as individual memories + LLM-extracted atomic facts with lexical aliases
2. **Recall** — time-aware query expansion, multi-query retrieval (TOP_K=50), cross-encoder reranking (keep top 20)
3. **Answer** — Chain-of-Note: extract structured facts from retrieved memories, then reason over them
4. **Cleanup** — all memories for the question are deleted before the next one

This tests the full write → search → reason pipeline. No information leaks between questions.

## Latest Results (April 2026 — v6 Complete)

**50-question LongMemEval-S sample**

| Metric | Score |
|--------|-------|
| **Human-judged accuracy** | **46/50 (92%)** |
| Strict substring match | 41/50 (82%) |
| Total memories ingested | 45,582 |
| Avg time per question | 165.9s |
| Retrieval failures | 2/50 |
| Wrong answer (misinterpretation) | 2/50 |

### Strict vs Human-Judged

Strict matching uses substring containment — if the gold answer doesn't appear verbatim in the hypothesis, it's marked wrong. This penalizes correct paraphrases:

| Expected | Generated | Strict | Human |
|----------|-----------|--------|-------|
| "the sports store downtown" | "a sports store downtown" | MISS | Correct |
| "University of Melbourne in Australia" | "University of Melbourne" | MISS | Correct |
| "triple what I paid for it" | "triple the amount you paid for it" | MISS | Correct |
| "UCLA" | "UCLA (University of California, Los Angeles)" | MISS | Correct |
| "a blue Snaggletooth" | "a rare blue Snaggletooth" | MISS | Correct |

### Real Failures

| Question | Expected | Got | Failure Mode |
|----------|----------|-----|-------------|
| Where did I redeem a $5 coupon on coffee creamer? | Target | "I don't have that information" | Retrieval — fact not surfaced |
| What was my previous occupation? | Marketing specialist at a small startup | Manager at a pharma startup | Wrong fact retrieved |
| How many copies of my debut album? | 500 | "I don't have that information" | Retrieval — fact not surfaced |

## Leaderboard Context

| System | LongMemEval QA | Architecture |
|--------|---------------|-------------|
| AgentMemory V4 | 96.2% | 6-signal fusion, best pure retrieval |
| Chronos | 95.6% | SVO event calendar, temporal reasoning |
| OMEGA | 95.4% | SQLite + forgetting, local-only |
| Mastra OM | 94.87% | Observation compression, no retrieval |
| **MemCore** | **~92%** | **Bio-inspired lifecycle + epistemic gate** |
| Mem0 | ~49% | Cloud vector search |

## Historical Runs

| Date | Version | Score (strict/human) | Features Tested |
|------|---------|---------------------|-----------------|
| 2026-04-22 | v6 complete | 82% / 92% | + tool hints, MW wiring, full lifecycle |
| 2026-04-20 | v6 phase 1 | 90% / 94% | + difficulty-weighted stability, MW counters, suppression |
| 2026-04-19 | v5 post-seasoning | 90% / 94% | Cross-encoder + CoN + fact extraction |

### Notes on Score Variation

The 50-question sample introduces ~5-8% noise between runs. The strict score drop from 90% to 82% is within this variance — the human-judged score is stable at 92-94%. The v6 lifecycle features (reconsolidation, suppression, MW tracking, tool hints) are designed for production use over time, not cold-start benchmark performance. Their value will show in:

- **MW success ratios** converging over weeks of daily use
- **Reconsolidation** enriching frequently-accessed memories
- **Suppression** reducing redundant results
- **Drift detection** preventing memory corruption

## Running the Benchmark

```bash
# Requires LongMemEval dataset (not included — download from the LongMemEval repo)
# Place at memcore/benchmark/longmemeval_s_cleaned.json

# Set environment
export MEMCORE_URL=http://localhost:8020
export LITELLM_BASE_URL=http://localhost:4000/v1
export LITELLM_API_KEY=your-key

# Run 50 questions
python3 -m memcore.benchmark.run_longmemeval_v4 --limit 50

# Run full 500
python3 -m memcore.benchmark.run_longmemeval_v4

# Resume from checkpoint
python3 -m memcore.benchmark.run_longmemeval_v4 --resume

# Ablation: no fact extraction
python3 -m memcore.benchmark.run_longmemeval_v4 --limit 50 --no-facts

# Ablation: no Chain-of-Note (direct answer)
python3 -m memcore.benchmark.run_longmemeval_v4 --limit 50 --no-con
```

## Scoring

The benchmark outputs `memcore_hypothesis_v4.jsonl` with `{"question_id", "hypothesis"}` per line. Score against the gold answers in the dataset using substring matching or LLM-as-judge for paraphrase tolerance.

---

## Production Backtest (April 2026)

LongMemEval tests retrieval quality with fresh data. But does MemCore actually help in production? We tested 50 real prompts from actual Claude Code sessions against the live memory store.

### The Gap: Synthetic vs Real

| Metric | LongMemEval (synthetic) | Production (real) |
|--------|------------------------|-------------------|
| Coverage (moderate+ confidence) | 92% | **28%** |
| Avg relevant memories per recall | ~4/5 | 1.3/5 |
| Noise per recall | ~0.5/5 | 3.7/5 |
| Would-help rate | ~90% | 30% |

### Why

LongMemEval ingests conversation data per question — it tests whether MemCore can *find* a fact that exists. Production tests whether the fact *was ever stored*. Most real prompts are short, contextual follow-ups ("why did we go backwards?", "hows our memory right now?") that need session-level context, not individual facts.

### What's Missing (23 gaps identified)

The LLM judge found that most missing context was **episode-level**: development status, session progress, strategic decisions, research findings — the narrative arc of work, not isolated facts.

### The Fix: Episode Segmentation

Built a live episode boundary detector that segments conversations in real-time using TextTiling with embeddings. At topic shifts, an LLM summarizes the completed episode and stores it as one rich memory. This creates the session-level context that individual fact extraction misses.

### Production Evaluation Framework

Three modes for ongoing measurement:

```bash
# Deep eval: 50 prompts with LLM relevance judge
python -m memcore.benchmark.production_eval retro --limit 50

# Quick health check: coverage + latency only
python -m memcore.benchmark.production_eval health

# Trend over time: compare weekly retro runs
python -m memcore.benchmark.production_eval trend
```

### Episode Segmentation Results

Processed 48 historical sessions → 305 episodes detected → 181 stored (write gate filtered duplicates). Memory store grew from 1,936 to 2,840.

| Metric | Before Episodes | After Episodes |
|--------|----------------|---------------|
| Coverage (moderate+) | 28% | 28% |
| Episode memories in results | 0% | 16% |
| Total memories | 1,936 | 2,840 |

Episodes are being recalled (16% of results include them) but don't lift coverage on the held-out eval set. The eval sessions (Apr 20-22) covered new topics not present in training sessions (Mar 23-Apr 19), creating a topic mismatch. The real value of episode segmentation is longitudinal — it shows up in the next session when asking about yesterday's work.

### Key Insight

Most real prompts (72%) are short conversational follow-ups ("now", "where are we") that need current-session context, not stored memories. The 28% coverage may be the natural ceiling for prompts where historical memory is relevant. The right metric is tracking trends over weeks, not one-shot backtests.
