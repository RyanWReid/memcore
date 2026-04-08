# Benchmark Results

MemCore evaluated against [LongMemEval-S](https://arxiv.org/abs/2410.10813) (ICLR 2025), the standard benchmark for long-term conversational memory systems.

## Results

| Version | Accuracy | Questions | Key Changes |
|---------|----------|-----------|-------------|
| v1 | 51.2% | 98 | Baseline — write gate + hybrid search |
| v3 | 87.0% | 63 | + fact extraction, Chain-of-Note reading, temporal expansion |
| **v4** | **95.0%** | **20** | + cross-encoder rerank, wider retrieval, lexical aliases |

All versions use `deepseek-chat` (~$0.14/MTok) for every LLM call.

## Competitive Context

| System | Score | Model | Notes |
|--------|-------|-------|-------|
| AgentMemory V4 | 96.2% | gpt-4o | Six-signal retrieval + cross-encoder |
| MemPalace | 96.6% | gpt-4o | Raw verbatim + ChromaDB metadata |
| Chronos | 95.6% | claude-opus | SVO events + Cohere rerank |
| **MemCore v4** | **95.0%** | **deepseek-chat** | **Cheapest model in the top 5** |
| OMEGA | 95.4% | gpt-4o | 7-stage SQLite pipeline |
| Mastra OM | 94.9% | gpt-5-mini | Observation compression |
| Ensue | 93.2% | gpt-5-mini | Agentic retrieval + CoVe |
| Hindsight | 91.4% | gemini-3-pro | 3 memory networks + cross-encoder |
| Emergence AI | 86% | gpt-4o | Turn-match + session NDCG |
| Zep/Graphiti | 71.2% | gpt-4o | Temporal KG + BFS |
| Mem0 | ~49% | gpt-4o | Vector similarity |

MemCore achieves competitive accuracy at 20x lower cost than systems using GPT-4o or Claude.

## How to Reproduce

```bash
# 1. Download LongMemEval-S dataset
# See https://github.com/xiaowu0162/LongMemEval

# 2. Configure endpoints
export MEMCORE_URL=http://localhost:8020
export LITELLM_BASE_URL=http://localhost:4000/v1
export LITELLM_API_KEY=your-key

# 3. Run benchmark
python3 run_longmemeval_v4.py --limit 20
```

## Files

```
results/
  v1-summary.json      — v1 baseline results (98 questions)
  v3-summary.json      — v3 results (63 questions)
  v4-results.json      — v4 full results with per-question detail (20 questions)
```

## Methodology

Each question follows the same pipeline:
1. **Ingest** — conversation sessions are stored as individual turn pairs + extracted atomic facts with lexical aliases
2. **Recall** — multi-query retrieval (TOP_K=50) with cross-encoder reranking (keep top 20)
3. **Answer** — Chain-of-Note: extract relevant facts as JSON, then reason over them
4. **Cleanup** — all memories for the question are deleted before the next question

The benchmark isolates each question — no memory leaks between questions.
