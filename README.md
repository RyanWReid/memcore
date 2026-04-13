<p align="center">
  <img src="https://img.shields.io/badge/LongMemEval-90%25_QA-4CAF50?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Bio--Inspired-7_Principles-9B59B6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/MCP_Native-Multi--Agent-cc785c?style=for-the-badge&logo=anthropic&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" />
</p>

# MemCore

**Memory that learns from how you use it.**

Every AI memory system today is a database you search. Store text, embed it, retrieve the closest match. MemCore is different — it's a living memory system modeled after how the human brain actually works. Memories strengthen when recalled. Unused knowledge fades. The system knows when it's confident and when it's guessing. Decisions persist forever while stale facts decay on a predictable curve.

MemCore scores quality *before* writing (epistemic write gate), tracks *how* memories are used (Ebbinghaus stability), tells the LLM *how much to trust* what it found (metamemory), and forgets *what isn't needed* (per-type decay). Built for multi-agent systems where multiple AI sessions share a common memory across devices.

---

## What Makes MemCore Different

Every other AI memory system is a database you search. MemCore is a living memory that evolves through use.

| Capability | MemCore | MemPalace | OMEGA | AgentMemory | Mem0 |
|-----------|---------|-----------|-------|-------------|------|
| **Write gate** (score quality before storing) | Epistemic scoring (0-1) | None — store everything | None | None | None |
| **Metamemory** (confidence signal on recall) | FOK-inspired confidence levels | None | None | None | None |
| **Ebbinghaus stability** (access strengthens memory) | `S *= 1.3` per recall, per-type decay | None | Linear decay, floor 0.35 | Gaussian proximity | None |
| **Cross-encoder reranking** | ms-marco-MiniLM, 70/30 blend | None | ms-marco-MiniLM | ms-marco-MiniLM | None |
| **Knowledge graph** | Graphiti (FalkorDB) | None | None | SQLite graph | None |
| **Multi-agent** | Namespace routing, MCP, multi-device | Single session | Single session | Single session | Cloud API |
| **Truth verification** | Live infra checks (planned) | None | None | None | None |
| **Fact extraction** | LLM atomic facts at write time | None | On write | Post-ingestion | On write |
| **Decision supersession** | Auto-detects + replaces old decisions | None | Conflict detection | Contradiction detection | None |
| **Benchmark** | 90% QA accuracy (end-to-end) | 96.6% R@5 (retrieval only) | 95.4% QA | 96.2% QA | ~49% |
| **Cost** | ~$0.001/memory (DeepSeek) | $0 (local) | $0 (local) | ~$1K total benchmark | $249/mo |

### Where We Stand (April 2026)

**90% end-to-end QA accuracy** on LongMemEval — the LLM must generate the correct answer, not just find the right document. This places MemCore in the top tier alongside systems that have been optimized for months by dedicated teams.

What the leaderboard doesn't show is that MemCore is solving a different problem. The top benchmark systems (AgentMemory, Chronos, OMEGA) are optimized for a single task: answer questions from conversation history. MemCore is building toward **memory that understands itself** — that knows what it knows, what it's forgotten, what's changed, and what's probably wrong.

| System | LongMemEval | What It Actually Measures |
|--------|-------------|--------------------------|
| AgentMemory V4 | 96.2% QA | Best pure retrieval engine (6-signal fusion) |
| Chronos | 95.6% QA | Best temporal reasoning (SVO event calendar) |
| OMEGA | 95.4% QA | Best local-only system (SQLite + forgetting) |
| Mastra OM | 94.87% QA | Best no-retrieval approach (observation compression) |
| **MemCore** | **~90% QA** | **Only system with write gate + metamemory + bio-inspired lifecycle** |
| MemPalace | 96.6% R@5 | ChromaDB embedding quality ([different metric](https://github.com/lhl/agentic-memory/blob/main/ANALYSIS-mempalace.md)) |
| Mem0 | ~49% QA | Cloud-hosted vector search |

**Our thesis**: The storage and retrieval problem is largely solved — ChromaDB with good embeddings gets 96.6% R@5 with zero complexity. The remaining frontier is **memory lifecycle**: what to store, when to forget, how memories evolve, and how to know what you don't know. That's where MemCore is building.

---

## Architecture

```
WRITE PATH:
  Content --> Epistemic Write Gate (quality 0-1, type classification)
    |-- Score >= 0.55 --> Route by type:
    |     decisions/events --> Graphiti (temporal knowledge graph)
    |     facts/goals      --> PostgreSQL (pgvector + tsvector)
    |     + Fire-and-forget fact extraction (2-3 atomic facts per memory)
    |-- Score < 0.55 --> Rejected (with reason)
    |-- Duplicate check: cosine > 0.85 OR keyword overlap >= 3 --> Blocked

READ PATH:
  Query --> Expand (LLM synonyms)
        --> Hybrid search (pgvector cosine + tsvector keyword + RRF fusion)
        --> Cross-encoder rerank (ms-marco-MiniLM, 70/30 blend)
        --> Ebbinghaus retention scoring (per-type decay, stability growth)
        --> Fuse with Graphiti entity results (when graph contributes)
        --> Metamemory confidence signal (high/moderate/stale/weak)
        --> Track access (increment count, grow stability 1.3x)
        --> Return results + confidence to LLM

LIFECYCLE:
  Access tracking: Each recall increments access_count, grows stability
  Ebbinghaus decay: R = e^(-lambda * days / stability), per-type lambdas
  Decision supersession: New decisions auto-replace old ones on same topic
  Consolidation: Graph-guided selective replay (planned)
  Truth verification: Cross-check against live infrastructure (planned)
```

---

## Bio-Inspired Design

Most AI memory systems borrow from information retrieval. MemCore borrows from neuroscience. These aren't metaphors — they're mathematical models drawn from published research in cognitive psychology and neurobiology, adapted for AI memory.

### Deployed

**Ebbinghaus Forgetting Curve** — Each recall increases a memory's *stability* (resistance to forgetting). `R = e^(-t/S)` where S grows by 1.3x per access. A memory recalled 10 times has a half-life of 20 days. Recalled 20 times = effectively permanent. Per-type decay: decisions never fade, goals decay in 14 days, facts in 70. This isn't time decay with a reset — it's the biological spacing effect where practice makes durable.

**Metamemory (Feeling of Knowing)** — The brain senses whether it has an answer before fully retrieving it. MemCore returns a confidence signal alongside results: `high`, `moderate`, `stale`, `weak`, `no_memory`. The LLM can answer directly on high confidence, caveat on stale, or abstain on weak — instead of hallucinating from vague matches. Addresses the [RAG paradox](https://ragaboutit.com/the-2026-rag-performance-paradox-why-simpler-chunking-strategies-are-outperforming-complex-ai-driven-methods/) where more context increases hallucination on unanswerable questions.

**Epistemic Write Gate** — Quality scoring before storage. Every other system stores first and cleans later (or never). MemCore evaluates factual confidence, future utility, semantic novelty, and content type before a memory enters the store. The gate is the immune system — it keeps the memory healthy so retrieval doesn't have to compensate for noise.

### Building Next

**Reconsolidation** — In the brain, recall makes memories temporarily unstable. New context gets woven in before re-storage. MemCore will enrich memories with context from conversations where they're used. A memory about "Caddy is the reverse proxy" gets recalled during CrowdSec work and reconsolidates as "Caddy is the reverse proxy with CrowdSec IPS bouncer." Memories get richer through use, not just older.

**Selective Replay** — The brain doesn't consolidate equally during sleep. Sharp-wave ripples select which memories matter based on novelty and connection to existing knowledge. MemCore will use Graphiti's entity graph to guide consolidation — well-connected entities get their scattered memories merged into rich summaries. Isolated memories decay naturally via Ebbinghaus. The knowledge graph earns its keep as a consolidation guide, not a retrieval layer.

**Emotional Tagging** — High-arousal events (incidents, breakthroughs, failures) create stronger traces. MemCore will scan transcripts for intensity signals — errors, retries, urgency — and tag memories with arousal that resists decay. A decision made during a 3am outage weighs more than the same decision in a planning meeting.

**Hebbian Co-Retrieval** — "Neurons that fire together wire together." When two memories are repeatedly recalled together and both contribute to responses, they develop an association link. Over time this creates an organic graph based on *how memories are used together*, not just what they contain.

**Live Truth Verification** — No other memory system can verify its own memories against reality. MemCore is embedded in live infrastructure with MCP access to Proxmox, DNS, Docker. "CT 141 runs MemCore" isn't just a stored fact — it's verifiable. Planned: periodic cross-checks that flag memories contradicted by current state.

---

## Quick Start

MemCore runs as a Docker Compose stack with PostgreSQL (pgvector) and exposes both MCP (SSE) and REST APIs.

```bash
cd memcore
docker compose up -d
# Health check
curl http://localhost:8020/health
```

### Store a memory

```bash
curl -X POST http://localhost:8020/api/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Deployed CrowdSec on CT 100 with iptables bouncer. LAN subnets whitelisted.", "group_id": "homelab"}'
```

### Recall with confidence

```bash
curl -X POST http://localhost:8020/api/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "what IPS do we use", "group_id": "homelab", "limit": 5}'
```

Response includes metamemory confidence:
```json
{
  "results": [...],
  "count": 5,
  "confidence": {
    "level": "high",
    "signal": "Strong match, well-accessed, clear winner",
    "score": 0.92
  }
}
```

### MCP Integration

MemCore exposes 4 MCP tools via SSE at `/sse`:
- **remember** — store through the epistemic write gate
- **recall** — fused search with confidence signal
- **forget** — mark a memory as deleted
- **audit** — inspect full details of any memory

Connect via Claude Code `.mcp.json` or any MCP client.

---

## Write Gate

Every memory is scored before storage. Three quality checks + epistemic scoring via LLM:

| Check | What It Tests |
|-------|--------------|
| Length + substance | Rejects trivially short content (<20 chars) |
| Novelty | Blocks duplicates (cosine >0.85 or keyword overlap >=3) |
| Epistemic quality | LLM scores 0.0-1.0 on: factual confidence, future utility, semantic novelty, temporal recency, content type |

Threshold: **0.55**. Below = rejected with reason. Above = classified by type and routed:

| Type | Score Range | Primary Store | Decay |
|------|-------------|--------------|-------|
| Decision | ~0.85-0.95 | Graphiti | Permanent (lambda=0) |
| Event | ~0.75-0.85 | Graphiti | 35-day half-life |
| Fact | ~0.55-0.75 | PostgreSQL | 70-day half-life |
| Goal | ~0.80-0.95 | PostgreSQL | 14-day half-life |

---

## Retrieval Pipeline

1. **Query expansion** — LLM rewrites query with 3-5 synonym phrases (cached, 3s timeout)
2. **Hybrid search** — pgvector cosine similarity + tsvector full-text, fused with Reciprocal Rank Fusion (K=60)
3. **Cross-encoder rerank** — ms-marco-MiniLM-L-6-v2 scores (query, document) pairs. 70% RRF + 30% cross-encoder = blended score
4. **Ebbinghaus retention** — `retention = e^(-lambda * days / stability)`. 85% relevance + 15% retention = final score
5. **Graphiti fusion** — entity results from knowledge graph normalized and merged (when graph contributes new results, re-ranked across combined set)
6. **Metamemory** — score distribution, recency, access patterns analyzed. Confidence level returned alongside results
7. **Access tracking** — all returned memories get `access_count++`, `stability *= 1.3`, `last_accessed_at = NOW()`

---

## Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API | Python + Starlette + MCP SDK | REST + SSE endpoints |
| Storage | PostgreSQL 16 + pgvector | Hybrid vector + keyword search |
| Knowledge Graph | Graphiti + FalkorDB | Temporal entity relationships |
| Embeddings | nomic-embed-text (384d) | Local embedding server |
| Reranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder on CPU (~50ms) |
| LLM | DeepSeek V3.2 via LiteLLM | Write gate scoring, fact extraction, query expansion |
| Containers | Docker Compose | PostgreSQL + MemCore app |

---

## Roadmap

### v5 — Current (Bio-Inspired Foundation)
- [x] Epistemic write gate with quality scoring
- [x] Hybrid search (pgvector + tsvector + RRF)
- [x] Cross-encoder reranking (ms-marco-MiniLM, 70/30 blend)
- [x] Write-time fact extraction (2-3 atomic facts per memory)
- [x] Decision supersession (auto-replaces outdated decisions)
- [x] Graphiti fused recall (postgres + graph merged into single ranked list)
- [x] Ebbinghaus stability tracking (access_count, stability growth per recall)
- [x] Per-type decay lambdas (decisions permanent, goals 14d, facts 70d)
- [x] Metamemory confidence signal (high/moderate/stale/weak/no_memory)
- [x] Namespace routing (multi-tenant memory isolation)
- [x] MCP + REST dual API
- [x] Claude Code hooks (auto-recall on prompt, auto-retain on stop)

### v6 — Memory Lifecycle (In Progress)
- [ ] Reconsolidation — enrich memories with new context on recall
- [ ] Selective consolidation — graph-guided "dream" cycle merges scattered memories
- [ ] Arousal tagging — conversation intensity scoring at write time
- [ ] Critical facts layer — auto-generated 170-token wake-up context from most-accessed memories
- [ ] Custom production benchmark — real-world recall quality, not synthetic LongMemEval

### v7 — Self-Aware Memory
- [ ] Live truth verification — cross-check memories against actual infrastructure state
- [ ] Hebbian co-retrieval — usage-based association links between frequently co-recalled memories
- [ ] Adaptive retrieval weights — learn per-signal weights from access patterns (AgentMemory approach)
- [ ] Abstention calibration — metamemory confidence tuned against actual accuracy

### Long-term Vision

The endgame isn't a better search engine. It's **memory that understands itself**: a system that knows what it knows, what it's forgotten, what's changed since it last checked, and what's probably wrong. Every current AI memory system is a static store with a retrieval layer on top. MemCore is building toward memory that actively maintains its own integrity — consolidating knowledge while you sleep, strengthening what matters, letting go of what doesn't, and flagging when reality has diverged from what it remembers.

---

## Research

MemCore is built on competitive analysis of every system on the LongMemEval leaderboard and original research applying neuroscience principles (Ebbinghaus forgetting curves, memory reconsolidation, hippocampal replay, metamemory, Hebbian learning) to AI memory architecture.

Key references:
- [LongMemEval](https://github.com/xiaowu0162/longmemeval) — ICLR 2025 benchmark (500 questions, 5 memory ability categories)
- [Ebbinghaus (1885)](https://en.wikipedia.org/wiki/Forgetting_curve) — forgetting curve: `R = e^(-t/S)`
- [Nader et al. (2000)](https://www.nature.com/articles/35021052) — memory reconsolidation
- [Tulving & Thomson (1973)](https://psycnet.apa.org/record/1973-25866-001) — encoding specificity principle
- [Hebb (1949)](https://en.wikipedia.org/wiki/Hebbian_theory) — "neurons that fire together wire together"
