# MemCore

**Governed memory for AI agents.** Store memories through an epistemic write gate that scores quality before storage — not after. Cross-encoder reranking, automatic fact extraction, hybrid search. Ships as a single Docker Compose stack.

Built for [Claude Code](https://claude.ai/code) and any MCP-compatible client.

---

## Why MemCore?

Every memory system stores first and cleans up later. MemCore rejects garbage at the door.

| Feature | Mem0 | Zep/Graphiti | Letta | **MemCore** |
|---------|------|-------------|-------|-------------|
| Write-time quality gate | No | No | No | **Yes** — epistemic scoring before storage |
| Cross-encoder reranking | No | No | No | **Yes** — ms-marco-MiniLM-L-6-v2 |
| Automatic fact extraction | No | No | No | **Yes** — LLM decomposes memories into atomic facts |
| Hybrid search (FTS + vector) | Vector only | BM25 + cosine | Vector only | **RRF fusion** of tsvector + pgvector |
| Self-hosted, single container | Cloud | Self-hosted | Cloud | **Docker Compose** |

**LongMemEval benchmark:** 51% (v1) → 87% (v4) using DeepSeek ($0.14/MTok) — no expensive models required.

## How It Works

```
WRITE (remember):
  Content → Heuristic precheck (0ms, $0)
          → LLM quality scoring (3s, ~$0.003) — only for borderline cases
          → Type classification (fact/event/decision/goal/...)
          → Store in PostgreSQL + pgvector
          → Background: extract 2-5 atomic facts, store separately
          → Route events/decisions to Graphiti knowledge graph

READ (recall):
  Query → LLM synonym expansion
        → Hybrid search (tsvector FTS + pgvector cosine, RRF fusion)
        → Cross-encoder reranking (70% RRF + 30% ms-marco, ~50ms CPU)
        → Return top results with blended scores
```

### The Write Gate

Three quality checks + five-factor epistemic scoring:

**Quality checks** (all must pass):
- **Coreference** — no dangling pronouns ("it", "they" without antecedent)
- **Self-contained** — meaningful without surrounding conversation
- **Temporal anchoring** — time references have explicit dates

**Epistemic scoring** (weighted combination):
- Future utility (25%) — will this be useful later?
- Factual confidence (20%) — backed by evidence?
- Semantic novelty (15%) — adds new information?
- Temporal recency (10%) — current or stale?
- Content type prior (30%) — decisions (0.90) > events (0.75) > facts (0.55)

Score >= 0.55 → stored. Below → rejected with reason.

**Fast path:** Obvious content (long decisions, clear events) is accepted with 1 LLM call. Trivial content ("ok", "thanks") is rejected with 0 LLM calls. Only borderline cases get the full 3-call evaluation.

## Quick Start

```bash
# Clone
git clone https://github.com/RyanWReid/memcore.git
cd memcore

# Configure
cp .env.example .env
# Edit .env with your LiteLLM/OpenAI endpoint and API key

# Run
docker compose up -d

# Verify
curl http://localhost:8020/health
```

### Connect to Claude Code

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "memcore": {
      "type": "sse",
      "url": "http://localhost:8020/sse"
    }
  }
}
```

Now Claude Code has persistent memory across sessions via 4 tools: `remember`, `recall`, `forget`, `audit`.

## MCP Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a memory through the write gate. Returns score, type, and storage layer. |
| `recall` | Search memory by query. Hybrid search + cross-encoder reranking. ~100ms. |
| `forget` | Soft-delete a memory by ID. |
| `audit` | Inspect a memory's full epistemic scores, quality checks, and gate decision. |

## REST API

For webhooks, automation, and non-MCP clients:

```bash
# Store a memory
curl -X POST http://localhost:8020/api/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Deployed Redis 7.2 on port 6379", "group_id": "homelab"}'

# Search memories
curl -X POST http://localhost:8020/api/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "Redis deployment", "limit": 5}'

# Bulk ingest (bypasses gate — for migrations)
curl -X POST http://localhost:8020/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "...", "group_id": "myproject", "created_at": "2024-01-15T00:00:00Z"}'
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  MemCore Server                  │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ MCP SSE  │  │ REST API │  │  Write Gate    │  │
│  │ /sse     │  │ /api/*   │  │  (epistemic)   │  │
│  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
│       │              │                │          │
│  ┌────┴──────────────┴────────────────┴───────┐  │
│  │              Storage Router                │  │
│  │  facts/goals → PostgreSQL + pgvector       │  │
│  │  events/decisions → Graphiti (optional)     │  │
│  └────────────────────┬───────────────────────┘  │
│                       │                          │
│  ┌────────────────────┴───────────────────────┐  │
│  │           Retrieval Pipeline               │  │
│  │  Query expansion → Hybrid search (RRF)     │  │
│  │  → Cross-encoder rerank → Return top-k     │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │        Fact Extraction (background)        │  │
│  │  Every remember → 2-5 atomic facts stored  │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
   ┌──────┴──────┐ ┌──┴───┐ ┌─────┴──────┐
   │ PostgreSQL  │ │ LLM  │ │  Graphiti  │
   │ + pgvector  │ │ API  │ │ (optional) │
   └─────────────┘ └──────┘ └────────────┘
```

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `http://localhost:4000/v1` | OpenAI-compatible LLM endpoint |
| `LITELLM_API_KEY` | — | API key for LLM |
| `GATE_MODEL` | `deepseek-chat` | Model for gate scoring (cheap + fast recommended) |
| `GATE_THRESHOLD` | `0.55` | Minimum score to store a memory |
| `EMBEDDING_URL` | `http://localhost:8100/v1/embeddings` | Embedding endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `EMBEDDING_DIM` | `384` | Embedding dimensions |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `FACT_EXTRACTION_ENABLED` | `true` | Auto-extract atomic facts on write |
| `GRAPHITI_URL` | `http://localhost:8000` | Graphiti MCP endpoint (optional) |
| `GRAPHITI_RECALL_TIMEOUT` | `10.0` | Graphiti search timeout in seconds |

## Memory Types

MemCore classifies memories and routes them to the appropriate storage:

| Type | Description | Storage |
|------|-------------|---------|
| `fact` | Preferences, settings, current state | PostgreSQL |
| `event` | Deployments, incidents, milestones | PostgreSQL + Graphiti |
| `decision` | Architectural choices with rationale | PostgreSQL + Graphiti |
| `relationship` | Entity connections, dependencies | PostgreSQL + Graphiti |
| `goal` | Current objectives, priorities | PostgreSQL |
| `document` | Long-form content, specs | PostgreSQL |
| `trajectory` | Action sequences + outcomes | PostgreSQL |

## Requirements

- Docker and Docker Compose
- An OpenAI-compatible LLM endpoint (LiteLLM, OpenAI, Ollama, etc.)
- An embedding endpoint (or use OpenAI embeddings)
- ~1GB RAM for the cross-encoder model (downloaded on first recall)

Graphiti is optional — MemCore works fully with just PostgreSQL.

## License

MIT
