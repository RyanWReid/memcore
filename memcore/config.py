"""MemCore configuration — reads from environment variables."""

import os


# LiteLLM gateway
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")

# Model for gate scoring (cheap, fast, reliable JSON)
GATE_MODEL = os.getenv("GATE_MODEL", "deepseek-chat")

# Storage backends
GRAPHITI_URL = os.getenv("GRAPHITI_URL", "http://localhost:8000")
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://memcore:memcore@memcore-db:5432/memcore",
)

# Write gate
GATE_THRESHOLD = float(os.getenv("GATE_THRESHOLD", "0.55"))

# Embeddings
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8100/v1/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# Cross-encoder reranking
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Write-time fact extraction
FACT_EXTRACTION_ENABLED = os.getenv("FACT_EXTRACTION_ENABLED", "true").lower() == "true"

# Graphiti recall timeout (fast path — don't block recall for slow graph search)
GRAPHITI_RECALL_TIMEOUT = float(os.getenv("GRAPHITI_RECALL_TIMEOUT", "10.0"))

# MCP server
MCP_PORT = int(os.getenv("MCP_PORT", "8020"))
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
