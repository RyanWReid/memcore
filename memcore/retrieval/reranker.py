"""Cross-encoder reranker — ms-marco-MiniLM-L-6-v2.

Proven by AgentMemory (96.2%), OMEGA (95.4%), Chronos (95.6%), Hindsight (91.4%).
Every system scoring 90%+ on LongMemEval uses a cross-encoder reranker.

Blending: 70% original RRF score + 30% cross-encoder sigmoid score (AgentMemory approach).
~50ms for 20 documents on CPU.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from memcore.config import RERANKER_ENABLED, RERANKER_MODEL

logger = logging.getLogger(__name__)

_cross_encoder = None
_load_failed = False


def _get_cross_encoder():
    """Lazy-load the cross-encoder model on first call."""
    global _cross_encoder, _load_failed
    if _load_failed:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder model: %s", RERANKER_MODEL)
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
        logger.info("Cross-encoder loaded successfully")
        return _cross_encoder
    except Exception as e:
        logger.error("Failed to load cross-encoder: %s", e)
        _load_failed = True
        return None


def _sigmoid(x: float) -> float:
    """Sigmoid function for normalizing cross-encoder logits."""
    return 1.0 / (1.0 + math.exp(-x))


async def rerank(
    query: str,
    results: list[dict],
    top_k: int = 20,
    blend_weight: float = 0.3,
) -> list[dict]:
    """Rerank search results using cross-encoder, blending with original scores.

    Args:
        query: The search query
        results: List of dicts with 'content' and 'rrf_score' keys
        top_k: Number of results to return after reranking
        blend_weight: Weight for cross-encoder score (1-blend_weight for original)

    Returns:
        Reranked results with added 'rerank_score' and 'blended_score' keys
    """
    if not RERANKER_ENABLED or not results:
        return results[:top_k]

    encoder = _get_cross_encoder()
    if encoder is None:
        return results[:top_k]

    try:
        # Build (query, document) pairs for cross-encoder
        pairs = [(query, r.get("content", "")[:512]) for r in results]

        # Score all pairs — this is CPU-bound, ~50ms for 20 docs
        scores = encoder.predict(pairs)

        # Normalize original RRF scores to [0, 1]
        max_rrf = max((float(r.get("rrf_score", 0)) for r in results), default=1.0)
        if max_rrf == 0:
            max_rrf = 1.0

        for r, ce_score in zip(results, scores):
            ce_norm = _sigmoid(float(ce_score))
            rrf_norm = float(r.get("rrf_score", 0)) / max_rrf
            r["rerank_score"] = round(ce_norm, 4)
            r["blended_score"] = round(
                (1 - blend_weight) * rrf_norm + blend_weight * ce_norm, 4
            )

        results.sort(key=lambda x: x.get("blended_score", 0), reverse=True)
        return results[:top_k]

    except Exception as e:
        logger.error("Reranking failed: %s", e)
        return results[:top_k]
