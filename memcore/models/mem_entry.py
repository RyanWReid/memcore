"""MemEntry — core data model for the governed memory system.

Inspired by MemOS TextualMemoryItem + A-MAC scoring + Governed Memory epistemic fields.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Classification of memory content — determines storage layer routing."""

    FACT = "fact"  # stable preferences, settings, current state → PostgreSQL
    EVENT = "event"  # timestamped incidents, deployments, milestones → Graphiti
    DECISION = "decision"  # architectural choices with rationale → Graphiti
    RELATIONSHIP = "relationship"  # entity connections, dependencies → Graphiti
    DOCUMENT = "document"  # long-form content, specs, logs → PostgreSQL
    GOAL = "goal"  # current objectives, priorities → PostgreSQL
    TRAJECTORY = "trajectory"  # tool call sequences + outcomes → PostgreSQL
    TOOL_HINT = "tool_hint"  # MCP tool descriptions for retrieval → PostgreSQL


class StorageLayer(str, Enum):
    """Target storage backend."""

    GRAPHITI = "graphiti"
    POSTGRES = "postgres"


class MemStatus(str, Enum):
    ACTIVE = "active"
    RESOLVING = "resolving"  # contradiction detected, pending resolution
    ARCHIVED = "archived"
    DELETED = "deleted"


# Type → Layer routing map
TYPE_TO_LAYER: dict[MemoryType, StorageLayer] = {
    MemoryType.FACT: StorageLayer.POSTGRES,
    MemoryType.EVENT: StorageLayer.GRAPHITI,
    MemoryType.DECISION: StorageLayer.GRAPHITI,
    MemoryType.RELATIONSHIP: StorageLayer.GRAPHITI,
    MemoryType.DOCUMENT: StorageLayer.POSTGRES,
    MemoryType.GOAL: StorageLayer.POSTGRES,
    MemoryType.TRAJECTORY: StorageLayer.POSTGRES,
    MemoryType.TOOL_HINT: StorageLayer.POSTGRES,
}


class QualityChecks(BaseModel):
    """Three quality checks from Governed Memory paper."""

    coreference_ok: bool = Field(
        description="No dangling pronouns — all entities explicitly named"
    )
    self_contained: bool = Field(
        description="Meaningful without surrounding conversation context"
    )
    temporal_anchored: bool = Field(
        description="Time references resolved to explicit dates or marked relative"
    )


class EpistemicScores(BaseModel):
    """Five-factor scoring from A-MAC paper."""

    future_utility: float = Field(ge=0.0, le=1.0, description="Predicted downstream relevance")
    factual_confidence: float = Field(ge=0.0, le=1.0, description="Evidence support level")
    semantic_novelty: float = Field(ge=0.0, le=1.0, description="Distance from existing memories")
    temporal_recency: float = Field(ge=0.0, le=1.0, description="Time-decay weight")
    content_type_prior: float = Field(ge=0.0, le=1.0, description="Category base rate")

    @property
    def combined(self) -> float:
        """Weighted combination. content_type_prior is most influential per A-MAC."""
        weights = {
            "future_utility": 0.25,
            "factual_confidence": 0.20,
            "semantic_novelty": 0.15,
            "temporal_recency": 0.10,
            "content_type_prior": 0.30,
        }
        return sum(
            getattr(self, k) * v for k, v in weights.items()
        )


class TemporalAnchor(BaseModel):
    """Three-timestamp model from Mastra/Governed Memory."""

    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    referenced_at: Optional[datetime] = Field(
        default=None,
        description="Explicit date referenced in content (e.g. 'my meeting Jan 31')",
    )
    relative_offset: Optional[str] = Field(
        default=None,
        description="Human-readable offset (e.g. '3 days before incident')",
    )


class MemEntry(BaseModel):
    """A single memory entry in the governed memory system."""

    # Identity
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: MemoryType
    layer: StorageLayer
    group_id: str = Field(default="homelab", description="homelab or personal")

    # Epistemic fields — MemCore's differentiator
    quality_checks: Optional[QualityChecks] = None
    epistemic_scores: Optional[EpistemicScores] = None
    epistemic_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Combined 0.0-1.0 trustworthiness score — stored, auditable",
    )

    # Temporal
    temporal: TemporalAnchor = Field(default_factory=TemporalAnchor)

    # Provenance
    source_agent: str = Field(default="unknown")
    session_id: Optional[str] = None

    # Lifecycle
    status: MemStatus = Field(default=MemStatus.ACTIVE)

    # Gate decision
    gate_passed: bool = Field(default=False)
    gate_reason: Optional[str] = Field(
        default=None, description="Why the gate accepted or rejected this entry"
    )
