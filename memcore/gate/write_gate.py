"""Epistemic Write Gate — the core intelligence of MemCore.

Three quality checks (Governed Memory paper) + five-factor scoring (A-MAC paper).
All heavy lifting at ingest time, not query time.

Fast-path heuristic: cheap checks first, LLM only for borderline cases.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional

from memcore.config import GATE_THRESHOLD
from memcore.models import (
    EpistemicScores,
    MemEntry,
    MemoryType,
    MemStatus,
    QualityChecks,
    StorageLayer,
    TemporalAnchor,
    TYPE_TO_LAYER,
)

from .llm_client import llm_json_call

logger = logging.getLogger(__name__)

# Content type priors — base rates from A-MAC. Most influential factor.
CONTENT_TYPE_PRIORS: dict[MemoryType, float] = {
    MemoryType.DECISION: 0.90,  # decisions are almost always worth storing
    MemoryType.EVENT: 0.75,
    MemoryType.RELATIONSHIP: 0.70,
    MemoryType.DOCUMENT: 0.65,
    MemoryType.GOAL: 0.60,
    MemoryType.FACT: 0.55,
    MemoryType.TRAJECTORY: 0.40,
}

# --- Heuristic fast-path ---

# Keywords that strongly signal memory types (skip LLM classification)
_DECISION_KEYWORDS = re.compile(
    r"\b(chose|decided|replaced|switched to|migrated to|picked|selected|over|instead of|rationale|because we)\b",
    re.IGNORECASE,
)
_EVENT_KEYWORDS = re.compile(
    r"\b(deployed|installed|created|removed|destroyed|fixed|patched|upgraded|migrated|incident|outage)\b",
    re.IGNORECASE,
)
_FACT_KEYWORDS = re.compile(
    r"\b(IP|port|password|key|token|URL|config|setting|version|running on|located at)\b",
    re.IGNORECASE,
)

# Minimum content length to be self-contained
_MIN_CONTENT_LENGTH = 20
# Maximum length before we consider it a document
_DOCUMENT_LENGTH_THRESHOLD = 2000


def heuristic_precheck(content: str) -> tuple[str, float | None, MemoryType | None]:
    """Fast heuristic check before LLM calls.

    Returns:
        (decision, score_estimate, type_hint)
        decision: "reject" | "accept" | "borderline"
        score_estimate: rough score or None
        type_hint: guessed type or None
    """
    stripped = content.strip()

    # Immediate reject: too short to be meaningful
    if len(stripped) < _MIN_CONTENT_LENGTH:
        return "reject", 0.1, MemoryType.FACT

    # Immediate reject: looks like a greeting or ack
    lower = stripped.lower()
    if lower in ("ok", "yes", "no", "thanks", "got it", "sure", "done", "hi", "hello"):
        return "reject", 0.05, MemoryType.FACT

    # Type hints from keywords
    type_hint = None
    if _DECISION_KEYWORDS.search(stripped):
        type_hint = MemoryType.DECISION
    elif _EVENT_KEYWORDS.search(stripped):
        type_hint = MemoryType.EVENT
    elif len(stripped) > _DOCUMENT_LENGTH_THRESHOLD:
        type_hint = MemoryType.DOCUMENT
    elif _FACT_KEYWORDS.search(stripped):
        type_hint = MemoryType.FACT

    # High-confidence accept: long, specific content with decision/event keywords
    word_count = len(stripped.split())
    if word_count >= 15 and type_hint in (MemoryType.DECISION, MemoryType.EVENT):
        prior = CONTENT_TYPE_PRIORS.get(type_hint, 0.5)
        return "accept", prior, type_hint

    return "borderline", None, type_hint

QUALITY_CHECK_PROMPT = """\
You are an epistemic quality checker for a memory system. Evaluate this memory candidate.

Return JSON with exactly these fields:
{
  "coreference_ok": true/false,
  "self_contained": true/false,
  "temporal_anchored": true/false,
  "explanation": "brief reason for each check"
}

Rules:
- coreference_ok: Are all entities explicitly named? No dangling "it", "they", "that service" without an antecedent.
- self_contained: Is this meaningful without the surrounding conversation? Would someone reading just this text understand what it refers to?
- temporal_anchored: If time is referenced ("yesterday", "last week", "recently"), is there an explicit date? If no time reference, this is true by default.
"""

SCORING_PROMPT = """\
You are a memory admission scorer. Rate this memory candidate on five factors.

Return JSON with exactly these fields, each a float 0.0 to 1.0:
{
  "future_utility": 0.0-1.0,
  "factual_confidence": 0.0-1.0,
  "semantic_novelty": 0.0-1.0,
  "temporal_recency": 0.0-1.0,
  "reasoning": "brief justification"
}

Scoring guide:
- future_utility: How likely is this to be useful in a future conversation? High for decisions, architecture, preferences. Low for ephemeral status, greetings.
- factual_confidence: Is this backed by evidence or observation, or is it speculation/hearsay? High for things the agent directly observed or verified.
- semantic_novelty: Does this add new information? High if it's genuinely new knowledge. Low if it restates something obvious or already known.
- temporal_recency: Is this about current/recent state (high) or historical context that may be stale (lower)?
"""

CLASSIFY_PROMPT = """\
You are a memory type classifier. Classify this memory into exactly one type and extract temporal information.

Return JSON:
{
  "memory_type": "fact|event|decision|relationship|document|goal|trajectory",
  "referenced_date": "YYYY-MM-DD or null if no specific date referenced",
  "relative_offset": "human-readable time offset or null",
  "reasoning": "brief justification"
}

Type definitions:
- fact: A stable preference, setting, configuration, or current state
- event: A timestamped incident, deployment, milestone, or change
- decision: An architectural or design choice with rationale
- relationship: A connection or dependency between entities/systems
- document: Long-form content, spec, log, or reference material
- goal: A current objective, priority, or planned work
- trajectory: A sequence of actions and their outcomes
"""


async def run_quality_checks(content: str, context: str = "") -> QualityChecks:
    """Run 3 quality checks via LLM."""
    user_msg = f"Memory candidate:\n{content}"
    if context:
        user_msg += f"\n\nContext provided:\n{context}"

    result = await llm_json_call(QUALITY_CHECK_PROMPT, user_msg)

    return QualityChecks(
        coreference_ok=result.get("coreference_ok", False),
        self_contained=result.get("self_contained", False),
        temporal_anchored=result.get("temporal_anchored", True),
    )


async def run_scoring(content: str, context: str = "") -> EpistemicScores:
    """Run 5-factor scoring via LLM (4 LLM-assessed + 1 content_type_prior set later)."""
    user_msg = f"Memory candidate:\n{content}"
    if context:
        user_msg += f"\n\nContext provided:\n{context}"

    result = await llm_json_call(SCORING_PROMPT, user_msg)

    return EpistemicScores(
        future_utility=max(0.0, min(1.0, float(result.get("future_utility", 0.5)))),
        factual_confidence=max(0.0, min(1.0, float(result.get("factual_confidence", 0.5)))),
        semantic_novelty=max(0.0, min(1.0, float(result.get("semantic_novelty", 0.5)))),
        temporal_recency=max(0.0, min(1.0, float(result.get("temporal_recency", 0.5)))),
        content_type_prior=0.5,  # set after classification
    )


async def classify_and_extract_temporal(
    content: str, context: str = ""
) -> tuple[MemoryType, TemporalAnchor]:
    """Classify memory type and extract temporal information."""
    user_msg = f"Memory candidate:\n{content}"
    if context:
        user_msg += f"\n\nContext provided:\n{context}"

    result = await llm_json_call(CLASSIFY_PROMPT, user_msg)

    # Parse memory type
    type_str = result.get("memory_type", "fact").lower()
    try:
        memory_type = MemoryType(type_str)
    except ValueError:
        logger.warning("Unknown memory type '%s', defaulting to fact", type_str)
        memory_type = MemoryType.FACT

    # Parse temporal
    referenced_at = None
    ref_date = result.get("referenced_date")
    if ref_date and ref_date != "null":
        try:
            referenced_at = datetime.strptime(ref_date, "%Y-%m-%d")
        except ValueError:
            pass

    temporal = TemporalAnchor(
        ingested_at=datetime.utcnow(),
        referenced_at=referenced_at,
        relative_offset=result.get("relative_offset"),
    )

    return memory_type, temporal


async def evaluate(
    content: str,
    context: str = "",
    group_id: str = "homelab",
    source_agent: str = "unknown",
    session_id: Optional[str] = None,
) -> MemEntry:
    """Full write gate evaluation pipeline.

    Fast-path: heuristic precheck rejects garbage and fast-tracks obvious content
    without any LLM calls (~0ms, $0). LLM only called for borderline cases (~3s, ~$0.003).
    """
    # --- Fast-path heuristic ---
    decision, score_est, type_hint = heuristic_precheck(content)

    if decision == "reject":
        memory_type = type_hint or MemoryType.FACT
        layer = TYPE_TO_LAYER[memory_type]
        entry = MemEntry(
            content=content,
            memory_type=memory_type,
            layer=layer,
            group_id=group_id,
            epistemic_score=round(score_est or 0.1, 3),
            temporal=TemporalAnchor(ingested_at=datetime.utcnow()),
            source_agent=source_agent,
            session_id=session_id,
            status=MemStatus.ARCHIVED,
            gate_passed=False,
            gate_reason=f"Heuristic reject: score {score_est:.2f} (too short or trivial)",
        )
        logger.info("Gate FAST-DROP: %s | %s", entry.gate_reason, content[:80])
        return entry

    if decision == "accept" and type_hint is not None:
        # Fast-track: still do LLM classification for accuracy, but skip quality+scoring
        memory_type, temporal = await classify_and_extract_temporal(content, context)
        layer = TYPE_TO_LAYER[memory_type]
        epistemic_score = score_est or CONTENT_TYPE_PRIORS.get(memory_type, 0.5)
        entry = MemEntry(
            content=content,
            memory_type=memory_type,
            layer=layer,
            group_id=group_id,
            epistemic_score=round(epistemic_score, 3),
            temporal=temporal,
            source_agent=source_agent,
            session_id=session_id,
            status=MemStatus.ACTIVE,
            gate_passed=True,
            gate_reason=f"Heuristic accept: score {epistemic_score:.2f} (1 LLM call for classification)",
        )
        logger.info(
            "Gate FAST-PASS: score=%.3f type=%s layer=%s | %s",
            epistemic_score, memory_type.value, layer.value, content[:80],
        )
        return entry

    # --- Full LLM evaluation (borderline cases) ---
    quality = await run_quality_checks(content, context)
    memory_type, temporal = await classify_and_extract_temporal(content, context)
    scores = await run_scoring(content, context)

    # Set content_type_prior from the classified type
    scores.content_type_prior = CONTENT_TYPE_PRIORS.get(memory_type, 0.5)

    # Compute combined epistemic score
    epistemic_score = scores.combined

    # Quality check penalty: each failed check reduces score by 0.1
    quality_penalty = 0.0
    if not quality.coreference_ok:
        quality_penalty += 0.1
    if not quality.self_contained:
        quality_penalty += 0.1
    if not quality.temporal_anchored:
        quality_penalty += 0.1
    epistemic_score = max(0.0, epistemic_score - quality_penalty)

    # Gate decision
    gate_passed = epistemic_score >= GATE_THRESHOLD
    if gate_passed:
        gate_reason = f"Score {epistemic_score:.2f} >= threshold {GATE_THRESHOLD} (full LLM eval)"
    else:
        gate_reason = f"Score {epistemic_score:.2f} < threshold {GATE_THRESHOLD}"
        failures = []
        if not quality.coreference_ok:
            failures.append("coreference")
        if not quality.self_contained:
            failures.append("self-containment")
        if not quality.temporal_anchored:
            failures.append("temporal anchoring")
        if failures:
            gate_reason += f" (failed: {', '.join(failures)})"

    layer = TYPE_TO_LAYER[memory_type]

    entry = MemEntry(
        content=content,
        memory_type=memory_type,
        layer=layer,
        group_id=group_id,
        quality_checks=quality,
        epistemic_scores=scores,
        epistemic_score=round(epistemic_score, 3),
        temporal=temporal,
        source_agent=source_agent,
        session_id=session_id,
        status=MemStatus.ACTIVE if gate_passed else MemStatus.ARCHIVED,
        gate_passed=gate_passed,
        gate_reason=gate_reason,
    )

    logger.info(
        "Gate %s: score=%.3f type=%s layer=%s | %s",
        "PASS" if gate_passed else "DROP",
        epistemic_score,
        memory_type.value,
        layer.value,
        content[:80],
    )

    return entry
