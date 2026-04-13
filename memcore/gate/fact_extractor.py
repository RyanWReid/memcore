"""Write-time fact extraction — extracts atomic facts from memory content.

Proven by LongMemEval paper: +9.4% recall, +5.4% QA accuracy.
Extracts 2-5 self-contained facts per memory, stores as separate searchable entries.
"""

from __future__ import annotations

import logging
import re

from .llm_client import llm_json_call

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
Extract 2-5 atomic facts from this memory that would be useful to recall later.

Each fact must be:
- Self-contained (no pronouns — use explicit names/entities)
- A single piece of durable information (NOT ephemeral actions like "pushed code" or "ran a script")
- Include any dates, versions, IPs, ports, or specific config values mentioned
- Worth remembering in a future conversation (skip trivial observations)

Do NOT extract facts about:
- What Claude/the assistant did (committed, pushed, updated, built, deployed code)
- Temporary task progress (completion times, log file paths, retry counts)
- Generic observations ("uses HTTP", "is free", "has RAM")
- Benchmark results or scores from memory system testing

Return JSON:
{
  "facts": ["fact1", "fact2", "fact3"]
}

If the content is too short or trivial to extract meaningful facts from, return:
{"facts": []}
"""

# Skip extraction for content shorter than this.
# Short memories ARE atomic facts — extracting smaller pieces just creates redundancy
# with the parent. Only extract from long, dense memories where facts are buried
# in narrative.
_MIN_CONTENT_LENGTH = 250

# Extracted facts must be at least this long to be stored
_MIN_FACT_LENGTH = 60


async def extract_facts(content: str) -> list[str]:
    """Extract atomic facts from memory content via LLM.

    Returns a list of 0-5 fact strings. Returns empty list on failure or
    if content is too short to extract meaningful facts from.
    """
    if len(content.strip()) < _MIN_CONTENT_LENGTH:
        return []

    try:
        result = await llm_json_call(EXTRACTION_PROMPT, content)
        facts = result.get("facts", [])
        if not isinstance(facts, list):
            return []
        # Filter: must be strings, non-trivial length, max 3 (was 5 — too many)
        return [f for f in facts if isinstance(f, str) and len(f) >= _MIN_FACT_LENGTH][:3]
    except Exception as e:
        logger.warning("Fact extraction failed: %s", e)
        return []
