"""Fuzzy-trace gist generation.

Brainerd & Reyna (1990): the brain stores two parallel representations of every
memory — a verbatim trace (exact detail) and a gist trace (meaning/category).
Verbatim decays fast; gist persists and generalizes better. People
preferentially reason from gist when answering conceptual questions.

MemCore mirrors this by generating a one-sentence gist per memory at write
time, embedding it separately, and RRF-fusing its vector leg into hybrid
search. A conceptual query ("reverse proxy decision") can match the gist
("Chose Caddy over NPM for reverse proxy") even when the verbatim
("Deployed Caddy on CT 100 at 192.168.8.100 with CrowdSec bouncer and
custom Caddyfile at /etc/caddy/...") is embedding-dominated by specifics.
"""

import logging

import httpx

from memcore.config import (
    GIST_MODEL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
)

logger = logging.getLogger(__name__)


GIST_PROMPT = (
    "Summarize this memory in ONE sentence that captures the core "
    "decision/fact/event without specific identifiers (IPs, ports, file "
    "paths, version numbers, exact names of hosts/variables). Keep the "
    "WHAT and WHY; drop the HOW. Output the sentence only, no preamble."
)


async def generate_gist(content: str) -> str | None:
    """Ask the gist model for a one-sentence abstract of the memory.

    Returns None on failure — callers should fall back to storing no gist
    rather than blocking the write.
    """
    if not content or len(content) < 40:
        # Too short to usefully abstract — the content IS its own gist.
        return content or None

    try:
        async with httpx.AsyncClient(timeout=6) as client:
            resp = await client.post(
                f"{LITELLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                json={
                    "model": GIST_MODEL,
                    "messages": [
                        {"role": "system", "content": GIST_PROMPT},
                        {"role": "user", "content": content[:2000]},
                    ],
                    "temperature": 0,
                    "max_tokens": 80,
                },
            )
            resp.raise_for_status()
            gist = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip quotes the model sometimes wraps the output in
            gist = gist.strip("\"'`")
            if not gist or len(gist) < 10:
                return None
            return gist[:500]
    except Exception as e:
        logger.warning("Gist generation failed: %s", e)
        return None
