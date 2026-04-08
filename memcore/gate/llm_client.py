"""Thin OpenAI-compatible client for LiteLLM gateway calls."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from memcore.config import LITELLM_API_KEY, LITELLM_BASE_URL, GATE_MODEL

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=LITELLM_BASE_URL,
            headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
            timeout=30.0,
        )
    return _client


async def llm_json_call(
    system_prompt: str,
    user_prompt: str,
    model: str = GATE_MODEL,
) -> dict[str, Any]:
    """Call LLM via LiteLLM and parse JSON response.

    Uses response_format for structured output when available,
    falls back to parsing JSON from markdown code blocks.
    """
    client = _get_client()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    resp = await client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Parse JSON — handle both raw and markdown-wrapped
    content = content.strip()
    if content.startswith("```"):
        # Strip markdown code fence
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM JSON response: %s", content[:200])
        raise
