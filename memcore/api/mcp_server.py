"""MemCore MCP Server — governed memory system exposed as MCP tools.

Tools: remember, recall, forget, audit
Transport: SSE (for MetaMCP registration)
"""

from __future__ import annotations

import asyncio
import json
import logging

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse

from memcore.config import GATE_THRESHOLD, FACT_EXTRACTION_ENABLED
from memcore.gate.write_gate import evaluate
from memcore.gate.fact_extractor import extract_facts
from memcore.storage import router

logger = logging.getLogger(__name__)

# Limit concurrent fact extraction tasks to avoid overwhelming postgres/LLM
_fact_extraction_sem = asyncio.Semaphore(3)

# MCP Server
server = Server("memcore")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="remember",
            description=(
                "Store a memory through the epistemic write gate. Content is scored "
                f"(0.0-1.0) and only stored if score >= {GATE_THRESHOLD}. "
                "Routes to Graphiti (events/decisions), LightRAG (documents), "
                "or PostgreSQL (facts/goals) based on type classification."
            ),
            inputSchema={
                "type": "object",
                "required": ["content"],
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to evaluate and potentially store.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional surrounding context to help the gate evaluate quality.",
                        "default": "",
                    },
                    "group_id": {
                        "type": "string",
                        "description": "Memory namespace: 'homelab' for infrastructure, 'personal' for everything else.",
                        "enum": ["homelab", "personal"],
                        "default": "homelab",
                    },
                    "source_agent": {
                        "type": "string",
                        "description": "Which agent is writing this memory.",
                        "default": "claude",
                    },
                },
            },
        ),
        Tool(
            name="recall",
            description=(
                "Search memory by query. Default: fast PostgreSQL search (~0.1s). "
                "Add 'graphiti' to layers for deep graph search (slow, ~minutes). "
                "Use memory-graph MCP tools directly for Graphiti-specific queries."
            ),
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for.",
                    },
                    "group_id": {
                        "type": "string",
                        "description": "Memory namespace to search.",
                        "enum": ["homelab", "personal"],
                        "default": "homelab",
                    },
                    "layers": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["graphiti", "postgres"]},
                        "description": "Which layers to search. Default: postgres only (fast). Add graphiti for deep search (slow).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results per layer.",
                        "default": 10,
                    },
                },
            },
        ),
        Tool(
            name="forget",
            description="Mark a memory as deleted by its ID. Does not delete from Graphiti/LightRAG, but excludes from future recall.",
            inputSchema={
                "type": "object",
                "required": ["memory_id"],
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "UUID of the memory to forget.",
                    },
                },
            },
        ),
        Tool(
            name="audit",
            description="Get full details of a memory entry including epistemic scores, quality checks, and gate decision.",
            inputSchema={
                "type": "object",
                "required": ["memory_id"],
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "UUID of the memory to audit.",
                    },
                },
            },
        ),
        Tool(
            name="intent",
            description=(
                "Store a prospective memory — remember to do something when a condition is met. "
                "Example: intent(content='Check if DNS resolved', trigger='after DNS reboot'). "
                "Intents auto-surface during recall when their trigger matches the current context."
            ),
            inputSchema={
                "type": "object",
                "required": ["content", "trigger"],
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "What to remember/surface when triggered.",
                    },
                    "trigger": {
                        "type": "string",
                        "description": "Natural language trigger condition (e.g., 'when deploying to CT 100', 'after next reboot').",
                    },
                    "group_id": {
                        "type": "string",
                        "description": "Memory namespace.",
                        "enum": ["homelab", "personal"],
                        "default": "homelab",
                    },
                },
            },
        ),
        Tool(
            name="complete_intent",
            description="Mark a prospective memory intent as completed/acknowledged.",
            inputSchema={
                "type": "object",
                "required": ["intent_id"],
                "properties": {
                    "intent_id": {
                        "type": "string",
                        "description": "UUID of the intent to complete.",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "remember":
        return await _handle_remember(arguments)
    elif name == "recall":
        return await _handle_recall(arguments)
    elif name == "forget":
        return await _handle_forget(arguments)
    elif name == "audit":
        return await _handle_audit(arguments)
    elif name == "intent":
        return await _handle_intent(arguments)
    elif name == "complete_intent":
        return await _handle_complete_intent(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _extract_and_store_facts(content: str, group_id: str, source_agent: str):
    """Background task: extract atomic facts from stored memory and store them."""
    async with _fact_extraction_sem:
        try:
            facts = await extract_facts(content)
            if not facts:
                return

            from memcore.models import MemEntry, MemoryType, StorageLayer, TemporalAnchor
            from memcore.storage import postgres_store

            stored = 0
            for fact in facts:
                entry = MemEntry(
                    content=fact,
                    memory_type=MemoryType.FACT,
                    layer=StorageLayer.POSTGRES,
                    group_id=group_id,
                    epistemic_score=0.65,
                    temporal=TemporalAnchor(),
                    source_agent=source_agent,
                    gate_passed=True,
                    gate_reason="fact-extraction-from-parent",
                )
                result = await postgres_store.store(entry)
                if result.get("status") == "stored":
                    stored += 1

            if stored:
                logger.info("Fact extraction: stored %d/%d facts for group=%s", stored, len(facts), group_id)
        except Exception as e:
            logger.error("Background fact extraction failed: %s", e)


async def _handle_remember(args: dict) -> list[TextContent]:
    content = args["content"]
    context = args.get("context", "")
    group_id = args.get("group_id", "homelab")
    source_agent = args.get("source_agent", "claude")

    # Run through epistemic write gate
    entry = await evaluate(
        content=content,
        context=context,
        group_id=group_id,
        source_agent=source_agent,
    )

    if not entry.gate_passed:
        return [TextContent(
            type="text",
            text=json.dumps({
                "stored": False,
                "epistemic_score": entry.epistemic_score,
                "reason": entry.gate_reason,
                "memory_type": entry.memory_type.value,
                "quality_checks": entry.quality_checks.model_dump() if entry.quality_checks else None,
            }, indent=2),
        )]

    # Store in appropriate layer
    store_result = await router.store(entry)

    blocked = store_result.get("blocked", False)

    # Fire-and-forget fact extraction (don't block the response)
    if not blocked and FACT_EXTRACTION_ENABLED:
        asyncio.create_task(_extract_and_store_facts(content, group_id, source_agent))

    return [TextContent(
        type="text",
        text=json.dumps({
            "stored": not blocked,
            "memory_id": entry.id,
            "epistemic_score": entry.epistemic_score,
            "memory_type": entry.memory_type.value,
            "layer": entry.layer.value,
            "group_id": entry.group_id,
            "reason": store_result.get("reason") if blocked else None,
            "quality_checks": entry.quality_checks.model_dump() if entry.quality_checks else None,
            "storage_results": store_result,
        }, indent=2),
    )]


async def _handle_recall(args: dict) -> list[TextContent]:
    query = args["query"]
    group_id = args.get("group_id", "homelab")
    layers = args.get("layers")
    limit = args.get("limit", 10)

    # If specific layers requested, use the old per-layer response
    if layers:
        results = await router.recall(
            query=query,
            group_id=group_id,
            layers=layers,
            limit=limit,
        )
        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2, default=str),
        )]

    # Default: fused recall (postgres + graphiti merged into single ranked list)
    fused, confidence = await router.recall_fused(
        query=query,
        group_id=group_id,
        limit=limit,
    )
    return [TextContent(
        type="text",
        text=json.dumps({
            "results": fused,
            "count": len(fused),
            "confidence": confidence,
        }, indent=2, default=str),
    )]


async def _handle_forget(args: dict) -> list[TextContent]:
    memory_id = args["memory_id"]
    result = await router.forget(memory_id)
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_audit(args: dict) -> list[TextContent]:
    memory_id = args["memory_id"]
    entry = await router.get_entry(memory_id)
    if entry is None:
        return [TextContent(type="text", text=json.dumps({"error": "not found"}))]
    return [TextContent(type="text", text=json.dumps(entry, indent=2, default=str))]


async def _handle_intent(args: dict) -> list[TextContent]:
    from memcore.lifecycle.prospective import store_intent
    result = await store_intent(
        content=args["content"],
        trigger_condition=args["trigger"],
        group_id=args.get("group_id", "homelab"),
    )
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _handle_complete_intent(args: dict) -> list[TextContent]:
    from memcore.lifecycle.prospective import complete_intent
    result = await complete_intent(args["intent_id"])
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# --- REST API endpoints (for hooks, curl, n8n — no MCP overhead) ---

async def health(request):
    return JSONResponse({"status": "ok", "service": "memcore", "gate_threshold": GATE_THRESHOLD})


async def api_stats(request):
    """System health stats — memory counts, MW distribution, stability, drift, suppression."""
    from memcore.storage.postgres_store import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                count(*) as total,
                sum(case when access_count > 0 then 1 else 0 end) as accessed,
                sum(case when access_count >= 5 then 1 else 0 end) as well_accessed,
                round(avg(stability)::numeric, 2) as avg_stability,
                round(avg(access_count)::numeric, 2) as avg_access,
                sum(case when reconsolidation_count > 0 then 1 else 0 end) as reconsolidated,
                round(avg(case when content_drift > 0 then content_drift end)::numeric, 3) as avg_drift,
                sum(case when suppression_count > 0 then 1 else 0 end) as suppressed,
                round(avg(case when mw_total > 0 then mw_success::numeric / mw_total end)::numeric, 3) as avg_mw_ratio,
                sum(mw_total) as total_mw_retrievals,
                sum(mw_success) as total_mw_successes
            FROM mem_entries
            WHERE status = 'active'
        """)
        types = await conn.fetch("""
            SELECT memory_type, count(*), round(avg(stability)::numeric, 1) as avg_stab,
                   round(avg(access_count)::numeric, 1) as avg_access
            FROM mem_entries WHERE status = 'active'
            GROUP BY memory_type ORDER BY count(*) DESC
        """)
        intents = await conn.fetchrow("""
            SELECT count(*) as total,
                   sum(case when status = 'active' then 1 else 0 end) as active,
                   sum(case when status = 'completed' then 1 else 0 end) as completed
            FROM mem_intents
        """)
    return JSONResponse({
        "memories": {
            "total": row["total"],
            "accessed": row["accessed"],
            "well_accessed_5plus": row["well_accessed"],
            "avg_stability": float(row["avg_stability"] or 0),
            "avg_access_count": float(row["avg_access"] or 0),
        },
        "lifecycle": {
            "reconsolidated": row["reconsolidated"],
            "avg_content_drift": float(row["avg_drift"] or 0),
            "suppressed": row["suppressed"],
        },
        "memory_worth": {
            "total_retrievals": row["total_mw_retrievals"],
            "total_successes": row["total_mw_successes"],
            "avg_success_ratio": float(row["avg_mw_ratio"] or 0),
        },
        "intents": {
            "total": intents["total"],
            "active": intents["active"],
            "completed": intents["completed"],
        },
        "by_type": [
            {"type": t["memory_type"], "count": t["count"],
             "avg_stability": float(t["avg_stab"] or 0),
             "avg_access": float(t["avg_access"] or 0)}
            for t in types
        ],
    })


async def api_mw_success(request):
    """Mark memories as successful retrievals (Memory Worth signal).

    POST {"memory_ids": ["id1", "id2", ...]}
    Called by the retain hook when recalled memories contributed to the response.
    """
    body = await request.json()
    memory_ids = body.get("memory_ids", [])
    if not memory_ids:
        return JSONResponse({"error": "memory_ids required"}, status_code=400)
    from memcore.storage.postgres_store import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE mem_entries
            SET mw_success = mw_success + 1
            WHERE id = ANY($1::text[])
        """, memory_ids)
    return JSONResponse({"status": "ok", "updated": len(memory_ids)})


async def api_recall(request):
    """REST endpoint for recall. POST {"query": "...", "group_id": "homelab", "limit": 10}

    Default: fused recall (postgres + graphiti merged).
    Pass "layers" to use per-layer response instead.
    """
    body = await request.json()
    query = body.get("query", "")
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)

    layers = body.get("layers")
    group_id = body.get("group_id", "homelab")
    limit = body.get("limit", 5)

    if layers:
        # Explicit layers requested — per-layer response (backward compat)
        results = await router.recall(
            query=query, group_id=group_id, layers=layers, limit=limit,
        )
        serializable = json.loads(json.dumps(results, default=str))
        return JSONResponse(serializable, status_code=200)

    # Default: fused recall with metamemory confidence
    fused, confidence = await router.recall_fused(query=query, group_id=group_id, limit=limit)
    serializable = json.loads(json.dumps({
        "results": fused, "count": len(fused), "confidence": confidence,
    }, default=str))
    return JSONResponse(serializable, status_code=200)


async def api_remember(request):
    """REST endpoint for remember. POST {"content": "...", "group_id": "homelab"}"""
    body = await request.json()
    content = body.get("content", "")
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)

    entry = await evaluate(
        content=content,
        context=body.get("context", ""),
        group_id=body.get("group_id", "homelab"),
        source_agent=body.get("source_agent", "hook"),
    )

    result = {"stored": False, "epistemic_score": entry.epistemic_score, "memory_type": entry.memory_type.value}

    if entry.gate_passed:
        store_result = await router.store(entry)
        blocked = store_result.get("blocked", False)
        result["stored"] = not blocked
        result["memory_id"] = entry.id
        result["storage_results"] = store_result
        if blocked:
            result["reason"] = store_result.get("reason", "duplicate")
        elif FACT_EXTRACTION_ENABLED:
            asyncio.create_task(_extract_and_store_facts(
                content, body.get("group_id", "homelab"), body.get("source_agent", "hook"),
            ))
    else:
        result["reason"] = entry.gate_reason

    return JSONResponse(result, status_code=200)


async def api_ingest(request):
    """Bulk ingest endpoint — bypasses write gate, accepts custom timestamps.

    POST {"content": "...", "group_id": "...", "created_at": "2023-01-15T14:30:00Z"}
    Used for benchmarks and historical data injection.
    """
    body = await request.json()
    content = body.get("content", "")
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)

    from memcore.models import MemEntry, MemoryType, StorageLayer, TemporalAnchor
    from memcore.storage import postgres_store
    from datetime import datetime

    group_id = body.get("group_id", "benchmark")
    created_at_str = body.get("created_at")
    temporal = TemporalAnchor()
    if created_at_str:
        try:
            temporal.ingested_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    entry = MemEntry(
        content=content,
        memory_type=MemoryType.FACT,
        layer=StorageLayer.POSTGRES,
        group_id=group_id,
        epistemic_score=0.8,
        temporal=temporal,
        source_agent=body.get("source_agent", "benchmark"),
        gate_passed=True,
        gate_reason="benchmark-ingest-bypass",
    )

    result = await postgres_store.store(entry)

    # Override created_at in DB if provided
    if created_at_str:
        try:
            pool = await postgres_store.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE mem_entries SET created_at = $1 WHERE id = $2",
                    temporal.ingested_at,
                    entry.id,
                )
        except Exception:
            pass

    return JSONResponse({
        "stored": True,
        "memory_id": entry.id,
        "group_id": group_id,
    }, status_code=200)


async def api_clear_group(request):
    """Delete all memories in a group. POST {"group_id": "lme_xxx"}"""
    body = await request.json()
    group_id = body.get("group_id", "")
    if not group_id:
        return JSONResponse({"error": "group_id required"}, status_code=400)

    from memcore.storage import postgres_store
    pool = await postgres_store.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM mem_entries WHERE group_id = $1", group_id
        )
    count = int(result.split()[-1]) if result else 0
    return JSONResponse({"deleted": count, "group_id": group_id}, status_code=200)


async def api_forget(request):
    """REST endpoint for forget. POST {"memory_id": "uuid"}"""
    body = await request.json()
    memory_id = body.get("memory_id", "")
    if not memory_id:
        return JSONResponse({"error": "memory_id required"}, status_code=400)

    result = await router.forget(memory_id)
    return JSONResponse(result, status_code=200)


def create_app() -> Starlette:
    """Create Starlette app with SSE transport for MCP + REST API."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    async def api_intent(request):
        """REST endpoint for prospective memory. POST {"content": "...", "trigger": "..."}"""
        body = await request.json()
        content = body.get("content", "")
        trigger = body.get("trigger", "")
        if not content or not trigger:
            return JSONResponse({"error": "content and trigger required"}, status_code=400)
        from memcore.lifecycle.prospective import store_intent
        result = await store_intent(
            content=content,
            trigger_condition=trigger,
            group_id=body.get("group_id", "homelab"),
        )
        return JSONResponse(json.loads(json.dumps(result, default=str)), status_code=200)

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/api/stats", api_stats),
            Route("/api/recall", api_recall, methods=["POST"]),
            Route("/api/remember", api_remember, methods=["POST"]),
            Route("/api/forget", api_forget, methods=["POST"]),
            Route("/api/ingest", api_ingest, methods=["POST"]),
            Route("/api/clear_group", api_clear_group, methods=["POST"]),
            Route("/api/intent", api_intent, methods=["POST"]),
            Route("/api/mw_success", api_mw_success, methods=["POST"]),
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )
    return app
