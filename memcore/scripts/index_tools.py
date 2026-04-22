#!/usr/bin/env python3
"""Index MCP tool descriptions into MemCore as tool_hint memories.

Stores one memory per MCP server with user-voice intent phrases.
These surface during recall to nudge Claude toward MCP tools over shell.

Usage:
    python -m memcore.scripts.index_tools [--base-url http://localhost:8020]
"""

import argparse
import os
import sys
import requests

# Hardcoded tool catalog — NOT LLM-generated.
# User-voice intent phrases for semantic matching.
TOOLS = {
    "proxmox": {
        "description": "Proxmox container/VM management (55 tools): create, destroy, start, stop, list, migrate, snapshot, configure resources",
        "intents": [
            "create a container",
            "list my VMs and containers",
            "restart a container",
            "check container status",
            "take a VM snapshot",
            "change container resources",
            "start or stop a VM",
        ],
    },
    "home-assistant": {
        "description": "Smart home control: lights, automations, sensors, presence (CT 107)",
        "intents": [
            "turn on the lights",
            "check sensor readings",
            "trigger an automation",
            "what devices are home",
            "set light brightness or color",
        ],
    },
    "github": {
        "description": "GitHub API: repos, PRs, issues, branches, commits, releases",
        "intents": [
            "create a pull request",
            "check PR status",
            "list open issues",
            "create a GitHub issue",
            "merge a pull request",
        ],
    },
    "n8n": {
        "description": "n8n workflow automation on CT 110: create, run, inspect workflows",
        "intents": [
            "create a workflow",
            "run an automation",
            "check workflow status",
            "list n8n workflows",
        ],
    },
    "fetch": {
        "description": "Fetch web content: GET pages, download files, scrape text",
        "intents": [
            "fetch a URL",
            "download a web page",
            "get content from a website",
        ],
    },
    "playwright": {
        "description": "Browser automation: navigate, click, fill forms, screenshot JS-rendered pages",
        "intents": [
            "automate a browser",
            "fill out a web form",
            "take a browser screenshot",
            "scrape a JavaScript page",
        ],
    },
    "postgres": {
        "description": "PostgreSQL queries against MetaMCP DB or project databases",
        "intents": [
            "run a SQL query",
            "check database tables",
            "query the database",
        ],
    },
    "docker": {
        "description": "Docker container management: list, start, stop, logs, exec into containers",
        "intents": [
            "list docker containers",
            "check container logs",
            "restart a docker container",
            "exec into a container",
        ],
    },
    "filesystem": {
        "description": "File operations on CT 103: read, write, list, search files",
        "intents": [
            "read a file on CT 103",
            "list files on the MCP hub",
        ],
    },
    "git": {
        "description": "Git repo operations: clone, commit, branch, diff, log",
        "intents": [
            "check git status",
            "create a git branch",
            "view git log",
        ],
    },
    "memcore": {
        "description": "Governed memory system: remember facts/decisions/events, recall with confidence scoring, forget, audit",
        "intents": [
            "remember something",
            "recall a memory",
            "what do we know about",
            "store a decision",
            "forget a memory",
        ],
    },
    "memory-graph": {
        "description": "Graphiti temporal knowledge graph: entity relationships, decisions, events, temporal queries",
        "intents": [
            "search the knowledge graph",
            "find entity relationships",
            "query decision history",
        ],
    },
    "gpt-researcher": {
        "description": "Deep web research: multi-source synthesis, quick search, write research reports via SearXNG + LiteLLM",
        "intents": [
            "research a topic",
            "do deep research",
            "find information online",
            "write a research report",
        ],
    },
    "coolify": {
        "description": "Coolify PaaS management on CT 111: deploy apps, manage env vars, restart services",
        "intents": [
            "deploy an app",
            "check deployment status",
            "update environment variables",
            "restart a Coolify service",
        ],
    },
    "paperless": {
        "description": "Paperless-ngx document management: search, tag, upload, organize documents",
        "intents": [
            "search documents",
            "find a receipt or document",
            "tag a document",
            "upload to paperless",
        ],
    },
    "tailscale": {
        "description": "Tailscale management (42 tools): devices, DNS, routes, keys, users, ACLs",
        "intents": [
            "list tailscale devices",
            "check tailscale status",
            "manage tailscale DNS",
            "add a tailscale route",
        ],
    },
    "mac-desktop": {
        "description": "Full Mac control from any session: commands, files, apps via Desktop Commander",
        "intents": [
            "run something on the Mac",
            "open an app on MacBook",
            "manage Mac files remotely",
        ],
    },
    "mac-peekaboo": {
        "description": "Mac screen capture, OCR, running apps inspection",
        "intents": [
            "take a Mac screenshot",
            "what's on the Mac screen",
            "list running Mac apps",
        ],
    },
    "xcodebuild-mcp": {
        "description": "Xcode build, test, simulator, debugging (27 tools)",
        "intents": [
            "build an Xcode project",
            "run iOS tests",
            "debug an Xcode build",
        ],
    },
    "ios-simulator-mcp": {
        "description": "iOS simulator: screenshots, tap, swipe, UI hierarchy inspection",
        "intents": [
            "take a simulator screenshot",
            "tap on the iOS simulator",
            "inspect simulator UI",
        ],
    },
    "maestro-mcp": {
        "description": "Mobile E2E test automation via Maestro CLI",
        "intents": [
            "run mobile E2E tests",
            "automate mobile testing",
        ],
    },
    "mermaid": {
        "description": "Mermaid diagram generation: flowcharts, ERDs, sequence diagrams, C4. Renders to SVG/PNG",
        "intents": [
            "create a diagram",
            "generate a flowchart",
            "make an architecture diagram",
            "draw a sequence diagram",
        ],
    },
    "drawio": {
        "description": "Draw.io diagram editor: create/edit .drawio files, export PNG/SVG/PDF",
        "intents": [
            "create a drawio diagram",
            "edit a diagram file",
            "export diagram to PNG",
        ],
    },
    "linear": {
        "description": "Linear project management: issues, projects, cycles, labels, teams",
        "intents": [
            "create a Linear issue",
            "list Linear tasks",
            "check project status in Linear",
        ],
    },
    "sequential-thinking": {
        "description": "Step-by-step reasoning for complex multi-step tasks",
        "intents": [
            "think through a complex problem",
            "break down a hard question",
        ],
    },
    "hevy": {
        "description": "Workout tracking via Hevy app: exercises, routines, history",
        "intents": [
            "check my workouts",
            "log an exercise",
            "view workout history",
        ],
    },
}


def build_content(name: str, info: dict) -> str:
    """Build the memory content string for a tool hint."""
    intents_str = ", ".join(info["intents"])
    return f"[{name} MCP] {info['description']}. Use when: {intents_str}."


def index_all(base_url: str, group_id: str = "mcp_tools"):
    """POST each tool hint to /api/ingest."""
    success = 0
    errors = 0

    for name, info in TOOLS.items():
        content = build_content(name, info)
        payload = {
            "content": content,
            "group_id": group_id,
            "memory_type": "tool_hint",
            "source_agent": "tool-indexer",
        }
        try:
            resp = requests.post(
                f"{base_url}/api/ingest",
                json=payload,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                print(f"  OK  {name}: {data.get('memory_id', '?')}")
                success += 1
            else:
                print(f"  ERR {name}: HTTP {resp.status_code} — {resp.text[:100]}")
                errors += 1
        except Exception as e:
            print(f"  ERR {name}: {e}")
            errors += 1

    print(f"\nDone: {success} indexed, {errors} errors, {len(TOOLS)} total")
    return errors == 0


def main():
    parser = argparse.ArgumentParser(description="Index MCP tools into MemCore")
    parser.add_argument(
        "--base-url",
        default=os.getenv("MEMCORE_URL", "http://localhost:8020"),
        help="MemCore API base URL",
    )
    parser.add_argument(
        "--group-id",
        default="mcp_tools",
        help="Group ID for tool hints",
    )
    args = parser.parse_args()

    print(f"Indexing {len(TOOLS)} MCP tools into MemCore at {args.base_url}...")
    ok = index_all(args.base_url, args.group_id)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
