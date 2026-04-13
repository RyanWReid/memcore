#!/usr/bin/env python3
"""MemCore precision benchmark — tests recall quality against known ground truths.

Two benchmark suites:
1. Direct queries — full descriptive questions
2. Conversational queries — short prompts enriched with last assistant context

Reports precision@3 and precision@5 for each.
"""

import json
import os
import glob
import httpx

MEMCORE_URL = "http://localhost:8020"

# Test cases: (query, namespace, expected_substring_in_answer)
# Each tests whether MemCore can surface the right memory for a real question.
TEST_CASES = [
    # === EASY: Direct factual recall ===
    ("what IP is pve-hack on", "homelab", "192.168.8.13"),
    ("reverse proxy for the homelab", "homelab", "Caddy"),
    ("what does CT 103 do", "homelab", "MCP"),
    ("music streaming in the homelab", "homelab", "Navidrome"),
    ("photo enhancement model", "lusterai", "GPT Image"),

    # === MEDIUM: Semantic / synonym queries (no exact keyword match) ===
    ("how do I get alerts on my phone", "homelab", "ntfy"),
    ("which services broke during the crash", "homelab", "reboot"),
    ("self-hosted authentication gateway", "homelab", "Authelia"),
    ("where do containers get their IPs from", "homelab", "192.168.8"),
    ("AI cost optimization in the homelab", "homelab", "DeepSeek"),
    ("how to access machines remotely from browser", "homelab", "Guacamole"),
    ("what got removed because it used too much memory", "homelab", "Authentik"),
    ("two-wheeled vehicle purchase plans", "personal", "KTM"),
    ("building game worlds from satellite data", "personal", "Minecraft"),

    # === HARD: Multi-hop / reasoning queries ===
    ("what happens when uptime kuma detects a service is down", "homelab", "ntfy"),
    ("why can't the hack lab reach the main network", "homelab", "iptables"),
    ("what changed when the second proxmox node was added", "homelab", "pve02"),
    ("what tools were evaluated and rejected for knowledge graphs", "homelab", "Cognee"),
    ("why is the osint VM not running kali", "homelab", "headless"),
    ("what hardware was salvaged from the broken laptop", "homelab", "Omen"),
    ("how does claude on my phone control homelab services", "homelab", "MetaMCP"),

    # === VERY HARD: Temporal / cross-domain / abstract ===
    ("what architectural decisions were made about the third node", "homelab", "pve-hack"),
    ("what's the full path from a grafana alert to my phone buzzing", "homelab", "ntfy"),
    ("which services had to be manually restarted after the outage", "homelab", "manual"),
    ("what open source tools did we almost use but decided against", "homelab", "Cognee"),
    ("what mobile app features were intentionally deferred", "lusterai", "background"),
    ("budget constraints on personal purchases", "personal", "3,000"),
    ("what DNS problems have we had and how were they fixed", "homelab", "Pi-hole"),
    ("security risks in the current setup", "homelab", "MCP"),
    ("what's broken right now that we know about", "homelab", "stale"),
]


# Conversational test cases: (last_assistant_context, short_user_prompt, namespace, expected_substring)
# Simulates what happens when the user types a short reply mid-conversation
CONVERSATIONAL_CASES = [
    # Context about Authentik replacement
    ("We replaced Authentik with Authelia because Authentik was using over 2GB of RAM",
     "why did we do that again", "homelab", "Authelia"),

    # Context about ntfy setup
    ("The notification pipeline goes through ntfy for push alerts to your phone",
     "how does that work", "homelab", "ntfy"),

    # Context about hack lab
    ("The OSINT VM on pve-hack uses Debian instead of Kali",
     "why not kali", "homelab", "headless"),

    # Context about Cognee evaluation
    ("We evaluated several knowledge graph tools before settling on our current stack",
     "which ones did we reject", "homelab", "Cognee"),

    # Context about MetaMCP
    ("Claude Code connects to all homelab services through a single MCP endpoint",
     "how does that work from my phone", "homelab", "MetaMCP"),

    # Context about monitoring
    ("Uptime Kuma checks all services every 60 seconds",
     "what happens when something goes down", "homelab", "ntfy"),

    # Context about the motorcycle
    ("You were researching motorcycles a while back",
     "which one did I like", "personal", "KTM"),

    # Context about DNS
    ("Pi-hole handles all the DNS resolution for the homelab",
     "what problems have we had with it", "homelab", "Pi-hole"),

    # Context about pve-hack VMs
    ("pve-hack runs the offensive security lab",
     "what VMs are on it", "homelab", "wraith"),

    # Context about Minecraft project
    ("The map project converts real-world data into Minecraft worlds",
     "what does it use for data", "personal", "satellite"),

    # Context about reverse proxy
    ("All the .home.lab domains go through Caddy",
     "does it have auth", "homelab", "Authelia"),

    # Context about cost
    ("DeepSeek is what we use for LLM calls in the homelab",
     "is it expensive", "homelab", "DeepSeek"),

    # Context about broken things
    ("There's a stale iptables rule forwarding to a dead container",
     "which one", "homelab", "3000"),

    # Context about hardware
    ("The Omen 15 laptop was stripped for parts",
     "what did we get from it", "homelab", "NVMe"),

    # Context about LusterAI
    ("The LusterAI app does AI photo enhancement for real estate",
     "what model does it use", "lusterai", "GPT Image"),
]


def enrich_short_prompt(context: str, prompt: str) -> str:
    """Simulate the hook's context enrichment for short prompts."""
    if len(prompt) < 50 and len(context) > 20:
        return f"{context[:150]} {prompt}"
    return prompt


def run_benchmark():
    hits_at_3 = 0
    hits_at_5 = 0
    total = len(TEST_CASES)
    failures = []

    print(f"Running {total} test cases against MemCore")
    print(f"{'='*70}")

    for query, namespace, expected in TEST_CASES:
        try:
            resp = httpx.post(
                f"{MEMCORE_URL}/api/recall",
                json={"query": query, "group_id": namespace, "limit": 5},
                timeout=10,
            )
            results = resp.json().get("postgres", [])
        except Exception as e:
            print(f"  ERROR: {query} — {e}")
            failures.append((query, "API error"))
            continue

        contents = [r.get("content", "") for r in results]
        top3 = contents[:3]
        top5 = contents[:5]

        found_in_3 = any(expected.lower() in c.lower() for c in top3)
        found_in_5 = any(expected.lower() in c.lower() for c in top5)

        if found_in_3:
            hits_at_3 += 1
        if found_in_5:
            hits_at_5 += 1

        status = "✓" if found_in_3 else ("~" if found_in_5 else "✗")
        print(f"  {status} [{namespace:10s}] {query}")
        if not found_in_5:
            # Show what we got instead
            snippet = contents[0][:80] if contents else "(empty)"
            failures.append((query, f"expected '{expected}', got: {snippet}"))

    print(f"\n{'='*70}")
    print(f"Precision@3: {hits_at_3}/{total} ({hits_at_3/total*100:.0f}%)")
    print(f"Precision@5: {hits_at_5}/{total} ({hits_at_5/total*100:.0f}%)")

    if failures:
        print(f"\nMisses ({len(failures)}):")
        for q, reason in failures:
            print(f"  - {q}: {reason}")


def run_conversational_benchmark():
    """Test recall with short prompts enriched by conversation context."""
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_3_raw = 0  # Without enrichment for comparison
    hits_at_5_raw = 0
    total = len(CONVERSATIONAL_CASES)
    failures = []

    print(f"\nRunning {total} CONVERSATIONAL test cases")
    print(f"{'='*70}")

    for context, prompt, namespace, expected in CONVERSATIONAL_CASES:
        # Test WITH context enrichment
        enriched = enrich_short_prompt(context, prompt)
        try:
            resp = httpx.post(
                f"{MEMCORE_URL}/api/recall",
                json={"query": enriched, "group_id": namespace, "limit": 5},
                timeout=15,
            )
            results = resp.json().get("postgres", [])
        except Exception as e:
            print(f"  ERROR: {prompt} — {e}")
            failures.append((prompt, "API error"))
            continue

        contents = [r.get("content", "") for r in results]
        found_3 = any(expected.lower() in c.lower() for c in contents[:3])
        found_5 = any(expected.lower() in c.lower() for c in contents[:5])
        if found_3: hits_at_3 += 1
        if found_5: hits_at_5 += 1

        # Test WITHOUT enrichment (raw short prompt)
        try:
            resp_raw = httpx.post(
                f"{MEMCORE_URL}/api/recall",
                json={"query": prompt, "group_id": namespace, "limit": 5},
                timeout=15,
            )
            raw_results = resp_raw.json().get("postgres", [])
        except Exception:
            raw_results = []

        raw_contents = [r.get("content", "") for r in raw_results]
        raw_3 = any(expected.lower() in c.lower() for c in raw_contents[:3])
        raw_5 = any(expected.lower() in c.lower() for c in raw_contents[:5])
        if raw_3: hits_at_3_raw += 1
        if raw_5: hits_at_5_raw += 1

        # Show result
        enriched_status = "✓" if found_3 else ("~" if found_5 else "✗")
        raw_status = "✓" if raw_3 else ("~" if raw_5 else "✗")
        improvement = ""
        if found_3 and not raw_3: improvement = " ← FIXED by context"
        if not found_5 and not raw_5: improvement = ""
        if raw_3 and not found_3: improvement = " ← REGRESSED"

        print(f"  {enriched_status} (raw:{raw_status}) [{namespace:10s}] \"{prompt}\"{improvement}")
        if not found_5:
            snippet = contents[0][:60] if contents else "(empty)"
            failures.append((prompt, f"expected '{expected}', got: {snippet}"))

    print(f"\n{'='*70}")
    print(f"WITH context:    P@3: {hits_at_3}/{total} ({hits_at_3/total*100:.0f}%)  P@5: {hits_at_5}/{total} ({hits_at_5/total*100:.0f}%)")
    print(f"WITHOUT context: P@3: {hits_at_3_raw}/{total} ({hits_at_3_raw/total*100:.0f}%)  P@5: {hits_at_5_raw}/{total} ({hits_at_5_raw/total*100:.0f}%)")
    print(f"Context boost:   P@3: +{hits_at_3 - hits_at_3_raw}  P@5: +{hits_at_5 - hits_at_5_raw}")

    if failures:
        print(f"\nMisses ({len(failures)}):")
        for q, reason in failures:
            print(f"  - \"{q}\": {reason}")


if __name__ == "__main__":
    run_benchmark()
    run_conversational_benchmark()
