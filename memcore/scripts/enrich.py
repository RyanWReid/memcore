#!/usr/bin/env python3
"""Enrich existing memories with search keyword aliases.

Reads memories via MemCore recall, asks DeepSeek for aliases,
then does a bulk SQL UPDATE via a single SQL file pushed through the SSH chain.
"""

import json
import subprocess
import sys
import time
import httpx

MEMCORE_URL = "http://192.168.8.141:8020"
LITELLM_URL = "http://192.168.8.150:4000/v1"
LITELLM_KEY = "sk-69415b996802d1a9fce35cad94e79a93"
EMBEDDING_URL = "http://192.168.8.141:8100/v1/embeddings"
MODEL = "deepseek-chat"


def get_all_memories() -> list[dict]:
    """Fetch all active memory IDs and content via broad recall queries."""
    all_mems = {}

    # Search with broad terms to get everything
    queries = [
        "infrastructure deployment service container",
        "decision architecture design",
        "monitoring alerting notification",
        "network firewall DNS IP",
        "project app mobile development",
        "AI model LLM pipeline",
        "hack security OSINT Kali",
        "storage disk NVMe hardware",
        "Minecraft map tile render",
        "motorcycle vehicle purchase",
        "photo enhancement real estate",
        "Proxmox node container VM",
        "MCP MetaMCP tool integration",
        "Caddy Authelia SSO proxy",
        "backup incident reboot crash",
        "research knowledge graph",
    ]

    for ns in ["homelab", "personal", "lusterai", "work"]:
        for q in queries:
            try:
                resp = httpx.post(
                    f"{MEMCORE_URL}/api/recall",
                    json={"query": q, "group_id": ns, "limit": 20},
                    timeout=15,
                )
                for r in resp.json().get("postgres", []):
                    all_mems[r["id"]] = {"content": r["content"], "group_id": ns}
            except Exception:
                pass

    return [{"id": k, **v} for k, v in all_mems.items()]


def get_aliases(content: str) -> str:
    """Ask DeepSeek for search keyword synonyms."""
    try:
        resp = httpx.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": (
                        "Given this memory, output ONLY a parenthetical with 3-5 search keywords/synonyms "
                        "that someone might use to find it. Use DIFFERENT vocabulary than what's already in the text. "
                        "Format: (keyword1, keyword2, keyword3, keyword4)\n"
                        "Output ONLY the parenthetical, nothing else."
                    )},
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
                "max_tokens": 50,
            },
            timeout=10,
        )
        resp.raise_for_status()
        aliases = resp.json()["choices"][0]["message"]["content"].strip()
        if not aliases.startswith("("):
            aliases = f"({aliases})"
        if not aliases.endswith(")"):
            aliases = f"{aliases})"
        return aliases
    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
        return ""


def get_embedding(text: str) -> list[float] | None:
    """Generate embedding."""
    try:
        resp = httpx.post(
            EMBEDDING_URL,
            json={"input": text[:8000], "model": "nomic-embed-text"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception:
        return None


def has_aliases(content: str) -> bool:
    """Check if content already ends with a keyword alias parenthetical."""
    tail = content.rstrip()
    if not tail.endswith(")"):
        return False
    paren_start = tail.rfind("(", max(0, len(tail) - 100))
    if paren_start == -1:
        return False
    inner = tail[paren_start + 1:-1]
    parts = [p.strip() for p in inner.split(",")]
    return len(parts) >= 3 and all(len(p) < 30 for p in parts)


def main():
    print("Fetching all memories...")
    memories = get_all_memories()
    print(f"Found {len(memories)} unique memories")

    # Filter to those needing enrichment
    to_enrich = [m for m in memories if not has_aliases(m["content"])]
    print(f"{len(to_enrich)} need alias enrichment")

    if not to_enrich:
        print("Nothing to enrich!")
        return

    # Build SQL updates
    sql_statements = []
    enriched = 0

    for i, mem in enumerate(to_enrich):
        aliases = get_aliases(mem["content"])
        if not aliases or len(aliases) < 10:
            print(f"  [{i+1}/{len(to_enrich)}] - Skipped (no aliases)")
            continue

        new_content = f"{mem['content']} {aliases}"
        embedding = get_embedding(new_content)
        if not embedding:
            print(f"  [{i+1}/{len(to_enrich)}] - Skipped (no embedding)")
            continue

        emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
        safe_content = new_content.replace("'", "''")

        sql_statements.append(
            f"UPDATE mem_entries SET content = '{safe_content}', "
            f"embedding = '{emb_str}'::vector, "
            f"updated_at = NOW() "
            f"WHERE id = '{mem['id']}';"
        )
        enriched += 1
        print(f"  [{i+1}/{len(to_enrich)}] ✓ {mem['content'][:50]}... + {aliases}")

        # Rate limit
        if i % 5 == 4:
            time.sleep(0.5)

    if not sql_statements:
        print("No updates to apply")
        return

    # Write all SQL to one file and execute
    sql_file = "/tmp/enrich_memories.sql"
    with open(sql_file, "w") as f:
        f.write("\n".join(sql_statements))

    print(f"\nApplying {enriched} updates...")
    cmds = [
        f"scp {sql_file} root@192.168.8.10:/tmp/enrich_memories.sql",
        "ssh root@192.168.8.10 'scp /tmp/enrich_memories.sql root@192.168.8.11:/tmp/enrich_memories.sql'",
        "ssh root@192.168.8.10 \"ssh root@192.168.8.11 'pct push 141 /tmp/enrich_memories.sql /tmp/enrich_memories.sql && pct exec 141 -- docker cp /tmp/enrich_memories.sql memcore-memcore-db-1:/tmp/enrich_memories.sql && pct exec 141 -- docker exec memcore-memcore-db-1 psql -U memcore -d memcore -f /tmp/enrich_memories.sql'\"",
    ]

    for cmd in cmds:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if "UPDATE" in result.stdout:
            # Count successful updates
            updates = result.stdout.count("UPDATE 1")
            print(f"Applied {updates} updates to database")

    print(f"\nDone! Enriched {enriched} memories with search aliases")


if __name__ == "__main__":
    main()
