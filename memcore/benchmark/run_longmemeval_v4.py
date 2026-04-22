#!/usr/bin/env python3
"""LongMemEval-S benchmark adapter for MemCore — v4 (cross-encoder + wider retrieval).

Changes from v3:
1. Cross-encoder reranking via production MemCore API (replaces failed LLM rerank)
   - ms-marco-MiniLM-L-6-v2, 70/30 blend with RRF, ~50ms on CPU
2. Wider retrieval: TOP_K 50 → keep 20 after cross-encoder (was 30→10)
3. Lexical aliases in fact extraction prompt (2-4 vocabulary alternatives per fact)
4. Removed LLM reranker entirely (saved cost, improved accuracy)

Usage:
  python3 run_longmemeval_v4.py                    # Run all 500
  python3 run_longmemeval_v4.py --limit 10         # Run first 10 (testing)
  python3 run_longmemeval_v4.py --resume            # Resume from checkpoint
  python3 run_longmemeval_v4.py --no-facts          # Skip fact extraction (ablation)
  python3 run_longmemeval_v4.py --no-con            # Skip Chain-of-Note (ablation)
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import httpx

MEMCORE_URL = os.getenv("MEMCORE_URL", "http://localhost:8020")
LITELLM_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_API_KEY", "")
MODEL = os.getenv("BENCHMARK_MODEL", "deepseek-chat")

DATA_FILE = os.path.join(os.path.dirname(__file__), "longmemeval_s_cleaned.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "memcore_hypothesis_v4.jsonl")
CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), ".checkpoint_v4.json")

TOP_K = 50  # wider retrieval — cross-encoder reranks down to RERANK_K
RERANK_K = 20  # keep top 20 after cross-encoder (was 10, too aggressive)
FACT_EXTRACTION_CONCURRENCY = 8  # parallel LLM calls for extraction
STORE_CONCURRENCY = 10  # parallel MemCore writes
TURNS_PER_BATCH = 4  # turns to batch into one extraction call
MIN_TURN_LENGTH = 40  # skip trivially short turns for extraction


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_date(date_str: str) -> str | None:
    """Convert LongMemEval date format '2023/05/20 (Sat) 02:21' to ISO 8601."""
    try:
        parts = date_str.split(" (")
        date_part = parts[0]
        time_part = parts[1].split(") ")[1] if len(parts) > 1 and ") " in parts[1] else "00:00"
        return f"{date_part.replace('/', '-')}T{time_part}:00Z"
    except Exception:
        return None


def parse_date_to_dt(date_str: str) -> datetime | None:
    """Parse LongMemEval date to datetime object."""
    iso = parse_date(date_str)
    if iso:
        try:
            return datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except Exception:
            pass
    return None


def date_label(date_str: str) -> str:
    """Extract just the date portion: '2023/05/20'."""
    return date_str.split(" (")[0] if " (" in date_str else date_str


# ---------------------------------------------------------------------------
# 1. Write-time fact extraction
# ---------------------------------------------------------------------------

BATCH_EXTRACTION_PROMPT = """Extract atomic facts from these conversation turns. For each turn:
- Extract 2-5 self-contained facts (no pronouns — use explicit names/entities)
- If event dates are mentioned (e.g., "graduated in 2020", "birthday is March 5"), include them in the fact text
- For each fact, append 2-4 lexical aliases in parentheses — alternative words/phrases someone might use to search for this fact

Example: "Sarah graduated from MIT in 2020 (Massachusetts Institute of Technology, college degree, university graduation, completed studies)"

Conversation date: {conv_date}

{turns_block}

Output ONLY a JSON object mapping turn numbers to arrays of fact strings:
{{"1": ["fact1", "fact2"], "2": ["fact1", "fact2", "fact3"]}}"""


async def extract_facts_batch(client: httpx.AsyncClient, turns: list[tuple[int, str]],
                              conv_date: str) -> dict[int, list[str]]:
    """Extract atomic facts from multiple turns in a single LLM call.

    Args:
        turns: list of (turn_index, turn_text) tuples
        conv_date: conversation date label

    Returns:
        dict mapping turn_index to list of extracted facts
    """
    if not turns:
        return {}

    turns_block = "\n\n".join(
        f"--- Turn {idx} ---\n{text[:1500]}" for idx, text in turns
    )

    try:
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": BATCH_EXTRACTION_PROMPT.format(
                        conv_date=conv_date, turns_block=turns_block
                    )},
                ],
                "temperature": 0,
                "max_tokens": 600,
            },
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            result = {}
            for k, v in parsed.items():
                idx = int(k)
                if isinstance(v, list):
                    result[idx] = [f for f in v if isinstance(f, str) and len(f) > 10][:5]
            return result
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Ingestion (fact extraction + dual temporal metadata)
# ---------------------------------------------------------------------------

async def ingest_sessions(client: httpx.AsyncClient, question_id: str,
                          sessions: list, dates: list,
                          extract_facts_flag: bool = True) -> int:
    """Ingest turns as individual memories + batched atomic fact extraction."""
    group_id = f"lme_{question_id}"
    stored = 0
    sem = asyncio.Semaphore(FACT_EXTRACTION_CONCURRENCY)
    store_sem = asyncio.Semaphore(STORE_CONCURRENCY)

    async def _store_memory(content: str, iso_date: str, retries: int = 2) -> bool:
        for attempt in range(retries + 1):
            async with store_sem:
                try:
                    resp = await client.post(
                        f"{MEMCORE_URL}/api/ingest",
                        json={
                            "content": content,
                            "group_id": group_id,
                            "source_agent": "longmemeval-v3",
                            "created_at": iso_date,
                        },
                        timeout=15,
                    )
                    return resp.status_code == 200
                except Exception:
                    if attempt < retries:
                        await asyncio.sleep(1)
        return False

    # Collect all turns first
    all_turns = []  # (memory_text, conv_label, iso_date)
    for session, date_str in zip(sessions, dates):
        iso_date = parse_date(date_str) or ""
        conv_label = date_label(date_str)

        i = 0
        while i < len(session):
            turn = session[i]
            content_parts = [f"[{conv_label}]"]

            if turn["role"] == "user":
                content_parts.append(f"User: {turn['content']}")
                if i + 1 < len(session) and session[i + 1]["role"] == "assistant":
                    content_parts.append(f"Assistant: {session[i + 1]['content']}")
                    i += 2
                else:
                    i += 1
            elif turn["role"] == "assistant":
                content_parts.append(f"Assistant: {turn['content']}")
                i += 1
            else:
                i += 1
                continue

            memory_text = "\n".join(content_parts)
            if len(memory_text) > 4000:
                memory_text = memory_text[:4000]
            all_turns.append((memory_text, conv_label, iso_date))

    # Step 1: Store all raw turns concurrently
    store_tasks = [_store_memory(text, iso) for text, _, iso in all_turns]
    store_results = await asyncio.gather(*store_tasks)
    stored = sum(1 for r in store_results if r)

    if not extract_facts_flag:
        return stored

    # Step 2: Batch fact extraction — group turns by date, TURNS_PER_BATCH each
    # Only extract from turns long enough to contain useful info
    extractable = [
        (idx, text, label, iso)
        for idx, (text, label, iso) in enumerate(all_turns)
        if len(text) >= MIN_TURN_LENGTH
    ]

    # Group into batches by conv_label (same date = same batch context)
    from itertools import groupby
    batches_by_date = {}
    for idx, text, label, iso in extractable:
        batches_by_date.setdefault(label, []).append((idx, text, iso))

    async def _run_batch(conv_label: str, batch_turns: list[tuple[int, str]],
                         iso_date: str) -> int:
        """Extract facts for a batch and store them."""
        async with sem:
            facts_map = await extract_facts_batch(client, batch_turns, conv_label)

        count = 0
        fact_store_tasks = []
        for turn_idx, facts in facts_map.items():
            for fact in facts:
                fact_content = f"[{conv_label}] [FACT]\n{fact}"
                fact_store_tasks.append(_store_memory(fact_content, iso_date))

        if fact_store_tasks:
            results = await asyncio.gather(*fact_store_tasks)
            count = sum(1 for r in results if r)
        return count

    batch_tasks = []
    for conv_label, turns_for_date in batches_by_date.items():
        iso_date = turns_for_date[0][2]  # same iso for all turns on same date
        # Split into batches of TURNS_PER_BATCH
        batch_items = [(idx, text) for idx, text, _ in turns_for_date]
        for i in range(0, len(batch_items), TURNS_PER_BATCH):
            chunk = batch_items[i:i + TURNS_PER_BATCH]
            batch_tasks.append(_run_batch(conv_label, chunk, iso_date))

    batch_results = await asyncio.gather(*batch_tasks)
    stored += sum(batch_results)
    return stored


# ---------------------------------------------------------------------------
# 2. Time-aware query expansion
# ---------------------------------------------------------------------------

TEMPORAL_PATTERNS = [
    # "how many days/weeks/months between X and Y"
    (r"how many (days?|weeks?|months?|years?)", "duration"),
    # "before/after <date>"
    (r"(before|after|since|until)\s+\d{4}", "boundary"),
    (r"(before|after|since|until)\s+(january|february|march|april|may|june|july|august|september|october|november|december)", "boundary"),
    # "in <month> <year>" or "<month> <year>"
    (r"(in\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}", "period"),
    # "last/first time"
    (r"(last|first|most recent|earliest|latest)\s+(time|session|conversation)", "ordinal"),
    # "how long ago"
    (r"how long ago", "duration"),
    # Explicit date references
    (r"\d{4}/\d{1,2}/\d{1,2}", "explicit"),
    # "N days/weeks/months ago"
    (r"\d+\s+(days?|weeks?|months?|years?)\s+(ago|before|after)", "relative"),
    # "when did"
    (r"when did", "temporal_query"),
]


def detect_temporal_intent(question: str) -> dict:
    """Detect temporal constraints in a question."""
    q_lower = question.lower()
    result = {"is_temporal": False, "types": [], "keywords": []}

    for pattern, ttype in TEMPORAL_PATTERNS:
        match = re.search(pattern, q_lower)
        if match:
            result["is_temporal"] = True
            result["types"].append(ttype)
            result["keywords"].append(match.group(0))

    return result


TEMPORAL_EXPANSION_PROMPT = """Analyze this question for temporal constraints and generate search queries.

Question: {question}
Question date (today): {question_date}

1. Identify any time references (dates, durations, "before/after", "first/last time", etc.)
2. Generate 2-3 search queries that would help find the answer in a conversation history.
   - One query focused on the core topic (remove temporal words)
   - One query focused on temporal anchoring (include dates/times)
   - If counting/aggregating, one query per potential sub-topic

Output ONLY a JSON object:
{{
  "temporal_filter": {{
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null",
    "ordering": "chronological" or "reverse_chronological" or null
  }},
  "queries": ["query1", "query2", "query3"]
}}"""


async def expand_temporal_query(client: httpx.AsyncClient, question: str,
                                question_date: str) -> tuple[list[str], dict | None]:
    """Expand a question with temporal awareness. Returns (queries, temporal_filter)."""
    temporal = detect_temporal_intent(question)

    if not temporal["is_temporal"]:
        # Fall back to v2-style decomposition for non-temporal questions
        queries = await decompose_question(client, question)
        return queries, None

    try:
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": TEMPORAL_EXPANSION_PROMPT.format(
                        question=question, question_date=question_date
                    )},
                ],
                "temperature": 0,
                "max_tokens": 250,
            },
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)

        queries = parsed.get("queries", [question])
        if question not in queries:
            queries.insert(0, question)
        queries = queries[:4]

        temporal_filter = parsed.get("temporal_filter")
        return queries, temporal_filter

    except Exception:
        return [question], None


async def decompose_question(client: httpx.AsyncClient, question: str) -> list[str]:
    """For complex non-temporal questions, decompose into sub-queries."""
    complexity_signals = ["how many", "and", "total", "all", "both", "each",
                          "compare", "difference between", "before and after"]
    is_complex = any(s in question.lower() for s in complexity_signals)

    if not is_complex:
        return [question]

    try:
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": (
                        "Break this question into 1-3 simple sub-questions that would help "
                        "find the answer in a conversation history. If the question is already "
                        "simple, return just the original question. "
                        "Output ONLY the sub-questions, one per line. No numbering or bullets."
                    )},
                    {"role": "user", "content": question},
                ],
                "temperature": 0,
                "max_tokens": 150,
            },
            timeout=10,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        sub_queries = [q.strip() for q in text.split("\n") if q.strip() and len(q.strip()) > 5]
        if question not in sub_queries:
            sub_queries.insert(0, question)
        return sub_queries[:3]
    except Exception:
        return [question]


# ---------------------------------------------------------------------------
# Retrieval (with temporal filtering/boosting)
# ---------------------------------------------------------------------------

async def recall_memories(client: httpx.AsyncClient, question_id: str,
                          queries: list[str],
                          temporal_filter: dict | None = None) -> list[dict]:
    """Recall memories with multi-query + cross-encoder reranking + temporal boosting.

    Uses production MemCore API which has ms-marco cross-encoder built in.
    Retrieves TOP_K per query, deduplicates, then applies temporal boosting.
    """
    group_id = f"lme_{question_id}"
    seen_ids = set()
    all_results = []

    for query in queries:
        try:
            resp = await client.post(
                f"{MEMCORE_URL}/api/recall",
                json={
                    "query": query,
                    "group_id": group_id,
                    "limit": TOP_K,
                },
                timeout=60,
            )
            data = resp.json()
            # Support both fused response {"results": [...]} and legacy {"postgres": [...]}
            results_list = data.get("results") or data.get("postgres", [])
            for r in results_list:
                rid = r.get("id", "")
                if rid and rid not in seen_ids:
                    seen_ids.add(rid)
                    all_results.append(r)
        except Exception:
            pass

    # Sort by blended_score (cross-encoder + RRF) if available, else rrf_score
    def _score(r):
        return float(r.get("blended_score", 0) or r.get("rrf_score", 0))

    # Apply temporal boosting if we have a filter
    if temporal_filter:
        start_date = temporal_filter.get("start_date")
        end_date = temporal_filter.get("end_date")
        ordering = temporal_filter.get("ordering")

        for r in all_results:
            base_score = _score(r)
            created = str(r.get("created_at", ""))[:10]

            in_range = True
            if start_date and created < start_date:
                in_range = False
            if end_date and created > end_date:
                in_range = False

            if in_range and (start_date or end_date):
                r["final_score"] = base_score * 1.5
            else:
                r["final_score"] = base_score

        if ordering == "chronological":
            all_results.sort(key=lambda x: str(x.get("created_at", "")))
        elif ordering == "reverse_chronological":
            all_results.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        else:
            all_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    else:
        all_results.sort(key=_score, reverse=True)

    return all_results[:RERANK_K]


# ---------------------------------------------------------------------------
# 3. Chain-of-Note reading strategy
# ---------------------------------------------------------------------------

CON_EXTRACT_PROMPT = """You are extracting relevant facts from retrieved memories to answer a question.

Question: {question}
Question date: {question_date}

Retrieved Memories:
{context}

Extract ALL facts from these memories that could help answer the question.
For each fact, note which memory it came from and its date.

Output ONLY a JSON array of objects:
[
  {{"fact": "...", "memory_id": N, "date": "YYYY-MM-DD", "relevance": "high/medium/low"}},
  ...
]

If no memories contain relevant information, output: []"""


CON_REASON_PROMPT = """You are an AI assistant answering a question based on extracted facts from past conversations.

RULES:
- Answer based ONLY on the extracted facts below.
- If no relevant facts exist, say "I don't have that information."
- When facts have changed over time, use the MOST RECENT version (check dates).
- When counting items, ensure you count ALL relevant facts — answers may span multiple conversations.
- For duration/time questions, calculate carefully using the dates.
- Preserve full details: include locations, qualifiers, full names (e.g., "University of Melbourne in Australia" not just "University of Melbourne").
- Be concise but complete. Don't paraphrase away important context.

Today's date: {question_date}

Extracted Facts:
{facts_json}

Question: {question}

Think step by step, then give your final answer. Keep the final answer concise."""


async def generate_answer_con(client: httpx.AsyncClient, question: str,
                              question_date: str, memories: list[dict]) -> str:
    """Chain-of-Note: extract facts first, then reason over them."""
    # Phase 1: Build context and extract relevant facts
    context_parts = []
    for i, mem in enumerate(memories, 1):
        created = str(mem.get("created_at", "unknown"))[:10]
        content = mem.get("content", "")
        context_parts.append(f"[Memory {i} — Date: {created}]\n{content}")

    context = "\n\n".join(context_parts) if context_parts else "(No memories)"

    if not context_parts:
        return "I don't have that information."

    try:
        # Phase 1: Extract structured facts
        resp1 = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": CON_EXTRACT_PROMPT.format(
                        question=question,
                        question_date=question_date,
                        context=context,
                    )},
                ],
                "temperature": 0,
                "max_tokens": 500,
            },
            timeout=60,
        )
        resp1.raise_for_status()
        facts_raw = resp1.json()["choices"][0]["message"]["content"].strip()

        # Validate it's parseable JSON
        facts_clean = re.sub(r"^```(?:json)?\s*", "", facts_raw)
        facts_clean = re.sub(r"\s*```$", "", facts_clean)
        try:
            facts_parsed = json.loads(facts_clean)
            if isinstance(facts_parsed, list) and len(facts_parsed) == 0:
                return "I don't have that information."
            facts_json = json.dumps(facts_parsed, indent=2)
        except json.JSONDecodeError:
            # If JSON parsing fails, use raw text
            facts_json = facts_raw

        # Phase 2: Reason over extracted facts
        resp2 = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": CON_REASON_PROMPT.format(
                        question=question,
                        question_date=question_date,
                        facts_json=facts_json,
                    )},
                ],
                "temperature": 0,
                "max_tokens": 300,
            },
            timeout=60,
        )
        resp2.raise_for_status()
        answer = resp2.json()["choices"][0]["message"]["content"].strip()

        # Strip reasoning prefix if present — keep just the final answer
        # Look for patterns like "Final answer:" or "Answer:"
        for marker in ["**Final Answer:**", "**Final answer:**", "Final Answer:",
                       "Final answer:", "**Answer:**", "Answer:"]:
            if marker in answer:
                answer = answer.split(marker, 1)[1].strip()
                break

        return answer

    except Exception as e:
        return f"Error: {e}"


async def generate_answer_direct(client: httpx.AsyncClient, question: str,
                                 question_date: str, memories: list[dict]) -> str:
    """Direct answer generation (v2-style fallback, used for --no-con ablation)."""
    context_parts = []
    for i, mem in enumerate(memories, 1):
        created = str(mem.get("created_at", "unknown"))[:10]
        content = mem.get("content", "")
        context_parts.append(f"[Memory {i} — Date: {created}]\n{content}")

    context = "\n\n".join(context_parts) if context_parts else "(No relevant memories found)"

    prompt = f"""You are an AI assistant recalling information from past conversations.

IMPORTANT RULES:
- Answer based ONLY on the retrieved memories below.
- If the information is not in any memory, say "I don't have that information."
- When facts have changed over time, use the MOST RECENT version (check the dates).
- When counting items, check ALL memories — answers may span multiple conversations.
- For date/time questions, calculate carefully using the dates shown in each memory.
- Be concise and direct. State facts, don't explain your reasoning.

Today's date: {question_date}

Retrieved Memories:
{context}

Question: {question}

Answer:"""

    try:
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 200,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

async def clear_question(client: httpx.AsyncClient, question_id: str):
    """Clean up memories for a question after processing. Retries to ensure cleanup."""
    group_id = f"lme_{question_id}"
    for attempt in range(3):
        try:
            resp = await client.post(
                f"{MEMCORE_URL}/api/clear_group",
                json={"group_id": group_id},
                timeout=30,
            )
            if resp.status_code == 200:
                result = resp.json()
                if result.get("deleted", 0) == 0 or attempt > 0:
                    return
                # If we deleted something, wait a beat for PG to settle
                await asyncio.sleep(0.5)
        except Exception:
            await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(completed: set):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(completed), f)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

async def run_benchmark(limit: int | None = None, resume: bool = False,
                        use_facts: bool = True, use_con: bool = True):
    print(f"Loading dataset from {DATA_FILE}...")
    with open(DATA_FILE) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    features = []
    if use_facts:
        features.append("fact-extraction+aliases")
    if use_con:
        features.append("chain-of-note")
    features.append("cross-encoder-rerank")
    features.append("temporal-expansion")
    features.append("dual-temporal-metadata")

    print(f"Running {len(data)} questions (v3: {', '.join(features)})")

    completed = load_checkpoint() if resume else set()
    if completed:
        print(f"Resuming: {len(completed)} already done")

    mode = "a" if resume and os.path.exists(OUTPUT_FILE) else "w"
    out_f = open(OUTPUT_FILE, mode)

    qtype_counts = {}
    qtype_times = {}
    total_done = len(completed)
    total_facts_stored = 0

    async with httpx.AsyncClient(
        timeout=60,
        limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
    ) as client:
        for i, entry in enumerate(data):
            qid = entry["question_id"]
            qtype = entry["question_type"]

            if qid in completed:
                continue

            q_start = time.time()
            print(f"\n[{total_done + 1}/{len(data)}] {qtype}: {entry['question'][:80]}...")

            # Step 1: Ingest (per-message pairs + fact extraction + event dates)
            t0 = time.time()
            stored = await ingest_sessions(
                client, qid,
                entry["haystack_sessions"],
                entry["haystack_dates"],
                extract_facts_flag=use_facts,
            )
            t_ingest = time.time() - t0
            total_facts_stored += stored
            print(f"  Ingested {stored} memories ({t_ingest:.1f}s)")

            # Step 2: Time-aware query expansion (or plain decomposition)
            t0 = time.time()
            queries, temporal_filter = await expand_temporal_query(
                client, entry["question"], entry["question_date"],
            )
            t_expand = time.time() - t0
            temporal_tag = " [TEMPORAL]" if temporal_filter else ""
            if len(queries) > 1:
                print(f"  Expanded into {len(queries)} queries ({t_expand:.1f}s){temporal_tag}")

            # Step 3: Recall with multi-query + temporal boosting
            t0 = time.time()
            memories = await recall_memories(client, qid, queries, temporal_filter)
            t_recall = time.time() - t0
            print(f"  Recalled {len(memories)} memories ({t_recall:.1f}s)")

            # Step 4: Generate answer (Chain-of-Note or direct)
            t0 = time.time()
            if use_con:
                answer = await generate_answer_con(
                    client, entry["question"],
                    entry["question_date"], memories,
                )
            else:
                answer = await generate_answer_direct(
                    client, entry["question"],
                    entry["question_date"], memories,
                )
            t_gen = time.time() - t0
            print(f"  Answer ({t_gen:.1f}s): {answer[:120]}")
            print(f"  Expected: {str(entry['answer'])[:120]}")

            # Write hypothesis
            out_f.write(json.dumps({"question_id": qid, "hypothesis": answer}) + "\n")
            out_f.flush()

            # Step 5: Clean up
            await clear_question(client, qid)

            completed.add(qid)
            total_done += 1
            qtype_counts[qtype] = qtype_counts.get(qtype, 0) + 1
            q_time = time.time() - q_start
            qtype_times.setdefault(qtype, []).append(q_time)

            if total_done % 10 == 0:
                save_checkpoint(completed)
                elapsed_avg = sum(sum(v) for v in qtype_times.values()) / total_done
                eta = elapsed_avg * (len(data) - total_done)
                print(f"\n  --- Checkpoint: {total_done}/{len(data)} done "
                      f"| avg {elapsed_avg:.1f}s/q | ETA {eta/60:.0f}m ---")

    out_f.close()
    save_checkpoint(completed)

    print(f"\n{'='*60}")
    print(f"Done! {total_done} questions processed")
    print(f"Total memories stored: {total_facts_stored}")
    print(f"Hypothesis file: {OUTPUT_FILE}")
    print(f"\nQuestions by type (avg time):")
    for qtype in sorted(qtype_counts.keys()):
        cnt = qtype_counts[qtype]
        avg_t = sum(qtype_times.get(qtype, [0])) / max(cnt, 1)
        print(f"  {qtype}: {cnt} ({avg_t:.1f}s avg)")


def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval-S v4 against MemCore")
    parser.add_argument("--limit", type=int, help="Only run first N questions")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--no-facts", action="store_true", help="Disable fact extraction (ablation)")
    parser.add_argument("--no-con", action="store_true", help="Disable Chain-of-Note (ablation)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        limit=args.limit,
        resume=args.resume,
        use_facts=not args.no_facts,
        use_con=not args.no_con,
    ))


if __name__ == "__main__":
    main()
