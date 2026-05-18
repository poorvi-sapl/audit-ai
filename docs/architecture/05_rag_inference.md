# RAG & Inference Design — Audit AI
**Version:** 1.0  
**Date:** April 2026  
**Status:** FINAL  
**Checklist items closed:** #13 (RAG), #14 (Rerankers), #15 (Long Context Problem)

---

## 1. Overview

The inference pipeline has two distinct retrieval goals that use separate Qdrant collections:

| Goal | Collection | Retrieves |
|---|---|---|
| **RAG context** | `workpaper_chunks_embeddings` | Similar workpaper sections from past engagements — gives model domain context for the current task |
| **Consistency** | `validated_findings_embeddings` | Similar validated findings from past engagements — ensures firm-wide consistency in how similar conditions are characterised and recommended |

Both retrievals run in parallel before context assembly. The model receives both as separate context blocks.

---

## 2. Embedding Pipeline

### Model: e5-mistral-7b-instruct

- **Hardware:** L40S on Server 2, ~14GB VRAM
- **Vector dimension:** 4096
- **Input encoding:** instruction-tuned — uses different prefixes for queries vs passages

| Use | Input format |
|---|---|
| Passage (chunk ingestion) | `{chunk_text}` (no prefix) |
| Query (RAG retrieval) | `Instruct: {task_description}\nQuery: {query_text}` |

Task descriptions by task_type:

| task_type | Instruction prefix |
|---|---|
| risk_classification | `Given an audit workpaper section, retrieve relevant workpaper content for risk assessment` |
| compliance_check | `Given an audit workpaper section, retrieve relevant content for compliance threshold verification` |
| finding_documentation | `Given an audit condition, retrieve similar workpaper sections for finding documentation` |
| summarization | `Given an audit workpaper, retrieve relevant sections for executive summary` |
| ner_extraction | `Given an audit document, retrieve sections containing financial entities and regulatory references` |

### When embeddings are generated

| Event | Action |
|---|---|
| New workpaper chunk created (ETL pipeline) | Embed chunk text → push to `workpaper_chunks_embeddings` → set `embedding_synced=TRUE` |
| model_output transitions to VALIDATED | Embed `finding` field → push to `validated_findings_embeddings` → set `embedding_synced=TRUE` |
| Qdrant sync worker (arq background job) | Polls for `embedding_synced=FALSE` rows every 60 seconds and processes any missed syncs |

### Embedding batch size

```python
# ETL pipeline: batch 32 chunks at a time
# Inference: single query embedding (latency-critical)
# Validation sync: batch 16 outputs at a time
```

---

## 3. Qdrant Retrieval Design

### 3.1 workpaper_chunks_embeddings — RAG Retrieval

**Goal:** Retrieve the most semantically relevant workpaper sections from past engagements to give the model context for the current task.

**Why cross-engagement?** Chunks in Qdrant are PII-scrubbed (the transformer chain runs before embedding). Retrieving across engagements gives the model richer context from similar situations at other clients. All identifying information has been replaced with `[CLIENT_ENTITY]`, `[EIN]`, etc.

```python
# model/inference/engine.py — qdrant retrieval logic

def retrieve_rag_context(
    query_embedding: list[float],
    workpaper_type: str,
    year: int,
    top_k: int = 20,
) -> list[ScoredPoint]:

    results = qdrant_client.search(
        collection_name="workpaper_chunks_embeddings",
        query_vector=query_embedding,
        limit=top_k,
        query_filter=Filter(
            must=[
                FieldCondition(key="workpaper_type",  match=MatchValue(value=workpaper_type)),
                FieldCondition(key="chunk_mode",      match=MatchValue(value="semantic")),
            ],
            must_not=[
                FieldCondition(key="is_rollforward",  match=MatchValue(value=True)),
            ]
        ),
        with_payload=True,
        score_threshold=0.65,   # discard low-relevance results
    )

    # Apply ETL-computed drift_weight before passing to reranker.
    # drift_weight was set by TemporalTagger (year-decay × time_sensitivity multiplier)
    # and stored in the Qdrant payload at embed time — no recalculation needed here.
    for r in results:
        r.score = r.score * r.payload["drift_weight"]

    return sorted(results, key=lambda x: x.score, reverse=True)
```

**Filter logic:**
- `workpaper_type` — hard filter; a bank rec query only retrieves bank rec chunks
- `chunk_mode = "semantic"` — hard filter; numeric chunks are consumed by math tools, not RAG
- `is_rollforward = True` — **must_not** (exclusion filter); rolled-forward chunks contain prior-year numbers that are stale relative to the current engagement. Original chunks from those same engagements are still retrieved normally. Rollforward chunks remain in the index for training data analysis but are never surfaced during inference.
- `engagement_id` — NOT filtered; cross-engagement retrieval of original chunks is intentional
- Year weighting — soft post-retrieval score multiplier, not a hard filter

---

### 3.2 validated_findings_embeddings — Consistency Retrieval

**Goal:** Surface similar validated findings from past engagements so the model maintains consistency in how it characterises conditions and writes recommendations.

```python
def retrieve_consistency_context(
    query_embedding: list[float],
    task_type: str,
    top_k: int = 5,
) -> list[ScoredPoint]:

    results = qdrant_client.search(
        collection_name="validated_findings_embeddings",
        query_vector=query_embedding,
        limit=top_k,
        query_filter=Filter(
            must=[
                FieldCondition(key="task_type", match=MatchValue(value=task_type)),
            ]
        ),
        with_payload=True,
        score_threshold=0.70,   # higher threshold — only highly similar past findings
    )
    return results
```

Only `VALIDATED` outputs are ever pushed to this collection. AI_DRAFT and REJECTED outputs are never embedded here — consistency must be built from auditor-approved content only.

---

## 4. Reranker: bge-reranker-v2-m3

### Why a cross-encoder after Qdrant

Qdrant's ANN search uses bi-encoder similarity (embedding similarity). This is fast but approximate — it sometimes retrieves chunks that are lexically similar but contextually irrelevant (e.g., a SEFA schedule mentions "compliance" and matches a query about compliance checks, but the content is just line items).

The cross-encoder reranker processes each `(query, candidate_chunk)` pair jointly, giving much higher relevance precision. It adds ~100-150ms latency per inference call — acceptable within the <2s p95 target.

**bge-reranker-v2-m3 properties:**
- Multilingual model (bge-reranker-v2-m3) — English-only use at HCLLP; Hindi support not needed
- ~560M params, ~2GB VRAM on L40S
- Outputs a single relevance float per pair

### Reranking flow

```
Qdrant returns top-20 candidates (after year multiplier applied)
         │
         ▼
Cross-encoder scores each (query, chunk_text) pair
  scores = reranker.compute_score(
      [[query_text, r.payload["text"]] for r in results]
  )
         │
         ▼
Sort by cross-encoder score descending
         │
         ▼
Take top-5 for RAG context
Take top-3 for consistency context (separate retrieval)
         │
         ▼
Pass to context assembly
```

```python
# model/inference/engine.py

from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

def rerank(query_text: str, candidates: list[ScoredPoint], top_n: int) -> list[ScoredPoint]:
    pairs = [[query_text, c.payload["text"]] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    for candidate, score in zip(candidates, scores):
        candidate.score = score
    return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_n]
```

---

## 5. Math Tools Engine

### Overview

The LLM is not permitted to perform arithmetic. All numeric computations — threshold comparisons, materiality calculations, bank reconciliation differences, trial balance checks, financial ratios, and variance analysis — are computed by the `model/tools/` Python library and injected as verified facts into the context window **before generation starts**.

**The model reads computed results. It explains and documents. It never calculates.**

This is enforced architecturally: the math tools run inside a LangGraph node that executes in parallel with RAG retrieval, before context assembly. The vLLM generation step never receives a prompt containing an unresolved numeric question.

---

### 5.0 AuditState — LangGraph State Schema

All LangGraph nodes share a single typed state object. The FastAPI handler populates the initial state before handing off to `graph.invoke()`; each node reads from and writes to its own fields. No node touches another node's output fields.

```python
# agents/state.py
from __future__ import annotations
from typing import Any, TypedDict
from uuid import UUID


class ActiveChunk(TypedDict):
    chunk_id:       UUID
    chunk_text:     str
    chunk_index:    int
    numeric_values: list[float]  # extracted by ETL numeric_extractor transformer
    rows:           list[dict]   # structured rows (trial balance, bank rec, etc.)
    content_json:   dict         # raw content_json from workpaper_chunks table


class AuditState(TypedDict):

    # ── Request identifiers ──────────────────────────────────────────────────
    workpaper_id:   UUID
    engagement_id:  UUID
    request_id:     str    # arq job ID / FastAPI request ID — used for Langfuse tracing

    # ── Routing context (set by router_node from PG; never mutated after routing) ──
    task_type:        str        # 'risk_classification' | 'compliance_check' |
                                 # 'finding_documentation' | 'summarization' | 'ner_extraction'
    workpaper_type:   str        # canonical WorkpaperType value — see db/enums.py
    client_type:      str        # 'NPO' | 'Govt' — drives materiality base + regulation cluster
    is_gagas:         bool       # True for all Govt; True for NPO receiving govt grants (Yellow Book)
    has_single_audit: bool       # True when federal expenditures ≥ $750K (2 CFR 200 applies)
    other_subtype:    str | None # Govt subtypes only (e.g. 'special_district'); None otherwise
    fiscal_year:      int

    # ── Engagement financials (set by FastAPI handler before graph.invoke()) ─
    # Unpacked from engagement.financial_context JSONB.
    # Keys: total_assets, total_revenue, total_federal_expenditures, total_expenses,
    #       net_income, materiality_amount, fiscal_year_end
    # materiality_basis removed — derived from client_type by materiality.py.
    # currency removed — always USD.
    # Never re-read from DB inside graph nodes.
    engagement_financials: dict[str, Any]

    # ── Active content ───────────────────────────────────────────────────────
    active_chunk:  ActiveChunk   # the workpaper chunk being analysed
    model_version: str           # from PG model_versions WHERE is_current = TRUE

    # ── Node outputs (None until the owning node completes) ─────────────────
    math_results:         dict[str, Any]  # tool_name → typed result object
                                          # set by math_tool_dispatch; {} when no tools registered
    rag_chunks:           list[dict]      # reranked RAG context; set by rag_retrieval_node
    consistency_findings: list[dict]      # reranked consistency context; set by rag_retrieval_node
    assembled_context:    str             # full prompt context string; set by context_assembly_node
    generated_output:     str | None      # raw vLLM output; set by generate_node
    post_processor_result: dict | None    # PostProcessorResult serialised to dict; set by generate_node
```

**Initialisation contract:** The FastAPI handler calls:
```python
initial_state: AuditState = {
    "workpaper_id":          wp.id,
    "engagement_id":         eng.id,
    "request_id":            request.state.request_id,
    "task_type":             body.task_type,
    "workpaper_type":        wp.workpaper_type,
    "client_type":           eng.client_type,
    "is_gagas":              eng.is_gagas,
    "has_single_audit":      eng.has_single_audit,
    "other_subtype":         eng.other_subtype,
    "fiscal_year":           eng.fiscal_year,
    "engagement_financials": dict(eng.financial_context),
    "active_chunk":          build_active_chunk(wp),
    "model_version":         current_model.version_tag,
    "math_results":          {},
    "rag_chunks":            [],
    "consistency_findings":  [],
    "assembled_context":     "",
    "generated_output":      None,
    "post_processor_result": None,
}
result: AuditState = await graph.ainvoke(initial_state)
```

---

### 5.1 The `model/tools/` Library

```
model/tools/
├── __init__.py         ← exports all six tools by name
├── thresholds.py       ← regulatory threshold comparisons (config-driven from thresholds.yaml)
├── materiality.py      ← planning materiality, performance materiality, clearly trivial
├── bank_rec.py         ← outstanding items, deposits in transit, unreconciled difference
├── trial_balance.py    ← debit/credit verification, out-of-balance detection, subtotal checks
├── ratios.py           ← current ratio, quick ratio, debt-to-equity, days receivable
├── variance.py         ← absolute and % variance per line; flags items exceeding threshold
└── manifest.py         ← TOOL_MANIFEST dict: workpaper_type → [tool_names to run]
```

Each tool exposes a single typed entry function:

| Module | Entry function | Primary input | Output type |
|---|---|---|---|
| `thresholds.py` | `check_thresholds(amounts, client_type, is_gagas, has_single_audit)` | floats + engagement context | `list[ThresholdResult]` |
| `materiality.py` | `compute_materiality(total_assets, total_revenue, total_federal_exp, total_expenses, client_type)` | engagement metadata from PG | `MaterialityResult` |
| `bank_rec.py` | `reconcile(rows, currency)` | `content_json.rows[]` | `BankRecResult` |
| `trial_balance.py` | `check_balance(rows)` | `content_json.rows[]` | `TrialBalanceResult` |
| `ratios.py` | `compute_ratios(rows)` | `content_json.rows[]` | `RatiosResult` |
| `variance.py` | `compute_variance(rows, threshold_pct, materiality)` | `content_json.rows[]` | `VarianceResult` |

All tools are stateless pure functions — no DB writes, no GPU, no network calls. Input → computation → typed output.

---

### 5.2 `manifest.py` — Tool Registry

`manifest.py` is a plain Python dict mapping `workpaper_type` to the ordered list of tools LangGraph should run for that workpaper type. LangGraph reads this registry; the LLM never sees it.

```python
# model/tools/manifest.py

TOOL_MANIFEST: dict[str, list[str]] = {
    "bank_reconciliation":   ["thresholds", "materiality", "bank_rec"],
    "trial_balance":         ["thresholds", "materiality", "trial_balance", "variance"],
    "financial_statements":  ["thresholds", "materiality", "ratios", "variance"],
    "analytical_procedure":  ["thresholds", "materiality", "variance"],
    "sefa_schedule":         ["thresholds", "materiality"],
    "compliance_test":       ["thresholds"],
    "risk_assessment":       ["thresholds", "materiality"],
    "finding_documentation": ["thresholds"],
    "management_letter":     ["thresholds"],
    "internal_control":      [],   # qualitative only — no numeric tools
    "planning_document":     ["materiality"],
    "other":                 [],
    "final_report":          [],   # qualitative only — AI assists review/summarization only
}
```

**Framework coverage:** `thresholds.py` and `materiality.py` load the correct benchmarks for Single Audit, GAGAS, and GAAS from `config/thresholds.yaml`, keyed by `client_type`, `is_gagas`, and `has_single_audit`. All US frameworks are covered by the same tool files — no framework-specific variants needed.

---

### 5.3 LangGraph Tool Dispatch Node

The tool dispatch node sits in the LangGraph StateGraph between the Router node and context assembly. It runs concurrently with the RAG retrieval node — both start immediately after routing and their results join at context assembly.

```python
# agents/graph.py — relevant StateGraph assembly

from model.tools.manifest import TOOL_MANIFEST
import model.tools as tools

async def math_tool_dispatch(state: AuditState) -> AuditState:
    """
    Runs all registered math tools for this workpaper_type.
    Results stored in state.math_results (dict[tool_name, result]).
    Executes in parallel with RAG retrieval — zero latency overhead.
    state.engagement_financials is pre-populated from engagement.financial_context JSONB
    by the FastAPI handler before LangGraph is invoked.
    """
    tool_names = TOOL_MANIFEST.get(state.workpaper_type, [])
    fin = state.engagement_financials  # unpacked from financial_context JSONB
    results: dict = {}

    for tool_name in tool_names:
        if tool_name == "thresholds":
            results[tool_name] = tools.thresholds.check_thresholds(
                amounts=state.active_chunk.numeric_values,
                client_type=state.client_type,
                is_gagas=state.is_gagas,
                has_single_audit=state.has_single_audit,
            )
        elif tool_name == "materiality":
            results[tool_name] = tools.materiality.compute_materiality(
                total_assets=fin.get("total_assets", 0.0),
                total_revenue=fin.get("total_revenue", 0.0),
                total_federal_exp=fin.get("total_federal_expenditures", 0.0),
                total_expenses=fin.get("total_expenses", 0.0),
                client_type=state.client_type,
            )
        else:
            # numeric tools: bank_rec, trial_balance, ratios, variance — always USD
            tool_fn = getattr(tools, tool_name)
            results[tool_name] = tool_fn.run(
                rows=state.active_chunk.rows,
                currency="USD",
            )

    state.math_results = results
    return state


# StateGraph wiring
graph = StateGraph(AuditState)

graph.add_node("router",           router_node)
graph.add_node("math_tools",       math_tool_dispatch)
graph.add_node("rag_retrieval",    rag_retrieval_node)    # parallel with math_tools
graph.add_node("context_assembly", context_assembly_node) # waits for both
graph.add_node("generate",         generate_node)

graph.add_edge("router",                           "math_tools")
graph.add_edge("router",                           "rag_retrieval")
graph.add_edge(["math_tools", "rag_retrieval"],    "context_assembly")  # fan-in join
graph.add_edge("context_assembly",                 "generate")
```

**Why workpaper_type determines tools, not the model:** The `workpaper_type` is already known from the routing step (read from PG `workpapers` table, set during ETL labelling). LangGraph reads the manifest and dispatches deterministically. The model is never asked "which tools do you need?" — this avoids an additional generation round trip and eliminates unpredictable tool selection.

---

### 5.4 Math Results Block in Context

The tool results are rendered into a `COMPUTED AUDIT FACTS` block and inserted into the context window as the fourth section, between the consistency findings and the active workpaper chunk. The model is explicitly instructed not to recalculate.

```
════════════════════════════════════════
COMPUTED AUDIT FACTS (Python-verified — do not recalculate)
════════════════════════════════════════
[Block is present only when TOOL_MANIFEST[workpaper_type] is non-empty]

Materiality (Single Audit — 2026):
  Planning materiality:     $47,250   (0.5% of total federal expenditures: $9,450,000)
  Performance materiality:  $23,625   (50% of planning)
  Clearly trivial:          $2,363    (5% of planning)

Bank Reconciliation:
  Balance per bank statement:      $1,847,234.00
  Less: Outstanding checks:          ($14,822.50)
  Add:  Deposits in transit:          $19,444.00
  Adjusted bank balance:           $1,851,855.50
  Balance per books:               $1,851,902.00
  Unreconciled difference:               $46.50  ← WITHIN tolerance ($500)  ✓

Threshold Check (2 CFR 200):
  Total federal expenditures:  $9,450,000 → EXCEEDS Single Audit threshold ($750,000)
  Type A threshold (3%):          $283,500  → use for major program determination
  Single Audit required:         YES
```

---

### 5.5 Numeric Input Dependency — ETL Chunker

The numeric tools (`bank_rec`, `trial_balance`, `ratios`, `variance`) require structured row data — not a text description of numbers. This is why the ETL Chunker runs in **numeric mode** for these workpaper types, storing `content_json.rows[]` alongside the text representation.

See [04_etl_pipeline.md](04_etl_pipeline.md) Section 5 Step 4 — Chunker for the numeric mode specification.

`thresholds.py` and `materiality.py` have no ETL dependency — they read from PG engagement metadata (totals already stored in the `engagements` table).

| Tool | Input source | ETL dependency |
|---|---|---|
| `thresholds.py` | PG `engagements` (amounts + client_type + is_gagas + has_single_audit) | None |
| `materiality.py` | PG `engagements` (total_assets, total_revenue, total_federal_exp, total_expenses, client_type) | None |
| `bank_rec.py` | `content_json.rows[]` | Numeric chunker required |
| `trial_balance.py` | `content_json.rows[]` | Numeric chunker required |
| `ratios.py` | `content_json.rows[]` | Numeric chunker required |
| `variance.py` | `content_json.rows[]` | Numeric chunker required |

---

## 6. Context Assembly

The final prompt fed to vLLM is assembled in this exact order. Sections are separated by explicit labels so the model can distinguish them.

```
<s>[INST]
════════════════════════════════════════
SYSTEM
════════════════════════════════════════
You are an audit assistant for a licensed CPA firm.
All outputs are AI-Assisted Drafts requiring senior auditor review before use.
Do not express opinions. Do not use language resembling a signed audit opinion.

Client type:      {client_type}
GAGAS applies:    {is_gagas}
Single Audit:     {has_single_audit}
Fiscal year:      {fiscal_year}
Currency:         USD

════════════════════════════════════════
SIMILAR WORKPAPER SECTIONS (reference only)
════════════════════════════════════════
{rag_chunk_1}

---
{rag_chunk_2}

[... up to 5 chunks ...]

════════════════════════════════════════
SIMILAR PAST FINDINGS (consistency reference)
════════════════════════════════════════
{past_finding_1}

---
{past_finding_2}

[... up to 3 findings ...]

════════════════════════════════════════
COMPUTED AUDIT FACTS (Python-verified — do not recalculate)
════════════════════════════════════════
[Block omitted when TOOL_MANIFEST[workpaper_type] is empty]

{math_results_block}   ← rendered from state.math_results by context_assembly_node
                          see Section 5.4 for example output

════════════════════════════════════════
CURRENT WORKPAPER SECTION
════════════════════════════════════════
Sheet: {sheet_name} | Chunk {chunk_index} of {total_chunks}
{[DOCUMENT TRUNCATED: X of Y rows shown] if truncated}

{active_chunk_content}

════════════════════════════════════════
TASK
════════════════════════════════════════
{task_specific_instruction}
[/INST]
```

### Math results block rendering

The `math_results_block` is assembled by `context_assembly_node` from `state.math_results` — the dict populated by the LangGraph math tool dispatch node (Section 5). Each tool result is rendered into plain labelled text. Example rendering for a `bank_reconciliation` workpaper under Single Audit:

```
Materiality (Single Audit — 2026):
  Planning materiality:     $47,250   (0.5% of total federal expenditures: $9,450,000)
  Performance materiality:  $23,625   (50% of planning)
  Clearly trivial:          $2,363    (5% of planning)

Bank Reconciliation:
  Balance per bank statement:      $1,847,234.00
  Less: Outstanding checks:          ($14,822.50)
  Add:  Deposits in transit:          $19,444.00
  Adjusted bank balance:           $1,851,855.50
  Balance per books:               $1,851,902.00
  Unreconciled difference:               $46.50  ← WITHIN tolerance ($500)  ✓

Threshold Check (2 CFR 200):
  Total federal expenditures:  $9,450,000 → EXCEEDS Single Audit threshold ($750,000)
  Type A threshold (3%):          $283,500
  Single Audit required:         YES
```

**The model never calculates whether a value exceeds a threshold, computes a reconciliation, or determines materiality. The `model/tools/` library does all arithmetic in Python and injects results as verified facts. (I-02 fix.)**

---

## 7. Per-Task-Type Prompt Templates

### task_type: risk_classification

```
TASK
Classify the risk level for the condition described in the workpaper section above.

Output format (strict — all fields required):
Risk Level: [CRITICAL | HIGH | MEDIUM | LOW | N_A]
Risk Factors: [bullet list of specific factors driving this classification]
Regulation: [applicable regulation ID(s) from the list below — verified only]
Recommendation: [one concise action item]

Valid regulation IDs for this engagement: {regulation_id_list}
```

- **CoT:** No
- **Temperature:** 0.1
- **Decoding:** `do_sample=False`
- **Max tokens:** 512

---

### task_type: compliance_check

```
TASK
Check whether the content above is compliant with the applicable thresholds 
and regulations for this engagement.

Pre-computed threshold check results (do not recalculate):
{threshold_check_results}   ← from tools/thresholds.py

Think through this step by step before giving your final answer.

Output format (strict — all fields required):
Compliant: [YES | NO | PARTIAL]
Violations: [list each violation with: regulation_id, description, severity]
Thresholds Triggered: [list each: threshold_name, value found, limit, currency]
Recommendation: [remediation steps if not compliant, or "No action required"]
```

- **CoT:** Yes — "think through step by step" instruction triggers reasoning
- **Temperature:** 0.2
- **Max tokens:** 2048
- Reasoning stored in MongoDB `reasoning_chains`

---

### task_type: finding_documentation

```
TASK
Draft a complete audit finding for the condition identified in the workpaper 
section above. Use the mandatory five-element structure.

Think through each element before writing.

Output format (strict — all five elements required):
Criteria:       [the standard, regulation, or policy that was not met]
Condition:      [what was actually found — specific, factual, no opinion language]
Cause:          [the root cause — why the condition exists]
Effect:         [the actual or potential impact on the entity]
Recommendation: [specific management action to correct the condition]

Also provide:
Risk Level:   [CRITICAL | HIGH | MEDIUM | LOW]
Regulation:   [applicable regulation ID — verified only]
```

- **CoT:** Yes
- **Temperature:** 0.2
- **Max tokens:** 2048
- Reasoning stored in MongoDB `reasoning_chains`

---

### task_type: summarization

```
TASK
Write a concise workpaper summary for partner review.

Output format (strict — all fields required):
Summary: [2-4 paragraph narrative covering purpose, scope, key findings]
Key Figures: [table of significant monetary amounts, dates, quantities]
Flagged Items: [bullet list of items requiring audit attention or follow-up]
Workpaper Type: [detected workpaper type from standard list]
```

- **CoT:** No
- **Temperature:** 0.3
- **Max tokens:** 1024

---

### task_type: ner_extraction

```
TASK
Extract all financial entities, regulatory references, and identifiers 
from the workpaper section above.

Output format (strict — all fields required):
Entities: [list each: entity_type, value, confidence (0.0-1.0), position]
Entity Types Found: [distinct types present]
PII Detected: [YES | NO]
PII Types: [list if PII Detected=YES — these should not appear if PII scrubbing worked]
```

- **CoT:** No
- **Temperature:** 0.1
- **Decoding:** `do_sample=False`
- **Max tokens:** 512

---

## 8. vLLM AsyncLLMEngine Wrapper

```python
# model/inference/engine.py

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest

SAMPLING_PARAMS = {
    'risk_classification':    SamplingParams(temperature=0.1, max_tokens=512,  do_sample=False),
    'compliance_check':       SamplingParams(temperature=0.2, max_tokens=2048, top_p=0.95),
    'finding_documentation':  SamplingParams(temperature=0.2, max_tokens=2048, top_p=0.95),
    'summarization':          SamplingParams(temperature=0.3, max_tokens=1024, top_p=0.90),
    'ner_extraction':         SamplingParams(temperature=0.1, max_tokens=512,  do_sample=False),
}

class AuditInferenceEngine:

    def __init__(self, base_model_path: str):
        args = AsyncEngineArgs(
            model=base_model_path,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=16,
            max_num_seqs=16,           # 16 concurrent users
            dtype='bfloat16',
            quantization='bitsandbytes',
            gpu_memory_utilization=0.50,  # 50% of L40S for model
                                           # remaining 50% for e5 embedder + reranker
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self._current_adapter_path: str | None = None

    def load_adapter(self, adapter_path: str, version_tag: str) -> None:
        self._lora_request = LoRARequest(
            lora_name=version_tag,
            lora_int_id=1,
            lora_local_path=adapter_path,
        )
        self._current_adapter_path = adapter_path

    async def generate(
        self,
        prompt: str,
        task_type: str,
        request_id: str,
    ) -> tuple[str, int]:
        """Returns (raw_output, processing_time_ms)"""
        import time
        t0 = time.monotonic()
        params = SAMPLING_PARAMS[task_type]
        async for output in self.engine.generate(
            prompt,
            params,
            request_id,
            lora_request=self._lora_request,
        ):
            if output.finished:
                ms = int((time.monotonic() - t0) * 1000)
                return output.outputs[0].text, ms

        raise RuntimeError(f"vLLM generation did not complete for request {request_id}")
```

**GPU memory allocation on Server 2 (48GB L40S):**

| Process | VRAM allocation |
|---|---|
| vLLM engine (`gpu_memory_utilization=0.50`) | ~24 GB |
| Mistral 22B 4-bit NF4 model weights | ~14 GB (within the 24GB budget) |
| e5-mistral-7b-instruct embedder | ~14 GB |
| bge-reranker-v2-m3 | ~2 GB |
| OS + CUDA overhead | ~2 GB |
| **Total** | **~38 GB / 48 GB** |

---

## 9. Post-Processor Design

```python
# model/inference/post_processor.py
```

The post-processor runs on every model output before it reaches the database. It is the last line of defense before an AI_DRAFT output is surfaced to an auditor.

### Step 1 — Parse raw output into Pydantic schema

```python
def parse_output(raw_text: str, task_type: str) -> dict:
    """
    Parse raw model text into the appropriate output schema.
    Uses regex field extraction rather than JSON parsing —
    model output is structured prose, not JSON.
    """
    parsers = {
        'risk_classification':   parse_risk_classification,
        'compliance_check':      parse_compliance_check,
        'finding_documentation': parse_finding_documentation,
        'summarization':         parse_summarization,
        'ner_extraction':        parse_ner_extraction,
    }
    return parsers[task_type](raw_text)
```

### Step 2 — Validate required fields

```python
REQUIRED_FIELDS = {
    'risk_classification':   ['risk_level', 'risk_factors', 'recommendation'],
    'compliance_check':      ['compliant', 'recommendation'],
    'finding_documentation': ['criteria', 'condition', 'cause', 'effect', 'recommendation'],
    'summarization':         ['summary', 'workpaper_type'],
    'ner_extraction':        ['entities', 'entity_types_found', 'pii_detected'],
}

def validate_required_fields(parsed: dict, task_type: str) -> list[str]:
    """Returns list of missing field names. Empty list = all present."""
    return [
        f for f in REQUIRED_FIELDS[task_type]
        if not parsed.get(f)
    ]
```

**If missing fields found:** Retry once with explicit reminder appended to prompt. If still missing after retry: surface `output_status=AI_DRAFT` with a `MISSING_FIELDS` flag — reviewer sees a warning banner. (I-06 fix.)

### Step 3 — Verify regulation citations

Schema and full validation behaviour documented in `06_config_output.md` Section 5. Summary: any citation ID absent from `regulations_master.json` OR not yet in effect for the engagement's fiscal year is stripped and sets `hallucination_flag=True`.

```python
import json
from datetime import date
from pathlib import Path

# Loaded once at module import — restart API server after regulations_master.json changes
ALL_REGULATIONS: dict = json.loads(
    Path('config/regulations_master.json').read_text()
)

def get_valid_regulations(fiscal_year: int) -> set[str]:
    """
    Returns regulation IDs valid for the given fiscal year.
    Excludes regulations whose effective_from date is after fiscal_year end (Dec 31).
    """
    cutoff = date(fiscal_year, 12, 31)
    return {
        reg_id
        for reg_id, meta in ALL_REGULATIONS.items()
        if date.fromisoformat(meta['effective_from']) <= cutoff
    }

def verify_citations(
    parsed: dict,
    fiscal_year: int,
) -> tuple[list[str], list[str]]:
    """
    Returns (verified_citations, hallucinated_citations).
    Hallucinated = present in output but not valid for this engagement's fiscal year.
    """
    valid = get_valid_regulations(fiscal_year)
    cited = parsed.get('regulation_cited', [])
    if isinstance(cited, str):
        cited = [cited]
    verified     = [c for c in cited if c in valid]
    hallucinated = [c for c in cited if c not in valid]
    return verified, hallucinated
```

`fiscal_year` is read from `state.fiscal_year` (set by the router node from PG `engagements.fiscal_year`).

**If hallucinated citations found:**
- Remove hallucinated IDs from `regulation_cited` field
- Set a `hallucination_flag=True` in the output metadata
- Log to MLflow for monitoring
- Output is still surfaced as AI_DRAFT but reviewer sees a `⚠️ Unverified citation removed` banner
- The `hallucination_after` metric in `training_runs` is tracked from this signal

### Step 4 — Scan for prohibited opinion language

```python
OPINION_PATTERNS = [
    r'\bin our opinion\b',
    r'\bwe believe\b',
    r'\bpresent fairly\b',
    r'\bwe have audited\b',
    r'\bwe express\b',
    r'\bin all material respects\b',
    r'\bour audit included\b',
    r'\bwe conducted our audit\b',
]

def scan_opinion_language(text: str) -> list[str]:
    """Returns list of matched patterns. Empty = clean."""
    import re
    matches = []
    for pattern in OPINION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(pattern)
    return matches
```

**If opinion language found:**
- Output is still stored as AI_DRAFT
- `opinion_language_flag=True` set in metadata
- Reviewer sees a `🚨 Opinion language detected — mandatory review` alert
- This is a HIGH priority review flag — escalated to manager tier minimum

### Step 5 — Extract and store CoT reasoning

```python
def extract_reasoning(raw_text: str, task_type: str) -> str | None:
    """
    For CoT task types, split raw output into reasoning and final answer.
    Reasoning text → MongoDB reasoning_chains
    Final answer → model_outputs fields
    """
    COT_TASKS = {'compliance_check', 'finding_documentation'}
    if task_type not in COT_TASKS:
        return None

    # Model is prompted to "think through step by step" — reasoning precedes structured output
    # Split on first occurrence of the structured output header
    parts = raw_text.split('Output format', 1)
    if len(parts) == 2:
        reasoning_text = parts[0].strip()
        final_answer = 'Output format' + parts[1]
        return reasoning_text
    return raw_text   # fallback: store full text as reasoning
```

### Full post-processor pipeline

```
raw_output (from vLLM)
     │
     ├── parse_output()         → parsed dict
     │
     ├── validate_required_fields()
     │     └── if missing: retry once → if still missing: flag
     │
     ├── verify_citations()
     │     └── remove hallucinated IDs, log count
     │
     ├── scan_opinion_language()
     │     └── flag if found
     │
     ├── extract_reasoning()    → reasoning_text | None
     │
     └── build PostProcessorResult:
           parsed_output:       dict
           hallucination_flag:  bool
           hallucinated_ids:    list[str]
           opinion_flag:        bool
           missing_fields:      list[str]
           reasoning_text:      str | None
```

---

## 10. Full Inference Request Lifecycle

```
FastAPI receives POST /workpapers/{id}/analyze
     │
     ├── Auth: validate JWT, check RBAC
     ├── Load workpaper from PG (verify status='ready')
     ├── Load engagement from PG
     │     Base fields: client_type, is_gagas, has_single_audit, other_subtype, fiscal_year
     │     Unpack financial_context JSONB → engagement_financials dict:
     │       {total_federal_expenditures, total_revenue, total_assets, total_expenses,
     │        net_income, materiality_amount, fiscal_year_end}
     │     If workpaper_type ∈ {bank_reconciliation, trial_balance, financial_statements,
     │     analytical_procedure} AND financial_context == {} → return HTTP 422
     │       "financial_context required for this workpaper type — complete engagement setup first"
     │
     ├── [PARALLEL]
     │   ├── Get active chunks from PG workpaper_chunks
     │   └── Load current model_version from PG (is_current=TRUE)
     │
     ├── LangGraph Router
     │     Detect workpaper_type (from PG workpapers.workpaper_type)
     │     Detect task_type (from request body)
     │
     ├── Embed query
     │     query_text = "{task_type} context: {active_chunk_text[:500]}"
     │     query_embedding = e5_embedder.encode(
     │         f"Instruct: {TASK_INSTRUCTIONS[task_type]}\nQuery: {query_text}"
     │     )
     │
     ├── [PARALLEL — all three run concurrently after routing]
     │   │
     │   ├── LangGraph math_tool_dispatch node
     │   │     Look up TOOL_MANIFEST[workpaper_type]
     │   │     Run registered tools (thresholds, materiality, bank_rec, etc.)
     │   │     → state.math_results dict (~5-15ms, Python-only)
     │   │
     │   ├── retrieve_rag_context(query_embedding, workpaper_type, year)
     │   │     → top-20 candidates, year-weighted (~20ms)
     │   │
     │   └── retrieve_consistency_context(query_embedding, task_type)
     │         → top-5 consistency candidates (~20ms)
     │
     ├── [PARALLEL]
     │   ├── rerank(query_text, rag_candidates, top_n=5)
     │   └── rerank(query_text, consistency_candidates, top_n=3)
     │
     ├── assemble_context()
     │     system_prompt + rag_chunks + consistency_findings
     │     + math_results_block + active_chunk + task_instruction
     │
     ├── engine.generate(prompt, task_type, request_id)
     │     → raw_output, processing_time_ms
     │
     ├── post_processor.process(raw_output, task_type)
     │     → PostProcessorResult
     │
     ├── [IF reasoning_text]:
     │     INSERT MongoDB reasoning_chains
     │
     ├── INSERT PG model_outputs
     │     output_status = 'AI_DRAFT'   ← always, enforced by DB default
     │     rag_used = TRUE
     │     confidence_score = logit_prob_of_top_token
     │     processing_time_ms = ms
     │     mongo_reasoning_id = reasoning_chains._id (if CoT)
     │
     ├── INSERT PG audit_trail (event_type='created', entity_type='model_outputs')
     │
     └── Return response to FastAPI
           {output_id, parsed_output, flags, processing_time_ms}
           Label: "AI-Assisted Draft — Requires Senior Auditor Review Before Use"
```

---

## 11. Consistency Mechanism — How Past Findings Are Used

When a reviewer validates an output (`output_status → VALIDATED`):

```
1. model_outputs.embedding_synced set to FALSE by status update trigger
2. Sync worker (arq, polls every 60s) detects embedding_synced=FALSE on VALIDATED rows
3. Embeds model_outputs.finding text via e5-mistral-7b-instruct
4. Upserts to Qdrant validated_findings_embeddings
   payload: {task_type, risk_level, engagement_id, regulation_cited[], model_version}
5. Sets model_outputs.embedding_synced=TRUE
```

At next inference involving a similar condition:
- Top-3 similar past validated findings surfaced in the consistency context block
- Displayed to auditor in UI as: "Similar findings from prior engagements:"
- Helps auditor verify: "Is this the same severity we used last time? Same recommendation?"
- Builds institutional consistency over time without the model "memorizing" client details

**Important:** The consistency context is visible to the auditor in the UI, not just injected silently into the model. The auditor can see what past findings are being used for reference.

---

## 12. Guardrails Summary

| Guardrail | Enforcement point | Action on violation |
|---|---|---|
| All outputs marked AI_DRAFT | DB default constraint on insert | Impossible to bypass — constraint, not application logic |
| VALIDATED requires reviewer_id | PG CHECK constraint | DB rejects PATCH attempt without reviewer_id |
| Hallucinated citations removed | Post-processor Step 3 | Removed from output; `⚠️` banner shown to reviewer |
| Opinion language flagged | Post-processor Step 4 | `🚨` alert shown; escalated to manager tier |
| All arithmetic done by math tools library | LangGraph Tool Dispatch node (pre-generation) | `model/tools/` runs before vLLM; `COMPUTED AUDIT FACTS` block explicitly instructs model "do not recalculate" (I-02 fix) |
| Math tools run deterministically, not model-driven | `manifest.py` registry + LangGraph dispatch | Tool selection based on `workpaper_type` from PG — never based on LLM tool-call output |
| Temperature locked per task | SamplingParams per task_type | Non-determinism eliminated for structured output tasks (I-03 fix) |
| vLLM for concurrency | Production serving config | Ollama blocked from production — `max_num_seqs=16` (I-04 fix) |
| flash-attn CUDA version | Startup assertion | Server refuses to start with version mismatch (I-07 fix) |
| VLAN egress block | Firewall + startup curl test | Egress blocked at network layer; tested after every infra change |

---

## 13. Latency Budget

Target: p95 < 2 seconds for a complete inference request.

| Step | Typical latency |
|---|---|
| Query embedding (e5-mistral-7b) | ~30ms |
| Qdrant ANN search ×2 (parallel) | ~20ms |
| Math tools dispatch (parallel with RAG) | ~5-15ms — absorbed into RAG window, zero added latency |
| bge-reranker-v2-m3 ×2 (parallel) | ~130ms |
| Context assembly | ~5ms |
| vLLM generation (risk_classification) | ~400ms |
| vLLM generation (finding_documentation) | ~900ms |
| Post-processor | ~15ms |
| DB writes (PG + MongoDB) | ~20ms |
| **Total (risk_classification)** | **~620ms** |
| **Total (finding_documentation)** | **~1,120ms** |
| **p95 safety margin** | Leaves ~880ms headroom before 2s limit |

Math tools run in parallel with RAG retrieval and complete in <15ms (pure Python, no GPU). They add zero latency to the end-to-end request time.

Bottleneck is generation length, not retrieval. Both task types are comfortably within target.

---

*Next document: [06_config_output.md](06_config_output.md) — Config & Output Format*
