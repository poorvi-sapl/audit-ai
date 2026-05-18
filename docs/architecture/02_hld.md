# High-Level Design — Audit AI
**Version:** 1.0  
**Date:** April 2026  
**Status:** FINAL  
**Checklist items closed:** #3 (HLD), #4 (Data Flow)

---

## 1. System Components

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                               Audit AI                                          ║
╠══════════════════════════════╦═══════════════════════════════════════════════════╣
║   SERVER 1 — TRAINING        ║          SERVER 2 — INFERENCE                    ║
║                              ║                                                  ║
║  ┌───────────────────────┐   ║  ┌─────────────────────────────────────────┐    ║
║  │   Apache Airflow      │   ║  │          FastAPI (async)                │    ║
║  │   (DAG orchestrator)  │   ║  │   /workpapers  /outputs  /feedback      │    ║
║  └──────────┬────────────┘   ║  └──────────────────┬────────────────────────┘  ║
║             │                ║                     │                            ║
║  ┌──────────▼────────────┐   ║  ┌──────────────────▼────────────────────────┐  ║
║  │    ETL Workers        │   ║  │          LangGraph StateGraph             │  ║
║  │  Extract→Transform    │   ║  │  Router → Tool Dispatch → Context Assembl │  ║
║  │  →Load→Embed→Sync     │   ║  └───────────┬──────────────────┬────────────┘  ║
║  └──────────┬────────────┘   ║              │   [parallel]     │               ║
║             │                ║  ┌───────────▼──────────┐  ┌───▼──────────────┐ ║
║  ┌──────────▼────────────┐   ║  │   model/tools/       │  │  e5-mistral-7b   │ ║
║  │  e5-mistral-7b        │   ║  │   Math Library       │  │  bge-reranker    │ ║
║  │  (training embeddings)│   ║  │   thresholds         │  │  Qdrant ANN      │ ║
║  └──────────┬────────────┘   ║  │   materiality        │  │  (RAG retrieval) │ ║
║             │                ║  │   bank_rec / tb      │  └───────┬──────────┘ ║
║  ┌──────────▼────────────┐   ║  │   ratios / variance  │          │            ║
║  │  QLoRA Training       │   ║  │   (Python-only <15ms)│          │            ║
║  │  PEFT + SFTTrainer    │   ║  └───────────┬──────────┘          │            ║
║  │  dpo.py               │   ║              └────────────┬─────────┘            ║
║  └──────────┬────────────┘   ║                           │ (computed facts      ║
║             │                ║                           │  + RAG context)      ║
║  /intake/   /models/         ║  ┌────────────────────────▼──────────────────┐  ║
║                              ║  │       vLLM AsyncLLMEngine                 │  ║
║                              ║  │   Mistral 22B + active LoRA adapter       │  ║
║                              ║  └────────────────────────┬──────────────────┘  ║
║                              ║                           │                     ║
║                              ║  ┌────────────────────────▼──────────────────┐  ║
║                              ║  │  PostgreSQL 16  │  MongoDB 7               │  ║
║                              ║  │  Qdrant         │  Redis                   │  ║
║                              ║  └──────────────────────────────────────────┘   ║
╚══════════════════════════════╩═══════════════════════════════════════════════════╝
                       │  arq jobs cross servers via Redis queues  │
                       └───────────────────────────────────────────┘
```

---

## 2. End-to-End Data Flow

There are two parallel pipelines that share the same four stores. The **Training Pipeline** builds and improves the model. The **Inference Pipeline** uses the model in production. Both are connected by the **DPO Feedback Flywheel**.

```
TRAINING PIPELINE                         INFERENCE PIPELINE
─────────────────                         ──────────────────
Caseware export                           Auditor uploads workpaper
     │                                          │
     ▼                                          ▼
/intake/{eng_code}/                       FastAPI receive
     │                                          │
     ▼                                          ▼
arq poller (5 min)                        ETL pipeline (same extractor/
     │ status=final gate                  transformer chain)
     ▼                                          │
Airflow DAG triggered                           ▼
     │                                    MongoDB raw_workpapers
     ▼                                    PG workpapers row
Extractor (per file type)                       │
     │                                          ▼
     ▼                                    Transformer chain
MongoDB raw_workpapers                          │
PG workpapers row                               ▼
     │                                    PG workpaper_chunks
     ▼                                    + content_json JSONB
Transformer chain                               │
     │                                          ▼
     ▼                                    e5-mistral-7b embed
PG cleaning_log (per action)              Qdrant push
     │                                    embedding_synced=TRUE
     ▼                                          │
PG workpaper_chunks                             ▼
+ content_json JSONB                      LangGraph Router
     │                                    detects workpaper_type
     ▼                                          │
e5-mistral-7b embed                             ▼
Qdrant push                               Qdrant top-k retrieval
embedding_synced=TRUE                     (filtered by jurisdiction
     │                                     workpaper_type, year)
     ▼                                          │
SFT Pair Builder                                ▼
     │                                    bge-reranker-v2-m3
     ▼                                    reranks results
MongoDB training_pairs_content                  │
PG sft_training_pairs                           ▼
     │                                    Context assembly
     ▼                                    (chunks + prompt template)
Senior auditor blind review                     │
PG eval_results                                 ▼
reviewer_approved=TRUE                    vLLM generates response
     │                                          │
     ▼                                          ▼
JSONL export (70/15/15 split)             Post-processor validates
     │                                    (citations vs
     ▼                                     regulations_master.json)
SFT training run                                │
PG training_runs                                ▼
PG model_versions                         PG model_outputs
is_current=TRUE                           output_status=AI_DRAFT
                                          MongoDB reasoning_chains
                                          (if CoT task_type)
                                                │
                                                ▼
                              ┌─────────────────▼──────────────────┐
                              │        AUDITOR REVIEW               │
                              │                                     │
                              │  approved → VALIDATED               │
                              │  rejected → REJECTED                │
                              │  corrected → feedback_events        │
                              └─────────────────┬──────────────────┘
                                                │ (if senior/partner
                                                │  + corrected)
                                                ▼
                                        PG dpo_candidates
                                        MongoDB training_pairs_content
                                        (pair_category="dpo")
                                                │
                                                ▼
                                    ╔═══════════════════════╗
                                    ║  DPO FEEDBACK FLYWHEEL ║
                                    ║  (quarterly cycle)     ║
                                    ╚═══════════════════════╝
```

---

## 3. Training Pipeline — Detailed

### 3.1 Caseware Intake Convention

Auditors manually export finalized engagements from Caseware into the watched `/intake/` folder on Server 1 using this naming convention:

```
/intake/
  SA-2022-001/              ← folder name = engagement_code
    C-1_CDBG_Compliance.pdf     ← prefix = workpaper_ref, body = section
    F-3_SEFA_Schedule.xlsx
    ML-1_ManagementLetter.docx
```

**Parsing rules:**
- Folder name → `engagements.engagement_code` (PG lookup, must exist and have `status=final`)
- Filename prefix (before first `_`) → `workpapers.workpaper_ref` (e.g. `C-1`, `F-3`)
- Filename body (between first `_` and extension) → `workpapers.section` (e.g. `CDBG_Compliance`)
- File extension → `workpapers.file_type` detection (confirmed against magic bytes)
- No manual metadata entry by the auditor — everything derived from path and filename

**Safety gate:** The arq poller (5-minute interval) only triggers the Airflow DAG if:
1. A new folder exists in `/intake/` that has not been processed
2. The matching `engagement_code` exists in PG with `status = 'final'`

WIP or draft engagements never enter the pipeline.

### 3.2 Airflow DAG Structure

```
DAG: intake_watcher
  ├── sense_new_folder      (FileSensor, polls every 5 min)
  └── trigger_etl_pipeline  (TriggerDagRunOperator)

DAG: etl_pipeline
  ├── validate_engagement   (PG lookup: status=final, dedup by file_hash)
  ├── extract               (per-format extractor → MongoDB raw_workpapers)
  ├── transform             (chain: Normalizer→PIIScrubber→TemporalTagger
  │                          →Chunker→InjectionSanitizer → PG cleaning_log)
  ├── load_chunks           (PG workpaper_chunks + content_json JSONB)
  ├── embed_chunks          (e5-mistral-7b → Qdrant push → embedding_synced=TRUE)
  └── build_pairs           (SFT pair builder → PG sft_training_pairs
                             + MongoDB training_pairs_content)

DAG: jsonl_export           (triggered manually; accepts --mode and --framework params)
  ├── query_approved_pairs  (PG: sft_training_pairs WHERE reviewer_approved=TRUE)
  │                          mass mode  → all client_types included
  │                          batch mode → filtered by --framework param
  ├── fetch_content         (MongoDB: training_pairs_content by mongo_pair_id)
  ├── split_by_engagement   (70/15/15 — all workpapers from one engagement
  │                          go entirely into one split)
  └── write_jsonl           (mass mode  → mass_train.jsonl / mass_val.jsonl / mass_test.jsonl)
                             (batch mode → {framework}_train.jsonl / val.jsonl / test.jsonl)

DAG: training_trigger       (triggered manually; accepts --stage and --framework params)
  │
  │  ── Stage 1: Mass SFT (--stage=mass) ──────────────────────────────────────
  ├── run_sft               (all-framework JSONL, 18-20 hrs, checkpoints/500 steps)
  ├── run_eval              (eval.py on holdout: F1, hallucination rate per framework)
  ├── log_training_run      (PG training_runs, run_type='mass_sft')
  └── register_version      (PG model_versions, tag='mass-v{n}', is_current=FALSE)
                             ↓  (Stage 1 checkpoint is the base for all Stage 2 runs)
  │  ── Stage 2: Batch SFT (--stage=batch --framework=single_audit|gagas|gaas) ──
  ├── run_sft               (framework-specific JSONL continued from Stage 1 checkpoint)
  │                          single_audit: 8-10 hrs | gagas: 8-10 hrs | gaas: 8-10 hrs
  ├── run_eval              (eval.py on framework-specific holdout)
  ├── log_training_run      (PG training_runs, run_type='batch_sft', framework=?)
  └── register_version      (PG model_versions, tag='{framework}-v{n}')
                             ↓  (partner sign-off promotes single_audit adapter to is_current)
  │  ── Stage 3: Targeted Refinement (--stage=refine --framework=?) ────────────
  ├── run_sft               (high-quality subset — curated workpapers only)
  ├── run_eval              (full eval suite + regression on prior holdout)
  ├── log_training_run      (PG training_runs, run_type='refinement')
  └── register_version      (new model_versions, notify partner for sign-off)

DAG: dpo_training           (triggered quarterly)
  ├── assemble_dpo_pairs    (PG dpo_candidates WHERE used_in_run=FALSE,
  │                          quality_score above threshold)
  ├── fetch_pair_content    (MongoDB training_pairs_content pair_category="dpo")
  ├── run_dpo               (dpo.py, TRL library)
  ├── run_eval              (eval.py on holdout)
  ├── log_training_run      (PG training_runs, run_type=dpo)
  └── register_version      (new model_versions, flip is_current)
```

### 3.3 SFT Training Strategy

Three-stage staged hybrid approach. All stages run on Server 1. Each stage continues from the previous checkpoint — never retrains from the base model.

```
Stage 1 — Mass SFT
─────────────────
ALL 15 years × ALL US framework types
(Single Audit + GAGAS + GAAS)
       │
       ▼ mass-v1 checkpoint (shared audit domain foundation)
       │  Goal: audit vocabulary, Caseware structure, shared C/C/C/E/R patterns
       │
       ├─────────────────────────────────────────────────────┐
       │                                                     │
Stage 2 — Batch SFT (per framework, continues from mass-v1)  │
──────────────────────────────────────────────────────────── │
                                                             │
Track A: Single Audit          Track B: GAGAS               │
  2 CFR 200 / Uniform Guidance   Yellow Book                 │
  → single_audit-v1              → gagas-v1                  │
                                                             │
Track C: GAAS                                               │
  AU-C series                                                 │
  → gaas-v1                                                   │
       │
       ▼ Goal: framework-specific judgment, citation precision
       │
Stage 3 — Targeted Refinement (per framework, post-blind-review)
────────────────────────────────────────────────────────────────
High-quality curated workpapers only
Fixes systematic errors surfaced in eval_results
→ {framework}-v2 (or incremental patch)
```

**Inference expansion gate (separate from training):**  
Phase 1 deployment serves Single Audit engagements (`has_single_audit=true`) only. Expansion to GAGAS-only Govt and NPO engagements in Phase 2 requires `would_sign_off ≥ 80%` in `eval_results`. The gate is enforced at the API layer — not in the training pipeline.

---

### 3.4 SFT Pair Types

| pair_type | Prompt | Completion |
|---|---|---|
| `procedure_conclusion` | Audit procedure performed + sample size + exceptions found | Conclusion + control assessment |
| `risk_response` | Risk factor identified + entity context | Audit response + expanded procedures |
| `finding_recommendation` | Condition + criteria + cause + effect | Management letter recommendation + corrective action |
| `analytical_narrative` | Numeric analytics from workpaper (ratios, variances) | Narrative interpretation for partner review |

Each pair tagged with: `framework_section` (e.g. `2 CFR 200.302`), `split_assignment`, `quality_score`.

---

## 4. Inference Pipeline — Detailed

### 4.1 Request Lifecycle

```
1. Auditor selects engagement + uploads workpaper file
        │
2. FastAPI receives file
   └── SHA-256 hash check → reject if already processed (dedup)
        │
3. arq job queued: etl:extract
   └── Same extractor chain as training pipeline
   └── MongoDB raw_workpapers created
   └── PG workpapers row: extraction_status=pending
        │
4. arq job queued: etl:chunk
   └── Transformer chain runs
   └── PG workpaper_chunks created (content_json JSONB)
   └── PG cleaning_log written
   └── workpapers.extraction_status → chunked
        │
5. arq job queued: etl:embed
   └── e5-mistral-7b-instruct encodes each chunk
   └── Qdrant push to workpaper_chunks_embeddings
   └── workpaper_chunks.embedding_synced = TRUE
        │
6. Auditor triggers analysis (selects task_type or auto-detected)
        │
7. LangGraph Router
   └── Reads workpaper_type from PG workpapers
   └── Routes to appropriate agent node
        │
8. Agent node: Qdrant retrieval
   └── Embed query with e5-mistral-7b-instruct
   └── ANN search on workpaper_chunks_embeddings
       Filters: jurisdiction, workpaper_type, year (recent weighted higher)
   └── Top-k=20 candidates returned
        │
9. bge-reranker-v2-m3
   └── Cross-encode (query, each_chunk) → relevance score
   └── Top-5 highest-score chunks selected
        │
10. Also query: validated_findings_embeddings
    └── Retrieve top-3 similar past validated findings
    └── Injected as "consistency context" in prompt
        │
11. Context assembly
    └── [System prompt with client_type, is_gagas, has_single_audit, thresholds]
    └── [Top-5 retrieved chunks as RAG context]
    └── [Top-3 similar past findings as consistency context]
    └── [Active workpaper chunk(s)]
    └── [Task-specific instruction from prompt template]
        │
12. vLLM AsyncLLMEngine generates response
        │
13. Post-processor
    └── Validate output structure (all required fields present)
    └── Verify every regulation_cited ID against regulations_master.json
        └── Unverified citation → flag, do not surface, log hallucination
    └── Scan for prohibited opinion language → flag if found
    └── Parse into Pydantic output schema for task_type
        │
14. PG model_outputs inserted
    └── output_status = AI_DRAFT (default — cannot be overridden on insert)
    └── rag_used = TRUE
    └── mongo_reasoning_id → MongoDB reasoning_chains (if CoT)
    └── processing_time_ms recorded
        │
15. Response returned to auditor UI
    └── Label: "AI-Assisted Draft — Requires Senior Auditor Review Before Use"
    └── Confidence score displayed
    └── Regulation citations shown with verified/unverified badge
```

### 4.2 Task Types & CoT Routing

| task_type | Uses CoT | Reasoning stored in MongoDB |
|---|---|---|
| `risk_classification` | No | No |
| `compliance_check` | Yes | Yes — threshold logic explained step-by-step |
| `finding_documentation` | Yes | Yes — C/C/C/E/R construction explained |
| `summarization` | No | No |
| `ner_extraction` | No | No |

---

## 5. LangGraph Agent Architecture

### 5.1 AuditState TypedDict

```python
class AuditState(TypedDict):
    engagement_id:     str
    workpaper_id:      str
    workpaper_type:    str          # detected by router
    task_type:         str          # requested by auditor or auto-detected
    client_type:       str          # 'NPO' | 'Govt'
    is_gagas:          bool         # Yellow Book applies
    has_single_audit:  bool         # federal exp ≥ $750K
    other_subtype:     str | None   # Govt metadata only
    chunks:            list[dict]   # active workpaper chunks
    rag_context:       list[dict]   # retrieved + reranked chunks
    consistency_ctx:   list[dict]   # similar past validated findings
    prompt:            str          # assembled prompt
    raw_output:        str          # vLLM raw generation
    parsed_output:     dict         # post-processor result
    hallucinations:    list[str]    # unverified citations found
    output_id:         str          # PG model_outputs UUID after insert
```

### 5.2 Graph Structure

```
                    ┌─────────────┐
   START ──────────▶│   Router    │
                    └──────┬──────┘
         ┌─────────────────┼─────────────────────┐
         │                 │                     │
         ▼                 ▼                     ▼
  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐
  │trial_balance│  │  bank_rec    │  │    compliance     │
  │   agent     │  │    agent     │  │      agent        │
  └──────┬──────┘  └──────┬───────┘  └────────┬──────────┘
         │                │                   │
         ├── (more agent nodes: ar, fixed_assets, etc.)
         │
         ▼ (all agents converge here)
  ┌─────────────────┐
  │  RAG Retriever  │ ← Qdrant top-k + bge-reranker
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  vLLM Engine    │
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ Post-Processor  │ ← validate, verify citations, parse schema
  └────────┬────────┘
           ▼
         END (output_id returned to FastAPI)
```

### 5.3 Tools Available to Agents

Defined in `model/tools/manifest.py` — passed to model at prompt time.

| Tool | Module | Purpose |
|---|---|---|
| `check_threshold` | tools/thresholds.py | Check amount vs thresholds.yaml (US + India sections) |
| `calculate_materiality` | tools/materiality.py | Compute materiality thresholds from total assets/revenue |
| `reconcile_trial_balance` | tools/trial_balance.py | Verify debits = credits, flag out-of-balance |
| `check_bank_rec` | tools/bank_rec.py | Verify book-to-bank reconciliation |
| `calculate_ratios` | tools/ratios.py | Liquidity, leverage, coverage ratios |
| `calculate_variance` | tools/variance.py | Budget-to-actual, prior year comparisons |

---

## 6. DPO Feedback Flywheel

```
    ┌─────────────────────────────────────────────┐
    │            CONTINUOUS IMPROVEMENT            │
    └────────────────────┬────────────────────────┘
                         │
    ┌────────────────────▼────────────────────────┐
    │  Auditor sees AI_DRAFT output in app        │
    │                                             │
    │  [👍 Approved]   [✏️ Corrected]  [👎 Rejected] │
    └────────┬──────────────┬──────────────┬──────┘
             │              │              │
             ▼              ▼              ▼
    output_status=   feedback_events   output_status=
      VALIDATED        row created       REJECTED
             │              │
             │    (if senior/partner tier only)
             │              │
             ▼              ▼
    Embed VALIDATED    dpo_candidates row
    finding → Qdrant   MongoDB training_pairs_content
    validated_findings  (pair_category="dpo")
    _embeddings        chosen=correction
                       rejected=original AI output
                              │
                              ▼
                   ┌──────────────────────┐
                   │  QUARTERLY DPO RUN   │
                   │                      │
                   │  Cycle 1 (Month 3-4) │
                   │  300-500 corrections │
                   │  focus: risk_class   │
                   │        + compliance  │
                   │                      │
                   │  Cycle 2 (Month 6)   │
                   │  800-1000 corrections│
                   │  focus: findings     │
                   │        + escalation  │
                   │                      │
                   │  Cycle 3 (Month 9-12)│
                   │  2000+ corrections   │
                   │  all task_types      │
                   └──────────┬───────────┘
                              │
                              ▼
                   training_runs (run_type=dpo)
                   model_versions (new is_current)
                   F1 + hallucination_rate measured
                   Partner signs off before deploy
```

**DPO eligibility rules (enforced at application layer):**
- `reviewer_tier` must be `senior` or `partner`
- `rating` must be `corrected` (not just rejected)
- `quality_score` check: pairs where chosen ≈ rejected are excluded
- One DPO pair per feedback_event (UNIQUE constraint on feedback_id)

---

## 7. Authentication Flow

### Phase 1 — JWT + bcrypt

```
1. POST /auth/login  {email, password}
        │
2. PG lookup: users WHERE email=? AND is_active=TRUE
        │
3. bcrypt verify password against hashed_password
        │
4. Issue JWT:  {sub: user_id, firm_id, role, tier, jti, exp: now+15min}
        │
5. Every authenticated request:
   └── Decode + verify JWT signature
   └── Check Redis: session:{jti}:revoked → 401 if found
   └── Check exp → 401 if expired
   └── RBAC check for endpoint (role + tier requirements)
        │
6. POST /auth/logout
   └── SET Redis session:{jti}:revoked = 1  (TTL: 24h)
```

### Phase 2 — M365 Azure AD SSO (additive migration)

```
1. User clicks "Sign in with Microsoft"
        │
2. OAuth2 redirect to Azure AD
        │
3. Azure AD returns id_token with oid claim
        │
4. PG lookup: users WHERE azure_oid = oid AND is_active=TRUE
        │
5. Issue internal JWT (same structure as Phase 1)
        │
6. hashed_password column remains — set NULL for SSO users
   azure_oid column populated
   No destructive migration needed
```

**No code changes to RBAC, rate limiting, or any downstream logic.**  
The only migration: `ALTER TABLE users ADD COLUMN azure_oid VARCHAR(100); ALTER TABLE users ALTER COLUMN hashed_password DROP NOT NULL;`

---

## 8. Multi-Firm & Tenant Isolation

The schema is multi-firm from day one. HCLLP is a single row in `firms`. Every query that touches tenant data is scoped by `firm_id`.

| Table | Isolation mechanism |
|---|---|
| engagements | `firm_id FK → firms` |
| users | `firm_id FK → firms` |
| workpapers | via engagement → firm |
| workpaper_chunks | via workpaper → engagement → firm |
| model_outputs | via workpaper → engagement → firm |
| Qdrant payloads | `engagement_id` in every point payload — filter on retrieval |

When/if additional firms are onboarded: add a row to `firms`, add row-level security policies to PG. Zero schema migration needed.

---

## 9. Engagement Lifecycle State Machine

```
           [created]
               │
               ▼
           planning
               │
               ▼
           fieldwork
               │
               ▼
          completion
               │
               ▼
           reporting
               │
      ┌────────┴────────┐
      ▼                 ▼
    final           excluded
      │
      ▼
  (intake watcher
   picks up folder
   and triggers ETL)
```

Only `status=final` engagements are processed by the training pipeline.  
`under_review` is set when a concern is raised post-finalization — pauses any re-processing.

---

## 10. Workpaper Status State Machines

### extraction_status (pipeline layer)

```
pending → extracted → chunked → [embedding triggered] → error
```

### status (app layer)

```
processing → ready → error
```

Both state machines are independent. `extraction_status` tracks the ETL job. `status` tracks readiness for auditor use. A workpaper moves to `status=ready` only after `extraction_status=chunked` and all chunks have `embedding_synced=TRUE`.

### model_outputs.output_status

```
AI_DRAFT  →  VALIDATED  (requires reviewer_id NOT NULL — CHECK constraint)
          →  REJECTED
```

There is no path from REJECTED back to AI_DRAFT. A new inference job must be triggered to generate a fresh output.

---

*Next document: [03_lld.md](03_lld.md) — Low-Level Design*
