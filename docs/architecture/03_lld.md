# Low-Level Design — Audit AI
**Version:** 1.0  
**Date:** April 2026  
**Status:** FINAL  
**Checklist items closed:** #5 (DB Flow), #6 (LLD), #8 (DB Schema)

---

## 1. Project Folder Structure

```
audit-ai/
│
├── etl/                            ← data pipeline (extraction, cleaning, versioning)
│   └── core/
│       ├── extractors/
│       │   ├── base.py             ← BaseExtractor abstract class
│       │   ├── excel.py            ← xlsx, xls, xlsm, csv
│       │   ├── word.py             ← docx, doc
│       │   ├── pdf_text.py         ← digital-native PDFs (pdfplumber)
│       │   ├── pdf_scanned.py      ← Docling + Surya + Tesseract fallback
│       │   ├── dbf.py              ← DBF, FPT, BAK, CGF
│       │   └── tally_xml.py        ← Tally ERP XML exports (.xml with <TALLYMESSAGE> root); maps LEDGER/VOUCHER elements to {label, amount, currency, row_type} rows for numeric chunker; no new dependency (stdlib xml.etree.ElementTree)
│       ├── transformers/
│       │   ├── base.py             ← BaseTransformer abstract class
│       │   ├── normalizer.py       ← currency, dates, whitespace
│       │   ├── pii_scrubber.py     ← Presidio + custom US/India patterns
│       │   ├── temporal.py         ← year tagging + drift weighting
│       │   ├── chunker.py          ← token-aware chunking at boundaries
│       │   └── injection_sanitizer.py ← prompt injection pattern removal
│       ├── loaders/
│       │   ├── base.py             ← BaseLoader abstract class
│       │   ├── jsonl.py            ← writes JSONL training files
│       │   ├── postgres.py         ← writes Records to PG via SQLAlchemy
│       │   ├── mongodb.py          ← writes raw content + pairs to MongoDB
│       │   └── qdrant.py           ← pushes embeddings to Qdrant
│       ├── pipeline.py             ← orchestrates E → T → L
│       └── record.py               ← Record + RawContent + Chunk pydantic schemas
│
├── connectors/
│   ├── base.py                     ← BaseConnector abstract class
│   └── audit_ai.py                 ← PII rules, field maps, pair builder config
│
├── config/
│   ├── thresholds.yaml             ← $750K federal, $10K CTR, materiality percentages (QC Partner sign-off required)
│   ├── pii_patterns.yaml           ← SSN, EIN, ITIN, bank routing, account patterns (US-only)
│   ├── regulations_master.json     ← valid regulation IDs for citation verification
│   ├── outdated_regulations.json   ← regulation transitions with effective dates
│   ├── regulation_validity_dates.json
│   └── field_mappings/
│       ├── busy.yaml               ← Busy accounting software field map
│       └── default.yaml            ← generic DBF field map
│
├── model/                          ← ML layer (training, inference, tools)
│   ├── training/
│   │   ├── train.py                ← QLoRA SFT training entry point
│   │   ├── dpo.py                  ← DPO training entry point (TRL library)
│   │   └── eval.py                 ← evaluation metrics suite
│   ├── inference/
│   │   ├── engine.py               ← vLLM AsyncLLMEngine wrapper
│   │   └── post_processor.py       ← validate, parse, verify citations
│   └── tools/                      ← audit math tools library (Python-only, no GPU, no LLM)
│       ├── __init__.py             ← exports: thresholds, materiality, bank_rec, trial_balance, ratios, variance
│       ├── thresholds.py           ← regulatory threshold comparisons (2 CFR 200, GAGAS/GAAS; config-driven from thresholds.yaml)
│       ├── materiality.py          ← planning materiality, performance materiality, clearly trivial (framework-aware)
│       ├── bank_rec.py             ← bank reconciliation: outstanding items, deposits in transit, unreconciled difference
│       ├── trial_balance.py        ← debit/credit balance verification, out-of-balance detection, subtotal checks
│       ├── ratios.py               ← current ratio, quick ratio, debt-to-equity, days receivable
│       ├── variance.py             ← absolute and % variance per line item; flags items exceeding materiality or 10%
│       └── manifest.py             ← TOOL_MANIFEST dict: workpaper_type → [tool_names]; read by LangGraph tool dispatch node
│
├── agents/                         ← LangGraph graph + agents
│   ├── graph.py                    ← main StateGraph assembly
│   ├── state.py                    ← AuditState TypedDict
│   ├── router.py                   ← workpaper type detection + routing
│   └── workpapers/
│       ├── trial_balance.py
│       ├── bank_rec.py
│       ├── ar.py
│       ├── fixed_assets.py
│       └── compliance.py
│
├── api/                            ← FastAPI backend
│   ├── main.py                     ← app factory, middleware registration
│   ├── routers/
│   │   ├── auth.py
│   │   ├── engagements.py
│   │   ├── workpapers.py
│   │   ├── outputs.py
│   │   ├── feedback.py
│   │   └── training.py
│   ├── schemas/
│   │   ├── output.py               ← 5 Pydantic output schemas
│   │   ├── engagement.py
│   │   ├── workpaper.py
│   │   ├── user.py
│   │   └── common.py               ← shared pagination, error schemas
│   ├── dependencies.py             ← get_db, get_current_user, require_role
│   └── middleware.py               ← rate limiting, request logging
│
├── db/
│   ├── enums.py                    ← canonical Python StrEnums (WorkpaperType, TaskType, etc.) — single source of truth; imported by ETL, Alembic migrations, and API schemas
│   ├── models/                     ← SQLAlchemy ORM models (one file per table)
│   │   ├── firms.py
│   │   ├── users.py
│   │   ├── engagements.py
│   │   ├── workpapers.py
│   │   ├── workpaper_chunks.py
│   │   ├── cleaning_log.py
│   │   ├── sft_training_pairs.py
│   │   ├── eval_results.py
│   │   ├── model_outputs.py
│   │   ├── feedback_events.py
│   │   ├── dpo_candidates.py
│   │   ├── training_runs.py
│   │   ├── model_versions.py
│   │   └── audit_trail.py
│   ├── repositories/               ← async CRUD (one file per aggregate)
│   │   ├── workpaper_repo.py
│   │   ├── output_repo.py
│   │   ├── feedback_repo.py
│   │   └── training_pair_repo.py
│   └── migrations/                 ← Alembic migration files
│       ├── env.py
│       └── versions/
│
├── airflow/
│   └── dags/
│       ├── intake_watcher.py
│       ├── etl_pipeline.py
│       ├── jsonl_export.py
│       ├── training_trigger.py
│       └── dpo_training.py
│
├── export/                         ← output renderers (one file per format)
│   ├── word_renderer.py            ← python-docx template replacement (includes merge_runs fix)
│   ├── excel_renderer.py           ← openpyxl workbook builder
│   ├── pdf_renderer.py             ← word → LibreOffice headless → PDF
│   └── caseware_exporter.py        ← generates .cwx XML import file for Caseware Working Papers; Phase 1: download only; Phase 2: direct push to Caseware Cloud API
│
├── templates/                      ← export renderer templates
│   ├── management_letter_word_v1.docx
│   ├── trial_balance_excel_v1.xlsx
│   └── generic_word_v1.docx
│
├── data/                           ← reference data (read-only at runtime)
│   └── (regulations_master.json symlinked from config/)
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── .env.example
├── alembic.ini
├── pyproject.toml
└── docker-compose.yml              ← local dev only (all 4 stores)
```

---

## 2. PostgreSQL — 14 Tables (Full Specification)

### ENUMs (define before tables)

```sql
CREATE TYPE user_role      AS ENUM ('admin','manager','senior','qualified','junior');
CREATE TYPE user_tier      AS ENUM ('partner','manager','senior','staff');
-- engagement_type ENUM removed — replaced by client_type (VARCHAR) + is_gagas (BOOL) + has_single_audit (BOOL) on engagements table.
-- These three columns together express all real combinations HCLLP runs. No display field stored.
CREATE TYPE eng_phase      AS ENUM ('planning','fieldwork','completion','reporting');
CREATE TYPE eng_status     AS ENUM ('active','final','excluded','under_review');
CREATE TYPE export_fmt     AS ENUM ('word','excel','pdf','caseware');
                                    -- caseware: generates .cwx XML import file for Caseware Working Papers
CREATE TYPE file_type      AS ENUM ('excel','word','pdf_text','pdf_scanned','dbf','caseview','tally_xml');
CREATE TYPE wp_type        AS ENUM ('bank_reconciliation','trial_balance','financial_statements',
                                    'analytical_procedure','sefa_schedule','compliance_test',
                                    'internal_control','risk_assessment','finding_documentation',
                                    'management_letter','planning_document','other','final_report');
                                    -- 13 canonical values; derived from db/enums.py WorkpaperType StrEnum
                                    -- Pre-deployment change: replaces substantive_test, disclosure_note, audit_program
                                    -- (those names existed in ENUM but had no TOOL_MANIFEST entries — dead code).
                                    -- DROP TYPE wp_type + recreate if run against an existing DB.
CREATE TYPE extract_status  AS ENUM ('pending','extracted','chunked','error','needs_password');
                                     -- needs_password: file is encrypted; extraction blocked until
                                     --   password supplied via PATCH /workpapers/{id}/password
CREATE TYPE wp_status       AS ENUM ('processing','ready','error');
CREATE TYPE cleaning_action AS ENUM ('pii_removed','duplicate_flagged','rollforward_detected',
                                     'boilerplate_stripped','incomplete_removed',
                                     'ocr_corrected','approved','password_protected');
CREATE TYPE split_assign   AS ENUM ('train','val','test');
CREATE TYPE pair_type      AS ENUM ('procedure_conclusion','finding_recommendation',
                                    'risk_response','analytical_narrative');
CREATE TYPE eval_role      AS ENUM ('partner','manager','senior');
CREATE TYPE task_type      AS ENUM ('risk_classification','compliance_check',
                                    'finding_documentation','summarization','ner_extraction');
CREATE TYPE risk_level     AS ENUM ('CRITICAL','HIGH','MEDIUM','LOW','N_A');
CREATE TYPE output_status  AS ENUM ('AI_DRAFT','VALIDATED','REJECTED');
CREATE TYPE reviewer_tier  AS ENUM ('partner','manager','senior','qualified','junior');
CREATE TYPE fb_rating      AS ENUM ('approved','rejected','corrected');
CREATE TYPE wp_permission      AS ENUM ('read','write','review','approve');
                                        -- read: view workpaper and outputs
                                        -- write: upload, edit draft outputs
                                        -- review: validate/reject outputs (senior+)
                                        -- approve: final sign-off (partner only)
CREATE TYPE run_type           AS ENUM ('sft','dpo','targeted_refinement');
CREATE TYPE run_status         AS ENUM ('queued','running','completed','failed');
CREATE TYPE inference_run_type AS ENUM ('live','regression','dpo_collection','batch_eval','debug');
                                        -- live: normal production inference
                                        -- regression: automated regression suite run (no real workpaper)
                                        -- dpo_collection: inference run specifically to gather preference pairs
                                        -- batch_eval: bulk eval round before model promotion
                                        -- debug: developer testing, excluded from all metrics
```

---

### Table 1 — firms

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK, default gen_random_uuid() |
| name | VARCHAR(200) | NO | e.g. "Harshwal & Company LLP" |
| slug | VARCHAR(100) | NO | UNIQUE. URL-safe e.g. "hcllp" |
| country | VARCHAR(10) | NO | "US" — HCLLP is US-only |
| is_active | BOOLEAN | NO | Default TRUE |
| created_at | TIMESTAMPTZ | NO | Default NOW() |

**Indexes:** `UNIQUE(slug)`

---

### Table 2 — users

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| firm_id | UUID | NO | FK → firms(id) RESTRICT |
| email | VARCHAR(255) | NO | UNIQUE |
| hashed_password | VARCHAR(255) | YES | NULL for SSO users (Phase 2) |
| full_name | VARCHAR(200) | NO | |
| role | user_role | NO | |
| tier | user_tier | NO | |
| is_active | BOOLEAN | NO | Default TRUE |
| last_login_at | TIMESTAMPTZ | YES | Updated on every successful auth |
| azure_oid | VARCHAR(100) | YES | NULL in Phase 1. Populated in Phase 2. |
| created_at | TIMESTAMPTZ | NO | Default NOW() |
| updated_at | TIMESTAMPTZ | YES | Auto-updated on row change |

**Indexes:** `UNIQUE(email)`, `BTREE(firm_id)`

---

### Table 3 — engagements

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| firm_id | UUID | NO | FK → firms(id) RESTRICT |
| engagement_code | VARCHAR(50) | NO | UNIQUE. e.g. "SA-2022-001" |
| name | VARCHAR(200) | NO | Display name |
| client_name | VARCHAR(200) | YES | Anonymized display label |
| entity_alias | VARCHAR(100) | YES | PII-scrubbed placeholder e.g. "[CLIENT_1]" |
| client_type | VARCHAR(10) | NO | `CHECK (client_type IN ('NPO','Govt'))`. NPO = FASB/US GAAP + AICPA GAAS. Govt = GASB + GAGAS. Determines materiality base and RAG regulation cluster. |
| is_gagas | BOOLEAN | NO | Default FALSE. TRUE for all Govt engagements; TRUE for NPO engagements that receive government grants (Yellow Book applies). |
| has_single_audit | BOOLEAN | NO | Default FALSE. TRUE when federal expenditures ≥ $750K (2 CFR 200 / Uniform Guidance applies). SEFA workpapers and federal program compliance tests only generated when TRUE. |
| other_subtype | VARCHAR(50) | YES | Nullable. Populated for Govt subtypes only — e.g. `'special_district'`, `'ftr_filer'`, `'housing_authority'`. Metadata only — no downstream routing in Phase 1. |
| fiscal_year | INTEGER | YES | e.g. 2024 |
| financial_context | JSONB | NO | Default `'{}'::jsonb`. Stores engagement-level financial figures consumed by `model/tools/` math library at inference time. See schema below. |
| phase | eng_phase | YES | App layer only |
| status | eng_status | NO | Default 'active' |
| total_workpapers | INTEGER | NO | Default 0. Incremented by pipeline. |
| default_export_format | export_fmt | YES | Engagement-level export default |
| default_export_template | VARCHAR(100) | YES | e.g. "management_letter_word_v1" |
| created_by | UUID | NO | FK → users(id) SET NULL |
| created_at | TIMESTAMPTZ | NO | Default NOW() |

**Indexes:** `UNIQUE(engagement_code)`, `BTREE(firm_id)`

**financial_context JSONB schema:**

```json
{
  "total_federal_expenditures": 0.00,
  "total_revenue":              0.00,
  "total_assets":               0.00,
  "total_expenses":             0.00,
  "net_income":                 0.00,
  "materiality_amount":         0.00,
  "fiscal_year_end":            "2024-03-31"
}
```

All fields optional at create time (default `{}`). `materiality_basis` removed — the base figure is now derived from `client_type` by `materiality.py` reading `config/thresholds.yaml` (NPO → total_expenses; Govt → revenues or expenditures, whichever larger). `currency` removed — always USD. `total_expenses` added for NPO materiality base. Collected via engagement creation form in the UI. Required before inference can run on `bank_reconciliation`, `trial_balance`, `financial_statements`, or `analytical_procedure` workpaper types — the API returns HTTP 422 if these types are analyzed and `financial_context` is `{}`.

---

### Table 4 — workpapers

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| engagement_id | UUID | NO | FK → engagements(id) RESTRICT |
| workpaper_ref | VARCHAR(50) | YES | e.g. "C-1", "F-3" (from filename prefix) |
| section | VARCHAR(100) | YES | e.g. "CDBG_Compliance" (from filename body) |
| file_name | VARCHAR(255) | NO | Original filename, sanitized on ingest |
| file_hash | VARCHAR(64) | NO | SHA-256 hex digest of file bytes. Two-level race protection: (1) Redis lock keyed on `wp_ingest:{file_hash}` held from dedup_check through INSERT — prevents concurrent workers from both passing the SELECT check; (2) `UNIQUE(file_hash)` DB constraint as last-resort guard if the Redis lock is bypassed (e.g. Redis outage). INSERT uses `ON CONFLICT DO NOTHING`; conflict is logged and treated as a clean skip, not an error. |
| file_type | file_type | NO | |
| file_size_bytes | INTEGER | YES | |
| workpaper_type | wp_type | YES | Detected by router |
| sheet_count | INTEGER | YES | Excel sheets or PDF pages |
| total_tokens | INTEGER | YES | Sum across all chunks (populated after chunking) |
| year_of_workpaper | INTEGER | YES | For temporal weighting in training |
| extraction_status | extract_status | NO | Default 'pending' |
| status | wp_status | NO | Default 'processing' |
| upload_by | UUID | NO | FK → users(id) SET NULL |
| ocr_used | BOOLEAN | NO | Default FALSE |
| pii_scrubbed | BOOLEAN | NO | Default FALSE. Fast flag; detail in cleaning_log. |
| mongo_raw_id | VARCHAR(50) | YES | Pointer to MongoDB raw_workpapers._id |
| extracted_at | TIMESTAMPTZ | YES | |
| created_at | TIMESTAMPTZ | NO | Default NOW() |
| updated_at | TIMESTAMPTZ | YES | Auto-updated |

**Indexes:** `UNIQUE(file_hash)`, `BTREE(engagement_id)`, `BTREE(extraction_status)`, `BTREE(status)`

---

### Table 4a — workpaper_permissions

Document-level RBAC. Each row grants a specific permission level to either a named user or an entire role for one workpaper. Checked by the API middleware before every workpaper or output read/write.

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| workpaper_id | UUID | NO | FK → workpapers(id) CASCADE DELETE |
| user_id | UUID | YES | FK → users(id) CASCADE DELETE. NULL = role-level grant. |
| role | user_role | YES | NULL = user-specific grant. |
| permission | wp_permission | NO | |
| granted_by | UUID | NO | FK → users(id) RESTRICT. Manager or admin who granted access. |
| granted_at | TIMESTAMPTZ | NO | Default NOW() |

**Constraints:**
- `CHECK ((user_id IS NOT NULL AND role IS NULL) OR (user_id IS NULL AND role IS NOT NULL))` — exactly one of user_id or role must be set; never both
- `UNIQUE(workpaper_id, user_id)` (partial: WHERE user_id IS NOT NULL)
- `UNIQUE(workpaper_id, role)` (partial: WHERE role IS NOT NULL)

**Indexes:** `BTREE(workpaper_id)`, `BTREE(user_id)`

**Default permission seeding:** When a workpaper is created (POST `/engagements/{id}/workpapers`), the API seeds default rows based on the engagement's firm RBAC policy:

| Role | Default permission |
|------|--------------------|
| `junior` | `write` — can upload and edit their own drafts |
| `senior` | `write` — full write access to all workpapers in engagement |
| `manager` | `review` — can validate/reject outputs |
| `partner` | `approve` — final sign-off authority |
| `admin` | `approve` — system admin, same as partner |

Overrides: a manager can grant a specific user elevated or restricted access via `POST /workpapers/{id}/permissions`. The `granted_by` column records accountability.

**API enforcement:** The FastAPI dependency `require_wp_permission(permission_level)` is injected into every workpaper-scoped endpoint. It queries `workpaper_permissions` for a matching `(workpaper_id, user_id)` row first (user-specific), then falls back to a `(workpaper_id, role)` row matching the user's role. If neither exists: HTTP 403.

---

### Table 5 — workpaper_chunks

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK — also used as Qdrant point ID |
| workpaper_id | UUID | NO | FK → workpapers(id) CASCADE DELETE |
| sheet_name | VARCHAR(200) | YES | Sheet name (Excel) or section heading (PDF) |
| chunk_index | INTEGER | NO | 0-based position within workpaper |
| total_chunks | INTEGER | YES | Total chunks for this workpaper |
| token_count | INTEGER | YES | Token count of this chunk |
| content_json | JSONB | NO | `{text, tables[], cells[], page_refs[], chunk_mode, rows[], numeric_columns[], currency}` |
| source_engagement_id | UUID | YES | FK → engagements(id) SET NULL. Set when this chunk's workpaper was rolled forward from a prior engagement. NULL for original workpapers. Used to trace rollforward lineage and prevent stale-number retrieval. |
| is_rollforward | BOOLEAN | NO | Default FALSE. TRUE when source_engagement_id is set. Denormalised boolean for fast Qdrant payload filter — rollforward chunks are excluded from cross-engagement RAG retrieval. |
| embedding_synced | BOOLEAN | NO | Default FALSE. TRUE after Qdrant push. |
| created_at | TIMESTAMPTZ | NO | Default NOW() |

**Indexes:** `BTREE(workpaper_id)`, `BTREE(embedding_synced)`, `BTREE(source_engagement_id)`

> **Rollforward retrieval rule:** Chunks with `is_rollforward = TRUE` are indexed in Qdrant (available for training data) but excluded from RAG retrieval at inference time via a `must_not` payload filter. This prevents stale prior-year numbers from appearing as context for this year's engagement. Original chunks (`is_rollforward = FALSE`) are retrieved normally across engagements.

---

### Table 6 — cleaning_log

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| workpaper_id | UUID | NO | FK → workpapers(id) CASCADE DELETE |
| cleaning_action | cleaning_action | NO | |
| pii_types_found | TEXT[] | YES | e.g. `["CLIENT_NAME","EIN","SSN"]` |
| is_duplicate_of | UUID | YES | workpaper_id of original if duplicate |
| similarity_score | DECIMAL(5,4) | YES | 0.0–1.0 for duplicate/rollforward detection |
| detail_json | JSONB | YES | `{patterns_matched, locations[], replacement_token}` |
| cleaned_by | VARCHAR(50) | NO | "auto" for pipeline; user UUID string for manual |
| cleaned_at | TIMESTAMPTZ | NO | Default NOW() |

**Indexes:** `BTREE(workpaper_id)`

---

### Table 7 — sft_training_pairs

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| workpaper_id | UUID | NO | FK → workpapers(id) RESTRICT |
| engagement_id | UUID | NO | FK → engagements(id) RESTRICT. Denormalised for split queries. |
| pair_type | pair_type | NO | |
| pair_hash | VARCHAR(64) | NO | SHA-256 hex digest of `prompt + completion` text. Idempotency key — prevents duplicate pairs when an engagement is re-submitted. |
| mongo_pair_id | VARCHAR(50) | NO | Pointer to MongoDB training_pairs_content._id |
| split_assignment | split_assign | NO | Assigned at engagement level, not pair level |
| quality_score | DECIMAL(3,2) | YES | 0.00–1.00 automated quality score |
| reviewer_approved | BOOLEAN | NO | Default FALSE. Must be TRUE before JSONL export. |
| framework_section | VARCHAR(100) | YES | e.g. "2 CFR 200.302" |
| reviewed_at | TIMESTAMPTZ | YES | NULL until approved or rejected |
| created_at | TIMESTAMPTZ | NO | Default NOW() |

**Constraints:** `UNIQUE(workpaper_id, pair_type, pair_hash)` — idempotency guard; `ON CONFLICT DO NOTHING` on insert ensures re-submissions never produce duplicate training pairs.  
**Indexes:** `BTREE(workpaper_id)`, `BTREE(engagement_id)`, `BTREE(split_assignment)`, `BTREE(reviewer_approved)`

---

### Table 8 — eval_results

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| model_output_id | UUID | NO | FK → model_outputs(id) RESTRICT. Primary link — the output being evaluated. |
| workpaper_id | UUID | YES | Nullable denormalized FK → workpapers(id) SET NULL. NULL for regression suite runs (no real workpaper). Populated for production blind review. |
| evaluator_id | UUID | NO | FK → users(id) RESTRICT. Must be senior/manager/partner tier. |
| evaluator_role | eval_role | NO | |
| evaluation_round | VARCHAR(50) | NO | e.g. `pre_golive_r1`, `dpo_cycle_1`, `regression_weekly` |
| task_type | task_type | NO | Denormalized from model_outputs for fast gate queries |
| would_sign_off | BOOLEAN | NO | Primary gate metric |
| overall_score | NUMERIC(3,2) | NO | CHECK(BETWEEN 1.00 AND 5.00) |
| hallucination_flag | BOOLEAN | NO | Default FALSE |
| hallucination_detail | TEXT | YES | NULL unless hallucination_flag=TRUE |
| notes | TEXT | YES | Free-text reviewer comments |
| evaluated_at | TIMESTAMPTZ | NO | Default NOW() |

**Constraints:** `UNIQUE(model_output_id, evaluator_id)` — one row per output-evaluator pair; prevents double-scoring.  
**Indexes:** `BTREE(model_output_id)`, `BTREE(evaluation_round)`, `BTREE(task_type)`

> **Why not `sft_training_pairs` FK?** Training pairs are ETL artefacts — they exist only for workpapers that completed the full pipeline. The regression suite and production blind review evaluate `model_outputs` directly, which have no corresponding training pair. Linking eval_results to `sft_training_pairs` would make it impossible to store scores from those runs.

---

### Table 9 — model_outputs

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK — also used as Qdrant point ID when VALIDATED |
| workpaper_id | UUID | YES | FK → workpapers(id) SET NULL. NULL for regression and batch_eval runs (no real workpaper). |
| chunk_id | UUID | YES | FK → workpaper_chunks(id) SET NULL |
| inference_run_type | inference_run_type | NO | Default `'live'`. Filters logs by purpose — live production, regression suite, DPO collection, etc. |
| task_type | task_type | NO | |
| risk_level | risk_level | YES | |
| finding | TEXT | YES | |
| regulation_cited | JSONB | YES | Array of validated regulation ID strings |
| recommendation | TEXT | YES | |
| confidence_score | FLOAT | YES | 0.0–1.0 |
| model_version | VARCHAR(50) | NO | Soft-ref to model_versions.version_tag |
| output_status | output_status | NO | Default AI_DRAFT |
| reviewer_id | UUID | YES | FK → users(id) SET NULL. CHECK: VALIDATED requires NOT NULL. |
| rag_used | BOOLEAN | NO | Default FALSE |
| processing_time_ms | INTEGER | YES | |
| mongo_reasoning_id | VARCHAR(50) | YES | Pointer to MongoDB reasoning_chains._id |
| embedding_synced | BOOLEAN | NO | Default FALSE. TRUE after VALIDATED output pushed to Qdrant. Only eligible when inference_run_type = 'live'. |
| reviewed_at | TIMESTAMPTZ | YES | NULL until reviewed |
| created_at | TIMESTAMPTZ | NO | Default NOW() |

**Constraints:**
- `CHECK (output_status != 'VALIDATED' OR reviewer_id IS NOT NULL)`
- `CHECK (inference_run_type != 'live' OR workpaper_id IS NOT NULL)` — live inference always has a real workpaper

**Indexes:** `BTREE(workpaper_id)`, `BTREE(output_status)`, `BTREE(embedding_synced)`, `BTREE(inference_run_type, created_at)` — composite used by weekly arq monitoring task to filter live outputs for dashboards

---

### Table 10 — feedback_events

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| output_id | UUID | NO | FK → model_outputs(id) RESTRICT |
| reviewer_id | UUID | NO | FK → users(id) RESTRICT |
| reviewer_tier | reviewer_tier | NO | |
| rating | fb_rating | NO | |
| corrected_output | JSONB | YES | Full corrected output. Only when rating=corrected. |
| feedback_note | TEXT | YES | Optional free-text |
| model_version | VARCHAR(50) | YES | Adapter version of output being reviewed |
| created_at | TIMESTAMPTZ | NO | Default NOW() |

**Indexes:** `BTREE(output_id)`

---

### Table 11 — dpo_candidates

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| feedback_id | UUID | NO | FK → feedback_events(id) CASCADE DELETE. UNIQUE. |
| mongo_pair_id | VARCHAR(50) | NO | Pointer to MongoDB training_pairs_content._id |
| quality_score | FLOAT | YES | Pairs where chosen ≈ rejected excluded from runs |
| reviewer_tier | reviewer_tier | YES | Tier of reviewer who provided correction |
| used_in_run | BOOLEAN | NO | Default FALSE |
| training_run_id | UUID | YES | FK → training_runs(id) SET NULL |
| created_at | TIMESTAMPTZ | NO | Default NOW() |

**Indexes:** `UNIQUE(feedback_id)`, `BTREE(used_in_run)`

---

### Table 12 — training_runs

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| run_name | VARCHAR(200) | NO | e.g. "SFT Stage 1 — Single Audit" |
| run_type | run_type | NO | |
| base_model_version | VARCHAR(50) | NO | Starting adapter this run builds on |
| dataset_hash | VARCHAR(64) | NO | SHA-256 of training JSONL — reproducibility anchor |
| pair_count | INTEGER | NO | Pairs consumed in this run |
| epochs | INTEGER | NO | |
| f1_before | FLOAT | YES | NULL for first run |
| f1_after | FLOAT | YES | NULL until run completes |
| hallucination_before | FLOAT | YES | |
| hallucination_after | FLOAT | YES | NULL until complete |
| status | run_status | NO | Default 'queued' |
| signed_off_by | UUID | YES | FK → users(id) SET NULL. Partner approval. |
| signed_off_at | TIMESTAMPTZ | YES | |
| created_at | TIMESTAMPTZ | NO | Default NOW() |

**Note:** Immutable once status=completed. No updates permitted after completion.

---

### Table 13 — model_versions

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| version_tag | VARCHAR(50) | NO | UNIQUE. e.g. "v1.0-sft", "v2.1-dpo-cycle3" |
| adapter_path | VARCHAR(500) | YES | Filesystem path to QLoRA adapter checkpoint |
| base_model | VARCHAR(100) | YES | "mistralai/Mistral-Small-22B" |
| training_run_id | UUID | YES | FK → training_runs(id) RESTRICT |
| f1_score | FLOAT | YES | F1 on holdout test set |
| hallucination_rate | FLOAT | YES | % outputs with unverified regulation citations |
| format_compliance | FLOAT | YES | % outputs with all required fields populated |
| is_current | BOOLEAN | NO | PARTIAL UNIQUE INDEX where is_current=TRUE |
| deployed_at | TIMESTAMPTZ | YES | |
| created_at | TIMESTAMPTZ | NO | Default NOW() |

**Indexes:** `UNIQUE(version_tag)`, `PARTIAL UNIQUE INDEX ON model_versions(is_current) WHERE is_current=TRUE`

---

### Table 14 — audit_trail

| Column | Type | Null | Constraint |
|---|---|---|---|
| id | UUID | NO | PK |
| entity_type | VARCHAR(50) | NO | Table name e.g. "workpapers", "model_outputs" |
| entity_id | UUID | NO | PK of the affected row |
| event_type | VARCHAR(50) | NO | created / status_changed / reviewed / signed_off / pii_scrubbed / exported |
| actor_id | UUID | YES | FK → users(id) SET NULL. NULL for automated pipeline events. |
| metadata | JSONB | YES | e.g. `{old_status:"AI_DRAFT", new_status:"VALIDATED"}` |
| event_time | TIMESTAMPTZ | NO | Default NOW() |

**Constraints:** No UPDATE or DELETE ever permitted on this table (enforced via trigger + role grants).  
**Indexes:** `BTREE(event_time)`, `COMPOSITE BTREE(entity_type, entity_id)`

---

### FK Cascade / Restrict Reference

| Relationship | On Delete |
|---|---|
| firms → engagements | RESTRICT |
| firms → users | RESTRICT |
| users → engagements (created_by) | SET NULL |
| users → workpapers (upload_by) | SET NULL |
| users → model_outputs (reviewer_id) | SET NULL |
| users → feedback_events | RESTRICT |
| users → training_runs (signed_off_by) | SET NULL |
| users → audit_trail (actor_id) | SET NULL |
| engagements → workpapers | RESTRICT |
| engagements → sft_training_pairs | RESTRICT |
| workpapers → workpaper_permissions | CASCADE DELETE |
| workpapers → workpaper_chunks | CASCADE |
| engagements → workpaper_chunks (source_engagement_id) | SET NULL |
| workpapers → cleaning_log | CASCADE |
| workpapers → sft_training_pairs | RESTRICT |
| workpapers → model_outputs (workpaper_id) | SET NULL |
| workpaper_chunks → model_outputs (chunk_id) | SET NULL |
| model_outputs → eval_results | RESTRICT |
| model_outputs → feedback_events | RESTRICT |
| feedback_events → dpo_candidates | CASCADE |
| dpo_candidates → training_runs | SET NULL |
| training_runs → model_versions | RESTRICT |

---

## 3. MongoDB — 3 Collections (Full Specification)

### Collection 1 — raw_workpapers

Stores full extracted output exactly as produced. No cleaning applied. Client name "City of Springfield" lives here in raw form. **Never expose outside the pipeline layer.**

```json
{
  "_id": "ObjectId",
  "workpaper_id": "uuid-string",
  "engagement_id": "uuid-string",
  "source_format": "caseview | excel | word | pdf_text | pdf_scanned | dbf",
  "raw_content": {
    "full_text": "complete extracted text as single string",
    "pages": [
      {
        "page_number": 1,
        "text_blocks": ["string"],
        "tables": [
          {
            "table_index": 0,
            "headers": ["string"],
            "rows": [["string"]]
          }
        ],
        "annotations": [
          {
            "type": "tickmark",
            "symbol": "✓",
            "note": "string"
          }
        ]
      }
    ],
    "sheets": [
      {
        "sheet_name": "string",
        "rows": [{"col_name": "value"}]
      }
    ]
  },
  "extraction_meta": {
    "extracted_at": "ISODate",
    "tool_used": "pdfplumber | docling | pandas | dbfread",
    "word_count": 0,
    "page_count": 0,
    "sheet_count": 0,
    "ocr_confidence_avg": 0.0,
    "error_pages": [0]
  }
}
```

**Indexes:** `workpaper_id` (unique), `engagement_id`

---

### Collection 2 — training_pairs_content

Final cleaned prompt–completion pairs for both SFT and DPO. `pair_category` differentiates them.

```json
{
  "_id": "ObjectId",
  "pair_id": "uuid-string",
  "pair_category": "sft | dpo",
  "pair_type": "procedure_conclusion | finding_recommendation | risk_response | analytical_narrative",
  "prompt": {
    "text": "full instruction + workpaper content — complete model input",
    "context": {
      "client_type": "Govt",
      "is_gagas": true,
      "has_single_audit": true,
      "section": "Federal Program Compliance",
      "federal_program": "CDBG",
      "cfr_reference": "2 CFR 200.302"
    }
  },
  "completion": {
    "text": "model completion text (SFT) or chosen text (DPO)",
    "quality_indicators": {
      "has_cfr_citation": true,
      "has_sample_size": true,
      "has_population_size": true,
      "has_clear_conclusion": true,
      "opinion_language_present": false
    }
  },
  "rejected": "DPO only — original model response. NULL for SFT pairs.",
  "cleaning_applied": {
    "pii_scrubbed": true,
    "pii_types_removed": ["CLIENT_NAME", "EIN"],
    "replaced_with": "[CLIENT_1]",
    "version": "presidio-2.x"
  },
  "metadata": {
    "client_type": "Govt",
    "is_gagas": true,
    "has_single_audit": true,
    "fiscal_year": 2022,
    "framework": "Uniform Guidance",
    "split": "train",
    "created_at": "ISODate"
  }
}
```

**Indexes:** `pair_id` (unique), `metadata.split`, `pair_category`

---

### Collection 3 — reasoning_chains

Full chain-of-thought reasoning text. Only for `compliance_check` and `finding_documentation` task types.

```json
{
  "_id": "ObjectId",
  "output_id": "uuid-string",
  "reasoning_text": "full step-by-step reasoning — 2,000–8,000 tokens",
  "reasoning_steps": [
    {
      "step_number": 1,
      "type": "observation | inference | citation",
      "text": "string"
    }
  ],
  "model_version": "v1.2-dpo",
  "created_at": "ISODate"
}
```

**Indexes:** `output_id` (unique)

---

## 4. Qdrant — 2 Collections

### Collection 1 — workpaper_chunks_embeddings

```yaml
collection_name: workpaper_chunks_embeddings
vectors:
  size: 4096
  distance: Cosine
hnsw_config:
  m: 16
  ef_construct: 100
  full_scan_threshold: 10000

point_id: chunk UUID from workpaper_chunks.id

payload_schema:
  workpaper_id:   uuid string   # filter by parent workpaper
  engagement_id:  uuid string   # tenant isolation filter
  workpaper_type: string        # filter by workpaper type
  year:           integer       # temporal-weighted retrieval (recent = higher weight)
  chunk_index:    integer
  token_count:    integer
  chunk_mode:           string        # "semantic" | "numeric" — numeric chunks skipped in RAG retrieval (math tools consume them instead)
  is_rollforward:       bool          # TRUE = rolled-forward from prior engagement; excluded from RAG retrieval via must_not filter
  source_engagement_id: uuid string   # engagement this chunk was rolled forward from; null for originals; stored for lineage queries
  drift_weight:         float         # ETL TemporalTagger output: combined year-decay × time_sensitivity multiplier (0.05–1.0); used by retrieve_rag_context() to boost/suppress scores instead of recomputing at query time

sync_trigger: SET workpaper_chunks.embedding_synced = TRUE after push
```

### Collection 2 — validated_findings_embeddings

```yaml
collection_name: validated_findings_embeddings
vectors:
  size: 4096
  distance: Cosine
hnsw_config:
  m: 16
  ef_construct: 100
  full_scan_threshold: 10000

point_id: output UUID from model_outputs.id
embed_field: model_outputs.finding (text field)

payload_schema:
  task_type:       string        # filter by task type
  risk_level:      string        # filter by risk level
  engagement_id:   uuid string   # tenant isolation
  regulation_cited: string[]     # filter by citation
  model_version:   string

sync_trigger: SET model_outputs.embedding_synced = TRUE after push
             Triggered when output_status transitions to VALIDATED
```

---

## 5. Module Interface Contracts

### BaseExtractor

```python
# etl/core/extractors/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from etl.core.record import RawContent

class BaseExtractor(ABC):

    @abstractmethod
    def can_handle(self, file_extension: str, magic_bytes: bytes) -> bool:
        """Return True if this extractor handles the given file."""

    @abstractmethod
    def extract(self, file_path: Path, workpaper_id: str) -> RawContent:
        """Extract raw content from file. Returns RawContent. Raises ExtractionError on failure."""

    @abstractmethod
    def get_supported_extensions(self) -> list[str]:
        """Return list of file extensions this extractor handles e.g. ['.xlsx', '.xls']"""

    def get_fallback(self) -> "BaseExtractor | None":
        """Return fallback extractor to try if this one fails. None = no fallback."""
        return None
```

### BaseTransformer

```python
# etl/core/transformers/base.py
from abc import ABC, abstractmethod
from etl.core.record import Record

class BaseTransformer(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique transformer name for cleaning_log entries."""

    @abstractmethod
    def transform(self, record: Record) -> Record:
        """Apply transformation. Returns modified Record.
        Must append CleaningAction entries to record.cleaning_actions."""
```

### BaseLoader

```python
# etl/core/loaders/base.py
from abc import ABC, abstractmethod
from etl.core.record import Record

class BaseLoader(ABC):

    @abstractmethod
    async def load(self, record: Record) -> str:
        """Persist record. Returns ID of created resource."""

    @abstractmethod
    async def bulk_load(self, records: list[Record]) -> list[str]:
        """Persist multiple records. Returns list of IDs."""
```

### Record Schema

```python
# etl/core/record.py
from pydantic import BaseModel
from typing import Any

class RawContent(BaseModel):
    full_text: str
    pages: list[dict] = []
    sheets: list[dict] = []
    tables: list[dict] = []
    annotations: list[dict] = []
    extraction_meta: dict = {}

class Chunk(BaseModel):
    chunk_index: int
    sheet_name: str | None
    token_count: int
    content_json: dict           # {text, tables[], cells[], page_refs[]}

class CleaningAction(BaseModel):
    action: str                  # matches cleaning_action ENUM
    pii_types_found: list[str] = []
    is_duplicate_of: str | None = None
    similarity_score: float | None = None
    detail: dict = {}

class Record(BaseModel):
    workpaper_id: str            # UUID from PG
    engagement_id: str
    source_format: str
    raw_content: RawContent
    cleaned_text: str | None = None
    chunks: list[Chunk] | None = None
    cleaning_actions: list[CleaningAction] = []
    metadata: dict = {}
```

---

## 6. API Endpoint Inventory

Base path: `/api/v1`  
Auth: All endpoints except `/health` and `/auth/login` require valid JWT.  
Rate limit: 60 req/min per user per endpoint (Redis sliding window).  
RBAC: Engagement-level endpoints check `role` claim from JWT. Workpaper-level endpoints additionally check `workpaper_permissions` table via `require_wp_permission()` FastAPI dependency. HTTP 403 returned when permission missing — no information leakage (same 403 for "not found" and "forbidden" on workpaper reads).

### Auth

| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/auth/login` | Public | Email + password → JWT |
| POST | `/auth/logout` | Any | Revoke JWT (Redis) |

### Engagements

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/engagements` | Any | List engagements (paginated, filter by status/type) |
| POST | `/engagements` | manager, admin | Create engagement |
| GET | `/engagements/{id}` | Any | Get engagement detail |
| PATCH | `/engagements/{id}` | manager, admin | Update phase/status/export defaults |

### Workpapers

| Method | Path | Min Permission | Description |
|---|---|---|---|
| POST | `/engagements/{eng_id}/workpapers` | `write` (seeded on create) | Upload workpaper file → queue ETL job; seeds default `workpaper_permissions` rows |
| GET | `/workpapers/{id}` | `read` | Get workpaper + extraction_status. HTTP 403 if no matching permission row. |
| GET | `/workpapers/{id}/chunks` | `read` | List chunks for workpaper |
| POST | `/workpapers/{id}/analyze` | `write` | Trigger inference → queue arq job |
| PATCH | `/workpapers/{id}/password` | `write` | Supply decryption password for a `needs_password` workpaper. Password transmitted over TLS, used in-memory for decryption only, **never persisted**. Re-queues ETL. Returns HTTP 400 if password is wrong. |
| GET | `/workpapers/{id}/permissions` | `review` | List current permission grants for this workpaper |
| POST | `/workpapers/{id}/permissions` | `approve` (manager+) | Grant or update a permission for a user or role |
| DELETE | `/workpapers/{id}/permissions/{perm_id}` | `approve` | Revoke a specific permission grant |

### Outputs

| Method | Path | Min Permission | Description |
|---|---|---|---|
| GET | `/workpapers/{id}/outputs` | `read` | List all outputs for workpaper |
| GET | `/outputs/{id}` | `read` | Get output detail (AI_DRAFT label enforced in response) |
| PATCH | `/outputs/{id}` | `review` | Set status VALIDATED or REJECTED (reviewer_id from JWT) |
| GET | `/outputs/{id}/export` | `read` | Download output. `?format=word\|excel\|pdf\|caseware` overrides engagement default. `caseware` returns a `.cwx` XML file for manual import into Caseware Working Papers. Only VALIDATED outputs. |

### Feedback

| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/outputs/{id}/feedback` | Any | Submit rating + optional correction |
| GET | `/outputs/{id}/feedback` | manager, admin | List feedback for output |

### Training (admin / manager only)

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/training/pairs` | manager, admin | List sft_training_pairs (filter by split/approved) |
| PATCH | `/training/pairs/{id}` | senior, manager, partner | Approve or reject pair |
| POST | `/training/export` | admin | Trigger JSONL export DAG |
| POST | `/training/runs` | admin | Trigger SFT or DPO training DAG |
| GET | `/training/runs` | manager, admin | List training runs |
| GET | `/training/runs/{id}` | manager, admin | Get run status + metrics |

### Model Versions

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/model/versions` | manager, admin | List all versions + metrics |
| PATCH | `/model/versions/{id}/activate` | admin | Set is_current=TRUE (flips old current to FALSE) |

### Admin

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/users` | admin | List users |
| POST | `/users` | admin | Create user |
| PATCH | `/users/{id}` | admin | Update role/tier/is_active |

### Health

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/health` | Public | Returns DB + queue + model status |

---

## 7. Export Renderer Design

Triggered by `GET /outputs/{id}/export`. Only `VALIDATED` outputs can be exported.

### Format Resolution Order

```
1. ?format= query param  (per-download override)
        │
        ▼ (if not provided)
2. engagements.default_export_format
        │
        ▼ (if NULL)
3. Hardcoded default: word for all engagements
   (trial_balance workpaper_type always defaults to excel regardless)
```

### Template Mapping

| client_type | has_single_audit | workpaper_type | format | template |
|---|---|---|---|---|
| Govt | true | management_letter | word | management_letter_word_v1.docx |
| Any | Any | trial_balance | excel | trial_balance_excel_v1.xlsx |
| Any | Any | Any | word | generic_word_v1.docx |
| Any | Any | Any | excel | generic_excel_v1.xlsx |
| Any | Any | Any | pdf | Render word first → LibreOffice headless → PDF |

### Renderer Flow

```
1. Load model_outputs row — verify output_status = VALIDATED
2. Load engagement row — get default_export_format + default_export_template
3. Resolve format (query param → engagement default → hardcoded default)
4. Load Pydantic output schema for task_type (parse from finding/regulation_cited/etc.)
5. Load template file from /templates/
6. Fill template fields with schema values
   Word:  python-docx paragraph/table manipulation
   Excel: openpyxl cell writes
   PDF:   render to Word first → LibreOffice headless convert → PDF bytes
7. Write audit_trail event_type="exported"
8. Stream file as response (Content-Disposition: attachment)
```

---

## 8. Database Index Summary

| Table | Column(s) | Type | Reason |
|---|---|---|---|
| firms | slug | UNIQUE | URL routing |
| users | email | UNIQUE | Login lookup |
| users | firm_id | BTREE | Tenant queries |
| engagements | firm_id | BTREE | Engagement list queries |
| engagements | engagement_code | UNIQUE | Pipeline dedup |
| workpapers | engagement_id | BTREE | Workpaper list |
| workpapers | file_hash | UNIQUE | Duplicate prevention |
| workpapers | extraction_status | BTREE | Pipeline queue polls |
| workpaper_chunks | workpaper_id | BTREE | Chunk fetch + CASCADE |
| workpaper_chunks | embedding_synced | BTREE | Qdrant sync worker polls FALSE rows |
| cleaning_log | workpaper_id | BTREE | Lineage queries |
| sft_training_pairs | workpaper_id | BTREE | Pair list |
| sft_training_pairs | engagement_id | BTREE | Split-level queries |
| sft_training_pairs | split_assignment | BTREE | JSONL export filter |
| sft_training_pairs | reviewer_approved | BTREE | Export gate filter |
| sft_training_pairs | (workpaper_id, pair_type, pair_hash) | UNIQUE | Idempotency — re-submission dedup |
| eval_results | model_output_id | BTREE | Eval lookup per output |
| eval_results | evaluation_round | BTREE | Gate query filter |
| eval_results | task_type | BTREE | Per-task metric aggregation |
| model_outputs | workpaper_id | BTREE | Output list per workpaper |
| model_outputs | output_status | BTREE | Reviewer dashboard |
| model_outputs | embedding_synced | BTREE | Qdrant sync worker |
| model_outputs | (inference_run_type, created_at) | COMPOSITE BTREE | Weekly arq monitoring — filters live outputs for dashboard metrics |
| feedback_events | output_id | BTREE | Feedback lookup |
| dpo_candidates | used_in_run | BTREE | Training run assembly |
| dpo_candidates | feedback_id | UNIQUE | One DPO pair per feedback |
| model_versions | is_current=TRUE | PARTIAL UNIQUE | One active version |
| model_versions | version_tag | UNIQUE | Version lookup |
| audit_trail | event_time | BTREE | Time-range compliance queries |
| audit_trail | (entity_type, entity_id) | COMPOSITE BTREE | Per-entity history lookup |

---

*Next document: [04_etl_pipeline.md](04_etl_pipeline.md) — ETL & Pipeline Design*
