# System Architecture — Audit AI
**Version:** 1.0  
**Date:** April 2026  
**Status:** FINAL  
**Firm:** Harshwal & Company LLP (HCLLP)  
**Checklist items closed:** #12 (Cost & Tech Stack)

---

## 1. System Purpose

Audit AI is a fully on-premise AI assistant that reads audit workpapers, classifies risk, checks regulatory compliance, drafts findings, and improves continuously from auditor corrections. It runs on firm-owned hardware inside an air-gapped network. No workpaper data, model weights, or inference traffic ever leaves the building.

**Training strategy:** Staged hybrid — Stage 1 mass SFT trains on all 15 years of data across all framework types (Single Audit, GAGAS, GAAS) to build a shared audit domain foundation. Stage 2 batch SFT runs separate per-framework tracks (continuing from the Stage 1 checkpoint) to sharpen framework-specific judgment. Stage 3 targeted refinement fixes systematic errors identified in blind review. HCLLP serves US-only engagements (NPO and Govt clients); ICAI-SA is permanently out of scope.

**Inference scope:** Phase 1 deployment serves Single Audit engagements (Uniform Guidance / 2 CFR 200) only. Expansion to GAGAS and GAAS in the live product is gated behind an 80% senior auditor sign-off threshold on eval_results. This is an **inference gate**, not a training gate — all US framework data is used in training from day one.

---

## 2. Infrastructure Topology

### 2.1 Physical Servers

| | Server 1 — Training | Server 2 — Inference |
|---|---|---|
| **Role** | ETL, training jobs, embedding generation | Live inference, app backend, all databases |
| **GPU** | 1× NVIDIA L40S 48GB VRAM | 1× NVIDIA L40S 48GB VRAM |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| **CUDA** | 12.2 | 12.2 |
| **Availability** | On-demand (jobs triggered via arq/Airflow) | Always-on |
| **Primary workloads** | Airflow DAGs, ETL workers, Stage 1 mass SFT (all frameworks, 18-20 hrs), Stage 2 batch SFT per framework track (8-12 hrs/track), DPO cycles, e5-mistral-7b-instruct embedding at training time | vLLM AsyncLLMEngine, FastAPI API, bge-reranker-v2-m3, e5-mistral-7b-instruct (live inference embedding), PostgreSQL 16, MongoDB 7, Qdrant, Redis |

### 2.2 GPU Memory Budget — Server 2

| Process | VRAM |
|---|---|
| Mistral 22B v24.09 (4-bit NF4 quantized) | ~14 GB |
| e5-mistral-7b-instruct (live embedding) | ~14 GB |
| bge-reranker-v2-m3 (cross-encoder reranker) | ~2 GB |
| vLLM engine overhead + KV cache | ~8 GB |
| **Total** | **~38 GB / 48 GB** |

Headroom: ~10GB. Sufficient for concurrent inference under normal load.

### 2.3 Network

```
[Auditor Workstations]
        │  HTTPS / TLS 1.3 (internal only)
        ▼
[Firm Internal Network]
        │  VLAN isolation
        ▼
[Air-Gapped VLAN]
  ┌─────┴──────┐
  │ Server 1   │   Server 2
  │ (Training) │   (Inference / App)
  └────────────┘
        │
   No internet route
   Firewall blocks all outbound from model endpoints
```

- All inter-server traffic: TLS 1.3 encrypted
- No external API calls permitted from either server
- Firewall rules enforced at VLAN boundary
- No cloud vendor keys, no external model APIs

---

## 3. Four-Store Architecture

### 3.1 The Golden Rule

> PostgreSQL owns all state and relationships.  
> MongoDB owns all variable-length content.  
> Qdrant owns all vectors.  
> Redis owns all ephemeral operations.  
> **Nothing is duplicated. UUID is the universal bridge.**

Every MongoDB document and every Qdrant point carries the UUID of its PostgreSQL counterpart. Postgres rows carry `mongo_*_id` and `qdrant_synced` flags pointing outward. You never query MongoDB or Qdrant standalone — Postgres drives all status and routing decisions.

### 3.2 Store Responsibilities

| Store | Owns | Never Stores |
|---|---|---|
| **PostgreSQL 16** | All metadata, FKs, status flags, counters, indexes, audit trail | Raw text, prompt/completion content, vectors, job queues |
| **MongoDB 7** | raw_workpapers, training_pairs_content, reasoning_chains | Anything with FK relationships or status flags |
| **Qdrant** | workpaper_chunks_embeddings, validated_findings_embeddings | SFT pairs, DPO candidates, raw text |
| **Redis** | arq job queues (ETL/inference/training), response cache, rate limits, JWT revocation | Persistent data of any kind |

### 3.3 PostgreSQL — 14 Tables (overview)

Full column specs in [03_lld.md](03_lld.md).

| # | Table | Layer | Purpose |
|---|---|---|---|
| 1 | firms | Tenant | Multi-firm container. Single row for HCLLP. |
| 2 | users | Auth | All human actors. JWT Phase 1, M365 SSO Phase 2. |
| 3 | engagements | Core | Top-level audit engagement. Shared by pipeline and app. |
| 4 | workpapers | Core | One row per file. SHA-256 dedup. |
| 5 | workpaper_chunks | AI/RAG | Token-bounded sections. chunk UUID = Qdrant point ID. |
| 6 | cleaning_log | ETL Lineage | Full row per cleaning action. Max traceability. |
| 7 | sft_training_pairs | Training | SFT pair registry. 70/15/15 engagement-level split. |
| 8 | eval_results | QA Gate | Blind review scores. Gates 80% sign-off threshold. |
| 9 | model_outputs | AI | Every inference result. Always AI_DRAFT on insert. |
| 10 | feedback_events | RLHF | Post-inference auditor corrections. Source for DPO. |
| 11 | dpo_candidates | Training | DPO preference pairs assembled from feedback. |
| 12 | training_runs | Training | One row per fine-tuning run (SFT and DPO). |
| 13 | model_versions | Registry | Adapter version registry. One is_current at a time. |
| 14 | audit_trail | Compliance | Append-only immutable log. Every state change. |

### 3.4 MongoDB — 3 Collections

| Collection | Purpose |
|---|---|
| raw_workpapers | Full extracted output before any cleaning. Client names live here in raw form. Never exposed outside pipeline layer. |
| training_pairs_content | Final cleaned prompt–completion pairs for both SFT and DPO (differentiated by pair_category field). |
| reasoning_chains | Full chain-of-thought reasoning text from model inference. Only for task_types that use CoT. |

### 3.5 Qdrant — 2 Collections

| Collection | Vector | Purpose |
|---|---|---|
| workpaper_chunks_embeddings | 4096-dim, HNSW, cosine | RAG context retrieval at inference time |
| validated_findings_embeddings | 4096-dim, HNSW, cosine | Consistency check — surfaces similar past findings |

Embedding model: **e5-mistral-7b-instruct** (runs on L40S, ~14GB VRAM).  
Point ID = chunk UUID from PostgreSQL (no separate mapping table).

### 3.6 Redis Key Design (summary)

| Key Pattern | TTL | Purpose |
|---|---|---|
| `arq:queue:etl` | None | ETL jobs |
| `arq:queue:inference` | None | Inference jobs |
| `arq:queue:training` | None | Training triggers |
| `cache:output:{output_id}` | 1h | Cached API response, invalidated on review |
| `cache:engagement:{id}:summary` | 15m | Engagement summary stats |
| `cache:qdrant:chunks:{workpaper_id}` | 30m | Cached RAG retrieval results |
| `ratelimit:{user_id}:{endpoint}` | 1m | Sliding window rate limit |
| `session:{jti}:revoked` | 24h | JWT revocation on logout |

---

## 4. Security Architecture

### L1 — Network Isolation

- Air-gapped VLAN; firewall blocks all outbound connections from model endpoints
- No external API calls permitted (no OpenAI, no AWS, no cloud vendors)
- TLS 1.3 enforced for all internal traffic between servers and workstations
- Model weights stored on Server 1 filesystem — no network-accessible model registry

### L2 — Authentication & Access Control

| Phase | Method |
|---|---|
| Phase 1 (launch) | JWT + bcrypt. `hashed_password` column populated. `azure_oid` NULL. |
| Phase 2 (additive migration) | M365 Azure AD SSO. Add `azure_oid` column, make `hashed_password` nullable. No destructive migration. |

**Roles:** admin / manager / senior / qualified / junior  
**Tiers:** partner / manager / senior / staff — controls DPO candidacy eligibility  
**Session timeout:** 15 minutes  
**MFA:** Enforced  
**RBAC enforcement:** API gateway layer — not optional per-endpoint logic

### L3 — Data Encryption

| At rest | AES-256 for all workpaper data on both servers |
|---|---|
| In transit | TLS 1.3 for all internal network traffic |
| Key management | Firm HSM — no cloud vendor key custody |

### L4 — Audit Trail

- Every state change across all 13 tables written to `audit_trail` (entity_type, entity_id, event_type, actor_id, metadata, event_time)
- Append-only — no UPDATE or DELETE ever permitted on `audit_trail`
- Full SIEM integration for security monitoring
- Every model query logged with user ID + timestamp
- Query hash stored for forensic recovery

### L5 — Data Residency

- Model weights: on-premise filesystem only
- Training data: never sent to any cloud service
- Inference: all processing inside firm data center
- Client workpaper content: never leaves the building
- All AI outputs marked **"AI-Assisted Draft"** — mandatory human review before workpaper entry

### AI Output Policy

- All `model_outputs` inserted with `output_status = AI_DRAFT` — enforced at DB level (default constraint)
- Transition to `VALIDATED` requires `reviewer_id NOT NULL` — enforced by CHECK constraint
- AI cannot auto-populate final workpapers
- Outputs surfaced to auditors always carry the label: *"AI-Assisted Draft — Requires Senior Auditor Review Before Use"*

---

## 5. Full Tech Stack

### 5.1 Runtime & Framework

| Component | Tool | Version |
|---|---|---|
| Language | Python | 3.11 |
| API framework | FastAPI | latest stable |
| Async ORM | SQLAlchemy | 2.x (async) |
| Migrations | Alembic | latest stable |
| Config | Pydantic Settings | v2 |
| Auth (Phase 1) | python-jose + passlib[bcrypt] | latest stable |
| Task queue | arq | latest stable |
| Pipeline orchestration | Apache Airflow | 2.x |

### 5.2 Databases

| Store | Tool | Version |
|---|---|---|
| Relational | PostgreSQL | 16 |
| Document | MongoDB | 7 |
| Vector | Qdrant | latest stable |
| Cache / Queue | Redis | 7.x |

### 5.3 AI / ML

| Component | Tool | Notes |
|---|---|---|
| Base model | Mistral 22B v24.09 | QLoRA fine-tuned |
| Quantization | 4-bit NF4 | ~14GB on L40S |
| Fine-tuning | HuggingFace PEFT + SFTTrainer | LoRA r=16, α=32, target q_proj + v_proj |
| DPO training | custom dpo.py (TRL library) | Quarterly re-training cycles |
| Inference engine | vLLM AsyncLLMEngine | Server 2, always-on |
| Embedding model | e5-mistral-7b-instruct | 4096-dim, ~14GB on L40S |
| Reranker | bge-reranker-v2-m3 | Cross-encoder, ~2GB |
| Agent framework | LangGraph | StateGraph + Tool Dispatch node + workpaper-type agents |
| Audit math library | model/tools/ (custom Python) | thresholds, materiality, bank_rec, trial_balance, ratios, variance — orchestrated by LangGraph, results injected as facts before generation |

### 5.4 ETL & Extraction

| Format | Primary Library | Version | Fallback |
|---|---|---|---|
| Excel .xlsx | pandas + openpyxl (data_only=True) | 2.1+ / 3.1+ | — |
| Excel .xls | xlrd | 2.0+ | LibreOffice → pandas |
| Excel .xlsm | xlwings | 0.30+ | — |
| Word .docx | python-docx | 1.1+ | — |
| Word .doc | LibreOffice headless | 7.5+ | — |
| PDF text | pdfplumber | 0.10+ | pdfminer.six (20221105) |
| PDF scanned | Docling + Surya | latest stable | Tesseract 5.3+ |
| DBF / FoxPro | dbfread | 2.0+ | simpledbf |
| CGF / BAK | chardet + configparser | latest | xmltodict |
| CDX, CVW, STY, MDX, NTX | **SKIP** — no data | — | — |

### 5.5 Transformation Pipeline (chain order is fixed)

| Step | Tool |
|---|---|
| 1. Normalizer | Custom + babel + dateutil |
| 2. PIIScrubber | Presidio (analyzer + anonymizer) + custom US/India regex patterns |
| 3. TemporalTagger | Custom + outdated_regulations.json |
| 4. Chunker | Token-aware, sentence/row/paragraph boundary-respecting |
| 5. InjectionSanitizer | Custom regex |

### 5.6 Monitoring & Observability

| Purpose | Tool |
|---|---|
| Training experiment tracking | MLflow (self-hosted) |
| Data versioning | DVC |
| Inference tracing (input, output, latency, feedback) | Langfuse (self-hosted) |
| Infrastructure monitoring (GPU utilization, response times) | Prometheus + Grafana |
| Model registry | HuggingFace Hub (private, local instance) |

---

## 6. Deployment Topology — Process Map

### Server 1 (Training — on-demand)

```
┌─────────────────────────────────────────────────┐
│  Apache Airflow 2.x                             │
│    DAG: intake_watcher (polls /intake/ every 5m)│
│    DAG: etl_pipeline (extract → transform → load)│
│    DAG: jsonl_export (train/val/test split)      │
│    DAG: training_trigger (SFT + DPO runs)        │
│                                                  │
│  ETL Workers (arq)                               │
│    extract → chunk → clean → embed → sync        │
│                                                  │
│  e5-mistral-7b-instruct                          │
│    (embedding generation for training pairs)     │
│                                                  │
│  QLoRA Training Jobs (PEFT + SFTTrainer)         │
│    18-20 hr runs, checkpoints every 500 steps    │
│    MLflow logging, DVC dataset versioning        │
│                                                  │
│  /intake/  ← Caseware manual exports land here  │
│  /models/  ← Adapter checkpoints stored here    │
└─────────────────────────────────────────────────┘
```

### Server 2 (Inference — always on)

```
┌─────────────────────────────────────────────────┐
│  FastAPI (async)                                 │
│    /api/v1/workpapers  (upload, chunk, analyze)  │
│    /api/v1/outputs     (review, validate)        │
│    /api/v1/feedback    (corrections, DPO)        │
│    /api/v1/engagements (CRUD)                    │
│                                                  │
│  vLLM AsyncLLMEngine                             │
│    Mistral 22B + active LoRA adapter             │
│    bge-reranker-v2-m3 (cross-encoder)            │
│    e5-mistral-7b-instruct (live embedding)       │
│                                                  │
│  LangGraph StateGraph                            │
│    Router → workpaper-type agents                │
│                                                  │
│  PostgreSQL 16  │  MongoDB 7                     │
│  Qdrant         │  Redis                         │
│                                                  │
│  Langfuse (self-hosted) — inference tracing      │
│  Prometheus + Grafana — GPU + latency monitoring │
└─────────────────────────────────────────────────┘
```

---

## 7. Cost Model

| Item | Cost | Notes |
|---|---|---|
| Hardware (2× L40S servers) | Existing firm asset | No additional hardware cost |
| One-time SFT fine-tuning run | ~$230K (internal estimate) | 18-20 hr GPU job, staff time for data prep and review |
| Ongoing DPO cycles (quarterly) | Incremental | Staff time for feedback collection + ~4-6 hr GPU job per cycle |
| Cloud costs | $0 | Zero — all inference and training on-prem |
| AWS Textract | $0 | Dropped — Docling + Surya on-prem replaces it |
| Software licenses | $0 | All open-source stack |

---

## 8. Key Architectural Decisions & Rationale

| Decision | Rationale |
|---|---|
| QLoRA over full fine-tune | 50MB adapter file per run instead of re-saving 44GB model. Fast to deploy, easy to roll back. |
| 4-bit NF4 quantization | Reduces Mistral 22B from 44GB → ~14GB without measurable quality loss for audit tasks. Fits on L40S with room for embedding + reranker. |
| On-premise only, zero cloud | Client workpapers are CUI (Controlled Unclassified Information). Air-gap eliminates data residency risk. |
| Four-store split (PG + Mongo + Qdrant + Redis) | Single responsibility per store. PostgreSQL enforces relational integrity. MongoDB handles variable-length content without schema friction. Qdrant purpose-built for ANN search. Redis purpose-built for ephemeral ops. |
| RAG over pure fine-tuning for regulation citations | Fine-tuning alone hallucinate regulation numbers. RAG grounds citations in the actual eCFR / AICPA / GAO standards corpus. Hallucination target: ≤ 2%. |
| bge-reranker-v2-m3 cross-encoder | Better relevance precision than score threshold alone. ~2GB footprint fits in L40S budget. Adds ~150ms latency — still within <2s p95 target. |
| LangGraph over simple prompt chaining | Workpaper-type routing is stateful. Graph structure makes agent flow inspectable, testable, and extensible per workpaper type. |
| Engagement-level train/val/test split | Prevents data leakage. Workpapers from the same engagement share terminology, entities, and style — splitting within an engagement inflates eval scores. |
| Airflow over Linux cron | ETL pipeline has DAG dependencies (extract must complete before chunk, chunk before embed). Airflow provides dependency management, retries, and observability that cron cannot. |
| Mass + Batch SFT over single-framework sequential training | Stage 1 mass SFT on all framework data prevents catastrophic forgetting when expanding frameworks later. Stage 2 batch SFT per framework sharpens judgment without re-learning shared audit vocabulary. The 80% would_sign_off gate controls inference expansion only — training uses all available data from the start. |
| Deterministic math tools over LLM arithmetic | LLMs make arithmetic errors on financial figures — an unreconciled difference of $4,668 reported as $4,670 is an audit error. The `model/tools/` library computes all numeric results in Python (deterministic, verifiable, zero latency overhead when run in parallel with RAG). The model reads computed facts and writes findings; it never calculates. |
| LangGraph Tool Dispatch over LLM-driven tool calling | LangGraph determines which math tools to run based on `workpaper_type` (already known from routing) — not by asking the model to generate tool-call JSON. This avoids: (1) latency from additional generation round trips, (2) the need to fine-tune the model for reliable tool-call syntax, (3) unpredictable tool selection at inference time. |

---

*Next document: [02_hld.md](02_hld.md) — High-Level Design*
