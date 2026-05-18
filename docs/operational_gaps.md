# Audit AI — Operational Gaps (O-01 to O-08)
**Status:** Pending decision / documentation  
**Date:** April 2026  
**Owner:** To be assigned per gap below

These 8 gaps are not code or schema defects — they are operational concerns that require a decision from the project team before they can be documented and closed. Each entry states what is missing, what questions need an answer, and who should own the decision.

---

## O-01 — No Runbook

**What is missing:**  
There is no operational runbook for the system. When something breaks in production — a failed inference request, a stuck Airflow DAG, a Qdrant sync backlog, a vLLM crash — the on-call person has no documented procedure to follow. Every recovery action is currently improvised.

**What needs to be decided / written:**

| Section | Contents needed |
|---------|----------------|
| Service startup / shutdown | Correct order for bringing up PostgreSQL → MongoDB → Qdrant → Redis → vLLM → FastAPI; reverse order for shutdown; health check commands per service |
| Common failure modes | Airflow DAG stuck (how to identify, how to clear); vLLM OOM (restart command, VRAM reclaim); Qdrant sync backlog (how to trigger manual re-sync); Redis eviction under load |
| Escalation path | Who gets called first (ML Eng vs Sysadmin vs Partner); at what severity; response time targets |
| Monitoring access | Grafana dashboard URL; Prometheus alert definitions; Langfuse trace URL; MLflow experiment URL |
| Rollback procedure | How to revert `model_versions.is_current` to prior version; how to drain in-flight requests before switching |

**Owner:** Systems Administrator + ML Engineer  
**Blocker:** None — can be written once the system is deployed to staging.

---

## O-02 — No Backup / DR Strategy

**What is missing:**  
There is no documented backup or disaster recovery plan for any of the four data stores (PostgreSQL, MongoDB, Qdrant, Redis) or for the Server 1 / Server 2 filesystems. A disk failure or accidental `DROP TABLE` currently means unrecoverable data loss.

**What needs to be decided:**

| Decision | Options to consider |
|----------|-------------------|
| Backup frequency | PostgreSQL: WAL-G continuous + daily full? Weekly full + daily incremental? |
| Backup destination | Local NAS within the VLAN? Encrypted offsite? Air-gap constraint affects this. |
| MongoDB backup | `mongodump` scheduled job? Replica set with oplog? |
| Qdrant backup | Qdrant snapshot API — how often, where stored? |
| Redis | Redis is ephemeral (no persistence) per current design — confirm this is acceptable or change |
| Recovery time objective (RTO) | How many hours of downtime is acceptable before auditors are materially impacted? |
| Recovery point objective (RPO) | How much data loss is acceptable — hours, days? |
| DR test cadence | Quarterly restore test? Semi-annual? Who runs it? |

**Owner:** Systems Administrator  
**Blocker:** Decision on backup destination — constrained by air-gap VLAN design (offsite backup may require a separate policy decision from IT Security Officer).

---

## O-03 — No Data Retention Policy

**What is missing:**  
There is no policy stating how long any category of data is kept before it is deleted. This affects regulatory compliance (AICPA record retention requirements for audit workpapers are typically 5–7 years), storage planning, and GDPR/data minimisation obligations for any PII that slips through the scrubber.

**What needs to be decided:**

| Data category | Questions to resolve |
|--------------|---------------------|
| Raw workpaper files (`/intake/`) | Deleted after ETL completes, or retained? If retained, for how long? |
| `workpaper_chunks` (PII-scrubbed text in PG + Qdrant) | 5 years? 7 years? Permanent for as long as the engagement exists? |
| `model_outputs` (AI drafts, `AI_DRAFT` status) | Retained indefinitely? Purged after auditor signs off? |
| `sft_training_pairs` + JSONL exports | JSONL is deleted post-training (documented). PG training pair rows — kept or purged after model is promoted? |
| `audit_trail` table | Append-only by design — retention period? Archival strategy? |
| `dpo_feedback` corrections | Retained permanently as training signal? |
| MongoDB `reasoning_chains` | Linked to model outputs — purge when output is purged? |
| `eval_results` scores | Permanent record for model governance, or purge after N DPO cycles? |

**Owner:** Engagement Partner + IT Security Officer  
**Blocker:** Legal / AICPA guidance review; any India-jurisdiction data (ICAI-SA engagements) may have different retention requirements under Indian IT Act.

---

## O-04 — No Model Weight Backup

**What is missing:**  
The Mistral 22B base model weights and all QLoRA adapter checkpoints live on Server 2's local filesystem. There is no backup. A disk failure on Server 2 means:  
- Re-downloading the base model (~44GB) from HuggingFace — not possible from an air-gapped VLAN without a deliberate transfer  
- Losing all trained adapter versions with no recovery path

**What needs to be decided:**

| Decision | Detail |
|----------|--------|
| Base model backup location | NAS within VLAN? Encrypted external drive stored in server room? |
| Adapter backup cadence | After every successful SFT/DPO run (i.e., every ~3–4 months)? |
| Backup format | Full checkpoint directory? Compressed archive? DVC-tracked? |
| Integrity verification | Checksum validation after backup copy completes |
| Transfer procedure for base model | If VLAN is truly air-gapped, initial model load and future updates require a documented physical media transfer procedure (USB drive scanned and logged per security policy) |

**Owner:** Systems Administrator + ML Engineer  
**Blocker:** IT Security Officer must approve the physical media transfer procedure before initial base model load can be documented.

---

## O-05 — No Audit Program Integration Scope Decision

**What is missing:**  
The architecture includes a `CasewireExporter` for downloading `.cwx` files and a Phase 2 note about pushing directly to Caseware Cloud via REST API. However, the **reverse direction** — pulling workpapers and engagement structure *from* Caseware into the intake pipeline automatically — has no documented scope decision.

Currently the audit team manually exports from Caseware and drops files into `/intake/`. Whether this remains manual forever or is replaced by a direct integration is unresolved.

**What needs to be decided:**

| Question | Options |
|----------|---------|
| Intake direction: Caseware → Audit AI | Remain manual (export + drop) forever; OR build a Caseware API pull in Phase 2; OR use a shared network drive watched by both systems |
| Scope of Phase 2 Caseware integration | Read-only pull (workpapers in); read-write (workpapers in + findings pushed back); full two-way sync |
| Which Caseware version is in use | Caseware Working Papers desktop vs. Caseware Cloud — the API surface is completely different |
| Impact on engagement_labels.csv | If Caseware holds workpaper metadata, can `engagement_labels.csv` be auto-generated from Caseware's workpaper index rather than manually prepared? |
| Timeline | Does Phase 2 integration block the ICAI-SA rollout, or are they independent? |

**Owner:** Engagement Partner + IT/Systems team  
**Blocker:** Requires knowing which version of Caseware HCLLP runs and whether the Caseware Cloud API is accessible from within the VLAN.

---

## O-06 — No Sampling Module

**What is missing:**  
Statistical sampling is a core audit procedure for both Single Audit (2 CFR 200 requires sampling of transactions) and GAAS engagements. The system has no sampling capability — no random sampling, no systematic sampling, no monetary unit sampling (MUS). Auditors using the system for sampling-dependent workpapers must do all sampling calculations manually outside the system.

**What needs to be decided:**

| Decision | Detail |
|----------|--------|
| In scope at all? | Is sampling a Phase 1 or Phase 2 feature, or explicitly out of scope? |
| Sampling methods to support | Random (simple random sample from a population); Systematic (every Nth item); Monetary Unit Sampling (probability proportional to size — most common for financial audits) |
| Integration point | New math tool in `model/tools/sampling.py` dispatched by TOOL_MANIFEST, or a standalone endpoint `/workpapers/{id}/sample`? |
| Input source | Does the population come from `content_json.rows[]` (already structured), or does the auditor upload a separate population file? |
| Output | Sample selection list + sampling parameters (confidence level, tolerable misstatement, expected deviation rate) injected into the `COMPUTED AUDIT FACTS` context block |
| Regulatory compliance | MUS calculations must be verifiable — the formula and seed must be reproducible and auditable |

**Owner:** ML Engineer + Audit QC Partner (to confirm which sampling methods HCLLP actually uses)  
**Blocker:** Scope decision — if out of scope, close as won't-fix. If in scope, needs placement in Phase 1 or Phase 2 roadmap.

---

## O-07 — No Management Response Tracking

**What is missing:**  
When an AI-drafted finding is finalized and issued to a client, the client's management is expected to provide a written response (agreeing, disagreeing, or providing a corrective action plan). There is no mechanism in the current system to:  
- Record that a finding has been issued and is awaiting a management response  
- Store the management response text  
- Link the response back to the originating `model_output` / finding  
- Track whether the corrective action was completed

**What needs to be decided:**

| Decision | Detail |
|----------|--------|
| In scope for Phase 1? | Or is management response tracking explicitly a Phase 2 / audit management system concern? |
| Data model | New `management_responses` table linked to `model_outputs`? Or is this handled entirely in Caseware and out of scope for Audit AI? |
| Who enters the response | Does the auditor paste the response into Audit AI, or does the client have a portal? |
| Status lifecycle | Proposed states: `awaiting_response → response_received → corrective_action_in_progress → closed` |
| DPO signal | Could management responses (agreement vs. disagreement with a finding) be used as a weak DPO signal? If yes, this needs a data model decision now. |

**Owner:** Engagement Partner  
**Blocker:** Scope decision. If management response tracking stays in Caseware (the current workflow), this closes as out-of-scope. If it moves into Audit AI, it needs a DB schema and API design before Phase 2.

---

## O-08 — No Multi-Turn Context

**What is missing:**  
Every inference call is stateless. The LangGraph pipeline starts fresh for each request with no memory of prior exchanges in the same session. An auditor cannot do:

> *"Document the finding for this trial balance exception."*  
> → AI produces finding  
> *"Now classify the risk for that finding."*  
> → AI has no memory of the finding it just produced

Each follow-up requires the auditor to re-upload context and re-specify everything. This significantly limits the usefulness of the system for multi-step audit workflows.

**What needs to be decided:**

| Decision | Detail |
|----------|--------|
| In scope at all? | Phase 1 or Phase 2 feature? Or explicitly out of scope (stateless only)? |
| Session definition | What constitutes a "session"? Single workpaper analysis? Single engagement? Time-bounded (e.g., 4-hour window)? |
| Storage | Session history stored in MongoDB (natural fit — variable-length document)? Redis (fast, ephemeral)? |
| Context window budget | The current 4,096-token active chunk budget leaves ~1,500 tokens for prior-turn history. How many prior turns fit? |
| State threading | Prior `model_output` IDs are the natural reference — does the auditor link turns explicitly ("based on output {id}") or does the system maintain an implicit session stack? |
| Privacy / isolation | Session history must be strictly scoped to one user + one engagement — no cross-session or cross-user context leakage |

**Owner:** ML Engineer  
**Blocker:** Scope decision. If stateless is acceptable for Phase 1 and multi-turn is Phase 2, close O-08 for now with a Phase 2 ticket created.

---

## Summary

| ID | Gap | Owner | Blocker |
|----|-----|-------|---------|
| O-01 | No runbook | Sysadmin + ML Eng | None — write after staging deployment |
| O-02 | No backup / DR | Sysadmin | Backup destination policy (air-gap constraint) |
| O-03 | No data retention policy | Eng Partner + IT Security | Legal / AICPA guidance review |
| O-04 | No model weight backup | Sysadmin + ML Eng | IT Security must approve physical media transfer |
| O-05 | No Caseware integration scope | Eng Partner + IT | Which Caseware version; VLAN API access |
| O-06 | No sampling module | ML Eng + QC Partner | Scope decision (Phase 1 vs Phase 2 vs out of scope) |
| O-07 | No management response tracking | Eng Partner | Scope decision (in Audit AI vs stays in Caseware) |
| O-08 | No multi-turn context | ML Eng | Scope decision (Phase 1 vs Phase 2) |

**O-02, O-03, O-04** are the highest operational risk right now — a disk failure before any backup strategy is in place means permanent data loss. These three should be resolved before Phase 1 deployment regardless of other scope decisions.
