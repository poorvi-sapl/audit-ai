# Risk Register — Audit AI
> Status: FINAL | Checklist item: #1 (Risk Register)
> Last updated: 2026-04-27

---

## 1. Purpose and Scope

This register identifies, quantifies, and assigns ownership for every material risk to the Audit AI system. It covers the full lifecycle: data ingestion → fine-tuning → RAG inference → auditor output → DPO feedback loop.

The register is a living document. Owners must review their assigned risks at each DPO cycle checkpoint (Month 3–4, Month 6, Month 9–12) and flag any status change.

---

## 2. Scoring Methodology

### 2.1 Probability Scale

| Score | Label | Definition |
|-------|-------|-----------|
| 1 | Rare | < 5% chance in any 12-month window |
| 2 | Unlikely | 5–20% chance |
| 3 | Possible | 20–50% chance |
| 4 | Likely | 50–80% chance |
| 5 | Almost Certain | > 80% chance |

### 2.2 Impact Scale

| Score | Label | Definition |
|-------|-------|-----------|
| 1 | Negligible | No user-visible effect; fixed silently |
| 2 | Minor | < 2h of work to remediate; no audit deliverable affected |
| 3 | Moderate | Audit deliverable delayed or requires rework; auditor trust dented |
| 4 | Major | Client data exposed OR audit finding materially wrong; leadership escalation |
| 5 | Critical | Regulatory sanction, professional liability, or system shutdown required |

### 2.3 Risk Rating Matrix

```
         Impact →
         1          2          3          4          5
P  5  │  5 (M)   │ 10 (H)  │ 15 (H)  │ 20 (C)  │ 25 (C)  │
r  4  │  4 (L)   │  8 (M)  │ 12 (H)  │ 16 (H)  │ 20 (C)  │
o  3  │  3 (L)   │  6 (M)  │  9 (M)  │ 12 (H)  │ 15 (H)  │
b  2  │  2 (L)   │  4 (L)  │  6 (M)  │  8 (M)  │ 10 (H)  │
↓  1  │  1 (L)   │  2 (L)  │  3 (L)  │  4 (L)  │  5 (M)  │

L = Low (1–4)   M = Medium (5–9)   H = High (10–16)   C = Critical (17–25)
```

### 2.4 Residual Risk Target

All risks must reach **Medium or below** before go-live. Critical and High risks require a documented mitigation plan with a named owner and deadline before any production deployment.

---

## 3. Risk Register

### R-01 — Data Privacy Breach

**Description:** Client PII (SSN, EIN, ITIN, bank account numbers) present in historical workpapers enters the training corpus and/or the Qdrant vector store, potentially surfacing in model outputs visible to unauthorized users.

| Attribute | Value |
|-----------|-------|
| **Category** | Data Governance / Compliance |
| **Inherent Probability** | 4 (Likely — 15 years of unprocessed workpapers) |
| **Inherent Impact** | 5 (Critical — client PII exposure triggers SOC 2 / AICPA liability) |
| **Inherent Risk Score** | 20 — **CRITICAL** |
| **Owner** | Lead Data Engineer + Engagement Partner (joint) |
| **Review Cadence** | Before every SFT batch; at each DPO cycle |

**Primary Mitigations:**

1. **Presidio pipeline (mandatory, not optional):** Every document passes through `PIIScrubber` (transformer chain step 2) before any storage write. Presidio analyzer runs US patterns (SSN, EIN, ITIN, bank routing, account) plus HCLLP custom regexes from `pii_patterns.yaml`.
2. **Cleaning log audit trail:** Every scrub action writes to `cleaning_log` (PostgreSQL) with `entity_type`, `scrub_method`, and `confidence_score`. The ETL DAG (`etl_pipeline`) fails the workpaper if scrub confidence < 0.85 on any detected entity.
3. **Training pair inspection gate:** Before each SFT export (`jsonl_export` DAG), a sampling script draws 50 random pairs and pipes them through Presidio again. Any hit blocks the export and pages the data engineer.
4. **Qdrant payload filter:** `pii_scrubbed: true` is a required payload field; the retrieval query adds `must: [{key: "pii_scrubbed", match: {value: true}}]` as a hard filter.
5. **Air-gap:** No outbound internet routes from the GPU VLAN; PII cannot exfiltrate via network even if a script bug re-injects it.
6. **Engagement-level access control:** `workpapers` table and `/files/` endpoint require `role IN ('senior_auditor','partner','admin')`; raw extracted text is never returned via API.
7. **Document-level RBAC (`workpaper_permissions` table):** Every workpaper has an explicit permission grant per user or role. The `require_wp_permission()` FastAPI dependency is injected into every workpaper-scoped endpoint. HTTP 403 is returned for both "not found" and "no permission" responses — prevents workpaper enumeration. Cross-engagement leaks are impossible: permission grants are scoped to individual `workpaper_id` rows, not engagement-wide. Default permissions seeded on upload (junior → write, senior → write, manager → review, partner → approve). See `03_lld.md` Table 4a and `06_config_output.md` Section 2.7.

**Residual Assessment:**
- Probability after mitigations: **2** (Unlikely — double scrub + blocked export)
- Impact after mitigations: **4** (Major — residual risk if a novel PII pattern escapes both passes)
- **Residual Score: 8 — MEDIUM** ✓ (meets go-live gate)

**Monitoring:**
- Prometheus counter `pii_entities_scrubbed_total{entity_type}` — spike alerts on anomaly
- Weekly spot-check: QA auditor manually reviews 10 random chunk samples from MongoDB

---

### R-02 — Model Hallucination (Invented Regulation Citations)

**Description:** The fine-tuned Mistral 22B model cites a regulation code (e.g., "2 CFR 200.519(b)(3)") that does not exist, or attributes a threshold value (e.g., "$500K" instead of "$750K") to the wrong fiscal year. An auditor signs off without cross-checking, creating a materially incorrect workpaper.

| Attribute | Value |
|-----------|-------|
| **Category** | Model Quality / Professional Liability |
| **Inherent Probability** | 4 (Likely — LLMs hallucinate citations by default) |
| **Inherent Impact** | 5 (Critical — incorrect audit finding = AICPA peer review failure, potential legal exposure) |
| **Inherent Risk Score** | 20 — **CRITICAL** |
| **Owner** | ML Engineer + Audit Quality Control Partner |
| **Review Cadence** | Every DPO cycle; monthly hallucination-rate metric review |

**Primary Mitigations:**

1. **Citation verifier (post-processor step 3):** After every inference call, `PostProcessor.verify_citations()` looks up every `regulation_cited` value in `regulations_master.json`. Any code not found in the master list sets `hallucination_flag = True` on the output.
2. **`regulations_master.json` coverage:** Maintained by the Audit QC partner. Covers 2 CFR 200 series (§200.500–200.521), AU-C series (AU-C 200–800), and SA series (SA 200–810, SA 300–499, SA 500–720). Update is a required step at each DPO cycle.
3. **Outdated regulation mapping:** `outdated_regulations.json` maps deprecated codes (e.g., OMB A-133 §.320) to current equivalents. Post-processor flags deprecated citations with `regulation_outdated: true`.
4. **Threshold YAML guard:** `thresholds.yaml` is loaded at startup. `PostProcessor.validate_fields()` checks that any dollar threshold in the output matches the value in the YAML for the given engagement year. Mismatch → `hallucination_flag = True`.
5. **AI-Assisted Draft label (mandatory):** Every output has `ai_draft_label: "AI-Assisted Draft — Requires Senior Auditor Review Before Use"` stamped in the Word/Excel export header. This is a non-suppressible field in `BaseOutput`.
6. **`opinion_flag` scanner (post-processor step 4):** Regex patterns in `pii_patterns.yaml → opinion_language_patterns` catch phrases like "In our opinion", "we conclude", "we certify". Any hit sets `opinion_flag = True` and the export renderer renders the cell/paragraph in red.
7. **Temperature discipline:** `compliance_check` tasks use `temperature=0.1`; `finding_documentation` uses `temperature=0.3`. High-temperature modes are locked out for any task touching `regulation_cited`.
8. **Blind human review in evaluation protocol:** See `08_eval_metrics.md` — 30 pairs per task type reviewed by a senior auditor who does not see the model confidence score before rating. Hallucination rate must be < 5% at go-live.

**Residual Assessment:**
- Probability after mitigations: **2** (Unlikely — verifier catches known codes; auditor required to sign off)
- Impact after mitigations: **3** (Moderate — flagged output requires rework but auditor is the last line of defense)
- **Residual Score: 6 — MEDIUM** ✓

**Monitoring:**
- Langfuse trace tag `hallucination_flag=true` rate; alert if 7-day rolling rate > 3%
- Monthly: ML engineer reviews all `hallucination_flag=true` outputs in Langfuse; any new unknown pattern added to `regulations_master.json`

---

### R-03 — Auditor Adoption / Trust Gap

**Description:** Senior auditors do not use the system because they distrust AI outputs, fear professional liability, or find the UX friction higher than manual methods. The system delivers technically correct results but achieves near-zero engagement, making the DPO flywheel impossible (no corrections → no Cycle 1 pairs → model stagnates).

| Attribute | Value |
|-----------|-------|
| **Category** | Change Management / Product |
| **Inherent Probability** | 4 (Likely — typical for AI tools in regulated professions) |
| **Inherent Impact** | 4 (Major — kills DPO cycle; entire investment yields no ROI) |
| **Inherent Risk Score** | 16 — **HIGH** |
| **Owner** | Engagement Partner (primary) + UX/Product lead |
| **Review Cadence** | Monthly during pilot; at each DPO cycle |

**Primary Mitigations:**

1. **Phase 1 scope constraint:** Only Single Audit (has_single_audit=true) workpapers in Phase 1. Focused scope = auditors can validate correctness against known criteria; trust builds before expanding to GAGAS-only Govt engagements and NPO engagements in Phase 2.
2. **Would-sign-off gate enforced:** Expansion to Phase 2 requires ≥ 80% `would_sign_off` rate in `eval_results`. The gate is a hard policy, not a suggestion. Premature expansion is the fastest way to destroy trust.
3. **Draft framing in UI:** The `ai_draft_label` is prominent. Auditors are explicitly positioned as reviewers, not rubber-stampers. The system never claims to be the author of a finding.
4. **Explainability via retrieved context:** The 5-section prompt structure (retrieved chunks in section 2) is surfaced to the auditor as "Evidence used." Auditors can see exactly which prior workpaper passages support each output.
5. **Correction UX first-class:** The `dpo_feedback` table and correction workflow are built in Phase 1. Making corrections easy (not buried) signals that auditor input directly improves the tool.
6. **Pilot champion program:** Identify 2–3 senior auditors who are early adopters. Their corrections seed Cycle 1 DPO pairs. Their public endorsement to peers is the highest-leverage adoption driver.
7. **Transparent confidence, not false certainty:** `hallucination_flag` and `opinion_flag` are shown to auditors. A system that admits uncertainty is more trustworthy than one that projects confidence it doesn't have.

**Residual Assessment:**
- Probability after mitigations: **2** (Unlikely — structured pilot + champion program + honest framing)
- Impact after mitigations: **3** (Moderate — slow adoption delays DPO timeline but doesn't kill the system)
- **Residual Score: 6 — MEDIUM** ✓

**Monitoring:**
- Weekly: `dpo_feedback` table row count per auditor; flag if any auditor has 0 corrections in a 30-day window
- Monthly: adoption survey (3 questions, < 2 min); NPS proxy tracked in project memory

---

### R-04 — Scope Creep

**Description:** Leadership or clients pressure the team to add capabilities (GAGAS-only Govt engagements, NPO engagements, new output types, Excel analytics) before the Single Audit pilot is stable and the 80% `would_sign_off` gate is passed. Each premature expansion dilutes training data quality, multiplies edge cases, and delays the DPO cycle.

| Attribute | Value |
|-----------|-------|
| **Category** | Project Management |
| **Inherent Probability** | 4 (Likely — common in AI projects with visible early results) |
| **Inherent Impact** | 3 (Moderate — delays timeline, degrades quality, burns engineering capacity) |
| **Inherent Risk Score** | 12 — **HIGH** |
| **Owner** | Project Manager + Engagement Partner |
| **Review Cadence** | Sprint review (every 2 weeks during active development) |

**Primary Mitigations:**

1. **Hard-coded phase gate in the codebase:** Phase 1 inference is restricted to engagements where `has_single_audit=true`. Enabling non-Single-Audit engagements (GAGAS-only Govt, NPO) requires a deliberate feature flag change — not a config change.
2. **80% gate is a go/no-go, not a guideline:** The `eval_results` table and evaluation protocol in `08_eval_metrics.md` define exactly how the gate is measured. The Engagement Partner must sign off that the gate is passed before any Phase 2 work begins.
3. **Documented scope in this architecture:** All 8 architecture docs explicitly state "Phase 1: Single Audit only." Any scope change requires updating these docs first — creating a paper trail and forcing a deliberate decision.
4. **Backlog triage process:** A formal backlog triage every sprint. New requests go into the backlog as `post-phase-1` tagged items, not the active sprint. The PM owns the gate.
5. **Cost-of-context framing:** Every scope addition requires a fresh SFT batch (estimate: $23K–$50K per batch from the cost model). Attaching a dollar cost to each addition reduces casual requests.

**Residual Assessment:**
- Probability after mitigations: **2** (Unlikely — hard code gate + documented policy)
- Impact after mitigations: **2** (Minor — if a small scope addition sneaks through, it is caught at sprint review before it hits production)
- **Residual Score: 4 — LOW** ✓

**Monitoring:**
- Sprint velocity tracking; any sprint where > 20% of story points are post-phase-1 items triggers a PM escalation
- Architecture doc version history as audit trail

---

### R-05 — Training Data Quality (Poor Pairs → Poor Model)

**Description:** The 15 years of historical workpapers contain inconsistent formatting, incomplete findings, superseded regulations, and varying quality between engagement partners. Poor training pairs produce a model that confidently generates plausible-sounding but structurally wrong audit outputs.

| Attribute | Value |
|-----------|-------|
| **Category** | Data Quality / ML |
| **Inherent Probability** | 4 (Likely — 15 years of heterogeneous data is inherently noisy) |
| **Inherent Impact** | 4 (Major — a degraded model is worse than no model; auditors learn to distrust it) |
| **Inherent Risk Score** | 16 — **HIGH** |
| **Owner** | ML Engineer + Lead Data Engineer |
| **Review Cadence** | Before every SFT batch; at each DPO cycle |

**Primary Mitigations:**

1. **Transformer chain quality gates:**
   - `Normalizer` → strips encoding artifacts, normalizes whitespace
   - `PIIScrubber` → confidence < 0.85 fails the workpaper
   - `TemporalTagger` → tags `fiscal_year_end` for temporal drift weighting
   - `Chunker` → rejects chunks < 50 tokens or > 8K tokens
   - `InjectionSanitizer` → strips prompt injection patterns
2. **Temporal drift weighting:** The `jsonl_export` DAG assigns `sample_weight` per the drift table: 2024–2026 → 1.00, 2022–2023 → 0.85, down to pre-2015 → 0.10. Old data has diminishing influence, not zero (historical context is valuable).
3. **Engagement-level split (not row-level):** The deterministic MD5 split ensures all pairs from one engagement stay in the same split. This prevents data leakage from partner-specific writing styles inflating val/test metrics.
4. **Pre-SFT pair sampling review:** Before each SFT batch, the ML engineer samples 100 pairs (stratified by task_type) for manual review. Any structural error pattern (e.g., missing `[INST]` tags, empty responses) blocks the batch.
5. **`sft_training_pairs` table tracking:** Every pair has `quality_score`, `source_doc_id`, and `split` fields. Low-quality pairs (score < 0.7) are excluded from training via a DAG task filter, but retained in the table for analysis.
6. **SFT eval loop:** After each SFT run, the model is evaluated on the held-out test split before weights are promoted to production. Promotion requires ROUGE-L ≥ 0.45 and hallucination rate ≤ 5%.
7. **PDF extraction timeout and page-split strategy:** Large or complex PDFs (> 50 pages) are pre-split into 25-page windows before OCR or text extraction. Each window runs under a signal-based timeout (90s for text PDFs, 600s per window for scanned). On timeout, the window is marked `error_window` and the Celery worker is released immediately — a hung PDF cannot starve the queue. Partial extractions (successfully processed windows) are kept rather than discarding the whole workpaper. If > 50% of pages are unrecoverable, the workpaper is flagged for manual queue and the engagement manager is notified. This prevents hung extractions from silently shrinking the training corpus.

**Residual Assessment:**
- Probability after mitigations: **2** (Unlikely — multi-layer quality gates block most bad data)
- Impact after mitigations: **3** (Moderate — some noise will always exist; model is imperfect but auditor-reviewable)
- **Residual Score: 6 — MEDIUM** ✓

**Monitoring:**
- Prometheus gauge `training_pairs_quality_score_p50{task_type}` — alert if drops below 0.7
- Post-training: Langfuse hallucination rate per task_type; regression test suite (see `08_eval_metrics.md`)

---

### R-06 — Infrastructure Failure (Single Point of Failure on Inference Server)

**Description:** Server 2 (inference GPU, 2× L40S) experiences hardware failure, CUDA driver crash, or vLLM OOM. With no redundant inference node, all AI-assisted workpaper generation is unavailable until the server is restored. Estimated MTTR for a GPU server: 4–24 hours.

| Attribute | Value |
|-----------|-------|
| **Category** | Infrastructure / Availability |
| **Inherent Probability** | 2 (Unlikely — enterprise GPU servers are reliable, but not infallible) |
| **Inherent Impact** | 3 (Moderate — system unavailable; auditors revert to manual; no data loss) |
| **Inherent Risk Score** | 6 — **MEDIUM** |
| **Owner** | Systems Administrator |
| **Review Cadence** | Quarterly hardware audit; after any incident |

**Primary Mitigations:**

1. **Graceful degradation design:** The FastAPI application layer (Server 1) remains up when Server 2 is down. Auditors get a clear "AI inference unavailable — manual mode" message rather than a 500 error. Workpapers can still be uploaded and stored; AI analysis is queued.
2. **arq job queue persistence:** Inference requests are queued in Redis (arq). When Server 2 comes back online, the queue drains automatically. No requests are lost.
3. **vLLM health endpoint:** Server 1 polls `GET /v1/models` on Server 2 every 30 seconds. On failure, it sets a `INFERENCE_AVAILABLE = False` flag in Redis. The API returns HTTP 503 with a `Retry-After` header.
4. **Prometheus + Grafana alerting:** `vllm_requests_running`, `gpu_memory_used_bytes`, and the health-check failure metric trigger PagerDuty-equivalent alerts to the sysadmin within 2 minutes.
5. **vLLM OOM recovery:** `enforce_eager=False` and `gpu_memory_utilization=0.50` (conservative) reduce OOM probability. If OOM occurs, a systemd `Restart=on-failure` service restarts vLLM automatically within 60 seconds.
6. **Backup inference path (dev only):** Ollama is available on Server 1 as an emergency fallback for single-request unblocking. **Not for production throughput.** Used only by the sysadmin to unblock a critical workpaper while Server 2 is being repaired.
7. **Hardware support contract:** HCLLP should maintain a next-business-day hardware replacement SLA on Server 2 GPUs. This is an operational control, not a software control.

**Residual Assessment:**
- Probability after mitigations: **2** (Unlikely — same; hardware failure probability is not changed by software)
- Impact after mitigations: **2** (Minor — graceful degradation + queue persistence = no data loss, known MTTR)
- **Residual Score: 4 — LOW** ✓

**Monitoring:**
- Grafana dashboard: GPU memory, vLLM queue depth, health-check status
- Alert: vLLM health-check fail for > 2 consecutive checks → sysadmin PagerDuty

---

### R-07 — ~~India Jurisdiction Accuracy~~ REMOVED

> **Removed April 2026.** HCLLP operates US-only engagements (NPO and Govt clients). ICAI-SA is permanently out of scope. Risk closed — no mitigations required.

---

### R-08 — Temporal Drift (Model Learns Outdated Regulation Thresholds)

**Description:** The model is trained on data spanning 2009–2024. Pre-2017 workpapers reference OMB A-133 (superseded by 2 CFR 200 in 2014) and pre-2024 data uses the old $750K Single Audit threshold. Without countermeasures, the model assigns equal weight to outdated thresholds and current ones.

| Attribute | Value |
|-----------|-------|
| **Category** | Data Quality / Regulatory |
| **Inherent Probability** | 5 (Almost Certain — 15-year corpus guarantees regulatory regime changes) |
| **Inherent Impact** | 4 (Major — an auditor relies on a $500K threshold that was correct in 2013 but wrong today) |
| **Inherent Risk Score** | 20 — **CRITICAL** |
| **Owner** | ML Engineer + Audit QC Partner |
| **Review Cadence** | Before every SFT batch; annually when federal thresholds update |

**Primary Mitigations:**

1. **Temporal drift weight table (enforced in DAG):** The `jsonl_export` DAG assigns `sample_weight` at export time:
   - 2024–2026: 1.00
   - 2022–2023: 0.85
   - 2020–2021: 0.70
   - 2018–2019: 0.55
   - 2016–2017: 0.35
   - 2014–2015: 0.20
   - Pre-2015: 0.10
   These weights are passed as `--sample_weights` to SFTTrainer. Pre-2015 data is retained for structural pattern learning but down-weighted to near-zero threshold influence.

2. **`outdated_regulations.json` hard filter:** The `InjectionSanitizer` transformer (step 5) replaces known superseded regulation codes with their current equivalents in the extracted text before chunking. The mapping includes:
   - `OMB A-133` → `2 CFR 200 Subpart F`
   - `OMB A-87` → `2 CFR 200 Subpart E`
   - `SAS 99` → `AU-C 240`
   This happens at ETL time, not inference time — the model never sees superseded codes in training data.

3. **`thresholds.yaml` is the runtime authority:** Even if the model generates a threshold value, `PostProcessor.validate_fields()` overrides it with the value from `thresholds.yaml` for the engagement's `fiscal_year_end`. The model output is a suggestion; the YAML is the truth.

4. **Annual threshold review process:** The Audit QC Partner is responsible for updating `thresholds.yaml` each October (when OMB updates federal thresholds). This is a calendar item, not an ad-hoc task. The update triggers a new SFT batch if any threshold changes.

5. **`TemporalTagger` metadata (transformer step 3):** Tags every chunk with `fiscal_year_end`, `regulation_version_at_time`, and `threshold_at_time` (looked up from `thresholds.yaml` using the fiscal year). Qdrant payload carries these fields, enabling year-filtered retrieval for historical analysis use cases.

**Residual Assessment:**
- Probability after mitigations: **1** (Rare — superseded codes replaced at ETL; runtime YAML overrides model output)
- Impact after mitigations: **2** (Minor — if a novel superseded code appears, the citation verifier flags it)
- **Residual Score: 2 — LOW** ✓

**Monitoring:**
- Annual: Audit QC Partner reviews OMB / AICPA regulatory updates and updates `thresholds.yaml` + `outdated_regulations.json`
- Post-SFT: Evaluation suite includes a dedicated "threshold accuracy" test set (20 pairs where the correct threshold is known)

---

### R-09 — Security Breach (Air-Gap Violation / Data Exfiltration)

**Description:** An adversary (external attacker or malicious insider) exfiltrates client workpaper data or model weights from the air-gapped VLAN, either via a misconfigured firewall rule, a compromised admin credential, or a physical media transfer.

| Attribute | Value |
|-----------|-------|
| **Category** | Security / Compliance |
| **Inherent Probability** | 2 (Unlikely — air-gap significantly reduces attack surface) |
| **Inherent Impact** | 5 (Critical — client confidential data breach triggers regulatory sanctions, client notification obligations, potential AICPA ethics investigation) |
| **Inherent Risk Score** | 10 — **HIGH** |
| **Owner** | IT Security Officer + Systems Administrator |
| **Review Cadence** | Quarterly security review; immediately after any firewall change |

**Primary Mitigations:**

1. **Network segmentation (Architecture Layer 1):** The GPU VLAN has no default gateway to the internet. All traffic to/from the VLAN passes through a stateful firewall. Outbound rules are deny-all; the only inbound rules allow TCP 443 from the internal corporate network to the FastAPI port.
2. **TLS 1.3 everywhere:** All inter-server communication (Server 1 ↔ Server 2, Server 1 ↔ PostgreSQL/MongoDB/Qdrant/Redis) is TLS 1.3. Certificates are self-signed with 2-year rotation (managed by the sysadmin). No TLS 1.2 fallback permitted.
3. **Authentication hardening:**
   - JWT tokens: 15-minute access token, 7-day refresh token, RS256 signing (asymmetric)
   - Passwords: bcrypt cost factor 12
   - Phase 2: M365 Azure AD SSO replaces password auth entirely (additive migration)
   - Admin endpoints require `role = 'admin'` claim in JWT; role claims are server-side, not client-supplied
4. **Encryption at rest:** PostgreSQL, MongoDB, and Qdrant data directories on LUKS-encrypted volumes. Model weights on encrypted storage. Redis is ephemeral (no persistence to disk for sensitive data).
5. **Audit log (non-repudiation):** Every API call writes to the `audit_log` table (PostgreSQL). The table is append-only (no DELETE permission granted to the application service account). The sysadmin reviews the log weekly.
6. **Physical security:** GPU servers in a locked server room; USB ports disabled in BIOS. Media transfer policy: all external media must be scanned and logged before connecting to any system in the VLAN.
7. **Injection prevention:** The `InjectionSanitizer` transformer (step 5) and API-layer `InjectionSanitizer.sanitize_api_input()` strip prompt injection attempts before they reach the model. This prevents an adversary from using a crafted workpaper to exfiltrate data via model output.
8. **Principle of least privilege:** Application service accounts have minimal PostgreSQL permissions (SELECT/INSERT/UPDATE on specific tables; no DROP/TRUNCATE). The `workpapers` table raw-text column is readable only by the ETL service account, not the inference service account.
9. **JSONL training file access control:** JSONL export files (which contain client workpaper text) are protected by two controls: (a) the `jsonl_export` Airflow DAG requires `role = 'admin'` to trigger — enforced by Airflow RBAC on the REST trigger endpoint; (b) written files are `chmod 600 / chown train_svc:train_svc` — only the training OS service account can read them. Files are automatically deleted by the `training_trigger` DAG's `post_training_cleanup` step immediately after `run_sft` completes. A `.pending_deletion` marker ensures deletion is not skipped. Deletion is logged to `audit_trail`. JSONL files exist on disk only during the ETL→training window — never retained beyond that.

**Residual Assessment:**
- Probability after mitigations: **1** (Rare — air-gap + TLS + encrypted storage + audit log)
- Impact after mitigations: **4** (Major — if a breach occurs despite mitigations, impact is still high; but likelihood is very low)
- **Residual Score: 4 — LOW** ✓

**Monitoring:**
- Prometheus: failed auth attempts counter; alert if > 10 failed logins from the same IP in 5 minutes
- Weekly: sysadmin reviews `audit_trail` for anomalous access patterns (off-hours access, bulk SELECT queries, missing `jsonl_deleted` events after training runs)
- Quarterly: firewall ruleset review; penetration test recommended annually
- Annual (or on suspected compromise): RSA key pair rotation — generate new 2048-bit pair with `openssl genrsa`, deploy new `jwt_public.pem` to Server 2, restart auth middleware on both servers; all outstanding tokens are invalidated (acceptable: 15-minute TTL means at most 15 min of disruption); old private key is securely deleted (`shred -u jwt_private.pem`)

---

### R-10 — DPO Feedback Volume Risk (Insufficient Corrections by Month 3–4)

**Description:** DPO Cycle 1 requires 300+ preference pairs by Month 3–4. If auditors don't actively use the system and submit corrections, the `dpo_feedback` table will have fewer than 300 complete, high-quality pairs. Running DPO on thin data produces a model that overfits the corrections and degrades on uncorrected tasks.

| Attribute | Value |
|-----------|-------|
| **Category** | ML / Change Management |
| **Inherent Probability** | 3 (Possible — 300 pairs in 3 months requires consistent engagement from multiple auditors) |
| **Inherent Impact** | 3 (Moderate — Cycle 1 is delayed or degraded; model improvement stalls; DPO flywheel doesn't spin up) |
| **Inherent Risk Score** | 9 — **MEDIUM** |
| **Owner** | ML Engineer + Engagement Partner |
| **Review Cadence** | Monthly during Months 1–4; weekly in Month 3 |

**Primary Mitigations:**

1. **Volume tracking with early warning:** The `dpo_feedback` table has `is_complete` and `quality_score` columns. A weekly cron job (arq task) counts `WHERE is_complete = true AND quality_score >= 0.7`. If count < 100 by end of Month 2, the ML engineer escalates to the Engagement Partner.
2. **Correction UX is frictionless:** Submitting a correction requires clicking "Improve This Output" on the output card, editing the preferred response inline, and clicking "Submit." Target: < 90 seconds per correction. If the UX takes longer, the team must fix it before Month 2.
3. **Structured correction prompts:** The correction UI pre-fills the output fields. Auditors correct only what is wrong, not re-write from scratch. This reduces correction time and increases completion rate.
4. **Targeted correction campaigns:** The ML engineer identifies the 2–3 task types with the highest hallucination rates (from Langfuse) and asks the pilot champion auditors to focus corrections on those tasks first. 300 focused pairs beats 500 scattered pairs.
5. **Synthetic pair augmentation (fallback):** If volume is < 200 by Month 3, the ML engineer generates synthetic preference pairs using the SFT model itself (sample 2 outputs at different temperatures; have an auditor rank them). Synthetic pairs are tagged `source = 'synthetic'` in `dpo_feedback` and limited to 30% of total Cycle 1 pairs to avoid synthetic-only overfitting.
6. **Cycle 1 threshold flexibility:** If genuine pairs reach 250–300 (not 300+) and quality scores are high (> 0.8 average), the ML engineer and QC Partner may approve Cycle 1 with the available data. The 300 target is a guideline for quality; the absolute minimum is 200 high-quality pairs.

**Residual Assessment:**
- Probability after mitigations: **2** (Unlikely — early warning at Month 2 + UX investment + synthetic fallback)
- Impact after mitigations: **2** (Minor — if Cycle 1 is delayed by 4–6 weeks, the overall timeline slips but the system is not damaged)
- **Residual Score: 4 — LOW** ✓

**Monitoring:**
- Weekly: `SELECT COUNT(*) FROM dpo_feedback WHERE is_complete = true AND quality_score >= 0.7` reported to ML engineer
- Langfuse: correction submission rate per auditor; champion auditors expected ≥ 5 corrections/week during active pilot

---

## 4. Risk Summary Matrix

| ID | Risk Name | Inherent Score | Inherent Rating | Residual Score | Residual Rating | Owner | Go-Live Gate |
|----|-----------|---------------|-----------------|----------------|-----------------|-------|-------------|
| R-01 | Data Privacy Breach | 20 | CRITICAL | 8 | **MEDIUM** | Lead Data Eng + Partner | ✓ Required |
| R-02 | Model Hallucination | 20 | CRITICAL | 6 | **MEDIUM** | ML Eng + QC Partner | ✓ Required |
| R-03 | Auditor Adoption Gap | 16 | HIGH | 6 | **MEDIUM** | Engagement Partner | ✓ Required |
| R-04 | Scope Creep | 12 | HIGH | 4 | **LOW** | Project Manager | — |
| R-05 | Training Data Quality | 16 | HIGH | 6 | **MEDIUM** | ML Eng + Data Eng | ✓ Required |
| R-06 | Infrastructure Failure | 6 | MEDIUM | 4 | **LOW** | Sysadmin | — |
| R-07 | ~~India Jurisdiction Accuracy~~ | REMOVED | — | — | — | — | Out of scope |
| R-08 | Temporal Drift | 20 | CRITICAL | 2 | **LOW** | ML Eng + QC Partner | ✓ Required |
| R-09 | Security Breach | 10 | HIGH | 4 | **LOW** | IT Security Officer | ✓ Required |
| R-10 | DPO Feedback Volume | 9 | MEDIUM | 4 | **LOW** | ML Eng + Partner | — |

**Go-Live Gate summary:** All 6 "Required" risks must be at MEDIUM or below, with documented mitigation evidence, before any production deployment. R-07 is closed (US-only scope confirmed).

---

## 5. Risk-to-Architecture Cross-Reference

| Risk | Mitigating Architecture Component | Document Reference |
|------|-----------------------------------|--------------------|
| R-01 | PIIScrubber transformer, `pii_patterns.yaml`, `cleaning_log` table | [04_etl_pipeline.md](04_etl_pipeline.md), [06_config_output.md](06_config_output.md) |
| R-01 | Qdrant `pii_scrubbed` payload filter | [05_rag_inference.md](05_rag_inference.md) |
| R-02 | Citation verifier, `regulations_master.json`, `hallucination_flag` | [05_rag_inference.md](05_rag_inference.md), [06_config_output.md](06_config_output.md) |
| R-02 | `thresholds.yaml` runtime override | [06_config_output.md](06_config_output.md) |
| R-03 | Phase gate, `would_sign_off` field, correction UX | [02_hld.md](02_hld.md), [08_eval_metrics.md](08_eval_metrics.md) |
| R-04 | `has_single_audit` flag, hard phase gate in code | [03_lld.md](03_lld.md), [02_hld.md](02_hld.md) |
| R-05 | Transformer chain, temporal weights, `sft_training_pairs` quality gate | [04_etl_pipeline.md](04_etl_pipeline.md) |
| R-06 | arq queue persistence, vLLM health check, graceful degradation | [02_hld.md](02_hld.md), [05_rag_inference.md](05_rag_inference.md) |
| R-07 | ~~Removed — US-only scope~~ | — |
| R-08 | Temporal weight table, `outdated_regulations.json` substitution, YAML override | [04_etl_pipeline.md](04_etl_pipeline.md), [06_config_output.md](06_config_output.md) |
| R-09 | Air-gap VLAN, TLS 1.3, JWT RS256, LUKS encryption, audit_log | [01_sys_arc.md](01_sys_arc.md), [03_lld.md](03_lld.md) |
| R-10 | `dpo_feedback` volume tracking, synthetic pair fallback | [02_hld.md](02_hld.md), [04_etl_pipeline.md](04_etl_pipeline.md) |

---

## 6. Residual Risk Acceptance

All residual risks are at **MEDIUM or below**, meeting the go-live gate criterion. No risk remains at HIGH or CRITICAL after mitigations are applied.

The Engagement Partner and IT Security Officer must sign off on this register before Phase 1 deployment. Signatures (or equivalent approval in the project's task tracker) constitute formal acceptance of residual risk on behalf of HCLLP.

| Residual Rating | Count | Risk IDs |
|----------------|-------|---------|
| LOW | 5 | R-04, R-06, R-08, R-09, R-10 |
| MEDIUM | 4 | R-01, R-02, R-03, R-05 |
| HIGH | 0 | — |
| CRITICAL | 0 | — |

---

## 7. Open Risk Items

| Item | Status | Due | Owner |
|------|--------|-----|-------|
| Confirm hardware support contract for Server 2 GPUs (NBD replacement SLA) | Open | Before go-live | Sysadmin |
| Annual threshold review process formalized as calendar item (October) | Open | Before go-live | Audit QC Partner |
| Audit QC Partner to complete `thresholds.yaml` materiality percentages (NPO and Govt sections) | Open | Before Phase 1 go-live | Audit QC Partner |
| Penetration test scheduled (annual recommendation) | Open | Month 6 | IT Security Officer |

---

*Next document: [08_eval_metrics.md](08_eval_metrics.md) — Evaluation Metrics (Checklist item #16)*
