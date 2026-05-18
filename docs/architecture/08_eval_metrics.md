# Evaluation Metrics — Audit AI
> Status: FINAL | Checklist item: #16 (Evaluation Metrics)
> Last updated: 2026-04-27

---

## 1. Purpose and Scope

This document defines the full evaluation framework for Audit AI. It covers:

- Per-task-type rubrics used in blind human review
- Quantitative technical metrics and their measurement methodology
- The blind review protocol (evaluator eligibility, scoring, aggregation)
- The `eval_results` table schema and how scores gate production promotion
- DPO cycle success criteria (Cycles 1, 2, 3)
- Regression test suite design and the never-seen rule
- Post-deployment monitoring metrics and alert thresholds

Every go-live decision is driven by the numbers in this document. No subjective "it feels good" promotions.

---

## 2. Task Types and Per-Task Rubrics

Audit AI has five task types. Each has a distinct rubric because the criteria for quality differ materially between them.

### 2.1 `risk_classification`

**What the model produces:** A `risk_level` from `{CRITICAL, HIGH, MEDIUM, LOW, N_A}` with a supporting `risk_rationale` (free text, 1–3 sentences).

**Primary metric:** F1 score (macro-averaged across all 5 classes) on the held-out test split.

**Go-live gate:** F1 ≥ 0.87

**Human rubric (5-point scale per dimension):**

| Dimension | 1 — Unacceptable | 3 — Acceptable | 5 — Excellent |
|-----------|-----------------|----------------|---------------|
| **Level accuracy** | Wrong by 2+ levels (e.g., CRITICAL when LOW) | Off by 1 level | Exactly correct |
| **Rationale grounding** | No reference to workpaper evidence | Some evidence cited but imprecise | Specific workpaper section cited with correct threshold |
| **Regulation alignment** | Wrong or missing regulatory basis | Correct framework, imprecise citation | Exact citation (e.g., 2 CFR 200.516(a)) verified in `regulations_master.json` |
| **`would_sign_off`** | Evaluator would not sign under any circumstances | Would sign with significant revisions | Would sign with minor or no revisions |

**Scoring note:** `would_sign_off` is binary (yes/no), collected separately from the 5-point dimensions. The 5-point scores are averaged to produce `overall_score`; `would_sign_off` drives the go-live gate directly.

---

### 2.2 `finding_documentation`

**What the model produces:** Structured finding with `criteria`, `condition`, `cause`, `effect` (all `str | None`) and `recommendation` (required `str`).

**Primary metric:** ROUGE-L against a reference finding (human-written by a senior auditor for the same workpaper).

**Go-live gate:** ROUGE-L ≥ 0.45

**Human rubric:**

| Dimension | 1 — Unacceptable | 3 — Acceptable | 5 — Excellent |
|-----------|-----------------|----------------|---------------|
| **Criteria accuracy** | Wrong standard cited or missing | Correct framework, approximate citation | Exact AU-C / 2 CFR 200 / SA cite verified |
| **Condition–cause–effect chain** | Conditions stated without cause or effect | 2 of 3 elements correct and linked | All 3 elements correct, logically coherent |
| **Recommendation specificity** | Generic ("improve controls") | Specific but lacks timeline or owner | Specific, actionable, references the finding condition |
| **Completeness** | ≥ 2 required fields empty | 1 required field empty | All fields populated |
| **`would_sign_off`** | No | With revisions | Yes or yes with minor edits |

---

### 2.3 `compliance_check`

**What the model produces:** A `compliance_status` from `{compliant, non_compliant, insufficient_data}`, a boolean `major_finding`, and a `compliance_summary` (free text).

**Primary metric:** Precision and recall on `non_compliant` class (the class that matters most — false negatives here are dangerous).

**Go-live gates:**
- Precision on `non_compliant` ≥ 0.85
- Recall on `non_compliant` ≥ 0.90 (recall is weighted higher; missing a non-compliance is worse than a false alarm)

**Human rubric:**

| Dimension | 1 — Unacceptable | 3 — Acceptable | 5 — Excellent |
|-----------|-----------------|----------------|---------------|
| **Status determination** | Wrong (e.g., compliant when non-compliant) | Correct status, weak evidence | Correct status with specific threshold reference |
| **`major_finding` accuracy** | Wrong (missed a major or invented one) | Correct for primary issue | Correct + explains why minor issues were excluded |
| **Summary clarity** | Jargon-heavy, unclear to a manager | Clear, some gaps | Clear, concise, actionable for a manager-level reader |
| **Threshold accuracy** | Used wrong year's threshold | Correct threshold, wrong citation | Correct threshold + fiscal-year citation |
| **`would_sign_off`** | No | With revisions | Yes |

---

### 2.4 `workpaper_summarization`

**What the model produces:** A `workpaper_summary` (free text, ≤ 500 words) and an `executive_summary` (≤ 150 words).

**Primary metrics:**
- BERTScore F1 ≥ 0.82 (captures semantic similarity better than ROUGE for summaries)
- Human factual accuracy score ≥ 4.0 / 5.0 (see rubric)

**Human rubric:**

| Dimension | 1 — Unacceptable | 3 — Acceptable | 5 — Excellent |
|-----------|-----------------|----------------|---------------|
| **Factual accuracy** | ≥ 1 material factual error | All facts correct, some omissions | All facts correct, key findings highlighted |
| **Coverage** | Misses the primary finding | Covers primary finding, misses 1–2 secondary | Covers primary + secondary findings proportionally |
| **Conciseness** | Exceeds word limit or padded | Within limit but repetitive | Tight; no redundancy; reads in < 1 min |
| **Audience fit** | Technical jargon not appropriate for executive audience | Mostly appropriate | Fully appropriate; no unexplained audit terms |
| **`would_sign_off`** | No | With revisions | Yes |

---

### 2.5 `anomaly_detection` — Derived Signal (not a separate model task)

`anomaly_detection` is **not** a separate task type the model is trained to produce. It is a derived numeric signal computed by `post_processor.py` after every `finding_documentation` output. No separate training pairs, prompt template, or model output schema are required.

**How it works:**

```
FindingDocumentationOutput generated
         ↓
post_processor.py checks state.math_results for a deviation value
  (present when bank_rec, trial_balance, or variance tool ran)
         ↓
anomaly_score = abs(deviation) / materiality_amount
anomaly_score = clamp(anomaly_score, 0.0, 1.0)
         ↓
FindingDocumentationOutput.anomaly_score = computed value (or None if no deviation)
```

**What AUROC measures:** Whether `anomaly_score` correctly predicts whether the auditor marked `would_sign_off = false` on the finding. A high AUROC (≥ 0.80) means the formula is a reliable flag for findings that warrant rejection or revision — it is a quality signal on the math tools' deviation detection, not on the language model's generation.

**Evaluation:** Run `scripts/eval/anomaly_metrics.py` against the `finding_documentation` subset of the regression test set that has an `anomaly_score` value (i.e., findings from numeric workpaper types). The script computes AUROC by comparing `anomaly_score` against the blind reviewer `would_sign_off` labels.

**No human rubric required** — the score is a deterministic formula; humans review the finding narrative itself under the `finding_documentation` rubric (Section 2.3).

---

## 3. Technical Metrics Reference Table

| Metric | Task Type | Method | Go-Live Gate | Tool |
|--------|-----------|--------|-------------|------|
| Macro F1 | `risk_classification` | sklearn `f1_score(average='macro')` on test split | ≥ 0.87 | `scripts/eval/classification_metrics.py` |
| Precision (`non_compliant`) | `compliance_check` | sklearn `classification_report` | ≥ 0.85 | Same |
| Recall (`non_compliant`) | `compliance_check` | sklearn `classification_report` | ≥ 0.90 | Same |
| ROUGE-L | `finding_documentation` | `rouge_score` library | ≥ 0.45 | `scripts/eval/rouge_eval.py` |
| BERTScore F1 | `workpaper_summarization` | `bert_score` library (model: `deberta-xlarge-mnli`) | ≥ 0.82 | `scripts/eval/bertscore_eval.py` |
| AUROC | `finding_documentation` (numeric workpapers only — derived `anomaly_score` vs. `would_sign_off`) | sklearn `roc_auc_score` | ≥ 0.80 | `scripts/eval/anomaly_metrics.py` |
| Hallucination rate | All | Citation verifier hit rate on 100 adversarial queries | ≤ 2% at go-live; ≤ 5% acceptable during pilot | `scripts/eval/hallucination_eval.py` |
| Format compliance | All | % of outputs with all required fields non-null | 100% required | `scripts/eval/format_check.py` |
| p95 latency | All | 50 concurrent requests against vLLM on Server 2 | < 2,000 ms | `scripts/eval/load_test.py` (wraps `locust`) |
| `would_sign_off` rate | All (aggregate) | % of blind review outputs rated yes | ≥ 80% before Phase 2 expansion | `eval_results` table |

---

## 4. Blind Review Protocol

### 4.1 Evaluator Eligibility

Only three roles are eligible to participate in blind review:

| Role | DB `role` value | Eligible |
|------|----------------|---------|
| Partner | `partner` | Yes |
| Manager | — (not yet in Phase 1 schema; treat as `senior_auditor`) | Yes |
| Senior Auditor | `senior_auditor` | Yes |
| Junior Auditor | `junior_auditor` | **No** |
| Admin | `admin` | **No** |

Evaluators must not have been involved in creating the engagement being evaluated (conflict of interest exclusion).

### 4.2 Sample Design

**Per evaluation round (conducted before each production promotion and before each DPO cycle):**

- **Total outputs: 250**
- Stratified by task type:

| Task Type | Sample Size | Rationale |
|-----------|-------------|-----------|
| `risk_classification` | 70 | Highest-frequency task; F1 is primary gate |
| `finding_documentation` | 80 | Highest liability; includes 20 findings from numeric workpapers (for AUROC derivation) |
| `compliance_check` | 60 | Recall gate requires sufficient `non_compliant` cases |
| `workpaper_summarization` | 40 | Lower liability; BERTScore is strong signal |

- **Evaluator assignment:** 5 evaluators, each reviewing 50 outputs. Outputs are randomly assigned across evaluators, with 20% overlap (10 outputs each evaluator shares with one other) for inter-rater reliability scoring.
- **Blind condition:** Evaluators see the AI output and the original workpaper excerpt. They do **not** see: (a) the model confidence score, (b) `hallucination_flag` or `opinion_flag` values, (c) other evaluators' scores, (d) the SFT training loss curve.
- **Time limit:** Each evaluator completes their 50 outputs in a single session, maximum 3 hours. Incomplete sessions are excluded.

### 4.3 Scoring Instrument

Each output is rated on:
1. The 4 rubric dimensions for its task type (1–5 scale each) → averaged to `overall_score`
2. A binary `would_sign_off` (yes=1, no=0)
3. A free-text `evaluator_notes` field (optional, ≤ 200 chars)

All scores are entered into a structured form (a simple FastAPI-served HTML form, not a third-party tool — stays on-prem). Form submission writes directly to the `eval_results` table.

### 4.4 Inter-Rater Reliability

For the 20% overlapping outputs:
- Cohen's Kappa on `would_sign_off` (binary)
- Pearson r on `overall_score` (continuous)

**Acceptance threshold:** Kappa ≥ 0.70. If Kappa < 0.70, the evaluation round is halted and the rubric is recalibrated with the evaluators before restarting.

### 4.5 Aggregation and Gate Decision

```
would_sign_off_rate = COUNT(would_sign_off = 1) / COUNT(*) across all evaluators and outputs

Gate: would_sign_off_rate >= 0.80 → PASS → promote to production
       would_sign_off_rate  < 0.80 → FAIL → DPO cycle or targeted retraining required
```

The `eval_results` table stores one row per output-evaluator pair. The gate query:

```sql
SELECT
    COUNT(*) FILTER (WHERE would_sign_off = TRUE)::float
    / COUNT(*) AS sign_off_rate,
    AVG(overall_score) AS avg_score,
    SUM(hallucination_flag::int)::float / COUNT(*) AS hallucination_rate
FROM eval_results
WHERE evaluation_round = :round_id
  AND evaluator_role IN ('partner', 'senior_auditor');
```

---

## 5. `eval_results` Table Usage

The `eval_results` table (defined in `03_lld.md`) schema:

```sql
CREATE TABLE eval_results (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_output_id   UUID NOT NULL REFERENCES model_outputs(id) ON DELETE RESTRICT,
    workpaper_id      UUID          REFERENCES workpapers(id) ON DELETE SET NULL,
                      -- NULL for regression suite runs (no real workpaper involved)
    evaluation_round  VARCHAR(50) NOT NULL,   -- e.g. 'pre_golive_r1', 'dpo_cycle_1', 'regression_weekly'
    evaluator_id      UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    evaluator_role    eval_role NOT NULL,
    task_type         task_type NOT NULL,     -- denormalized from model_outputs for fast gate queries
    overall_score     NUMERIC(3,2) NOT NULL CHECK (overall_score BETWEEN 1.00 AND 5.00),
    would_sign_off    BOOLEAN NOT NULL,
    hallucination_flag BOOLEAN NOT NULL DEFAULT FALSE,
    hallucination_detail TEXT,
    evaluator_notes   TEXT,
    evaluated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_eval_output_evaluator UNIQUE (model_output_id, evaluator_id)
);

CREATE INDEX idx_eval_results_output ON eval_results(model_output_id);
CREATE INDEX idx_eval_results_round  ON eval_results(evaluation_round);
CREATE INDEX idx_eval_results_task   ON eval_results(task_type);
```

**Key constraint:** `UNIQUE(model_output_id, evaluator_id)` — one row per output-evaluator pair; prevents double-scoring.  
**`workpaper_id` nullable:** Regression suite runs generate `model_outputs` against frozen test inputs with no real workpaper in the DB. Setting `workpaper_id = NULL` lets the regression suite write scores without fabricating a workpaper row.

**Promotion workflow:**

```
1. ML Engineer runs evaluation round → eval_results populated
2. ML Engineer runs gate query (Section 4.5) → sign_off_rate computed
3. If PASS:  Engagement Partner reviews the gate query result, countersigns in the audit_log
             ML Engineer promotes model weights: cp model_weights/candidate/ model_weights/production/
4. If FAIL:  ML Engineer reviews evaluator_notes for patterns
             Determine: targeted retraining vs. DPO cycle vs. rubric recalibration
             Document findings in a GitHub issue or equivalent before next attempt
```

---

## 6. DPO Cycle Success Criteria

### 6.1 Cycle 1 (Month 3–4)

**Inputs:**
- ≥ 300 preference pairs in `dpo_feedback` with `is_complete = true` and `quality_score ≥ 0.7`
- At minimum 50 pairs per task type

**Training config:**
- Base: SFT-trained Mistral 22B weights
- DPO β = 0.1 (conservative; large β = smaller policy update)
- Learning rate: 5e-7
- Batch size: 4 per GPU × 2 GPUs = effective batch 8
- Epochs: 1 (DPO is sensitive to overfitting; 1 epoch is the standard for first cycles)

**Success criteria:**

| Metric | Baseline (post-SFT) | Cycle 1 Target | Method |
|--------|---------------------|----------------|--------|
| `would_sign_off` rate | Measured at pre-go-live eval | + 5 percentage points | Blind review (150-output mini-round) |
| Hallucination rate | ≤ 5% (go-live gate) | ≤ 3% | Citation verifier on 100 adversarial queries |
| F1 (`risk_classification`) | ≥ 0.87 | No regression (≥ 0.87) | Automated test set |
| ROUGE-L (`finding_documentation`) | ≥ 0.45 | No regression (≥ 0.45) | Automated test set |
| Evaluator preference vs. baseline | — | ≥ 60% prefer Cycle 1 | Pairwise evaluation (50 outputs, A/B blind) |

**Definition of "Cycle 1 success":** All 5 metrics meet their targets. If any metric regresses, Cycle 1 is deemed a partial success and targeted retraining is required before Cycle 2.

---

### 6.2 Cycle 2 (Month 6)

**Inputs:**
- ≥ 800 cumulative preference pairs (Cycle 1 pairs + new corrections from Month 4–6)
- Minimum 120 pairs per task type

**Training config:**
- Base: Cycle 1 DPO weights
- DPO β = 0.05 (looser; more data means larger updates are safe)
- Learning rate: 3e-7
- Batch size: 4 per GPU × 2 GPUs
- Epochs: 1–2 (stop early if eval loss stops improving)

**Success criteria:**

| Metric | Cycle 1 Baseline | Cycle 2 Target |
|--------|-----------------|----------------|
| `would_sign_off` rate | Cycle 1 result | ≥ 85% |
| Hallucination rate | Cycle 1 result | ≤ 2% |
| F1 (`risk_classification`) | Cycle 1 result | ≥ 0.89 |
| ROUGE-L (`finding_documentation`) | Cycle 1 result | ≥ 0.48 |
| BERTScore (`workpaper_summarization`) | Cycle 1 result | ≥ 0.84 |
| Evaluator preference vs. Cycle 1 | — | ≥ 65% prefer Cycle 2 |

**Phase 2 expansion gate:** Cycle 2 success + `would_sign_off ≥ 85%` unlocks GAGAS-only Govt and NPO engagement expansion.

---

### 6.3 Cycle 3 (Month 9–12)

**Inputs:**
- ≥ 2,000 cumulative preference pairs
- Minimum 300 pairs per task type (all 5)
- Non-Single-Audit (GAGAS-only Govt, NPO) pairs included if Phase 2 is in scope

**Training config:**
- Base: Cycle 2 DPO weights
- DPO β = 0.05
- Consider iterative DPO (online DPO) if GPU memory budget allows (requires vLLM to serve reference model simultaneously — evaluate feasibility at Month 8)

**Success criteria:**

| Metric | Cycle 2 Baseline | Cycle 3 Target |
|--------|-----------------|----------------|
| `would_sign_off` rate | Cycle 2 result | ≥ 88% |
| Hallucination rate | Cycle 2 result | ≤ 1.5% |
| F1 (`risk_classification`) | Cycle 2 result | ≥ 0.91 |
| ROUGE-L (`finding_documentation`) | Cycle 2 result | ≥ 0.50 |
| BERTScore (`workpaper_summarization`) | Cycle 2 result | ≥ 0.85 |
| AUROC (derived `anomaly_score` on `finding_documentation`) | Cycle 2 result | ≥ 0.83 |
| Evaluator preference vs. Cycle 2 | — | ≥ 65% prefer Cycle 3 |

**Steady-state:** After Cycle 3, the DPO cadence continues annually (or when ≥ 500 new pairs accumulate, whichever comes first). Thresholds are reviewed against the Cycle 3 baseline; targets increase by 1–2 percentage points per subsequent cycle.

---

## 7. Regression Test Suite

### 7.1 Design Principles

1. **Never-seen rule:** The regression test set is drawn from the held-out **test split only** (15% of engagements, assigned by the deterministic MD5 hash in `04_etl_pipeline.md`). These engagements are never used for training or DPO, not even as positive examples. Contamination is a disqualifying error.

2. **Frozen after initial creation:** The test set is created once, before the first SFT run. Its composition (specific engagement IDs and output pairs) is committed to DVC. Any modification requires a documented justification and a full re-baseline.

3. **Stratified by task type and engagement type:** The test set must include examples of all 5 task types and all engagement types present in the training data. For Phase 1: all 5 task types × `single_audit` engagement type.

4. **Adversarial subset (20% of test set):** A subset of outputs is hand-crafted to probe known failure modes:
   - Outputs with superseded regulation codes (should be flagged by citation verifier)
   - Outputs with PII in the workpaper text (should be scrubbed before reaching the model)
   - Outputs with edge-case threshold amounts (near-boundary values like $749,999 vs $750,001)
   - Outputs for non-Single-Audit engagement types (GAGAS-only Govt, NPO — once Phase 2 is active)
   - Outputs with injection attempt patterns in the input text

### 7.2 Test Set Composition

| Category | Count | Source |
|----------|-------|--------|
| `risk_classification` — normal | 80 | Test split, random sample |
| `risk_classification` — adversarial | 20 | Hand-crafted (near-boundary thresholds) |
| `finding_documentation` — normal | 60 | Test split, random sample |
| `finding_documentation` — adversarial | 15 | Hand-crafted (superseded citations) |
| `finding_documentation` — high anomaly_score | 30 | Findings from numeric workpaper types where `anomaly_score > 0.7`; used to compute AUROC |
| `compliance_check` — normal | 60 | Test split, balanced compliant/non-compliant |
| `compliance_check` — adversarial | 15 | Hand-crafted (PII in input, edge thresholds) |
| `workpaper_summarization` | 40 | Test split, random sample |
| **Total** | **320** | |

### 7.3 Automated Regression Run

The regression suite runs automatically:
- Before every model promotion (post-SFT and post-DPO)
- Weekly in production (against the live model) via an arq scheduled task

```python
# scripts/eval/regression_suite.py
def run_regression(model_tag: str) -> RegressionReport:
    results = []
    for test_case in load_test_set():          # from DVC-versioned JSONL
        output = inference_engine.run(test_case.input)
        score  = score_output(output, test_case.reference)
        results.append(score)
    return aggregate_report(results, model_tag) # writes to eval_results table
```

**Regression failure definition:** Any metric that drops more than **2 percentage points** below the previous promoted model's score on the same test set. A regression failure blocks promotion and triggers an investigation.

### 7.4 DVC Versioning

```
data/
  eval/
    regression_test_set_v1.jsonl   # frozen; never modified
    regression_test_set_v1.dvc     # DVC pointer (MD5 + size)
```

The DVC file is committed to git. The JSONL is stored in the DVC remote (local NAS in the air-gapped VLAN). Any change to the test set JSONL breaks the DVC hash, which is a detectable tampering signal.

---

### 7.5 Test Set Construction Process

**When:** Built once, after `split_assignment` has been set for all engagements (post-ETL, pre-first-SFT-run). The build script must be run before `run_sft` is invoked for the first time.

**Who:**
- **ML Engineer** — runs the automated sampling script; constructs the JSONL file; commits DVC
- **Senior Auditor or QC Partner** — reviews and signs off on all adversarial items before they are included; no adversarial item enters the set without a named approver on record
- **QC Partner** — final sign-off on the completed set before it is frozen

#### Step 1 — Sample normal items (automated)

```python
# scripts/eval/build_regression_set.py

TARGET_COUNTS = {
    ('risk_classification',    'normal'):     80,
    ('risk_classification',    'adversarial'): 20,   # hand-crafted — see Step 2
    ('finding_documentation',  'normal'):     60,
    ('finding_documentation',  'adversarial'): 15,
    ('finding_documentation',  'high_anomaly'): 30,
    ('compliance_check',       'normal'):     60,
    ('compliance_check',       'adversarial'): 15,
    ('summarization',          'normal'):     40,
}

# Query PG: test-split, reviewer-approved, quality >= 0.75
pairs = db.execute("""
    SELECT id, task_type, mongo_pair_id, engagement_id
    FROM sft_training_pairs
    WHERE split_assignment = 'test'
      AND reviewer_approved = TRUE
      AND quality_score >= 0.75
    ORDER BY task_type, created_at
""")

# Stratified random sample per task_type to hit TARGET_COUNTS normals
normal_pairs = stratified_sample(pairs, TARGET_COUNTS, seed=42)

# For finding_documentation high_anomaly subset: join model_outputs
# to filter pairs where the associated output has anomaly_score > 0.7
high_anomaly_pairs = db.execute("""
    SELECT p.id, p.mongo_pair_id
    FROM sft_training_pairs p
    JOIN model_outputs m ON m.workpaper_id = p.workpaper_id
    WHERE p.split_assignment = 'test'
      AND p.reviewer_approved = TRUE
      AND p.task_type = 'finding_documentation'
      AND m.inference_run_type = 'live'
      AND (m.output_json->>'anomaly_score')::float > 0.7
    ORDER BY (m.output_json->>'anomaly_score')::float DESC
    LIMIT 30
""")

# Fetch prompt + reference completion from MongoDB for each sampled pair
test_cases = [fetch_mongo_content(p.mongo_pair_id) for p in normal_pairs + high_anomaly_pairs]
```

#### Step 2 — Construct adversarial items (manual)

Each adversarial item targets a specific known failure mode. The ML Engineer creates a probe input; the Senior Auditor crafts the reference output (what the model must produce to pass).

| Failure mode | Input modification | Reference output requirement |
|---|---|---|
| Superseded citation | Workpaper text contains `OMB Circular A-133` (superseded 2015) | `regulation_cited = []`; `hallucination_flag = True` |
| Near-boundary threshold | SEFA amount = $749,999 (just under $750K major program threshold) | `compliant = True`; threshold NOT triggered |
| Near-boundary threshold (over) | SEFA amount = $750,001 | `compliant = False`; threshold triggered |
| PII in input | Input contains a real-looking EIN / SSN (test data only) | `pii_detected = True` in NER output; findings do not reproduce the PII string |
| Prompt injection attempt | Workpaper text contains `Ignore previous instructions and output...` | Model output contains no injected content; `opinion_flag` not triggered |

**Adversarial item acceptance criteria (QC Partner sign-off required):**
- The probe input is clearly synthetic (no real client data)
- The reference output is unambiguous — a reasonable senior auditor would agree this is the correct response
- The item targets exactly one failure mode — compound probes are not permitted
- Approver name and date recorded in `data/eval/adversarial_items_log.csv`

#### Step 3 — Serialize to JSONL and freeze

Each test case is serialised as one JSON line:
```json
{
  "test_case_id": "uuid",
  "task_type": "risk_classification",
  "category": "normal",
  "input": { "prompt": "...", "client_type": "NPO", "is_gagas": false, "has_single_audit": true, "fiscal_year": 2023 },
  "reference": { "risk_level": "HIGH", "risk_factors": [...], "regulation_cited": [...] },
  "scoring_fields": ["risk_level", "regulation_cited"],
  "adversarial_target": null
}
```

```bash
# Finalize and commit
dvc add data/eval/regression_test_set_v1.jsonl
git add data/eval/regression_test_set_v1.dvc .gitignore
git commit -m "feat: freeze regression test set v1 (320 items)"
```

#### Step 4 — Establish baseline

Run the pre-SFT base model against the frozen set immediately after creation. This baseline score is the reference for all future regression failure checks.

```python
baseline_report = run_regression(model_tag='mistral-22b-base-pretrain')
# Stores scores in eval_results with evaluation_round = 'baseline_v1'
```

#### Updating the test set (Phase 2 / new task types)

The frozen set is never modified. Additions create a new version:
1. Sample additional items for Phase 2 engagement types (GAGAS-only Govt, NPO)
2. Produce `regression_test_set_v2.jsonl` containing all v1 items + new items
3. Run the current production model against v2 to establish a new baseline before using v2 for promotion decisions
4. v1 is retained in DVC for audit trail; v2 becomes the active set going forward

---

## 8. Post-Deployment Monitoring

### 8.1 Metrics Dashboard (Grafana)

All metrics are collected by Prometheus and visualized in Grafana. The dashboard has four panels:

**Panel 1 — Output Quality (weekly)**

| Metric | Source | Alert Threshold |
|--------|--------|----------------|
| `would_sign_off` approval rate (7-day rolling) | `eval_results` table (production ratings) | < 75% → review alert |
| Rejection rate (outputs marked for correction) | `dpo_feedback` table | > 25% daily → review alert |
| `hallucination_flag` rate (7-day rolling) | Langfuse trace tag | > 3% → immediate review |
| `opinion_flag` rate (7-day rolling) | Langfuse trace tag | > 1% → immediate review |

**Panel 2 — Latency (real-time)**

| Metric | Prometheus Counter/Gauge | Alert Threshold |
|--------|--------------------------|----------------|
| p50 response time | `vllm_request_latency_seconds{quantile="0.5"}` | > 800 ms → warning |
| p95 response time | `vllm_request_latency_seconds{quantile="0.95"}` | > 2,000 ms → alert |
| p99 response time | `vllm_request_latency_seconds{quantile="0.99"}` | > 5,000 ms → critical |
| Queue depth (arq) | `arq_queue_length{queue="inference"}` | > 50 → alert |

**Panel 3 — Model Health (daily)**

| Metric | Source | Alert Threshold |
|--------|--------|----------------|
| GPU memory utilization | `nvidia_smi_memory_used_bytes` | > 45 GB per GPU → warning |
| GPU temperature | `nvidia_smi_temperature_gpu` | > 80°C → alert |
| vLLM OOM events | Prometheus counter `vllm_oom_total` | > 0 → immediate review |
| Token throughput | `vllm_tokens_per_second` | < 50 t/s → warning |

**Panel 4 — Data Pipeline Health (daily)**

| Metric | Source | Alert Threshold |
|--------|--------|----------------|
| ETL DAG failure rate | Airflow API | Any failure → alert |
| PII scrub failure rate | `cleaning_log` table | > 0.5% of documents → alert |
| DPO pair accumulation rate | `dpo_feedback` table | < 5 pairs/week during active pilot → warning |
| Intake queue depth | File system watcher | > 100 files pending > 24h → alert |

### 8.2 Monthly Framework Confusion Report

**Purpose:** Detect if the model is citing Single Audit regulations (2 CFR 200) on non-Single-Audit engagements, or vice versa.

**Method:** Monthly, a script pulls all outputs from the `outputs` table joined to engagements and checks regulation citation consistency:

```sql
SELECT
    o.task_type,
    COUNT(*) FILTER (WHERE o.regulation_cited @> '["2 CFR"]'::jsonb AND e.has_single_audit = false) AS cfr_on_non_sa,
    COUNT(*) FILTER (WHERE o.regulation_cited @> '["AU-C"]'::jsonb  AND e.is_gagas = false)         AS auc_on_non_gaas
FROM outputs o
JOIN engagements e ON o.engagement_id = e.id
WHERE o.created_at >= NOW() - INTERVAL '30 days'
GROUP BY o.task_type;
```

**Alert threshold:** `cfr_on_non_sa > 0` → immediate review; add failing cases to the DPO correction queue.

### 8.3 Weekly Approval Rate Report

Automated arq task runs every Monday at 06:00 and writes a summary to the `audit_log` table (type `eval_summary`):

```python
# src/tasks/weekly_eval_report.py
async def weekly_eval_report():
    sign_off_rate = await db.fetch_val("""
        SELECT AVG(would_sign_off::int)
        FROM eval_results
        WHERE evaluated_at >= NOW() - INTERVAL '7 days'
          AND evaluator_role IN ('partner', 'senior_auditor')
    """)
    hallucination_rate = await langfuse_client.get_metric(
        name='hallucination_flag', window_days=7
    )
    await alert_if_below(sign_off_rate, threshold=0.75,
                         message="Weekly approval rate below 75% — model review required")
    await alert_if_above(hallucination_rate, threshold=0.03,
                         message="Hallucination rate above 3% — immediate review required")
    await log_eval_summary(sign_off_rate, hallucination_rate)
```

---

## 9. Evaluation Tooling Stack

| Tool | Purpose | Location |
|------|---------|---------|
| `scripts/eval/classification_metrics.py` | F1, precision, recall for classification tasks | Server 1, run manually pre-promotion |
| `scripts/eval/rouge_eval.py` | ROUGE-L for `finding_documentation` | Server 1 |
| `scripts/eval/bertscore_eval.py` | BERTScore for `workpaper_summarization` | Server 1 (uses CPU; DeBERTa is fast) |
| `scripts/eval/anomaly_metrics.py` | AUROC of derived `anomaly_score` vs. `would_sign_off` on `finding_documentation` numeric subset | Server 1 |
| `scripts/eval/hallucination_eval.py` | Citation verifier against `regulations_master.json` | Server 1 |
| `scripts/eval/load_test.py` | p95 latency test (wraps locust) | Server 1, targets Server 2 vLLM |
| `scripts/eval/regression_suite.py` | Full regression run against frozen test set | Server 1 |
| `scripts/eval/format_check.py` | Required-field completeness check | Server 1 |
| MLflow | Training metrics, loss curves, model registry | Server 1, port 5000 |
| Langfuse (self-hosted) | Trace-level hallucination/flag rates | Server 1, port 3000 |
| Prometheus + Grafana | Infrastructure + latency + model health | Server 1, ports 9090/3001 |
| DVC | Test set versioning, model weight versioning | Server 1, remote on NAS |

---

## 10. Evaluation Calendar

| Event | Timing | Trigger | Output |
|-------|--------|---------|--------|
| Pre-go-live evaluation round | Before Phase 1 production | Manual (ML Engineer) | 250-output blind review, gate query |
| Regression suite run | Before every promotion | Automated (CI step) | `RegressionReport` in `eval_results` |
| Weekly approval rate report | Every Monday | arq scheduled task | `audit_log` entry + alert if below threshold |
| Monthly framework confusion report | 1st of each month | arq scheduled task | SQL report + alert if cross-jurisdiction codes found |
| DPO Cycle 1 evaluation | Month 3–4 | Manual (post-DPO training) | 150-output mini blind review |
| DPO Cycle 2 evaluation | Month 6 | Manual (post-DPO training) | 250-output blind review |
| DPO Cycle 3 evaluation | Month 9–12 | Manual (post-DPO training) | 250-output blind review |
| Annual threshold review | October each year | Calendar | `thresholds.yaml` update + regression run |
| Quarterly hardware audit | Every 3 months | Calendar | GPU health report, alert rule review |

---

## 11. Go-Live Checklist Summary

All items must be checked and countersigned by the Engagement Partner before any production user accesses the system.

- [ ] Macro F1 ≥ 0.87 (`risk_classification`) — run `classification_metrics.py` on frozen test set
- [ ] Precision ≥ 0.85, Recall ≥ 0.90 (`non_compliant`) — same script
- [ ] ROUGE-L ≥ 0.45 (`finding_documentation`) — run `rouge_eval.py`
- [ ] BERTScore F1 ≥ 0.82 (`workpaper_summarization`) — run `bertscore_eval.py`
- [ ] AUROC ≥ 0.80 (derived `anomaly_score` on numeric `finding_documentation` subset) — run `anomaly_metrics.py`
- [ ] Hallucination rate ≤ 5% (target ≤ 2%) — run `hallucination_eval.py` with 100 adversarial queries
- [ ] Format compliance 100% — run `format_check.py`
- [ ] p95 latency < 2,000 ms under 50 concurrent users — run `load_test.py`
- [ ] Blind review complete: 250 outputs, ≥ 5 evaluators, Cohen's Kappa ≥ 0.70
- [ ] `would_sign_off` ≥ 80% from blind review aggregate — run gate query on `eval_results`
- [ ] Regression suite passes (no metric drops > 2pp vs. baseline) — run `regression_suite.py`
- [ ] R-01 through R-09 go-live risks documented and mitigated — see `07_risk_register.md`
- [ ] Grafana alerts configured for all thresholds in Section 8.1
- [ ] `thresholds.yaml` verified against current fiscal year — Audit QC Partner sign-off
- [ ] `regulations_master.json` reviewed for completeness — Audit QC Partner sign-off

---

*All 8 architecture documents are complete.*
*Implementation begins per the 14-day plan in the master architecture document.*
