# 09 — R7 Classifier Architecture: Locked Invariants

**Status:** Active  
**Last updated:** 2026-05-15  
**Scope:** `raw_to_training_pair/field_classifier.py`, `claim_mapper.py` (R7-B section), `completion_renderer.py`, `completion_drafter.py` (`score_r7` / `set_review_gate_r7`), `pipeline/pipeline.py` (`_process_single_variant_r7`)

---

## Component map

| Layer | Module | Responsibility |
|---|---|---|
| R7-A | `field_classifier.py` | Gemma + outlines constrained decoding → raw `field_states` dict (`absent`/`present`/`uncertain`) |
| R7-B | `claim_mapper.py` (ClassificationValidator) | Schema check (Signal B) → Pass 2 rule-based verifier → corrected `field_states` → cross-pass agreement (Signal C) → classification confidence (Signal A) → `ClassificationSignals` dataclass |
| R7-C | `completion_renderer.py` | Pure Python, deterministic → `FIELD_TO_SOP` lookup (compile-time table) → absent fields → numbered findings → uncertain fields → metadata only |
| R7-D | `completion_drafter.py` | Scoring → rubric (structural quality) → 3 independent grounding signals from `ClassificationSignals` → uncertainty penalty (Tier 1 blocks, Tier 2 discounts) → no hallucination penalty (structurally impossible) |
| R7-E | `pipeline.py` | Wiring: classifier → validator → renderer → drafter → `FIELD_TO_SOP` built at startup, passed to renderer |

---

## Infrastructure reuse invariant

**Rule:** Pass 2 (`validate_classification()` in `claim_mapper.py`) must not introduce new infrastructure. Every capability it needs is already present in `claim_mapper.py`.

| Pass 2 need | Existing component |
|---|---|
| Alias evidence patterns | `_load_reverse_aliases()` in `claim_mapper.py` |
| Embedding model | `_get_semantic_model()` in `claim_mapper.py` |
| Description vectors | `_get_desc_embeddings()` in `claim_mapper.py` |
| Field tiers (Tier 1 / Tier 2 distinction) | `_load_tier1_fields()` in `claim_mapper.py` |

**Why:** `claim_mapper.py` already owns alias lookup, embedding inference, and field tier metadata for the legacy grounding pipeline. Pass 2 is a natural extension of that role — it is the `ClassificationValidator`, not a separate module. Adding a new module for the same infrastructure would create two caches for the same data, two load paths for the same model, and two alias indexes that can drift out of sync.

**Corollary:** If a future Pass 2 capability requires infrastructure not in `claim_mapper.py`, the infrastructure should be added to `claim_mapper.py` (or its existing helpers), not to a new file.

---

## Purpose

The R7 architecture eliminates hallucinated findings by separating classification from narrative generation. Gemma acts as a constrained classifier only. All narrative is rendered deterministically from precompiled SOP mappings.

These invariants are non-negotiable. Any change that violates them requires an explicit architectural decision with a documented rationale.

---

## Invariant 1 — The classifier is a function, not a generator

**Rule:** `field_classifier.classify()` produces `{field → absent|present|uncertain}` for every canonical field. It produces no narrative text, no severity labels, no citation strings, no free-form output.

**Enforcement:**
- Output schema is a Pydantic model built at runtime via `create_model()` from the canonical field list. The schema enforces that every field is a required key with an enum value of `["absent", "present", "uncertain"]`.
- Constrained decoding is applied via `outlines.from_ollama()` (outlines ≥ 1.3.0), which passes the JSON schema to ollama's `format` parameter. This uses llama.cpp's GBNF grammar internally — token-level enforcement, not post-hoc filtering.
- The generator is never called with `temperature > 0.0`. `_TEMPERATURE = 0.0` is a module-level constant, not a parameter.

**Why:** Any narrative capability in the classifier reintroduces the hallucination surface that R7 was designed to eliminate. A MISANCHORED claim cannot be generated if the model never generates claims.

---

## Invariant 2 — Full keyspace enforcement

**Rule:** `|field_states| == |canonical_fields|`. The classification output must contain exactly one state for every canonical field. Partial key omission is a schema violation, not a recoverable condition.

**Enforcement:**
- The dynamic Pydantic model marks all fields as required. outlines/pydantic will reject any response that omits a key.
- `validate_classification()` checks `|field_states|` against `|canonical_fields|` after parsing. If `|field_states| < |canonical_fields|`, `signals.schema_violation_count` is incremented and `signals.structural_valid = False`.
- Schema violations are surfaced as `ClassificationSignals.drift_count` (unknown keys) and `schema_violation_count` (missing keys). They are never silently corrected by filling in defaults.

**Why:** Silent correction would mask model degradation. A missing key means the classifier skipped a field — that is a signal worth alarming on, not patching over.

---

## Invariant 3 — Pass 2 asymmetry: downgrade only, never upgrade

**Rule:** Pass 2 (`validate_classification()`) can downgrade `present → uncertain` when workpaper evidence is insufficient. Pass 2 **cannot** promote `uncertain → present` regardless of embedding similarity or Llama response.

**Evidence tiers (Pass 2 only):**

| Evidence found | Result |
|---|---|
| Keyword alias match in workpaper | `strong_present` (stays `present`) |
| Embedding similarity ≥ threshold, no alias | `provisional_present` (stays `present`, flagged) |
| No match at either layer | Stays `uncertain` (cannot be promoted) |

**Why:** False negatives (missing a field) are preferable to false positives (fabricating presence). A `present` classification that goes unchallenged will suppress a finding in the rendered completion. An `uncertain` classification triggers an auditor review flag (`tier1_uncertain_block`). The asymmetry is intentional — prefer surfacing uncertainty over asserting confidence the evidence does not support.

---

## Invariant 4 — Three independent grounding signals, never pre-summed

**Rule:** The three grounding signals from `ClassificationSignals` are stored separately and evaluated separately by `score_r7()`. They are never collapsed into a single composite score before scoring.

**Signals:**

| Signal | Field | Measures |
|---|---|---|
| A | `uncertain_rate` | Classifier confidence — fraction of fields uncertain |
| B | `structural_valid`, `drift_count`, `schema_violation_count` | Schema integrity — did the classifier respect the contract |
| C | `pass2_rejection_rate` | Pass 2 agreement — how often Pass 1 `present` was downgraded |

**Why:** Pre-summing destroys information. A pair with Signal A = 0.0 (all certain) but Signal B = False (schema drift) is structurally broken. A pair with Signal B = True but Signal C = 0.5 (half of present fields lack workpaper evidence) is factually fragile. These are different failure modes and should be visible as such in the review queue metadata.

**Downstream use:** `score_r7()` applies penalties for each signal independently. The metadata field `classification_signals` stores all three in the JSONL pair so auditors can inspect the breakdown.

---

## Invariant 5 — FIELD_TO_SOP table is compile-time and versioned

**Rule:** The mapping from canonical field → SOP section is built once at startup from `field_tiers.yaml` and `sop_field_classes.yaml`. It is never computed at render time. Every pair carries `sop_mapping_version` in metadata.

**Version format:** `YYYY-MM-DD_{8-hex-chars}` where the hex suffix is the first 8 characters of the SHA256 hash of both config files' content concatenated. Example: `2026-05-15_b33e4d14`.

**Enforcement:**
- `build_sop_mapping_table(client_type)` is `@lru_cache(maxsize=4)` — one table per client type, built once per process.
- `sop_mapping_version()` is `@lru_cache(maxsize=4)` — recomputed only when the cache is invalidated.
- `render_completion()` receives `sop_table` and `mapping_version` as explicit parameters. It never reads config files directly.

**Why:** Changing the SOP mapping mid-run would create pairs with inconsistent citation patterns in the same JSONL file. The version hash in metadata lets the dataset observer detect when a JSONL file spans multiple mapping versions — a signal that the SOP config changed between pipeline runs.

---

## Invariant 6 — Schema drift alarms are non-suppressible

**Rule:** When the classifier returns a field key not in `canonical_fields` (unknown field) or omits a field that is in `canonical_fields` (missing field), the anomaly is recorded in `ClassificationSignals` and propagated to the pair metadata. It is never silently discarded.

**Fields:**
- `signals.drift_count` — count of unknown keys returned by the classifier
- `signals.schema_violation_count` — count of canonical fields absent from the output
- `signals.structural_valid` — `False` if either count > 0

**Downstream effect:** `structural_valid = False` applies `structural_invalid_penalty` (0.20) in `score_r7()`, which typically pushes the pair below the quality gate and forces auditor review.

**Why:** Schema drift is the earliest indicator of model degradation or config/model version mismatch. Suppressing it hides the signal until a more expensive downstream failure surfaces it.

---

## Invariant 7 — Tier 1 uncertain is a hard block

**Rule:** Any Tier 1 field in `uncertain_fields` after Pass 2 sets `review_confidence = 0.0` unconditionally. No rubric score, no partial credit.

**Threshold config key:** `review.r7.tier1_uncertain_block: true`

**Why:** Tier 1 fields (`engagement_partner`, `fiscal_year_end`, `engagement_decision`, `opinion_type`, `report_date`, `includes_gagas`, `includes_single_audit`, `peer_review_date`) are the structural spine of every audit completion. A training pair where one of these fields is uncertain cannot safely teach the model — the ground truth itself is ambiguous. The pair must go to human review before it enters JSONL.

**Tier 2 uncertain:** Soft penalty only (`tier2_uncertain_penalty: 0.05` per field, deducted from rubric score). Pair is included with `label_confidence: low` metadata.

---

## Technology constraints (do not change without design review)

| Component | Choice | Rationale |
|---|---|---|
| Constrained decoding | `outlines.from_ollama()` ≥ 1.3.0 | Uses ollama's `format` parameter → llama.cpp GBNF grammar. Token-level enforcement. No double model loading. |
| Pass 1 model | `gemma3:12b` (via ollama) | Same model already used for extraction. No additional infra. |
| Pass 2 Layer A | Keyword alias lookup (`_load_field_aliases()`) | Zero-cost, deterministic, cache-warm after first call. |
| Pass 2 Layer B | Embedding cosine similarity (sentence-transformers) | Semantic coverage without LLM inference. Threshold: `pass2_embedding_threshold: 0.55`. |
| Pass 2 Layer C | `llama3.1:8b` via `ollama.chat()` | Better instruction following than Gemma for evidence spot-check. `temperature=0.0, num_predict=10`. Fires only when Layer B is inconclusive. |
| SOP narrative | `completion_renderer.py` — static `_FIELD_RISK_TEXT` dict | Deterministic. Zero variance across runs. Auditors see identical text for identical fields. |

**Do not introduce free-text generation in the classifier path.** If the rendering templates need updating, edit `_FIELD_RISK_TEXT` in `completion_renderer.py` and the `FIELD_TO_SOP` table — do not add a generation call.
