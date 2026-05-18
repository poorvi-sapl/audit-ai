"""
auditai_data_normalization/confidence.py
==========================================
Phase A2 — tier-based confidence scoring.

Replaces the flat weighted-mean approach with a two-component formula
that reflects real audit criticality as defined in config/field_tiers.yaml.

Tier structure (post audit-team review, Phase A1 sign-off)
-----------------------------------------------------------
Tier 1 — CRITICAL (8 fields):
    client_name, fiscal_year_end, engagement_decision, engagement_partner,
    audit_type, includes_gagas, includes_single_audit, reporting_framework

Tier 2 — IMPORTANT (8 fields):
    document_reference, includes_gaas_audit, includes_grant_compliance,
    preparation_date, partner_sign_date, ein, includes_nonattest_services,
    financial_statement_use

Tier 3 — INFORMATIONAL:
    All financial line items, tabular column labels, admin metadata.
    Zero weight. Zero penalty. Never affects aggregate score.

Scoring formula (Phase A2)
--------------------------
    base_score = (tier1_found / TIER1_TOTAL) * 0.70
               + (tier2_found / TIER2_TOTAL) * 0.30

    Floors applied after base_score:
        tier1_found >= 3  →  score = max(score, 0.55)
        tier1_found >= 5  →  score = max(score, 0.65)
        tier1_found >= 7  →  score = max(score, 0.75)
        tier1_found == 8  →  score = max(score, 0.82)

    Tier 3 fields never enter the formula — not counted in found or total.

Gate thresholds (Phase A4)
--------------------------
    extraction_confidence >= 0.50  →  proceed to completion drafter
    extraction_confidence >= 0.70  →  passes quality gate, eligible for JSONL
    extraction_confidence <  0.50  →  LLM fallback triggered (Phase B)
    extraction_confidence <  0.70  →  flagged; auditor review required

Per-field scoring table (unchanged from Phase 1)
-------------------------------------------------
    Extractors ran | Agreement    | Score
    -------------- | ------------ | -----
    3              | all agree    | 1.0
    3              | two agree    | 0.7
    3              | all disagree | 0.2
    3              | all empty    | 0.0
    2              | both agree   | 0.9
    2              | both disagree| 0.3
    2              | both empty   | 0.0
    1              | has value    | 0.6
    1              | empty        | 0.0
    0 / all None   | —            | 0.0

Public API
----------
    score_confidence(field_name, val_a, val_b, val_c) -> float
    score_fields(fields_dict) -> dict[str, float]
    score_record(per_field_scores) -> float          ← rewritten for A2
    summarise(per_field_scores) -> ConfidenceSummary ← extended for A2
    load_tiers(yaml_path) -> TierConfig              ← new in A2
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Tier configuration — loaded from field_tiers.yaml
# ---------------------------------------------------------------------------

@dataclass
class TierConfig:
    """
    Resolved tier membership loaded from config/field_tiers.yaml.

    Attributes
    ----------
    tier1 : frozenset[str]
        Critical fields. Drive the confidence floor logic.
    tier2 : frozenset[str]
        Important fields. Contribute 30% of the base score.
    tier3 : frozenset[str]
        Informational fields. Zero weight, zero penalty.
    """
    tier1: frozenset[str]
    tier2: frozenset[str]
    tier3: frozenset[str]

    def tier_of(self, field_name: str) -> str:
        """Return 'tier1', 'tier2', 'tier3', or 'unknown'."""
        if field_name in self.tier1:
            return "tier1"
        if field_name in self.tier2:
            return "tier2"
        if field_name in self.tier3:
            return "tier3"
        return "unknown"

    def is_scored(self, field_name: str) -> bool:
        """True for Tier 1 and Tier 2 fields — the only ones that affect score."""
        return field_name in self.tier1 or field_name in self.tier2


# Module-level cache so yaml is only read once per process.
_TIER_CACHE: TierConfig | None = None

# Default path — resolved relative to this file's location.
_DEFAULT_YAML = Path(__file__).parent.parent / "config" / "field_tiers.yaml"


def load_tiers(yaml_path: str | Path | None = None) -> TierConfig:
    """
    Load tier membership from field_tiers.yaml.

    Results are cached after the first call. Pass yaml_path explicitly
    in tests to override the default location.

    Parameters
    ----------
    yaml_path : str | Path | None
        Path to field_tiers.yaml. Defaults to config/field_tiers.yaml
        relative to the project root.

    Returns
    -------
    TierConfig
    """
    global _TIER_CACHE
    if _TIER_CACHE is not None and yaml_path is None:
        return _TIER_CACHE

    path = Path(yaml_path) if yaml_path else _DEFAULT_YAML

    if not path.exists():
        # Graceful fallback — use hardcoded tiers so pipeline never crashes
        # if config file is missing. Log a warning.
        import warnings
        warnings.warn(
            f"field_tiers.yaml not found at {path}. "
            "Using hardcoded fallback tiers. Run `make install` to set up config.",
            stacklevel=2,
        )
        return _hardcoded_fallback_tiers()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def _extract_fields(tier_list: list) -> frozenset[str]:
        """Handle both {field: name, reason: ...} and bare {field: name} entries."""
        names = set()
        for item in (tier_list or []):
            if isinstance(item, dict):
                names.add(item["field"])
            elif isinstance(item, str):
                names.add(item)
        return frozenset(names)

    config = TierConfig(
        tier1=_extract_fields(raw.get("tier1", [])),
        tier2=_extract_fields(raw.get("tier2", [])),
        tier3=_extract_fields(raw.get("tier3", [])),
    )

    if yaml_path is None:
        _TIER_CACHE = config

    return config


def _hardcoded_fallback_tiers() -> TierConfig:
    """
    Fallback tier config matching the audit-team-approved field_tiers.yaml.
    Used only when the yaml file cannot be found (e.g. first run before setup).
    """
    return TierConfig(
        tier1=frozenset({
            "client_name",
            "fiscal_year_end",
            "engagement_decision",
            "engagement_partner",
            "audit_type",
            "includes_gagas",
            "includes_single_audit",
            "reporting_framework",       # promoted T2 → T1 at audit team review
        }),
        tier2=frozenset({
            "document_reference",
            "includes_gaas_audit",
            "includes_grant_compliance",
            "preparation_date",
            "partner_sign_date",
            "ein",
            "includes_nonattest_services",
            "financial_statement_use",
        }),
        tier3=frozenset({
            # financial line items, tabular labels, extraction metadata
            "document_title", "engagement_code", "fiscal_year", "report_date",
            "client_type", "client_address", "predecessor_auditor",
            "total_assets", "total_current_assets", "total_noncurrent_assets",
            "cash_and_investments", "accounts_receivable", "prepaid_expenses",
            "security_deposits", "intangible_assets", "right_of_use_assets",
            "total_liabilities", "total_current_liabilities", "total_noncurrent_liabilities",
            "accounts_payable", "accrued_liabilities", "operating_lease_liabilities",
            "long_term_loan", "net_assets", "net_assets_with_restrictions",
            "total_revenue", "total_expenses", "service_revenue", "grant_revenue",
            "program_expenses", "management_general_expenses", "salaries_wages",
            "fringe_benefits", "professional_fees", "net_income",
            "net_cash_operating", "net_cash_investing", "net_cash_financing",
            "cash_beginning_of_year", "cash_end_of_year",
            "account_code", "account_name", "amount", "debit", "credit",
            "transaction_date", "row_type", "sheet_name",
            "source_year", "extraction_method", "confidence_score",
            # demoted at audit team review (Phase A1)
            "preparer_id", "reviewer_id", "opinion_type",
        }),
    )


# ---------------------------------------------------------------------------
# Normalisation helpers  (unchanged from Phase 1)
# ---------------------------------------------------------------------------

def _normalise(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).replace("\u2002", " ").replace("\u200b", "").replace("\u00a0", " ")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # Normalize date formats so MM/DD/YYYY == YYYY-MM-DD
    date_match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if date_match:
        m, d, y = date_match.groups()
        s = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    return s


# Placeholder patterns produced by PII scrubber or pre-redacted source docs.
# Values matching these patterns are treated as missing for confidence scoring —
# a field containing [CLIENT_ENTITY] is no more useful than an empty field.
_PLACEHOLDER_RE = re.compile(
    r"^\[("
    r"client_entity|preparer|date|ein|ssn|itin|routing_num|account_num"
    r"|name|address|phone|email|organization|entity|person"
    r"|redacted|pii_removed|confidential"
    r")\]$",
    re.IGNORECASE,
)


def _is_placeholder(val: Any) -> bool:
    """Return True if value is a PII placeholder like [CLIENT_ENTITY]."""
    if val is None:
        return False
    s = str(val).strip()
    return bool(_PLACEHOLDER_RE.match(s))


def _is_empty(val: Any) -> bool:
    """Return True if value is empty OR a PII placeholder."""
    return _normalise(val) == "" or _is_placeholder(val)


# ---------------------------------------------------------------------------
# Single-field scoring  (unchanged from Phase 1)
# ---------------------------------------------------------------------------

def score_confidence(
    field_name: str,
    val_a: Any = None,
    val_b: Any = None,
    val_c: Any = None,
) -> float:
    """
    Score one field from up to three extractor outputs.

    Parameters
    ----------
    field_name : str
        Field name (used only for logging — scoring is value-based).
    val_a, val_b, val_c : Any
        Extractor outputs. Pass None for extractors that did not run.

    Returns
    -------
    float in [0.0, 1.0]
    """
    na = _normalise(val_a)
    nb = _normalise(val_b)
    nc = _normalise(val_c)

    a_ran = val_a is not None
    b_ran = val_b is not None
    c_ran = val_c is not None

    a_empty = _is_empty(val_a)
    b_empty = _is_empty(val_b)
    c_empty = _is_empty(val_c)

    ran_count = sum([a_ran, b_ran, c_ran])

    if ran_count == 0:
        return 0.0

    if ran_count == 1:
        val = na if a_ran else (nb if b_ran else nc)
        empty = a_empty if a_ran else (b_empty if b_ran else c_empty)
        return 0.0 if empty else 0.6

    if ran_count == 2:
        vals, empties = [], []
        for ran, v, e in [(a_ran, na, a_empty), (b_ran, nb, b_empty), (c_ran, nc, c_empty)]:
            if ran:
                vals.append(v)
                empties.append(e)
        if all(empties):
            return 0.0
        return 0.9 if vals[0] == vals[1] else 0.3

    # All three ran
    if a_empty and b_empty and c_empty:
        return 0.0
    if na == nb == nc:
        return 1.0
    if (na == nb) or (nb == nc) or (na == nc):
        return 0.7
    return 0.2


# ---------------------------------------------------------------------------
# Batch field scoring  (unchanged from Phase 1)
# ---------------------------------------------------------------------------

def score_fields(fields: dict[str, list[Any]]) -> dict[str, float]:
    """
    Score a batch of fields.

    Parameters
    ----------
    fields : dict[str, list[Any]]
        {field_name: [val_a, val_b, val_c]}. Shorter lists padded with None.

    Returns
    -------
    dict[str, float]
    """
    return {
        fname: score_confidence(fname, *(vals + [None, None, None])[:3])
        for fname, vals in fields.items()
    }


# ---------------------------------------------------------------------------
# B3 — LLM confidence calibration
# ---------------------------------------------------------------------------

# Calibrated scores for LLM-extracted fields
# (replaces the flat 0.6 single-source score from score_confidence)
_LLM_ONLY_BASE        = 0.65   # LLM found it, no deterministic value at all
_LLM_MATCHES_DET      = 0.85   # LLM value matches at least one deterministic value
_LLM_CONTRADICTS_DET  = 0.30   # LLM value contradicts deterministic (flag for review)
_LLM_CONFIDENT_BONUS  = 0.05   # added when FieldResult.llm_confident=True

# Tier 1 hard cap for LLM-only values (no deterministic corroboration)
# Prevents high-stakes fields from auto-passing the quality gate on LLM alone
_TIER1_LLM_ONLY_CAP   = 0.72


def calibrate_llm_scores(
    per_field_scores: dict[str, float],
    fields_for_scoring: dict[str, list[Any]],
    fallback_results: dict,           # dict[str, FieldResult] from llm_extractor
    tiers: "TierConfig",
) -> tuple[dict[str, float], list[str]]:
    """
    Post-process per-field scores for fields filled by LLM fallback.

    Called by normalize.py after B2 merges fallback results and re-runs
    score_fields(). Replaces the generic 0.6 single-source score with
    calibrated scores that reflect whether the LLM corroborated or
    contradicted the deterministic extractor.

    Calibration rules (Phase B3)
    ----------------------------
    LLM value matches deterministic (slot A or B):
        score = 0.85  (+0.05 if llm_confident=True → 0.90)

    LLM-only value (slots A and B both None), llm_confident=True:
        score = 0.70  (confident single source)
        Tier 1 cap: max 0.72 — never auto-passes quality gate on LLM alone

    LLM-only value, llm_confident=False:
        score = 0.65  (uncertain single source)
        Tier 1 cap: max 0.65

    LLM contradicts deterministic (all three different):
        score stays at 0.20 (from score_confidence)
        field added to flagged list for auditor review

    Parameters
    ----------
    per_field_scores : dict[str, float]
        Output of score_fields() after B2 merge. Modified in place.
    fields_for_scoring : dict[str, list[Any]]
        {field: [val_a, val_b, val_c]} — val_c is the LLM value.
    fallback_results : dict[str, FieldResult]
        Output of extract_all_fields() from llm_extractor.py.
    tiers : TierConfig

    Returns
    -------
    tuple[dict[str, float], list[str]]
        (calibrated_scores, newly_flagged_fields)
        newly_flagged_fields: fields where LLM contradicted deterministic.
    """
    calibrated = dict(per_field_scores)
    newly_flagged: list[str] = []

    for fname, fresult in fallback_results.items():
        if not fresult.found:
            continue   # LLM returned null — no calibration needed

        vals = fields_for_scoring.get(fname, [None, None, None])
        val_a, val_b, val_c = (vals + [None, None, None])[:3]

        det_a = _normalise(val_a) if val_a is not None else None
        det_b = _normalise(val_b) if val_b is not None else None
        llm_v = _normalise(val_c) if val_c is not None else None

        has_det_a = det_a is not None and det_a != ""
        has_det_b = det_b is not None and det_b != ""
        has_det   = has_det_a or has_det_b

        in_tier1  = fname in tiers.tier1

        if not has_det:
            # LLM-only — no deterministic value to compare against
            if fresult.llm_confident:
                new_score = _LLM_ONLY_BASE + _LLM_CONFIDENT_BONUS  # 0.70
            else:
                new_score = _LLM_ONLY_BASE                          # 0.65

            # Tier 1 cap — never let LLM-only cross quality gate unchecked
            if in_tier1:
                new_score = min(new_score, _TIER1_LLM_ONLY_CAP)    # max 0.72

            calibrated[fname] = round(new_score, 4)

        else:
            # Deterministic value(s) exist — check agreement
            # Use normalised comparison (same logic as score_confidence)
            matches = (
                (has_det_a and llm_v is not None and llm_v == det_a) or
                (has_det_b and llm_v is not None and llm_v == det_b)
            )
            if matches:
                new_score = _LLM_MATCHES_DET                        # 0.85
                if fresult.llm_confident:
                    new_score += _LLM_CONFIDENT_BONUS               # 0.90
                calibrated[fname] = round(new_score, 4)
            else:
                # LLM contradicts deterministic — flag for auditor review
                # score_confidence already returns 0.2 for all-disagree
                # we leave the score at 0.2 and add to flagged list
                newly_flagged.append(fname)
                # score stays as set by score_confidence (0.2)

    return calibrated, newly_flagged


# ---------------------------------------------------------------------------
# Record-level aggregation  (Phase A2 rewrite)
# ---------------------------------------------------------------------------

# Floor table: (min_tier1_found, floor_score)
# Applied in order — highest applicable floor wins.
_TIER1_FLOORS = [
    (8, 0.82),
    (7, 0.75),
    (5, 0.65),
    (3, 0.55),
]

# Weights for the two-component formula.
_TIER1_WEIGHT = 0.70
_TIER2_WEIGHT = 0.30


def score_record(
    per_field_scores: dict[str, float],
    tiers: TierConfig | None = None,
) -> float:
    """
    Aggregate per-field scores into a single record-level confidence score.

    Phase A2 formula:
        base = (tier1_found / TIER1_TOTAL) * 0.70
             + (tier2_found / TIER2_TOTAL) * 0.30

    Floors applied after base:
        tier1_found >= 3  →  max(base, 0.55)
        tier1_found >= 5  →  max(base, 0.65)
        tier1_found >= 7  →  max(base, 0.75)
        tier1_found == 8  →  max(base, 0.82)

    Tier 3 fields are ignored entirely — not counted, not penalised.
    Fields not in any tier (unknown) are also ignored.

    Parameters
    ----------
    per_field_scores : dict[str, float]
        Output of score_fields(). Keys are field names, values are [0.0, 1.0].
    tiers : TierConfig | None
        Tier membership. Loaded from field_tiers.yaml if not supplied.

    Returns
    -------
    float
        Score in [0.0, 1.0], rounded to 4 decimal places.
    """
    if not per_field_scores:
        return 0.0

    if tiers is None:
        tiers = load_tiers()

    tier1_total = len(tiers.tier1)
    tier2_total = len(tiers.tier2)

    # Sum scores for each tier, count how many were found (score > 0)
    tier1_score_sum = 0.0
    tier1_found = 0
    tier2_score_sum = 0.0
    tier2_found = 0

    for fname, score in per_field_scores.items():
        if fname in tiers.tier1:
            tier1_score_sum += score
            if score > 0.0:
                tier1_found += 1
        elif fname in tiers.tier2:
            tier2_score_sum += score
            if score > 0.0:
                tier2_found += 1
        # tier3 and unknown: skip entirely

    # Fields not present in per_field_scores count as 0 (missing penalty)
    # — already handled by dividing by the full tier totals below.

    t1_component = (tier1_score_sum / tier1_total) * _TIER1_WEIGHT if tier1_total else 0.0
    t2_component = (tier2_score_sum / tier2_total) * _TIER2_WEIGHT if tier2_total else 0.0
    base = t1_component + t2_component

    # Apply tier1 floors
    for min_found, floor_val in _TIER1_FLOORS:
        if tier1_found >= min_found:
            base = max(base, floor_val)
            break

    return round(min(base, 1.0), 4)


# ---------------------------------------------------------------------------
# ConfidenceSummary  (extended for Phase A2)
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceSummary:
    """
    Full confidence breakdown for one document.

    Phase A2 additions:
        tier1_found / tier1_total  — how many critical fields were extracted
        tier2_found / tier2_total  — how many important fields were extracted
        tier1_missing              — which Tier 1 fields are absent (audit deficiencies)
        floor_applied              — which floor rule fired, if any
        extraction_gate            — True if score >= 0.50 (proceed to drafter)
        quality_gate               — True if score >= 0.70 (eligible for JSONL)
    """

    per_field_scores: dict[str, float] = field(default_factory=dict)
    aggregate_score: float = 0.0

    # Tier breakdown
    tier1_found: int = 0
    tier1_total: int = 0
    tier2_found: int = 0
    tier2_total: int = 0

    # Field lists
    fields_present: list[str] = field(default_factory=list)
    fields_missing: list[str] = field(default_factory=list)
    tier1_missing: list[str] = field(default_factory=list)
    low_confidence_fields: list[str] = field(default_factory=list)

    # Diagnostics
    floor_applied: str | None = None   # e.g. "tier1_found>=5 → 0.65"

    # Gate results
    extraction_gate: bool = False      # >= 0.50 — proceed to LLM drafter
    quality_gate: bool = False         # >= 0.70 — eligible for JSONL

    EXTRACTION_THRESHOLD: float = 0.50
    QUALITY_THRESHOLD: float = 0.70

    def __str__(self) -> str:
        lines = [
            f"aggregate={self.aggregate_score:.4f}  "
            f"extraction={'PASS' if self.extraction_gate else 'FAIL'}  "
            f"quality={'PASS' if self.quality_gate else 'FAIL'}",
            f"tier1={self.tier1_found}/{self.tier1_total}  "
            f"tier2={self.tier2_found}/{self.tier2_total}",
        ]
        if self.floor_applied:
            lines.append(f"floor applied: {self.floor_applied}")
        if self.tier1_missing:
            lines.append(f"tier1 missing (deficiencies): {self.tier1_missing}")
        if self.low_confidence_fields:
            lines.append(f"low_conf_fields: {self.low_confidence_fields}")
        return "\n".join(lines)


def summarise(
    per_field_scores: dict[str, float],
    tiers: TierConfig | None = None,
    extraction_threshold: float = 0.50,
    quality_threshold: float = 0.70,
) -> ConfidenceSummary:
    """
    Build a full ConfidenceSummary from per-field scores.

    Parameters
    ----------
    per_field_scores : dict[str, float]
        Output of score_fields().
    tiers : TierConfig | None
        Loaded from field_tiers.yaml if not supplied.
    extraction_threshold : float
        Gate for proceeding to LLM drafter. Default 0.50.
    quality_threshold : float
        Gate for JSONL eligibility. Default 0.70.

    Returns
    -------
    ConfidenceSummary
    """
    if tiers is None:
        tiers = load_tiers()

    aggregate = score_record(per_field_scores, tiers=tiers)

    # Tier counts
    tier1_found = sum(
        1 for f, s in per_field_scores.items() if f in tiers.tier1 and s > 0.0
    )
    tier2_found = sum(
        1 for f, s in per_field_scores.items() if f in tiers.tier2 and s > 0.0
    )

    # Missing Tier 1 fields — potential audit deficiencies
    # Includes fields in Tier 1 that weren't attempted at all
    attempted = set(per_field_scores.keys())
    tier1_missing = [
        f for f in tiers.tier1
        if f not in attempted or per_field_scores.get(f, 0.0) == 0.0
    ]

    present = [f for f, s in per_field_scores.items() if s > 0.0]
    missing = [f for f, s in per_field_scores.items() if s == 0.0]
    low_conf = [
        f for f, s in per_field_scores.items()
        if 0.0 < s < quality_threshold
    ]

    # Determine which floor fired (for diagnostics)
    floor_applied = None
    tier1_found_all = sum(
        1 for f, s in per_field_scores.items() if f in tiers.tier1 and s > 0.0
    )
    for min_found, floor_val in _TIER1_FLOORS:
        if tier1_found_all >= min_found:
            # Check if floor was actually binding
            raw_base = (
                sum(s for f, s in per_field_scores.items() if f in tiers.tier1)
                / len(tiers.tier1) * _TIER1_WEIGHT
                + sum(s for f, s in per_field_scores.items() if f in tiers.tier2)
                / len(tiers.tier2) * _TIER2_WEIGHT
                if tiers.tier2 else 0.0
            )
            if floor_val > raw_base:
                floor_applied = f"tier1_found>={min_found} → {floor_val}"
            break

    return ConfidenceSummary(
        per_field_scores=per_field_scores,
        aggregate_score=aggregate,
        tier1_found=tier1_found,
        tier1_total=len(tiers.tier1),
        tier2_found=tier2_found,
        tier2_total=len(tiers.tier2),
        fields_present=present,
        fields_missing=missing,
        tier1_missing=tier1_missing,
        low_confidence_fields=low_conf,
        floor_applied=floor_applied,
        extraction_gate=aggregate >= extraction_threshold,
        quality_gate=aggregate >= quality_threshold,
    )

    