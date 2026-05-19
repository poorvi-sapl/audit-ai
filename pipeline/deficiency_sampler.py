"""
pipeline/deficiency_sampler.py
================================
Randomised deficiency field combination sampler for training pair generation.

This module enforces a strict separation between schema, policy, and sampling behavior.

Field classification — three-way logic
---------------------------------------
Every field is classified against two registries:

    Registry A: sop_field_classes.yaml   — SOP-specific Q-number policy
    Registry B: field_tiers.yaml tier1+2 — canonical field universe

Outcomes:

    governed            → in sop_field_classes. Class and sampling_weight
                          are respected (deficiency_eligible / fixed_admin /
                          informational_only / zero_weight).

    assumed_eligible    → in canonical registry (tier1+tier2) but no
                          sop_field_classes entry. Treated as deficiency_eligible
                          at tier weight. Safe default — no warning.

    unregistered_field  → in present_fields but NOT in canonical registry.
                          Excluded from pool. Increments schema_drift_count
                          and sets coverage_alert=True. This is the real
                          drift signal — not a noisy unknown bucket.

Pool coverage
-------------
pool_coverage = eligible_pool_size / total_canonical_fields (tier1+tier2 total).
Measures SOP-exclusion shrinkage of the canonical field space, not per-call
sampling diversity. Unregistered fields are excluded from the calculation.

Layer architecture
------------------
Layer 1 — Canonical schema (field_tiers.yaml)
    Source of truth for field existence. A field not in tier1+tier2 is
    unregistered regardless of SOP content.

Layer 2 — SOP policy (sop_field_classes.yaml)
    Behavior overlay only. Governs deficiency_allowed / citation_allowed /
    client_type_overrides / sampling_weight. Does not affect field existence.

Layer 3 — Sampling behavior (this module)
    Uses Layer 1 to define the universe of valid fields.

    pool_coverage measures Layer 2 shrinkage of Layer 1:
        pool_coverage = eligible_pool / total_canonical

    sample_diversity measures within-call usage of Layer 3:
        sample_diversity = unique_sampled_fields / eligible_pool

Key invariants
--------------
- SOP cannot define new fields (only canonical schema can)
- Canonical schema cannot define behavior (only existence)
- Sampling logic must never treat SOP as a source of truth for existence
- Drift detection is based ONLY on Layer 1
- Eligibility filtering is based on Layer 2

Public API
----------
    sample(present_fields, file_name, client_type) -> SampleResult
    shadow_compare(present_fields, file_name, client_type) -> ShadowResult
    coverage_report(log_path) -> CoverageReport
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_PROJECT_DIR    = Path(__file__).parent.parent
_THRESHOLD_PATH = _PROJECT_DIR / "auditai_data_normalization" / "alias_registry" / "threshold_config.yaml"
_TIERS_PATH     = _PROJECT_DIR / "config" / "field_tiers.yaml"
_COVERAGE_LOG   = _PROJECT_DIR / "data" / "deficiency_coverage.jsonl"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FieldExclusion:
    """One field excluded from the sampling pool, with its reason."""
    field:  str
    reason: str   # "fixed_admin" | "client_override" | "zero_weight" | "informational" | "no_sfc_entry"


@dataclass
class FieldCoverageEntry:
    """Per-field classification in the coverage audit."""
    field:          str
    classification: str   # "sampled" | "eligible_not_sampled" |
                          # "assumed_eligible" |
                          # "excluded_fixed_admin" | "excluded_zero_weight" |
                          # "excluded_informational" | "unregistered"


@dataclass
class SampleResult:
    """Full output of one sample() call."""
    combinations:          list[list[str]]
    excluded_fields:       list[FieldExclusion]
    coverage_audit:        list[FieldCoverageEntry]
    pool_coverage:         float   # eligible_pool / total_canonical (SOP-driven, doc-independent)
    pool_coverage_warning: bool    # True when pool_coverage < structural threshold
    # Schema drift signals — unregistered fields are in neither registry
    schema_drift_count:    int
    unregistered_fields:   list[str]
    coverage_alert:        bool    # True when schema_drift_count > 0


@dataclass
class ShadowResult:
    """Comparison of strict vs relaxed sampling modes."""
    strict_combinations:   list[list[str]]
    relaxed_combinations:  list[list[str]]
    lost_fields:           list[str]   # eligible in relaxed, excluded in strict
    lost_fixed_admin:      list[str]   # subset of lost_fields with known reason
    lost_unknown:          list[str]   # subset with unknown classification


@dataclass
class CoverageReport:
    """Per-field sampling frequency across the coverage log."""
    total_combinations: int
    field_counts:       dict[str, int]
    undersampled:       list[str]
    oversampled:        list[str]


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def _load_thresholds() -> dict:
    if not _THRESHOLD_PATH.exists():
        return {}
    with open(_THRESHOLD_PATH) as f:
        return yaml.safe_load(f) or {}


def _load_tier_fields() -> tuple[list[str], list[str]]:
    if not _TIERS_PATH.exists():
        return [], []
    with open(_TIERS_PATH) as f:
        tiers = yaml.safe_load(f) or {}
    tier1 = [e["field"] for e in (tiers.get("tier1") or []) if isinstance(e, dict)]
    tier2 = [e["field"] for e in (tiers.get("tier2") or []) if isinstance(e, dict)]
    return tier1, tier2


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

def _make_seed(file_name: str, client_type: str) -> int:
    raw = f"{file_name}::{client_type}".encode("utf-8")
    return int(hashlib.md5(raw).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

def _classify_fields(
    present_fields: list[str],
    tier1_fields:   list[str],
    tier2_fields:   list[str],
    client_type:    str,
    graph,                       # SOPGraph | None
    t1_weight: float,
    t2_weight: float,
) -> tuple[
    list[tuple[str, float]],     # eligible_t1: (field, weight)
    list[tuple[str, float]],     # eligible_t2: (field, weight)
    list[FieldExclusion],        # excluded
    list[str],                   # unregistered_fields (in neither registry)
]:
    """
    Three-way field classification using the compiled SOPGraph.

    SOPGraph is the single authority. All eligibility, lock, and weight
    decisions come from one compiled object — no separate authority table.

    governed (node exists in SOPGraph):
        deficiency_allowed   → added to eligible pool at configured weight
        is_locked            → excluded: "fixed_admin" or "client_override"
        is_informational     → excluded: "informational"
        sampling_weight==0.0 → excluded: "zero_weight"

    no_sfc_entry (canonical but no SOPGraph node):
        excluded: "no_sfc_entry"  (strict SFC-authority rule)

    unregistered (in neither tier registry):
        excluded — schema drift signal
    """
    present_set   = set(present_fields)
    tier1_set     = set(tier1_fields)
    tier2_set     = set(tier2_fields)
    canonical_set = tier1_set | tier2_set

    eligible_t1:  list[tuple[str, float]] = []
    eligible_t2:  list[tuple[str, float]] = []
    excluded:     list[FieldExclusion]    = []
    unregistered: list[str]               = []

    # ── Canonical fields (tier1+tier2) ────────────────────────────────────
    for f in (tier1_fields + tier2_fields):
        if f not in present_set:
            continue

        base_weight = t1_weight if f in tier1_set else t2_weight
        pool        = eligible_t1 if f in tier1_set else eligible_t2

        if graph is None:
            # No graph available — degrade gracefully, include at tier weight
            pool.append((f, base_weight))
            continue

        node = graph.get(f)

        if node is None:
            # Strict SFC-authority rule: no SOPGraph node → not eligible.
            excluded.append(FieldExclusion(field=f, reason="no_sfc_entry"))
            continue

        sw = node.sampling_weight
        if sw is not None and sw == 0.0:
            excluded.append(FieldExclusion(field=f, reason="zero_weight"))
            continue

        if graph.is_locked(f, client_type):
            # client_override: locked only for this client_type, not in base
            reason = "fixed_admin" if node.locked_in_base else "client_override"
            excluded.append(FieldExclusion(field=f, reason=reason))
            continue

        if graph.is_informational(f, client_type):
            excluded.append(FieldExclusion(field=f, reason="informational"))
            continue

        if not graph.allowed(f, client_type):
            excluded.append(FieldExclusion(field=f, reason="no_sfc_entry"))
            continue

        pool.append((f, sw if sw is not None else base_weight))

    # ── Unregistered fields (in present_fields but not in canonical registry) ──
    for f in present_fields:
        if f not in canonical_set:
            unregistered.append(f)
            logger.warning(
                "deficiency_sampler: unregistered field '%s' (client_type=%s) — "
                "not in field_tiers.yaml. Schema drift signal. Excluded from sampling.",
                f, client_type,
            )

    return eligible_t1, eligible_t2, excluded, unregistered


# ---------------------------------------------------------------------------
# Entropy computation
# ---------------------------------------------------------------------------

def _eligible_pool_size(
    tier1_fields: list[str],
    tier2_fields: list[str],
    graph,
    client_type: str,
) -> int:
    """
    Count canonical fields (across all of tier1+tier2) that would be eligible
    for deficiency sampling for this client_type, regardless of which fields
    appear in a specific workpaper.

    Document-independent. Measures SOP-exclusion shrinkage of the canonical
    field space for pool_coverage reporting.
    """
    count = 0
    for f in tier1_fields + tier2_fields:
        if graph is None:
            count += 1
            continue
        node = graph.get(f)
        if node is None:
            continue   # strict SFC-authority rule: no node → not eligible
        if node.sampling_weight is not None and node.sampling_weight == 0.0:
            continue
        if graph.allowed(f, client_type):
            count += 1
    return count


def _compute_pool_coverage(eligible_pool: int, total_canonical: int) -> float:
    """
    Pool coverage: fraction of canonical fields still eligible after SOP exclusions.
    eligible_pool is computed across ALL canonical fields for this client_type
    (not conditioned on present_fields in any specific document).
    """
    if total_canonical == 0:
        return 0.0
    return round(eligible_pool / total_canonical, 4)


def _pool_coverage_threshold(client_type: str, cfg: dict) -> float:
    def_cfg = cfg.get("deficiency", {})
    return float(
        def_cfg.get("min_pool_coverage_by_client_type", {}).get(
            client_type,
            def_cfg.get("min_pool_coverage_default", 0.80),
        )
    )


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------

def _build_coverage_audit(
    tier1_fields:    list[str],
    tier2_fields:    list[str],
    present_set:     set[str],
    eligible_t1:     list[tuple[str, float]],
    eligible_t2:     list[tuple[str, float]],
    excluded:        list[FieldExclusion],
    unregistered:    list[str],
    combinations:    list[list[str]],
    client_type:     str,
) -> list[FieldCoverageEntry]:
    sampled       = {f for combo in combinations for f in combo}
    eligible_set  = {f for f, _ in eligible_t1 + eligible_t2}
    excluded_map  = {e.field: e.reason for e in excluded}
    unreg_set     = set(unregistered)

    audit: list[FieldCoverageEntry] = []
    for f in (tier1_fields + tier2_fields):
        if f not in present_set:
            continue
        if f in sampled:
            cls = "sampled"
        elif f in eligible_set:
            cls = "eligible_not_sampled"
        elif f in excluded_map:
            reason = excluded_map[f]
            cls = (
                "excluded_zero_weight"   if reason == "zero_weight"   else
                "excluded_informational" if reason == "informational"  else
                "excluded_no_sfc_entry"  if reason == "no_sfc_entry"  else
                "excluded_fixed_admin"
            )
        else:
            cls = "eligible_not_sampled"
        audit.append(FieldCoverageEntry(field=f, classification=cls))

    # Unregistered fields appear in present_fields but not in tier1+tier2
    for f in unreg_set:
        audit.append(FieldCoverageEntry(field=f, classification="unregistered"))

    return audit


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------

def sample(
    present_fields: list[str],
    file_name:      str,
    client_type:    str,
    sop_id:         str = "npo-cx-1.1",
) -> SampleResult:
    """
    Sample randomised deficiency field combinations for one (doc, client_type).

    Returns SampleResult — use .combinations for the field lists,
    .excluded_fields for audit, .pool_coverage / .pool_coverage_warning for SOP coverage checks.
    """
    cfg    = _load_thresholds()
    def_cfg = cfg.get("deficiency", {})
    n_variants = int(def_cfg.get("variants_per_workpaper",   2))
    min_tier1  = int(def_cfg.get("min_tier1_per_variant",    1))
    max_fields = int(def_cfg.get("max_fields_per_variant",   3))
    t1_weight  = float(def_cfg.get("tier1_sample_weight",   0.70))
    t2_weight  = float(def_cfg.get("tier2_sample_weight",   0.30))

    tier1_fields, tier2_fields = _load_tier_fields()
    present_set = set(present_fields)

    # Load compiled SOPGraph — None on failure (degrades gracefully)
    graph = None
    _graph_version = ""
    try:
        from pipeline.sop_compiler import compiled_sop as _compiled_sop
        graph = _compiled_sop(sop_id)
        _graph_version = graph.graph_version
        logger.debug(
            "deficiency_sampler: SOPGraph ready sop_id=%r version=%s",
            sop_id, _graph_version,
        )
    except Exception as _e:
        logger.debug("deficiency_sampler: SOPGraph unavailable — %s", _e)

    eligible_t1, eligible_t2, excluded, unregistered = _classify_fields(
        present_fields, tier1_fields, tier2_fields,
        client_type, graph, t1_weight, t2_weight,
    )

    for ex in excluded:
        logger.debug(
            "deficiency_sampler: excluded '%s' from %s pool — reason=%s",
            ex.field, client_type, ex.reason,
        )

    total_canonical  = len(tier1_fields) + len(tier2_fields)
    eligible_count   = len(eligible_t1) + len(eligible_t2)   # present + eligible
    eligible_pool    = _eligible_pool_size(tier1_fields, tier2_fields, graph, client_type)
    schema_drift     = len(unregistered)
    coverage_alert   = schema_drift > 0

    if len(eligible_t1) < min_tier1:
        logger.warning(
            "deficiency_sampler: %s/%s — only %d Tier 1 fields eligible "
            "(need %d). Cannot form valid deficiency combination.",
            file_name, client_type, len(eligible_t1), min_tier1,
        )
        audit = _build_coverage_audit(
            tier1_fields, tier2_fields, present_set,
            eligible_t1, eligible_t2, excluded, unregistered, [], client_type,
        )
        return SampleResult(
            combinations=[], excluded_fields=excluded,
            coverage_audit=audit,
            pool_coverage=_compute_pool_coverage(eligible_pool, total_canonical),
            pool_coverage_warning=True,
            schema_drift_count=schema_drift,
            unregistered_fields=unregistered,
            coverage_alert=coverage_alert,
        )

    rng = random.Random(_make_seed(file_name, client_type))

    # Build flat weighted pools for rng.choices
    t1_names    = [f for f, _ in eligible_t1]
    t1_weights  = [w for _, w in eligible_t1]
    all_names   = [f for f, _ in eligible_t1 + eligible_t2]
    all_weights = [w for _, w in eligible_t1 + eligible_t2]

    combinations: list[list[str]] = []
    attempts = 0
    max_attempts = n_variants * 20

    while len(combinations) < n_variants and attempts < max_attempts:
        attempts += 1

        max_possible  = min(max_fields, len(all_names))
        size_weights  = [1.0 / i for i in range(1, max_possible + 1)]
        total_sw      = sum(size_weights)
        size_weights  = [w / total_sw for w in size_weights]
        combo_size    = rng.choices(range(1, max_possible + 1), weights=size_weights, k=1)[0]

        n_t1     = min(min_tier1, len(t1_names), combo_size)
        t1_picks = rng.choices(t1_names, weights=t1_weights, k=n_t1)
        t1_picks = list(dict.fromkeys(t1_picks))   # deduplicate

        remaining_slots = combo_size - len(t1_picks)
        remaining_pool  = [(n, w) for n, w in zip(all_names, all_weights) if n not in t1_picks]

        if remaining_slots > 0 and remaining_pool:
            r_names   = [n for n, _ in remaining_pool]
            r_weights = [w for _, w in remaining_pool]
            extra = rng.choices(r_names, weights=r_weights, k=min(remaining_slots, len(r_names)))
            extra = list(dict.fromkeys(extra))
            combo = sorted(set(t1_picks + extra))
        else:
            combo = sorted(t1_picks)

        if combo in combinations:
            continue
        combinations.append(combo)

    if len(combinations) < n_variants:
        logger.info(
            "deficiency_sampler: %s/%s — produced %d/%d combinations "
            "(eligible: t1=%d t2=%d)",
            file_name, client_type, len(combinations), n_variants,
            len(eligible_t1), len(eligible_t2),
        )

    # Pool coverage guard — eligible_pool / total_canonical (SOP-driven, doc-independent)
    pool_coverage      = _compute_pool_coverage(eligible_pool, total_canonical)
    min_pool_coverage  = _pool_coverage_threshold(client_type, cfg)
    pool_coverage_warn = pool_coverage < min_pool_coverage and total_canonical > 0

    if pool_coverage_warn:
        logger.warning(
            "deficiency_sampler: pool coverage %.3f < %.3f for %s/%s "
            "— SOP exclusions are shrinking the eligible canonical field space. "
            "Excluded fields: %s",
            pool_coverage, min_pool_coverage, client_type, file_name,
            [e.field for e in excluded],
        )

    audit = _build_coverage_audit(
        tier1_fields, tier2_fields, present_set,
        eligible_t1, eligible_t2, excluded, unregistered, combinations,
        client_type,
    )

    _log_coverage(file_name, client_type, combinations, excluded, pool_coverage, unregistered)

    logger.debug(
        "deficiency_sampler: %s/%s sop_id=%r version=%s → combos=%s pool_coverage=%.3f "
        "excluded=%s unregistered=%s",
        file_name, client_type, sop_id, _graph_version, combinations, pool_coverage,
        [e.field for e in excluded], unregistered,
    )

    return SampleResult(
        combinations=combinations,
        excluded_fields=excluded,
        coverage_audit=audit,
        pool_coverage=pool_coverage,
        pool_coverage_warning=pool_coverage_warn,
        schema_drift_count=schema_drift,
        unregistered_fields=unregistered,
        coverage_alert=coverage_alert,
    )


# ---------------------------------------------------------------------------
# Shadow compare — strict vs relaxed
# ---------------------------------------------------------------------------

def shadow_compare(
    present_fields: list[str],
    file_name:      str,
    client_type:    str,
    sop_id:         str = "npo-cx-1.1",
) -> ShadowResult:
    """
    Run sample() in strict mode (R5 guards active) and relaxed mode
    (sop_field_classes bypassed). Returns the set of fields that strict
    mode excluded from the eligible pool vs relaxed mode.

    Use this to detect when sop_field_classes.yaml is over-constraining
    training diversity across a batch of documents.
    """
    strict  = sample(present_fields, file_name, client_type, sop_id=sop_id)

    # Relaxed: pass sfc=None so _classify_fields treats all fields as eligible
    cfg    = _load_thresholds()
    def_cfg = cfg.get("deficiency", {})
    t1_weight = float(def_cfg.get("tier1_sample_weight", 0.70))
    t2_weight = float(def_cfg.get("tier2_sample_weight", 0.30))
    tier1_fields, tier2_fields = _load_tier_fields()

    relaxed_t1, relaxed_t2, _, _ = _classify_fields(
        present_fields, tier1_fields, tier2_fields,
        client_type, None, t1_weight, t2_weight,
    )
    relaxed_eligible = {f for f, _ in relaxed_t1 + relaxed_t2}
    strict_eligible  = {
        e.field for e in strict.coverage_audit
        if e.classification in ("sampled", "eligible_not_sampled")
    }

    lost_fields     = sorted(relaxed_eligible - strict_eligible)
    excluded_map    = {e.field: e.reason for e in strict.excluded_fields}
    lost_fixed_admin = [f for f in lost_fields if excluded_map.get(f) in
                        ("fixed_admin", "client_override", "zero_weight", "informational")]
    lost_unknown    = [f for f in lost_fields if f not in excluded_map]

    return ShadowResult(
        strict_combinations=strict.combinations,
        relaxed_combinations=[],   # relaxed combos not needed — pool comparison is enough
        lost_fields=lost_fields,
        lost_fixed_admin=lost_fixed_admin,
        lost_unknown=lost_unknown,
    )


# ---------------------------------------------------------------------------
# Coverage logging
# ---------------------------------------------------------------------------

def _log_coverage(
    file_name:     str,
    client_type:   str,
    combinations:  list[list[str]],
    excluded:      list[FieldExclusion],
    pool_coverage: float,
    unregistered:  list[str],
) -> None:
    try:
        _COVERAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "file_name":    file_name,
            "client_type":  client_type,
            "combinations": combinations,
            "excluded":     [{"field": e.field, "reason": e.reason} for e in excluded],
            "pool_coverage": pool_coverage,
            "unregistered": unregistered,
        }
        with open(_COVERAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.debug("deficiency_sampler: coverage log write failed — %s", e)


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def coverage_report(log_path: Path | None = None) -> CoverageReport:
    """
    Read the coverage log and return per-field sampling statistics.
    """
    path = log_path or _COVERAGE_LOG
    if not path.exists():
        return CoverageReport(total_combinations=0, field_counts={},
                              undersampled=[], oversampled=[])

    field_counts: dict[str, int] = {}
    total = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                for combo in entry.get("combinations", []):
                    total += 1
                    for field_name in combo:
                        field_counts[field_name] = field_counts.get(field_name, 0) + 1
            except json.JSONDecodeError:
                continue

    if not field_counts:
        return CoverageReport(total_combinations=total, field_counts={},
                              undersampled=[], oversampled=[])

    avg = sum(field_counts.values()) / len(field_counts)
    tier1_fields, tier2_fields = _load_tier_fields()
    all_trackable = set(tier1_fields + tier2_fields)
    for f in all_trackable:
        if f not in field_counts:
            field_counts[f] = 0

    undersampled = sorted(f for f in all_trackable if field_counts.get(f, 0) < avg * 0.5)
    oversampled  = sorted(f for f in all_trackable if field_counts.get(f, 0) > avg * 2.0)

    return CoverageReport(
        total_combinations=total,
        field_counts=dict(sorted(field_counts.items(), key=lambda x: -x[1])),
        undersampled=undersampled,
        oversampled=oversampled,
    )
