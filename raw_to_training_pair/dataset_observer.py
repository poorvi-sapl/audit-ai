"""
raw_to_training_pair/dataset_observer.py
==========================================
Training dataset observability layer.

This is NOT a pipeline gate, ETL step, or validation filter.
It is a read-only system that answers three questions about your
training dataset at any point in time:

    1. What do I have?
       Distributions across client type, pair type, confidence,
       and deficiency field frequency.

    2. What am I missing?
       Tier 1 fields never seen in deficient pairs, severity levels
       never used, client types with zero coverage.

    3. Is drift happening over time?
       Delta between cumulative distribution and the last N approved
       pairs (configurable batch_window_size). Surfaces cases like
       "first 40 pairs were balanced, last 20 are 90% NPO."

No thresholds are enforced. Imbalance is not treated as a problem
unless you define an expected_distribution in threshold_config.yaml,
in which case the report shows actual vs expected delta so you can
judge whether the gap is intentional or not.

Public API
----------
    snapshot(stage2_path, stage3_path)  -> DatasetSnapshot
        Read current JSONL state. Fast — O(n) scan.

    drift_report(index_path, window)    -> DriftReport
        Compute recent vs cumulative distribution using fingerprint
        index timestamps.

    print_report(snapshot, drift)       -> str
        Formatted terminal / Streamlit output.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_DIR    = Path(__file__).parent.parent
_THRESHOLD_PATH = _PROJECT_DIR / "auditai_data_normalization" / "alias_registry" / "threshold_config.yaml"
_TIERS_PATH     = _PROJECT_DIR / "config" / "field_tiers.yaml"
_DEFAULT_STAGE2 = _PROJECT_DIR / "data" / "stage2_domain.jsonl"
_DEFAULT_STAGE3 = _PROJECT_DIR / "data" / "stage3_firm.jsonl"
_DEFAULT_INDEX  = _PROJECT_DIR / "data" / "pair_fingerprint_index.jsonl"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DatasetSnapshot:
    """Answer to Question 1 and 2 — what do I have, what am I missing."""

    total_pairs:        int
    stage2_count:       int
    stage3_count:       int

    # Q1 distributions
    by_client_type:     dict[str, int]    # {"NPO": 12, "Government": 8, ...}
    by_pair_type:       dict[str, int]    # {"clean": 10, "deficient": 10}
    by_stage:           dict[str, int]
    confidence_hist:    dict[str, int]    # {"<0.55": 3, "0.55-0.65": 5, ...}
    field_frequency:    dict[str, int]    # {field: times_in_missing}

    # Q2 gaps
    tier1_never_missing:   list[str]      # Tier 1 fields never in fields_missing
    severity_never_used:   list[str]      # severity levels never in any completion
    client_types_missing:  list[str]      # client types with 0 pairs

    # Optional: expected vs actual delta
    expected_distribution: dict[str, float]   # from config (empty if not set)
    distribution_delta:    dict[str, float]   # actual% - expected% (empty if no expected)

    # Rare patterns
    rare_deficiency_combos: list[dict]    # combos seen < rare_pattern_threshold times


@dataclass
class DriftReport:
    """Answer to Question 3 — is drift happening over time."""

    window_size:          int
    total_pairs:          int
    window_pairs:         int

    # Cumulative vs recent distributions
    cumulative_by_client: dict[str, float]   # percentages
    recent_by_client:     dict[str, float]
    client_drift:         dict[str, float]   # recent% - cumulative%

    cumulative_by_type:   dict[str, float]
    recent_by_type:       dict[str, float]
    type_drift:           dict[str, float]

    # Notable drift signals
    high_drift_fields:    list[str]   # client types with |drift| > 0.20
    drift_direction:      str         # "stable" | "drifting" | "insufficient_data"


# ---------------------------------------------------------------------------
# Config and tier loaders
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    try:
        with open(_THRESHOLD_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_tier1_fields() -> list[str]:
    try:
        with open(_TIERS_PATH) as f:
            tiers = yaml.safe_load(f) or {}
        return [e["field"] for e in (tiers.get("tier1") or []) if isinstance(e, dict)]
    except Exception:
        return []


_ALL_SEVERITY_LEVELS = ["High", "Medium", "Low", "Informational"]
_ALL_CLIENT_TYPES    = ["NPO", "Government", "For-Profit", "Tribal"]
_CONF_BUCKETS        = ["<0.55", "0.55-0.65", "0.65-0.75", "0.75-0.85", ">=0.85"]


def _conf_bucket(val: float) -> str:
    if val < 0.55:  return "<0.55"
    if val < 0.65:  return "0.55-0.65"
    if val < 0.75:  return "0.65-0.75"
    if val < 0.85:  return "0.75-0.85"
    return ">=0.85"


# ---------------------------------------------------------------------------
# JSONL readers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pairs


def _read_index(path: Path) -> list[dict]:
    return _read_jsonl(path)


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def snapshot(
    stage2_path: Path | None = None,
    stage3_path: Path | None = None,
) -> DatasetSnapshot:
    """
    Build a DatasetSnapshot from the current approved JSONL files.

    Reads stage2_domain.jsonl and stage3_firm.jsonl. Fast O(n) scan.
    Returns a DatasetSnapshot answering Q1 (distributions) and Q2 (gaps).

    Parameters
    ----------
    stage2_path, stage3_path : Path | None
        Override default data/ paths. Used in tests.
    """
    cfg      = _load_cfg()
    obs_cfg  = cfg.get("observability", {})
    min_app  = int(obs_cfg.get("min_field_appearances", 1))
    rare_thr = int(obs_cfg.get("rare_pattern_threshold", 3))
    expected_dist = obs_cfg.get("expected_distribution") or {}

    tier1_fields = _load_tier1_fields()

    p2 = stage2_path or _DEFAULT_STAGE2
    p3 = stage3_path or _DEFAULT_STAGE3

    pairs2 = _read_jsonl(p2)
    pairs3 = _read_jsonl(p3)
    all_pairs = pairs2 + pairs3

    total        = len(all_pairs)
    stage2_count = len(pairs2)
    stage3_count = len(pairs3)

    # Q1 — distributions
    by_client:   dict[str, int] = defaultdict(int)
    by_type:     dict[str, int] = defaultdict(int)
    by_stage:    dict[str, int] = defaultdict(int)
    conf_hist:   dict[str, int] = {b: 0 for b in _CONF_BUCKETS}
    field_freq:  dict[str, int] = defaultdict(int)
    combo_freq:  dict[str, int] = defaultdict(int)
    severities_seen: set[str]   = set()

    for pair in all_pairs:
        meta = pair.get("metadata", {})

        ct    = meta.get("client_type", "unknown")
        pt    = meta.get("pair_type", "unknown")
        stage = meta.get("stage", "unknown")
        conf  = float(meta.get("extraction_confidence", 0.0))

        by_client[ct]  += 1
        by_type[pt]    += 1
        by_stage[stage] += 1
        conf_hist[_conf_bucket(conf)] += 1

        # Field frequency from fields_missing
        missing = meta.get("fields_missing") or []
        for f in missing:
            field_freq[f.lower()] += 1

        # Deficiency combo frequency (sorted tuple for canonical form)
        if missing:
            combo_key = "|".join(sorted(f.lower() for f in missing))
            combo_freq[combo_key] += 1

        # Severity levels used — scan assistant message
        msgs = pair.get("messages", [])
        for msg in msgs:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                import re
                for sev in _ALL_SEVERITY_LEVELS:
                    if re.search(rf"Severity:\s*{sev}", content, re.IGNORECASE):
                        severities_seen.add(sev)

    # Q2 — gaps
    tier1_never_missing = [
        f for f in tier1_fields
        if field_freq.get(f.lower(), 0) < min_app
    ]

    severity_never_used = [
        s for s in _ALL_SEVERITY_LEVELS
        if s not in severities_seen
    ]

    client_types_missing = [
        ct for ct in _ALL_CLIENT_TYPES
        if by_client.get(ct, 0) == 0
    ]

    # Rare deficiency combos
    rare_combos = []
    for combo_key, count in sorted(combo_freq.items(), key=lambda x: x[1]):
        if count < rare_thr:
            rare_combos.append({
                "fields": combo_key.split("|"),
                "count":  count,
            })

    # Expected vs actual delta
    dist_delta: dict[str, float] = {}
    if expected_dist and total > 0:
        for ct, exp_frac in expected_dist.items():
            actual_frac = by_client.get(ct, 0) / total
            dist_delta[ct] = round(actual_frac - exp_frac, 4)

    return DatasetSnapshot(
        total_pairs           = total,
        stage2_count          = stage2_count,
        stage3_count          = stage3_count,
        by_client_type        = dict(by_client),
        by_pair_type          = dict(by_type),
        by_stage              = dict(by_stage),
        confidence_hist       = conf_hist,
        field_frequency       = dict(sorted(field_freq.items(), key=lambda x: -x[1])),
        tier1_never_missing   = tier1_never_missing,
        severity_never_used   = severity_never_used,
        client_types_missing  = client_types_missing,
        expected_distribution = dict(expected_dist),
        distribution_delta    = dist_delta,
        rare_deficiency_combos = rare_combos,
    )


# ---------------------------------------------------------------------------
# Drift report
# ---------------------------------------------------------------------------

def drift_report(
    index_path: Path | None = None,
    window: int | None = None,
) -> DriftReport:
    """
    Compute recent vs cumulative distribution using fingerprint index timestamps.

    Compares the last `window` approved pairs against the cumulative
    distribution to detect dataset drift over time.

    Parameters
    ----------
    index_path : Path | None
        Path to pair_fingerprint_index.jsonl.
    window : int | None
        Rolling window size. Defaults to observability.batch_window_size.
    """
    cfg     = _load_cfg()
    obs_cfg = cfg.get("observability", {})
    win     = window or int(obs_cfg.get("batch_window_size", 20))

    entries = _read_index(index_path or _DEFAULT_INDEX)
    total   = len(entries)

    if total == 0:
        return DriftReport(
            window_size=win, total_pairs=0, window_pairs=0,
            cumulative_by_client={}, recent_by_client={}, client_drift={},
            cumulative_by_type={}, recent_by_type={}, type_drift={},
            high_drift_fields=[], drift_direction="insufficient_data",
        )

    # Sort by added_at timestamp
    def _parse_ts(e: dict) -> datetime:
        try:
            return datetime.fromisoformat(e.get("added_at", "2000-01-01T00:00:00+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    entries_sorted = sorted(entries, key=_parse_ts)
    recent_entries = entries_sorted[-win:]
    window_count   = len(recent_entries)

    def _dist(ents: list[dict], key: str) -> dict[str, float]:
        counts: dict[str, int] = defaultdict(int)
        for e in ents:
            counts[e.get(key, "unknown")] += 1
        n = len(ents)
        return {k: round(v / n, 4) for k, v in counts.items()} if n else {}

    cum_client  = _dist(entries_sorted, "client_type")
    rec_client  = _dist(recent_entries, "client_type")
    cum_type    = _dist(entries_sorted, "pair_type")
    rec_type    = _dist(recent_entries, "pair_type")

    # Drift deltas
    all_clients = set(cum_client) | set(rec_client)
    client_drift = {
        ct: round(rec_client.get(ct, 0.0) - cum_client.get(ct, 0.0), 4)
        for ct in all_clients
    }

    all_types = set(cum_type) | set(rec_type)
    type_drift = {
        pt: round(rec_type.get(pt, 0.0) - cum_type.get(pt, 0.0), 4)
        for pt in all_types
    }

    high_drift = [ct for ct, delta in client_drift.items() if abs(delta) > 0.20]

    if total < win * 2:
        direction = "insufficient_data"
    elif high_drift:
        direction = "drifting"
    else:
        direction = "stable"

    return DriftReport(
        window_size          = win,
        total_pairs          = total,
        window_pairs         = window_count,
        cumulative_by_client = cum_client,
        recent_by_client     = rec_client,
        client_drift         = client_drift,
        cumulative_by_type   = cum_type,
        recent_by_type       = rec_type,
        type_drift           = type_drift,
        high_drift_fields    = high_drift,
        drift_direction      = direction,
    )


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def print_report(
    snap: DatasetSnapshot,
    drift: DriftReport | None = None,
) -> str:
    """
    Format a human-readable observability report.
    Returns the report string (also prints to stdout).
    """
    lines = []
    sep   = "─" * 60

    lines.append(sep)
    lines.append("TRAINING DATASET OBSERVABILITY REPORT")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(sep)

    # Q1 — What do I have?
    lines.append(f"\n▸ TOTAL PAIRS: {snap.total_pairs}  "
                 f"(stage2={snap.stage2_count}  stage3={snap.stage3_count})")

    if snap.total_pairs == 0:
        lines.append("  No approved pairs yet.")
        report = "\n".join(lines)
        print(report)
        return report

    total = snap.total_pairs

    lines.append("\n▸ BY CLIENT TYPE")
    for ct in _ALL_CLIENT_TYPES:
        count = snap.by_client_type.get(ct, 0)
        pct   = count / total * 100
        bar   = "█" * int(pct / 5)
        delta_str = ""
        if ct in snap.distribution_delta:
            d = snap.distribution_delta[ct] * 100
            delta_str = f"  (expected {snap.expected_distribution[ct]*100:.0f}%,  Δ{d:+.1f}%)"
        lines.append(f"  {ct:14s} {count:4d}  {pct:5.1f}%  {bar}{delta_str}")

    lines.append("\n▸ BY PAIR TYPE")
    for pt in ["clean", "deficient"]:
        count = snap.by_pair_type.get(pt, 0)
        pct   = count / total * 100
        lines.append(f"  {pt:12s} {count:4d}  {pct:5.1f}%")

    lines.append("\n▸ EXTRACTION CONFIDENCE DISTRIBUTION")
    for bucket in _CONF_BUCKETS:
        count = snap.confidence_hist.get(bucket, 0)
        pct   = count / total * 100
        bar   = "█" * int(pct / 5)
        lines.append(f"  {bucket:12s} {count:4d}  {pct:5.1f}%  {bar}")

    lines.append("\n▸ DEFICIENCY FIELD FREQUENCY (top 10)")
    top_fields = list(snap.field_frequency.items())[:10]
    for fname, count in top_fields:
        lines.append(f"  {fname:35s} {count:4d}x")

    # Q2 — What am I missing?
    lines.append("\n▸ GAPS")

    if snap.tier1_never_missing:
        lines.append("  Tier 1 fields never seen in fields_missing (model blind spots):")
        for f in snap.tier1_never_missing:
            lines.append(f"    ✗ {f}")
    else:
        lines.append("  ✓ All Tier 1 fields have appeared in deficient pairs")

    if snap.client_types_missing:
        lines.append(f"  Client types with 0 pairs: {snap.client_types_missing}")
    else:
        lines.append("  ✓ All client types represented")

    if snap.severity_never_used:
        lines.append(f"  Severity levels never used: {snap.severity_never_used}")
    else:
        lines.append("  ✓ All severity levels used")

    if snap.rare_deficiency_combos:
        lines.append(f"\n▸ RARE DEFICIENCY PATTERNS (seen < {3}x)")
        for combo in snap.rare_deficiency_combos[:10]:
            lines.append(f"  {combo['count']}x  {combo['fields']}")

    # Q3 — Drift
    if drift:
        lines.append(f"\n▸ DRIFT  (last {drift.window_size} pairs vs cumulative)")
        lines.append(f"  Status: {drift.drift_direction.upper()}")

        if drift.drift_direction != "insufficient_data":
            lines.append("  Client type drift:")
            for ct in _ALL_CLIENT_TYPES:
                cum = drift.cumulative_by_client.get(ct, 0.0) * 100
                rec = drift.recent_by_client.get(ct, 0.0) * 100
                d   = drift.client_drift.get(ct, 0.0) * 100
                flag = "  ◄ DRIFT" if abs(d) > 20 else ""
                lines.append(f"    {ct:14s} cumul={cum:5.1f}%  recent={rec:5.1f}%  Δ{d:+.1f}%{flag}")

            lines.append("  Pair type drift:")
            for pt in ["clean", "deficient"]:
                cum = drift.cumulative_by_type.get(pt, 0.0) * 100
                rec = drift.recent_by_type.get(pt, 0.0) * 100
                d   = drift.type_drift.get(pt, 0.0) * 100
                lines.append(f"    {pt:12s} cumul={cum:5.1f}%  recent={rec:5.1f}%  Δ{d:+.1f}%")
        else:
            lines.append(f"  Insufficient data for drift analysis (need >{drift.window_size * 2} pairs)")

    lines.append(f"\n{sep}")

    report = "\n".join(lines)
    print(report)
    return report