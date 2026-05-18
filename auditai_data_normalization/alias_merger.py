"""
auditai_data_normalization/alias_merger.py
===========================================
Validated merge of pending alias updates into field_aliases.yaml.

This is the only authorised way to write new aliases to field_aliases.yaml.
Direct edits to that file outside of this module are not allowed in production.

Merge flow
----------
    1. Load pending_alias_updates.yaml
    2. For each pending entry, run 5 validation gates:
          Gate 1 — canonical field exists in field_tiers.yaml
          Gate 2 — label not in blocked_patterns.yaml
          Gate 3 — (label, canonical) not in rejected_mappings.yaml
          Gate 4 — label not already an alias for a DIFFERENT canonical field
          Gate 5 — gold label precision stays >= threshold after this merge
    3. Entries that pass all gates → written to field_aliases.yaml
    4. Entries that fail any gate → status set to rejected_at_merge with reason
    5. alias_versioning.py creates snapshot + changelog entry atomically
    6. reset_alias_cache() + reset_fuzzy_cache() called to invalidate caches
    7. pending entries marked merged/rejected — file updated

Gate 5 detail
-------------
Gold labels are loaded from evaluation/gold_labels/benchmark_mappings.yaml.
After tentatively applying the pending batch, the full alias file is tested
against gold labels. If precision drops below merge.min_gold_precision in
threshold_config.yaml, the entire batch is blocked unless an explicit
override reason is supplied by the caller.

Public API
----------
    merge_pending(override_reason=None, triggered_by="system") -> MergeResult
        Main entry point. Runs the full merge flow.
        override_reason: if set, allows merge even if gold precision drops.

    validate_entry(entry) -> ValidationResult
        Validate a single pending entry without writing. Used by Streamlit UI
        to show per-entry status before the merge button is clicked.

    MergeResult
        .merged          list[str]   — raw_labels successfully merged
        .rejected        list[RejectedEntry]
        .version         str         — new version string e.g. "1.3"
        .precision_before float
        .precision_after  float
        .blocked_by_precision bool
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PKG_DIR      = Path(__file__).parent
_PROJECT_DIR  = _PKG_DIR.parent
_REGISTRY_DIR = _PKG_DIR / "alias_registry"

_ALIASES_PATH   = _PKG_DIR / "field_aliases.yaml"
_PENDING_PATH   = _REGISTRY_DIR / "pending_alias_updates.yaml"
_REJECTED_PATH  = _REGISTRY_DIR / "rejected_mappings.yaml"
_BLOCKED_PATH   = _REGISTRY_DIR / "blocked_patterns.yaml"
_THRESHOLD_PATH = _REGISTRY_DIR / "threshold_config.yaml"

_TIERS_PATH     = _PROJECT_DIR / "config" / "field_tiers.yaml"
_GOLD_PATH      = _PROJECT_DIR / "evaluation" / "gold_labels" / "benchmark_mappings.yaml"
_SNAPSHOTS_DIR  = _PROJECT_DIR / "config" / "alias_registry" / "snapshots"
_CHANGELOG_DIR  = _PROJECT_DIR / "config" / "alias_registry" / "changelog"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid:   bool
    gate:    str   # which gate failed, or "pass"
    reason:  str   # human-readable explanation


@dataclass
class RejectedEntry:
    raw_label:       str
    canonical_field: str
    gate:            str
    reason:          str


@dataclass
class MergeResult:
    merged:                list[str]         = field(default_factory=list)
    rejected:              list[RejectedEntry] = field(default_factory=list)
    version:               str               = ""
    precision_before:      float             = 0.0
    precision_after:       float             = 0.0
    blocked_by_precision:  bool              = False


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_canonical_fields() -> set[str]:
    """All canonical field names from field_tiers.yaml + field_aliases.yaml."""
    fields: set[str] = set()
    tiers = _load_yaml(_TIERS_PATH)
    for tier_key in ("tier1", "tier2", "tier3"):
        for entry in tiers.get(tier_key, []):
            if isinstance(entry, dict) and "field" in entry:
                fields.add(entry["field"])
    aliases = _load_yaml(_ALIASES_PATH)
    fields.update(aliases.keys())
    return fields


def _load_blocked_exact() -> set[str]:
    data = _load_yaml(_BLOCKED_PATH)
    return set(str(s).lower().strip() for s in (data.get("exact_strings") or []))


def _load_blocked_patterns() -> list[re.Pattern]:
    data = _load_yaml(_BLOCKED_PATH)
    return [re.compile(p) for p in (data.get("regex_patterns") or [])]


def _load_rejected_pairs() -> set[tuple[str, str]]:
    data = _load_yaml(_REJECTED_PATH)
    return {
        (str(e.get("raw_label", "")).lower().strip(),
         str(e.get("rejected_canonical", "")).strip())
        for e in (data.get("rejections") or [])
        if e.get("raw_label") and e.get("rejected_canonical")
    }


def _load_existing_aliases() -> dict[str, list[str]]:
    return _load_yaml(_ALIASES_PATH) or {}


def _load_pending() -> list[dict]:
    data = _load_yaml(_PENDING_PATH)
    return [e for e in (data.get("pending") or []) if e.get("status") == "pending"]


def _load_gold_labels() -> list[dict]:
    """Load gold benchmark mappings. Returns [] if file doesn't exist."""
    data = _load_yaml(_GOLD_PATH)
    return data.get("mappings") or []


def _load_thresholds() -> dict:
    return _load_yaml(_THRESHOLD_PATH)


# ---------------------------------------------------------------------------
# Gold precision scoring
# ---------------------------------------------------------------------------

def _compute_gold_precision(aliases: dict[str, list[str]]) -> float:
    """
    Test alias file against gold labels.
    Precision = correct_mappings / total_gold_labels.
    Returns 1.0 if no gold labels exist (no penalty for missing file).
    """
    gold = _load_gold_labels()
    if not gold:
        return 1.0

    # Build reverse lookup: variant → canonical
    reverse: dict[str, str] = {}
    for canonical, variants in aliases.items():
        for v in (variants or []):
            reverse[str(v).lower().strip()] = canonical

    correct = 0
    for entry in gold:
        raw   = str(entry.get("raw_label", "")).lower().strip()
        expected = str(entry.get("correct_canonical", "")).strip()
        if reverse.get(raw) == expected:
            correct += 1

    return round(correct / len(gold), 4)


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def validate_entry(entry: dict) -> ValidationResult:
    """
    Run all 4 structural validation gates on a single pending entry.
    Gate 5 (gold precision) is run at the batch level in merge_pending().

    Returns ValidationResult with .valid=True if all gates pass.
    """
    raw_label       = str(entry.get("raw_label", "")).strip()
    canonical_field = str(entry.get("canonical_field", "")).strip()
    normalized      = raw_label.lower()

    # Gate 1 — canonical field must exist
    known_fields = _load_canonical_fields()
    if canonical_field not in known_fields:
        return ValidationResult(
            valid=False, gate="gate1_unknown_canonical",
            reason=f"'{canonical_field}' is not a known canonical field. "
                   f"Add it to field_tiers.yaml first."
        )

    # Gate 2 — label must not be blocked
    blocked_exact    = _load_blocked_exact()
    blocked_patterns = _load_blocked_patterns()
    if normalized in blocked_exact:
        return ValidationResult(
            valid=False, gate="gate2_blocked_exact",
            reason=f"'{raw_label}' is in blocked_patterns.yaml (exact match). "
                   f"This label is noise and must not become an alias."
        )
    for pat in blocked_patterns:
        if pat.fullmatch(normalized):
            return ValidationResult(
                valid=False, gate="gate2_blocked_pattern",
                reason=f"'{raw_label}' matches blocked pattern '{pat.pattern}'. "
                       f"This label is noise and must not become an alias."
            )

    # Gate 3 — (label, canonical) must not be in rejected_mappings
    rejected = _load_rejected_pairs()
    if (normalized, canonical_field) in rejected:
        return ValidationResult(
            valid=False, gate="gate3_previously_rejected",
            reason=f"The mapping '{raw_label}' → '{canonical_field}' was previously "
                   f"rejected. Check rejected_mappings.yaml for the rejection reason."
        )

    # Gate 4 — label must not already be an alias for a DIFFERENT canonical field
    existing = _load_existing_aliases()
    for field_name, variants in existing.items():
        if field_name == canonical_field:
            continue
        if normalized in [str(v).lower().strip() for v in (variants or [])]:
            return ValidationResult(
                valid=False, gate="gate4_alias_conflict",
                reason=f"'{raw_label}' is already an alias for '{field_name}'. "
                       f"Remove it from that field first or choose a different label."
            )

    return ValidationResult(valid=True, gate="pass", reason="")


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_alias_to_file(
    raw_label: str,
    canonical_field: str,
    aliases: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Return updated alias dict with new entry. Does not write to disk."""
    updated = {k: list(v or []) for k, v in aliases.items()}
    if canonical_field not in updated:
        updated[canonical_field] = []
    normalized = raw_label.lower().strip()
    existing_normalized = [str(v).lower().strip() for v in updated[canonical_field]]
    if normalized not in existing_normalized:
        updated[canonical_field].append(raw_label.strip())
    return updated


def _save_aliases(aliases: dict[str, list[str]]) -> None:
    with open(_ALIASES_PATH, "w", encoding="utf-8") as f:
        yaml.dump(aliases, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=True)


def _update_pending_statuses(
    updates: list[tuple[str, str, str]]  # (raw_label, status, reason)
) -> None:
    """Update status fields in pending_alias_updates.yaml in place."""
    data = _load_yaml(_PENDING_PATH)
    entries = data.get("pending") or []
    status_map = {raw.lower(): (status, reason) for raw, status, reason in updates}
    for entry in entries:
        key = str(entry.get("raw_label", "")).lower()
        if key in status_map:
            entry["status"], entry["merge_rejection_reason"] = status_map[key]
    with open(_PENDING_PATH, "w", encoding="utf-8") as f:
        yaml.dump({"pending": entries}, f, default_flow_style=False,
                  allow_unicode=True)


# ---------------------------------------------------------------------------
# Version management
# ---------------------------------------------------------------------------

def _current_version() -> str:
    """Read current version from latest snapshot directory name."""
    if not _SNAPSHOTS_DIR.exists():
        return "1.0"
    versions = sorted(
        [d.name for d in _SNAPSHOTS_DIR.iterdir() if d.is_dir() and d.name.startswith("v")],
        key=lambda v: [int(x) for x in v.lstrip("v").split(".")]
    )
    return versions[-1].lstrip("v") if versions else "1.0"


def _next_version(current: str) -> str:
    """Increment minor version: '1.2' → '1.3'."""
    parts = current.split(".")
    return f"{parts[0]}.{int(parts[1]) + 1}"


def _write_snapshot(version: str) -> None:
    """Write immutable snapshot of all alias system files."""
    snap_dir = _SNAPSHOTS_DIR / f"v{version}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    files = [
        (_ALIASES_PATH,   "field_aliases.yaml"),
        (_THRESHOLD_PATH, "threshold_config.yaml"),
        (_BLOCKED_PATH,   "blocked_patterns.yaml"),
        (_REGISTRY_DIR / "ambiguous_labels.yaml", "ambiguous_labels.yaml"),
    ]
    for src, name in files:
        if src.exists():
            shutil.copy2(src, snap_dir / name)
    logger.info("alias_merger: snapshot written → %s", snap_dir)


def _write_changelog(
    version: str,
    triggered_by: str,
    merged: list[dict],
    rejected: list[RejectedEntry],
    precision_before: float,
    precision_after: float,
    override_reason: str | None,
) -> None:
    """Write behavioral delta changelog entry."""
    _CHANGELOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "version":    version,
        "timestamp":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "triggered_by": triggered_by,
        "precision_before": precision_before,
        "precision_after":  precision_after,
        "override_reason":  override_reason or "",
        "aliases_added": [
            {
                "raw_label":       e.get("raw_label"),
                "canonical_field": e.get("canonical_field"),
                "approval_type":   e.get("approval_type"),
                "reviewer_id":     e.get("reviewer_id"),
                "confidence":      e.get("confidence"),
                "source_workpaper": e.get("source_workpaper"),
            }
            for e in merged
        ],
        "aliases_rejected_at_merge": [
            {
                "raw_label": r.raw_label,
                "canonical_field": r.canonical_field,
                "gate": r.gate,
                "reason": r.reason,
            }
            for r in rejected
        ],
    }
    path = _CHANGELOG_DIR / f"v{version}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(entry, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
    logger.info("alias_merger: changelog written → %s", path)


# ---------------------------------------------------------------------------
# Main merge entry point
# ---------------------------------------------------------------------------

def merge_pending(
    override_reason: str | None = None,
    triggered_by: str = "system",
) -> MergeResult:
    """
    Validate and merge all pending alias updates into field_aliases.yaml.

    Parameters
    ----------
    override_reason : str | None
        If set, allows merge to proceed even if gold precision drops below
        threshold. The reason is recorded in the changelog.
    triggered_by : str
        Reviewer ID or system identifier. Recorded in changelog.

    Returns
    -------
    MergeResult
        Summary of what was merged, rejected, and precision impact.
    """
    cfg           = _load_thresholds()
    min_precision = cfg.get("merge", {}).get("min_gold_precision", 0.85)
    max_batch     = cfg.get("merge", {}).get("max_batch_size", 50)

    pending = _load_pending()
    result  = MergeResult()

    if not pending:
        logger.info("alias_merger: no pending entries to merge")
        return result

    if len(pending) > max_batch:
        logger.warning(
            "alias_merger: %d pending entries exceeds max_batch_size %d — "
            "split into smaller batches", len(pending), max_batch
        )
        pending = pending[:max_batch]

    # --- Gate 1-4: per-entry structural validation ---
    valid_entries:    list[dict]          = []
    rejected_entries: list[RejectedEntry] = []

    for entry in pending:
        vr = validate_entry(entry)
        if vr.valid:
            valid_entries.append(entry)
        else:
            rejected_entries.append(RejectedEntry(
                raw_label       = entry.get("raw_label", ""),
                canonical_field = entry.get("canonical_field", ""),
                gate            = vr.gate,
                reason          = vr.reason,
            ))
            logger.warning(
                "alias_merger: rejected '%s' → '%s' at %s: %s",
                entry.get("raw_label"), entry.get("canonical_field"),
                vr.gate, vr.reason,
            )

    if not valid_entries:
        result.rejected = rejected_entries
        logger.info("alias_merger: all %d entries failed validation", len(pending))
        _update_pending_statuses([
            (r.raw_label, "rejected_at_merge", r.reason) for r in rejected_entries
        ])
        return result

    # --- Gate 5: gold precision check on tentative merged state ---
    current_aliases = _load_existing_aliases()
    result.precision_before = _compute_gold_precision(current_aliases)

    tentative_aliases = current_aliases
    for entry in valid_entries:
        tentative_aliases = _write_alias_to_file(
            entry["raw_label"], entry["canonical_field"], tentative_aliases
        )

    result.precision_after = _compute_gold_precision(tentative_aliases)

    if (result.precision_after < min_precision
            and result.precision_after < result.precision_before
            and not override_reason):
        result.blocked_by_precision = True
        result.rejected = rejected_entries
        logger.warning(
            "alias_merger: merge BLOCKED — precision would drop %.3f → %.3f "
            "(min=%.3f). Supply override_reason to force.",
            result.precision_before, result.precision_after, min_precision,
        )
        return result

    # --- Commit: write to field_aliases.yaml ---
    _save_aliases(tentative_aliases)
    logger.info(
        "alias_merger: wrote %d aliases to field_aliases.yaml",
        len(valid_entries),
    )

    # --- Version + snapshot + changelog ---
    current_ver = _current_version()
    new_ver     = _next_version(current_ver)
    _write_snapshot(new_ver)
    _write_changelog(
        version          = new_ver,
        triggered_by     = triggered_by,
        merged           = valid_entries,
        rejected         = rejected_entries,
        precision_before = result.precision_before,
        precision_after  = result.precision_after,
        override_reason  = override_reason,
    )

    # --- Update pending file statuses ---
    status_updates = [
        (e["raw_label"], "merged", "") for e in valid_entries
    ] + [
        (r.raw_label, "rejected_at_merge", r.reason) for r in rejected_entries
    ]
    _update_pending_statuses(status_updates)

    # --- Invalidate caches ---
    try:
        from auditai_data_normalization.normalize import reset_alias_cache
        reset_alias_cache()
    except Exception:
        pass
    try:
        from auditai_data_normalization.alias_fuzzy import reset_fuzzy_cache
        reset_fuzzy_cache()
    except Exception:
        pass

    result.merged   = [e["raw_label"] for e in valid_entries]
    result.rejected = rejected_entries
    result.version  = new_ver

    logger.info(
        "alias_merger: merge complete → v%s  merged=%d rejected=%d "
        "precision %.3f → %.3f",
        new_ver, len(result.merged), len(result.rejected),
        result.precision_before, result.precision_after,
    )
    return result