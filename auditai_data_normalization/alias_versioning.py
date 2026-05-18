"""
auditai_data_normalization/alias_versioning.py
===============================================
Public interface for alias system version history, inspection, and rollback.

alias_merger.py owns writing snapshots and changelogs during a merge.
This module owns READING and ACTING on that version history:
    - listing all versions with their deltas
    - inspecting what changed between versions
    - rolling back to a previous snapshot
    - identifying which version was active when a pair_hash was produced
    - health reporting across the version history

Nothing in this module writes to field_aliases.yaml directly.
Rollback is the only mutating operation — it copies a snapshot back into
place and triggers alias_merger.py to record it as a new version entry.

Public API
----------
    list_versions()                        -> list[VersionSummary]
    get_changelog(version)                 -> dict
    diff_versions(version_a, version_b)    -> VersionDiff
    rollback(target_version, reason, by)   -> str   (new version string)
    current_version()                      -> str
    version_for_pair(pair_hash)            -> str | None
    health_report()                        -> VersionHealthReport
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PKG_DIR     = Path(__file__).parent
_PROJECT_DIR = _PKG_DIR.parent

_ALIASES_PATH   = _PKG_DIR / "field_aliases.yaml"
_REGISTRY_DIR   = _PKG_DIR / "alias_registry"
_THRESHOLD_PATH = _REGISTRY_DIR / "threshold_config.yaml"
_BLOCKED_PATH   = _REGISTRY_DIR / "blocked_patterns.yaml"
_AMBIGUOUS_PATH = _REGISTRY_DIR / "ambiguous_labels.yaml"

_SNAPSHOTS_DIR = _PROJECT_DIR / "config" / "alias_registry" / "snapshots"
_CHANGELOG_DIR = _PROJECT_DIR / "config" / "alias_registry" / "changelog"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VersionSummary:
    version:          str
    timestamp:        str
    triggered_by:     str
    aliases_added:    int
    aliases_rejected: int
    precision_before: float
    precision_after:  float
    override_reason:  str


@dataclass
class VersionDiff:
    version_a:      str
    version_b:      str
    added:          list[dict]    # aliases in B not in A
    removed:        list[dict]    # aliases in A not in B
    threshold_changes: list[dict] # thresholds that changed between versions


@dataclass
class VersionHealthReport:
    total_versions:       int
    current_version:      str
    current_alias_count:  int
    precision_trend:      list[tuple[str, float]]  # [(version, precision)]
    rollbacks:            list[str]                # versions that were rollbacks
    largest_batch:        tuple[str, int]          # (version, count)
    precision_drops:      list[tuple[str, float, float]]  # (ver, before, after)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _sorted_versions() -> list[str]:
    """Return all version strings sorted ascending."""
    if not _SNAPSHOTS_DIR.exists():
        return []
    dirs = [
        d.name.lstrip("v")
        for d in _SNAPSHOTS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("v")
    ]
    return sorted(dirs, key=lambda v: [int(x) for x in v.split(".")])


def _snapshot_dir(version: str) -> Path:
    return _SNAPSHOTS_DIR / f"v{version}"


def _changelog_path(version: str) -> Path:
    return _CHANGELOG_DIR / f"v{version}.yaml"


def _load_aliases_at(version: str) -> dict[str, list[str]]:
    path = _snapshot_dir(version) / "field_aliases.yaml"
    return _load_yaml(path)


def _load_thresholds_at(version: str) -> dict:
    path = _snapshot_dir(version) / "threshold_config.yaml"
    return _load_yaml(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def current_version() -> str:
    """Return the latest version string, e.g. '1.3'."""
    versions = _sorted_versions()
    return versions[-1] if versions else "1.0"


def list_versions() -> list[VersionSummary]:
    """
    Return a summary of every version in the history, oldest first.
    """
    summaries = []
    for ver in _sorted_versions():
        cl = _load_yaml(_changelog_path(ver))
        summaries.append(VersionSummary(
            version          = ver,
            timestamp        = cl.get("timestamp", ""),
            triggered_by     = cl.get("triggered_by", ""),
            aliases_added    = len(cl.get("aliases_added") or []),
            aliases_rejected = len(cl.get("aliases_rejected_at_merge") or []),
            precision_before = cl.get("precision_before", 0.0),
            precision_after  = cl.get("precision_after", 0.0),
            override_reason  = cl.get("override_reason", ""),
        ))
    return summaries


def get_changelog(version: str) -> dict:
    """Return the full changelog dict for a specific version."""
    path = _changelog_path(version)
    if not path.exists():
        raise FileNotFoundError(f"No changelog found for version {version}")
    return _load_yaml(path)


def diff_versions(version_a: str, version_b: str) -> VersionDiff:
    """
    Compare two versions and return what changed between them.

    Parameters
    ----------
    version_a : str  — older version
    version_b : str  — newer version

    Returns
    -------
    VersionDiff with added, removed aliases and threshold changes.
    """
    aliases_a = _load_aliases_at(version_a)
    aliases_b = _load_aliases_at(version_b)

    # Build flat sets of (canonical_field, variant) tuples
    def _flat(aliases: dict) -> set[tuple[str, str]]:
        result = set()
        for canonical, variants in aliases.items():
            for v in (variants or []):
                result.add((canonical, str(v).lower().strip()))
        return result

    flat_a = _flat(aliases_a)
    flat_b = _flat(aliases_b)

    added   = [{"canonical_field": f, "variant": v} for f, v in sorted(flat_b - flat_a)]
    removed = [{"canonical_field": f, "variant": v} for f, v in sorted(flat_a - flat_b)]

    # Threshold changes
    thr_a = _load_thresholds_at(version_a)
    thr_b = _load_thresholds_at(version_b)
    threshold_changes = _diff_thresholds(thr_a, thr_b)

    return VersionDiff(
        version_a          = version_a,
        version_b          = version_b,
        added              = added,
        removed            = removed,
        threshold_changes  = threshold_changes,
    )


def _diff_thresholds(a: dict, b: dict, prefix: str = "") -> list[dict]:
    """Recursively find changed threshold values."""
    changes = []
    all_keys = set(a.keys()) | set(b.keys())
    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        val_a = a.get(key)
        val_b = b.get(key)
        if isinstance(val_a, dict) or isinstance(val_b, dict):
            changes.extend(_diff_thresholds(
                val_a or {}, val_b or {}, prefix=full_key
            ))
        elif val_a != val_b:
            changes.append({
                "key":   full_key,
                "before": val_a,
                "after":  val_b,
            })
    return changes


def rollback(
    target_version: str,
    reason: str,
    triggered_by: str = "system",
) -> str:
    """
    Roll back the alias system to a previous snapshot.

    Copies the target snapshot files back into place, then calls
    alias_merger to record the rollback as a new version entry
    (so history is never rewritten — rollback appears as a forward event).

    Parameters
    ----------
    target_version : str  — version to restore, e.g. "1.2"
    reason         : str  — mandatory explanation recorded in changelog
    triggered_by   : str  — reviewer ID

    Returns
    -------
    str — the new version string created for the rollback event.

    Raises
    ------
    FileNotFoundError if target_version snapshot does not exist.
    ValueError if reason is empty.
    """
    if not reason or not reason.strip():
        raise ValueError("rollback() requires a non-empty reason")

    snap_dir = _snapshot_dir(target_version)
    if not snap_dir.exists():
        raise FileNotFoundError(
            f"Snapshot for version {target_version} not found at {snap_dir}"
        )

    logger.warning(
        "alias_versioning: ROLLBACK to v%s triggered by %s — reason: %s",
        target_version, triggered_by, reason,
    )

    # Restore files from snapshot
    restore_map = {
        "field_aliases.yaml":   _ALIASES_PATH,
        "threshold_config.yaml": _THRESHOLD_PATH,
        "blocked_patterns.yaml": _BLOCKED_PATH,
        "ambiguous_labels.yaml": _AMBIGUOUS_PATH,
    }
    for snap_file, dest in restore_map.items():
        src = snap_dir / snap_file
        if src.exists():
            shutil.copy2(src, dest)
            logger.info("alias_versioning: restored %s", snap_file)

    # Invalidate caches
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

    # Record rollback as a new forward version entry
    from auditai_data_normalization.alias_merger import (
        _current_version, _next_version, _write_snapshot, _write_changelog
    )
    cur_ver = _current_version()
    new_ver = _next_version(cur_ver)
    _write_snapshot(new_ver)
    _write_changelog(
        version          = new_ver,
        triggered_by     = triggered_by,
        merged           = [],
        rejected         = [],
        precision_before = 0.0,
        precision_after  = 0.0,
        override_reason  = f"ROLLBACK to v{target_version}: {reason}",
    )

    logger.info(
        "alias_versioning: rollback complete — restored v%s, recorded as v%s",
        target_version, new_ver,
    )
    return new_ver


def version_for_pair(pair_hash: str) -> str | None:
    """
    Find which alias version was active when a training pair was produced.

    Searches changelogs for the pair_hash in aliases_added metadata.
    Returns the version string or None if not found.

    Note: pair_hash linkage requires that the pipeline records pair_hashes
    in the changelog at write time. This is a forward-looking lookup —
    pairs produced before this system was in place will not be found.
    """
    for ver in _sorted_versions():
        cl = _load_yaml(_changelog_path(ver))
        for entry in (cl.get("pair_hashes") or []):
            if entry == pair_hash:
                return ver
    return None


def health_report() -> VersionHealthReport:
    """
    Return a health summary across all version history.
    Useful for the Streamlit sidebar and debugging sessions.
    """
    versions    = _sorted_versions()
    summaries   = list_versions()

    if not versions:
        return VersionHealthReport(
            total_versions=0, current_version="none",
            current_alias_count=0, precision_trend=[],
            rollbacks=[], largest_batch=("none", 0),
            precision_drops=[],
        )

    # Current alias count
    current_aliases = _load_aliases_at(versions[-1])
    alias_count = sum(len(v or []) for v in current_aliases.values())

    # Precision trend
    precision_trend = [
        (s.version, s.precision_after) for s in summaries
    ]

    # Rollback versions
    rollbacks = [
        s.version for s in summaries
        if "ROLLBACK" in (s.override_reason or "")
    ]

    # Largest batch
    largest = max(summaries, key=lambda s: s.aliases_added, default=None)
    largest_batch = (largest.version, largest.aliases_added) if largest else ("none", 0)

    # Precision drops
    precision_drops = [
        (s.version, s.precision_before, s.precision_after)
        for s in summaries
        if s.precision_after < s.precision_before
    ]

    return VersionHealthReport(
        total_versions      = len(versions),
        current_version     = versions[-1],
        current_alias_count = alias_count,
        precision_trend     = precision_trend,
        rollbacks           = rollbacks,
        largest_batch       = largest_batch,
        precision_drops     = precision_drops,
    )