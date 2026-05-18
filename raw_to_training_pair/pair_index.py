"""
raw_to_training_pair/pair_index.py
====================================
Structural fingerprint index for cross-workpaper pair deduplication.

Problem
-------
The existing pair_hash (sha256 of full content) only catches exact duplicates.
Two engagement forms for slightly different clients with the same missing fields
produce different hashes but identical training signal — the model learns nothing
new from the second pair.

Approach
--------
Compute a canonical structural fingerprint from:
    (client_type, pair_type, sorted_missing_fields, sorted_present_fields)

All components are lowercased and sorted before hashing so the fingerprint is
stable across runs and independent of input ordering.

An index of seen fingerprints is maintained in a flat JSONL file alongside the
output JSONL files. Each entry records the fingerprint, the pair_hash it came
from, and the source workpaper — giving reviewers enough context to make an
informed decision when a duplicate is flagged.

Policy (from threshold_config.yaml)
------------------------------------
"soft"   — identical fingerprint → status "review_duplicate" in review queue.
           Reviewer sees which prior pair matches and from which workpaper.
           If approved, pair enters JSONL with duplicate_reviewed: true.
"strict" — identical fingerprint → hard block, pair never enters queue or JSONL.

Start with "soft". Move to "strict" once the dataset is stable and you have
reviewed enough duplicates to trust the threshold.

Public API
----------
    compute_fingerprint(pair, include_client_type) -> str
        Compute canonical fingerprint for a pair dict.

    check_duplicate(pair, index_path)  -> DuplicateResult
        Check if pair fingerprint exists in index.

    add_to_index(pair, index_path)     -> None
        Add pair fingerprint to index after approval.

    load_index(index_path)             -> list[dict]
        Load all fingerprint entries from index file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_DIR    = Path(__file__).parent.parent
_THRESHOLD_PATH = _PROJECT_DIR / "auditai_data_normalization" / "alias_registry" / "threshold_config.yaml"
_DEFAULT_INDEX  = _PROJECT_DIR / "data" / "pair_fingerprint_index.jsonl"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DuplicateResult:
    is_duplicate:      bool
    policy:            str        # "soft" | "strict"
    fingerprint:       str
    matching_entry:    dict | None = None   # the prior index entry that matched
    message:           str = ""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_dedup_config() -> dict:
    try:
        with open(_THRESHOLD_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("deduplication", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

def compute_fingerprint(
    pair: dict,
    include_client_type: bool = True,
) -> str:
    """
    Compute a canonical structural fingerprint for a training pair.

    The fingerprint is derived from:
        - client_type       (optional, from metadata)
        - pair_type         (clean | deficient)
        - missing_fields    (sorted, lowercased)
        - present_fields    (sorted, lowercased)

    All components are normalised before hashing so the same logical pair
    always produces the same fingerprint regardless of input field ordering
    or capitalisation variations.

    Parameters
    ----------
    pair : dict
        A training pair dict as written to review_queue.jsonl.
        Expected shape: {"messages": [...], "metadata": {...}}
    include_client_type : bool
        When True (default), client_type is part of the fingerprint —
        NPO and Government pairs with identical fields are NOT duplicates.

    Returns
    -------
    str
        16-character hex fingerprint (truncated sha256).
    """
    meta = pair.get("metadata", {})

    # Normalise each component
    client_type    = meta.get("client_type", "unknown").strip().lower()
    pair_type      = meta.get("pair_type", "unknown").strip().lower()
    missing_fields = sorted(
        f.strip().lower() for f in (meta.get("fields_missing") or [])
    )
    present_fields = sorted(
        f.strip().lower() for f in (meta.get("fields_present") or [])
    )

    # Build deterministic string representation
    components = []
    if include_client_type:
        components.append(f"client_type:{client_type}")
    components.append(f"pair_type:{pair_type}")
    components.append(f"missing:{','.join(missing_fields)}")
    components.append(f"present:{','.join(present_fields)}")

    canonical = "|".join(components)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Index operations
# ---------------------------------------------------------------------------

def load_index(index_path: Path | None = None) -> list[dict]:
    """Load all fingerprint entries from the index file."""
    path = index_path or _DEFAULT_INDEX
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def check_duplicate(
    pair: dict,
    index_path: Path | None = None,
) -> DuplicateResult:
    """
    Check if a pair's structural fingerprint already exists in the index.

    Parameters
    ----------
    pair : dict
        Training pair to check.
    index_path : Path | None
        Path to the fingerprint index. Defaults to data/pair_fingerprint_index.jsonl.

    Returns
    -------
    DuplicateResult
        .is_duplicate = True if fingerprint found in index.
        .matching_entry = the prior index entry for reviewer context.
        .policy = "soft" | "strict" from threshold_config.
    """
    cfg                  = _load_dedup_config()
    policy               = cfg.get("policy", "soft")
    include_client_type  = cfg.get("fingerprint_include_client_type", True)

    fingerprint = compute_fingerprint(pair, include_client_type=include_client_type)
    entries     = load_index(index_path)

    for entry in entries:
        if entry.get("fingerprint") == fingerprint:
            pair_hash     = pair.get("metadata", {}).get("pair_hash", "")[:16]
            prior_hash    = entry.get("pair_hash", "")[:16]
            prior_source  = entry.get("source_workpaper", "unknown")
            prior_version = entry.get("alias_version", "")

            message = (
                f"Structural duplicate of pair {prior_hash} "
                f"from workpaper '{prior_source}'"
                + (f" (alias v{prior_version})" if prior_version else "")
                + f". Policy: {policy}."
            )

            logger.info(
                "pair_index: duplicate fingerprint %s — current=%s prior=%s source=%s",
                fingerprint, pair_hash, prior_hash, prior_source,
            )

            return DuplicateResult(
                is_duplicate   = True,
                policy         = policy,
                fingerprint    = fingerprint,
                matching_entry = entry,
                message        = message,
            )

    return DuplicateResult(
        is_duplicate = False,
        policy       = policy,
        fingerprint  = fingerprint,
    )


def add_to_index(
    pair: dict,
    index_path: Path | None = None,
    alias_version: str = "",
) -> None:
    """
    Add a pair's fingerprint to the index after it is approved.

    Call this from pipeline.py after write_approved() succeeds.
    Only approved pairs enter the index — rejected or pending pairs
    do not, so the index reflects the actual JSONL content.

    Parameters
    ----------
    pair : dict
        The approved training pair.
    index_path : Path | None
        Defaults to data/pair_fingerprint_index.jsonl.
    alias_version : str
        Current alias system version at time of approval (for provenance).
    """
    cfg                 = _load_dedup_config()
    include_client_type = cfg.get("fingerprint_include_client_type", True)

    path        = index_path or _DEFAULT_INDEX
    meta        = pair.get("metadata", {})
    fingerprint = compute_fingerprint(pair, include_client_type=include_client_type)

    entry = {
        "fingerprint":      fingerprint,
        "pair_hash":        meta.get("pair_hash", "")[:16],
        "source_workpaper": meta.get("source_file", meta.get("file_name", "")),
        "client_type":      meta.get("client_type", ""),
        "pair_type":        meta.get("pair_type", ""),
        "fields_missing":   sorted(
            f.lower() for f in (meta.get("fields_missing") or [])
        ),
        "fields_present":   sorted(
            f.lower() for f in (meta.get("fields_present") or [])
        ),
        "alias_version":    alias_version,
        "added_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.debug(
        "pair_index: added fingerprint %s for %s/%s",
        fingerprint, meta.get("client_type", ""), meta.get("pair_type", ""),
    )


# ---------------------------------------------------------------------------
# Coverage utility
# ---------------------------------------------------------------------------

def index_stats(index_path: Path | None = None) -> dict:
    """
    Return summary statistics for the fingerprint index.
    Useful for Streamlit sidebar and batch health checks.
    """
    entries = load_index(index_path)
    if not entries:
        return {"total": 0, "by_client_type": {}, "by_pair_type": {}}

    by_client: dict[str, int] = {}
    by_type:   dict[str, int] = {}
    for e in entries:
        ct = e.get("client_type", "unknown")
        pt = e.get("pair_type", "unknown")
        by_client[ct] = by_client.get(ct, 0) + 1
        by_type[pt]   = by_type.get(pt, 0) + 1

    return {
        "total":          len(entries),
        "by_client_type": by_client,
        "by_pair_type":   by_type,
    }