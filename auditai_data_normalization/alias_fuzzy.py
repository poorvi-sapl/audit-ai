"""
auditai_data_normalization/alias_fuzzy.py
==========================================
Fuzzy matching layer for alias resolution.

Called by normalize.py resolve_aliases() when a label survives:
    1. blocked_patterns check     — label is not noise
    2. alias_normalizer pipeline  — label is normalized to clean units
    3. deterministic alias lookup — no exact match found

For each normalized unit this module:
    1. Checks ambiguous_labels.yaml → routes to manual bucket with context note
    2. Checks rejected_mappings.yaml → suppresses previously rejected proposals
    3. Runs rapidfuzz WRatio against all 422 known alias variants
    4. Returns a FuzzyMatch with structured confidence object {score, source,
       method_weight} and a routing bucket decision

Routing buckets (from threshold_config.yaml)
--------------------------------------------
    raw_score >= auto_approve       → HIGH_CONFIDENCE  (quick one-click confirm)
    raw_score >= quick_confirm      → QUICK_CONFIRM     (side-by-side review)
    raw_score <  quick_confirm      → to LLM suggester  (None returned here)
    ambiguous label                 → MANUAL_REVIEW     (context note attached)
    previously rejected             → MANUAL_REVIEW     (rejection note attached)

Routing thresholds apply to the RAW fuzzy score (0.0–1.0).
The effective_score (raw * method_weight) is stored in the confidence object
for downstream comparison only — e.g. so a fuzzy 0.91 and LLM 0.91 are
distinguishable when audit trail records are reviewed.

Public API
----------
    match(normalized_label, source_label) -> FuzzyMatch | None
        Main entry point. Returns FuzzyMatch if score meets quick_confirm
        floor, None if score is too low (routes to LLM).

    FuzzyMatch
        .canonical_field  str
        .raw_label        str   — original pre-normalization label
        .matched_variant  str   — the alias variant that was matched
        .confidence       ConfidenceObject
        .bucket           RoutingBucket
        .rejection_note   str | None
        .context_note     str | None
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).parent
_CFG_DIR = _PKG_DIR / "alias_registry"
_ALIASES_PATH = _PKG_DIR / "field_aliases.yaml"

_BLOCKED_PATH   = _CFG_DIR / "blocked_patterns.yaml"
_REJECTED_PATH  = _CFG_DIR / "rejected_mappings.yaml"
_AMBIGUOUS_PATH = _CFG_DIR / "ambiguous_labels.yaml"
_THRESHOLD_PATH = _CFG_DIR / "threshold_config.yaml"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RoutingBucket(str, Enum):
    HIGH_CONFIDENCE = "high_confidence"   # one-click confirm in Streamlit
    QUICK_CONFIRM   = "quick_confirm"     # side-by-side score review
    MANUAL_REVIEW   = "manual_review"     # ambiguous or previously rejected
    # NOTE: no LLM bucket here — None return signals "send to LLM"


@dataclass
class ConfidenceObject:
    """Structured confidence — never a bare float."""
    score:         float   # raw fuzzy score 0.0–1.0
    source:        str     # always "fuzzy" from this module
    method_weight: float   # from threshold_config method_weights.fuzzy
    effective_score: float  # score * method_weight — used for routing

    def as_dict(self) -> dict:
        return {
            "score":          round(self.score, 4),
            "source":         self.source,
            "method_weight":  self.method_weight,
            "effective_score": round(self.effective_score, 4),
        }


@dataclass
class FuzzyMatch:
    """Result of a successful fuzzy match attempt."""
    canonical_field:  str
    raw_label:        str              # pre-normalization label
    normalized_label: str              # post-normalization unit that matched
    matched_variant:  str              # alias variant in field_aliases.yaml
    confidence:       ConfidenceObject
    bucket:           RoutingBucket
    rejection_note:   str | None = None   # set if previously rejected pair found
    context_note:     str | None = None   # set for ambiguous labels


# ---------------------------------------------------------------------------
# Config loaders (all lru_cached — invalidated by reset_fuzzy_cache())
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_thresholds() -> dict:
    with open(_THRESHOLD_PATH) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_alias_variants() -> list[tuple[str, str]]:
    """
    Load all (variant, canonical_field) pairs from field_aliases.yaml.
    Returns a flat list used as the rapidfuzz matching pool.
    """
    if not _ALIASES_PATH.exists():
        return []
    with open(_ALIASES_PATH) as f:
        aliases = yaml.safe_load(f) or {}
    pairs = []
    for canonical, variants in aliases.items():
        for variant in (variants or []):
            pairs.append((str(variant).lower().strip(), canonical))
    logger.debug("alias_fuzzy: loaded %d variant pairs", len(pairs))
    return pairs


@lru_cache(maxsize=1)
def _load_rejected_pairs() -> set[tuple[str, str]]:
    """
    Load all (normalized_label, canonical_field) rejected pairs.
    Returns a set for O(1) lookup.
    """
    if not _REJECTED_PATH.exists():
        return set()
    with open(_REJECTED_PATH) as f:
        data = yaml.safe_load(f) or {}
    pairs = set()
    for entry in (data.get("rejections") or []):
        label = str(entry.get("raw_label", "")).lower().strip()
        canonical = str(entry.get("rejected_canonical", "")).strip()
        if label and canonical:
            pairs.add((label, canonical))
    logger.debug("alias_fuzzy: loaded %d rejected pairs", len(pairs))
    return pairs


@lru_cache(maxsize=1)
def _load_ambiguous_labels() -> dict[str, dict]:
    """
    Load ambiguous labels as {normalized_label: entry_dict}.
    """
    if not _AMBIGUOUS_PATH.exists():
        return {}
    with open(_AMBIGUOUS_PATH) as f:
        data = yaml.safe_load(f) or {}
    result = {}
    for entry in (data.get("ambiguous_labels") or []):
        label = str(entry.get("label", "")).lower().strip()
        if label:
            result[label] = entry
    return result


@lru_cache(maxsize=1)
def _load_blocked_exact() -> set[str]:
    """Load exact blocked strings for fast pre-check."""
    if not _BLOCKED_PATH.exists():
        return set()
    with open(_BLOCKED_PATH) as f:
        data = yaml.safe_load(f) or {}
    return set(str(s).lower().strip() for s in (data.get("exact_strings") or []))


@lru_cache(maxsize=1)
def _load_blocked_patterns() -> list[re.Pattern]:
    """Load compiled regex blocked patterns."""
    if not _BLOCKED_PATH.exists():
        return []
    with open(_BLOCKED_PATH) as f:
        data = yaml.safe_load(f) or {}
    return [re.compile(p) for p in (data.get("regex_patterns") or [])]


def reset_fuzzy_cache() -> None:
    """
    Invalidate all lru_caches. Call after alias file updates or
    after alias_merger.py completes a merge.
    """
    _load_thresholds.cache_clear()
    _load_alias_variants.cache_clear()
    _load_rejected_pairs.cache_clear()
    _load_ambiguous_labels.cache_clear()
    _load_blocked_exact.cache_clear()
    _load_blocked_patterns.cache_clear()
    logger.debug("alias_fuzzy: all caches cleared")


# ---------------------------------------------------------------------------
# Blocked check
# ---------------------------------------------------------------------------

def _is_blocked(normalized: str) -> bool:
    """Return True if label matches any blocked pattern."""
    if normalized in _load_blocked_exact():
        return True
    for pattern in _load_blocked_patterns():
        if pattern.fullmatch(normalized):
            return True
    return False


# ---------------------------------------------------------------------------
# Core match function
# ---------------------------------------------------------------------------

def match(
    normalized_label: str,
    raw_label: str = "",
) -> FuzzyMatch | None:
    """
    Attempt fuzzy match for a single normalized label unit.

    Parameters
    ----------
    normalized_label : str
        Post-normalization label unit from alias_normalizer.normalize_label().
    raw_label : str
        Original pre-normalization label (for provenance in FuzzyMatch).

    Returns
    -------
    FuzzyMatch | None
        FuzzyMatch if effective_score >= quick_confirm_floor.
        None if score is too low — caller should route to LLM suggester.
        FuzzyMatch with bucket=MANUAL_REVIEW if ambiguous or rejected.
    """
    cfg          = _load_thresholds()
    fuzzy_cfg    = cfg["fuzzy"]
    weights      = cfg["method_weights"]
    method_wt    = weights["fuzzy"]
    auto_thr     = fuzzy_cfg["auto_approve"]
    confirm_thr  = fuzzy_cfg["quick_confirm_floor"]
    min_len      = fuzzy_cfg["min_label_length"]
    top_n        = fuzzy_cfg["top_n_candidates"]

    normalized = normalized_label.strip().lower()

    # Guard: minimum length
    if len(normalized) < min_len:
        return None

    # Guard: blocked patterns (safety net — should have been caught earlier)
    if _is_blocked(normalized):
        logger.debug("alias_fuzzy: blocked label slipped through: %r", normalized)
        return None

    # Check ambiguous labels — route to MANUAL_REVIEW immediately
    ambiguous = _load_ambiguous_labels()
    if normalized in ambiguous:
        entry = ambiguous[normalized]
        candidates = entry.get("candidate_fields", [])
        signals    = entry.get("context_signals", [])
        note = (
            f"Ambiguous label — could map to: {', '.join(candidates)}. "
            f"Context needed: {'; '.join(signals)}"
        )
        logger.info("alias_fuzzy: ambiguous label %r → MANUAL_REVIEW", normalized)
        return FuzzyMatch(
            canonical_field  = "",
            raw_label        = raw_label or normalized_label,
            normalized_label = normalized,
            matched_variant  = "",
            confidence       = ConfidenceObject(
                score=0.0, source="fuzzy",
                method_weight=method_wt, effective_score=0.0
            ),
            bucket       = RoutingBucket.MANUAL_REVIEW,
            context_note = note,
        )

    # Build variant pool and choices list
    variant_pairs = _load_alias_variants()
    if not variant_pairs:
        logger.warning("alias_fuzzy: no alias variants loaded — skipping fuzzy")
        return None

    choices = [v for v, _ in variant_pairs]

    # rapidfuzz WRatio — handles token order differences well
    # e.g. "year end fiscal" still matches "fiscal year end"
    results = process.extract(
        normalized,
        choices,
        scorer=fuzz.WRatio,
        limit=top_n,
        score_cutoff=confirm_thr * 100,  # raw score cutoff — routing on raw
    )

    if not results:
        return None

    # Take best result
    best_variant, raw_score_pct, best_idx = results[0]
    raw_score       = raw_score_pct / 100.0
    effective_score = raw_score * method_wt  # stored in confidence obj only

    if raw_score < confirm_thr:
        return None

    # Resolve canonical field from variant index
    canonical_field = variant_pairs[best_idx][1]

    # Check rejected pairs — suppress if this exact (label, field) was rejected
    rejected = _load_rejected_pairs()
    rejection_note = None
    if (normalized, canonical_field) in rejected:
        rejection_note = (
            f"Previously rejected: {repr(normalized)} → {repr(canonical_field)}. "
            "Review rejection reason in rejected_mappings.yaml before approving."
        )
        logger.info(
            "alias_fuzzy: suppressed rejected pair (%r → %r)",
            normalized, canonical_field,
        )
        return FuzzyMatch(
            canonical_field  = canonical_field,
            raw_label        = raw_label or normalized_label,
            normalized_label = normalized,
            matched_variant  = best_variant,
            confidence       = ConfidenceObject(
                score=raw_score, source="fuzzy",
                method_weight=method_wt,
                effective_score=round(effective_score, 4),
            ),
            bucket         = RoutingBucket.MANUAL_REVIEW,
            rejection_note = rejection_note,
        )

    # Route to correct bucket based on raw score
    if raw_score >= auto_thr:
        bucket = RoutingBucket.HIGH_CONFIDENCE
    else:
        bucket = RoutingBucket.QUICK_CONFIRM

    logger.info(
        "alias_fuzzy: %r → %r (variant=%r raw=%.3f eff=%.3f bucket=%s)",
        normalized, canonical_field, best_variant,
        raw_score, effective_score, bucket.value,
    )

    return FuzzyMatch(
        canonical_field  = canonical_field,
        raw_label        = raw_label or normalized_label,
        normalized_label = normalized,
        matched_variant  = best_variant,
        confidence       = ConfidenceObject(
            score=raw_score, source="fuzzy",
            method_weight=method_wt,
            effective_score=round(effective_score, 4),
        ),
        bucket = bucket,
    )


# ---------------------------------------------------------------------------
# Batch entry point (used by normalize.py for a full label dict)
# ---------------------------------------------------------------------------

def match_all(
    unknown_labels: dict[str, Any],
) -> dict[str, FuzzyMatch | None]:
    """
    Run fuzzy matching on all unknown labels from resolve_aliases().

    Parameters
    ----------
    unknown_labels : dict[str, Any]
        {raw_label: extracted_value} for labels that failed deterministic lookup.

    Returns
    -------
    dict[str, FuzzyMatch | None]
        {raw_label: FuzzyMatch} for matched labels.
        {raw_label: None} for labels that should go to LLM suggester.
    """
    from auditai_data_normalization.alias_normalizer import normalize_label

    results: dict[str, FuzzyMatch | None] = {}

    for raw_label in unknown_labels:
        units = normalize_label(raw_label)
        if not units:
            results[raw_label] = None
            continue

        # Try each normalized unit — take first successful match
        best: FuzzyMatch | None = None
        for unit in units:
            m = match(unit, raw_label=raw_label)
            if m is not None:
                if best is None or m.confidence.effective_score > best.confidence.effective_score:
                    best = m

        results[raw_label] = best

    return results