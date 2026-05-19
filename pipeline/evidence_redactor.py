"""
pipeline/evidence_redactor.py
==============================
Evidence-consistent text redaction for counterfactual training pair generation.

Purpose
-------
When generating deficient training pairs, the user message must NOT contain
evidence that contradicts the assistant's deficiency findings. The previous
approach (label-flip with [TRAINING NOTE]) produced contradiction learning:
    input:  "workpaper shows field X is present"
    output: "finding: field X not documented"

This module removes the text evidence that supports a field being present
before the user message is assembled. The label flip then accurately reflects
what the model sees — no contradiction.

Redaction strategy: alias-based line-level removal
---------------------------------------------------
For each deficiency field:
  1. Look up its aliases from field_aliases.yaml (forward lookup)
  2. Filter to precision aliases (>= 2 words) to avoid false positives
  3. Scan each line of the workpaper text
  4. Remove lines containing any precision alias
  5. Insert [NOT DOCUMENTED: {field}] placeholder

If no evidence is found for a field, `redacted` = False and `failed_fields`
records it. The caller should either skip the variant or flag it for the
hard gate (Step 3).

Phase 1 reconciliation interaction
-----------------------------------
Phase 1 extraction ran on the pre-scrub original text, so it confirms the
field was present in the real workpaper. After evidence redaction, the
model text no longer contains that evidence — but Phase 1 reconciliation
would override the label flip back to "present" if not guarded.

Fix: pipeline.py must skip Phase 1 reconciliation for deficiency_fields
when building counterfactual pairs. See _process_single_variant_r7().

Public API
----------
    redact_fields(text, fields) -> RedactionResult
    load_aliases() -> dict[str, list[str]]    (forward alias lookup, cached)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_PROJECT_DIR  = Path(__file__).parent.parent
_ALIASES_PATH = _PROJECT_DIR / "auditai_data_normalization" / "field_aliases.yaml"

# Aliases shorter than this word count are too generic (e.g. "date", "partner",
# "services") and would incorrectly remove unrelated lines.
_MIN_ALIAS_WORDS = 2

# Maps canonical fields to the PII placeholder their extracted value was scrubbed
# to by the PII pipeline.  Used for span-level redaction: all occurrences of the
# placeholder across the full cleaned_text are replaced with
# [NOT DOCUMENTED: {field}] so the model cannot infer the field value from prose
# references either.
#
# Only field-UNIQUE placeholders belong here — shared ones like [PERSON] are not
# safe because replacing them would wipe evidence for unrelated fields (e.g.
# redacting preparer_id would also erase the engagement_partner's name).
_PII_SPAN_MAP: dict[str, str] = {
    "client_name": "[CLIENT_ENTITY]",
    "ein":         "[EIN]",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FieldRedactionResult:
    """Redaction outcome for one canonical field."""
    field:           str
    redacted:        bool        # True if evidence was removed (line-level OR span-level)
    removed_lines:   list[str]   # verbatim lines removed by alias matching
    matched_aliases: list[str]   # alias terms that triggered line removal
    span_redacted:   bool = False  # True if _PII_SPAN_MAP redaction also ran
    span_count:      int  = 0      # number of PII placeholder occurrences replaced


@dataclass
class RedactionResult:
    """Full redaction outcome for one deficiency variant."""
    redacted_text:  str                        # text with evidence lines removed
    field_results:  list[FieldRedactionResult]
    fully_redacted: bool                       # True when ALL fields had evidence removed
    failed_fields:  list[str]                  # fields with no evidence found in text


# ---------------------------------------------------------------------------
# Alias loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_aliases() -> dict[str, list[str]]:
    """
    Load field_aliases.yaml → {canonical_field: [lowercased alias terms]}.
    Cached after first call.
    """
    if not _ALIASES_PATH.exists():
        logger.warning("evidence_redactor: field_aliases.yaml not found at %s", _ALIASES_PATH)
        return {}
    with open(_ALIASES_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    result: dict[str, list[str]] = {}
    for canonical, variants in raw.items():
        terms = [canonical.lower().replace("_", " ")]
        for v in (variants or []):
            terms.append(str(v).lower().strip())
        result[canonical] = terms
    return result


def _precision_aliases(field_name: str, alias_lookup: dict[str, list[str]]) -> list[str]:
    """
    Return aliases for `field_name` that are >= _MIN_ALIAS_WORDS words.
    Sorted longest-first so the most specific pattern is tried first.

    Single-word aliases are excluded — they match too broadly in continuous
    prose (e.g. "date" or "services" would remove unrelated lines).
    """
    all_aliases = alias_lookup.get(field_name, [])
    precise = [a for a in all_aliases if len(a.split()) >= _MIN_ALIAS_WORDS]
    return sorted(precise, key=lambda a: -len(a.split()))


# ---------------------------------------------------------------------------
# Core redaction
# ---------------------------------------------------------------------------

def redact_fields(
    text:         str,
    fields:       list[str],
    alias_lookup: dict[str, list[str]] | None = None,
) -> RedactionResult:
    """
    Remove evidence lines for each field in `fields` from `text`.

    Processes fields sequentially — each field's redaction operates on the
    text as already modified by prior fields in the list.

    Parameters
    ----------
    text : str
        Workpaper text (cleaned_text from DocumentRecord).
    fields : list[str]
        Canonical field names designated as absent in this deficiency variant.
    alias_lookup : dict | None
        Forward alias map {canonical_field: [alias_terms]}. If None, loads
        from field_aliases.yaml.

    Returns
    -------
    RedactionResult
        .redacted_text   — text with evidence removed, ready for user message
        .field_results   — per-field outcome (redacted / failed, matched aliases)
        .fully_redacted  — True when every field had at least one line removed
        .failed_fields   — fields where no alias match was found in the text
    """
    if alias_lookup is None:
        alias_lookup = load_aliases()

    current_text = text
    field_results: list[FieldRedactionResult] = []

    for field_name in fields:
        aliases = _precision_aliases(field_name, alias_lookup)

        if not aliases:
            logger.debug(
                "evidence_redactor: no precision aliases (>=%d words) for '%s' — "
                "cannot redact evidence",
                _MIN_ALIAS_WORDS, field_name,
            )
            field_results.append(FieldRedactionResult(
                field=field_name, redacted=False,
                removed_lines=[], matched_aliases=[],
            ))
            continue

        lines      = current_text.split("\n")
        kept:     list[str] = []
        removed:  list[str] = []
        matched:  list[str] = []

        for line in lines:
            line_lower = line.lower()
            hit = next((a for a in aliases if a in line_lower), None)

            if hit:
                removed.append(line)
                if hit not in matched:
                    matched.append(hit)
                kept.append(f"[NOT DOCUMENTED: {field_name}]")
            else:
                kept.append(line)

        # Collapse consecutive duplicate placeholders for the same field
        deduped: list[str] = []
        placeholder = f"[NOT DOCUMENTED: {field_name}]"
        for ln in kept:
            if ln == placeholder and deduped and deduped[-1] == placeholder:
                continue
            deduped.append(ln)

        current_text = "\n".join(deduped)

        # Span-level redaction: replace PII placeholder occurrences field-wide.
        # Handles fields whose label is a single word ("Organization:") that
        # falls below _MIN_ALIAS_WORDS, yet whose value is uniquely identifiable
        # via a field-specific PII placeholder (e.g. [CLIENT_ENTITY] for
        # client_name).  Runs after line removal so the placeholder is only
        # targeted in lines that line-level redaction did not already remove.
        span_count = 0
        pii_placeholder = _PII_SPAN_MAP.get(field_name)
        if pii_placeholder:
            _pii_re = re.compile(re.escape(pii_placeholder), re.IGNORECASE)
            _pii_hits = _pii_re.findall(current_text)
            if _pii_hits:
                span_count = len(_pii_hits)
                current_text = _pii_re.sub(
                    f"[NOT DOCUMENTED: {field_name}]", current_text
                )
                logger.debug(
                    "evidence_redactor: '%s' — span-redacted %d occurrence(s) of '%s'",
                    field_name, span_count, pii_placeholder,
                )

        field_results.append(FieldRedactionResult(
            field=field_name,
            redacted=bool(removed) or bool(span_count),
            removed_lines=removed,
            matched_aliases=matched,
            span_redacted=bool(span_count),
            span_count=span_count,
        ))

        if removed:
            logger.debug(
                "evidence_redactor: '%s' — removed %d line(s) via aliases %s",
                field_name, len(removed), matched,
            )
        elif not span_count:
            logger.warning(
                "evidence_redactor: '%s' — no evidence lines found in text "
                "(aliases tried: %s). Pair may be contradictory.",
                field_name, aliases[:3],
            )

    failed_fields  = [r.field for r in field_results if not r.redacted]
    fully_redacted = len(failed_fields) == 0

    if failed_fields:
        logger.warning(
            "evidence_redactor: partial redaction — %d/%d fields not redacted: %s. "
            "Step 3 hard gate should flag these pairs.",
            len(failed_fields), len(fields), failed_fields,
        )

    return RedactionResult(
        redacted_text  = current_text,
        field_results  = field_results,
        fully_redacted = fully_redacted,
        failed_fields  = failed_fields,
    )


def reset_cache() -> None:
    """Force alias reload on next call. Useful in tests."""
    load_aliases.cache_clear()
