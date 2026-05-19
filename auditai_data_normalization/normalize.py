"""
auditai_data_normalization/normalize.py
=========================================
The single public entry point for the ETL normalization pipeline.

    from auditai_data_normalization.normalize import normalize_document
    record = normalize_document("/path/to/any/file.docx")

One call. Any format. Returns a DocumentRecord.

Phase A4 pipeline
-----------------
1.  Route
2.  Primary extraction
3.  Parallel secondary extraction (PDF only)
4.  PII stripping
5.  Field extraction + alias resolution
6.  LLM tiebreaker on disagreed Tier 1/2 fields  (tier-driven, not hardcoded)
7.  Confidence scoring  (tier-based, Phase A2)
8.  Set extraction_gate  (>= 0.50)
9.  LLM fallback         (Phase B2 — fires when extraction_gate=False, re-scores)
10. Provenance flags     (llm_assisted, flagged_fields, extraction_method — B4)
11. needs_review + metadata

Gate logic
----------
    extraction_gate = extraction_confidence >= 0.50
        True  → record.is_ready_for_drafting() → proceed to completion drafter
        False → LLM fallback (Phase B2), then re-score

    quality_gate    = review_confidence >= 0.70
        Owned by completion_drafter.py — normalize.py leaves it False.

    needs_review    = extraction_gate=False OR extraction failed/partial
        quality_gate=False also sets needs_review, but drafter owns that.
"""

from __future__ import annotations

import importlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from auditai_data_normalization.confidence import (
    ConfidenceSummary,
    TierConfig,
    load_tiers,
    score_confidence,
    score_fields,
    summarise,
)
from auditai_data_normalization.pii import scrub_record
from auditai_data_normalization.router import route, RouteResult
from auditai_data_normalization.schema import DocumentRecord
from auditai_data_normalization.text_normalizer import normalize_text
from auditai_data_normalization.extractors.structural_extractor import (
        extract as structural_extract,
        FieldEvidence,
    )

logger = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).parent
_ALIASES_PATH = _PKG_DIR / "field_aliases.yaml"
_PARALLEL_TIMEOUT = 60

# Phase A4 — extraction gate threshold
_EXTRACTION_GATE = 0.50


# ---------------------------------------------------------------------------
# Alias loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_aliases() -> dict[str, str]:
    if not _ALIASES_PATH.exists():
        logger.warning("field_aliases.yaml not found at %s", _ALIASES_PATH)
        return {}
    with open(_ALIASES_PATH) as f:
        raw = yaml.safe_load(f) or {}
    reverse: dict[str, str] = {}
    for canonical, variants in raw.items():
        if isinstance(variants, list):
            for v in variants:
                reverse[str(v).lower().strip()] = canonical
        reverse[canonical.lower().strip()] = canonical
    logger.debug("Loaded %d alias variants", len(reverse))
    return reverse


def resolve_aliases(raw_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve raw label keys to canonical field names.

    Resolution pipeline (per label):
        1. Deterministic alias lookup  (field_aliases.yaml exact match)
        2. alias_normalizer            (compound split, OCR fix, abbrev expand)
        3. alias_fuzzy                 (rapidfuzz WRatio against known variants)
            -> HIGH_CONFIDENCE / QUICK_CONFIRM  -> written to pending_alias_updates
            -> MANUAL_REVIEW (ambiguous/rejected) -> logged with context note
            -> None (score too low)              -> logged to LLM suggester (D1)
        4. Unresolved labels pass through as-is + logged to LLM suggester

    Fuzzy-matched labels are NOT immediately applied to resolved{}.
    They are written to pending_alias_updates.yaml for human review + merge.
    Only deterministic matches update the live resolved dict.
    """
    aliases  = load_aliases()
    resolved: dict[str, Any] = {}
    unmapped: dict[str, Any] = {}

    # Step 1 — deterministic alias lookup
    for key, val in raw_dict.items():
        canonical = aliases.get(key.lower().strip())
        if canonical:
            resolved[canonical] = val
        else:
            unmapped[key] = val

    if not unmapped:
        return resolved

    logger.debug("resolve_aliases: %d unmapped after deterministic lookup", len(unmapped))

    # Steps 2+3 — normalizer + fuzzy matching
    fuzzy_matched:  dict[str, Any] = {}
    still_unmapped: dict[str, Any] = {}

    try:
        from auditai_data_normalization.alias_fuzzy import match_all, RoutingBucket
        fuzzy_results = match_all(unmapped)

        for raw_label, fuzzy_match in fuzzy_results.items():
            val = unmapped[raw_label]

            if fuzzy_match is None:
                still_unmapped[raw_label] = val
                continue

            if fuzzy_match.bucket == RoutingBucket.MANUAL_REVIEW:
                still_unmapped[raw_label] = val
                logger.info(
                    "resolve_aliases: MANUAL_REVIEW for %r -- %s",
                    raw_label,
                    fuzzy_match.context_note or fuzzy_match.rejection_note or "",
                )
                _log_fuzzy_to_pending(raw_label, val, fuzzy_match)
                continue

            # HIGH_CONFIDENCE or QUICK_CONFIRM: queue for merge, pass through as-is
            fuzzy_matched[raw_label] = val
            _log_fuzzy_to_pending(raw_label, val, fuzzy_match)
            logger.info(
                "resolve_aliases: fuzzy %r -> %r (raw=%.3f bucket=%s) -> pending",
                raw_label, fuzzy_match.canonical_field,
                fuzzy_match.confidence.score, fuzzy_match.bucket.value,
            )

    except Exception as e:
        logger.debug("resolve_aliases: fuzzy matching failed -- %s", e)
        still_unmapped = unmapped
        fuzzy_matched  = {}

    resolved.update(fuzzy_matched)
    resolved.update(still_unmapped)

    # Step 4 — log still-unmapped to LLM suggester (D1)
    if still_unmapped:
        try:
            from auditai_data_normalization.alias_suggester import log_unknown
            for label, val in still_unmapped.items():
                log_unknown(
                    raw_label=label,
                    extracted_value=str(val)[:200] if val else "",
                    source_file="",
                    run_llm=False,
                )
        except Exception as e:
            logger.debug("resolve_aliases: alias logging failed -- %s", e)

    return resolved


def _log_fuzzy_to_pending(
    raw_label: str,
    extracted_value: Any,
    fuzzy_match: Any,
) -> None:
    """Write a fuzzy match result to pending_alias_updates.yaml."""
    try:
        from datetime import datetime, timezone
        import yaml as _yaml

        pending_path = Path(__file__).parent / "alias_registry" / "pending_alias_updates.yaml"
        if not pending_path.exists():
            return

        with open(pending_path, encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
        entries = data.get("pending") or []

        normalized = raw_label.lower().strip()

        # Final blocked guard on raw label — catches cases where normalization
        # expands a blocked token (e.g. "No" -> "number") past the fuzzy guard
        from auditai_data_normalization.alias_fuzzy import _load_blocked_exact, _load_blocked_patterns
        if normalized in _load_blocked_exact():
            return
        for _pat in _load_blocked_patterns():
            if _pat.fullmatch(normalized):
                return

        already = any(
            str(e.get("raw_label", "")).lower().strip() == normalized
            and e.get("canonical_field") == fuzzy_match.canonical_field
            and e.get("status") == "pending"
            for e in entries
        )
        if already:
            return

        entry = {
            "raw_label":        raw_label.strip(),
            "canonical_field":  fuzzy_match.canonical_field,
            "reviewer_id":      "",
            "approval_type":    fuzzy_match.bucket.value,
            "confidence":       fuzzy_match.confidence.as_dict(),
            "source_workpaper": "",
            "timestamp":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes":            fuzzy_match.context_note or fuzzy_match.rejection_note or "",
            "status":           "pending",
        }
        entries.append(entry)

        with open(pending_path, "w", encoding="utf-8") as f:
            _yaml.dump({"pending": entries}, f, default_flow_style=False, allow_unicode=True)

    except Exception as e:
        logger.debug("resolve_aliases: failed to write fuzzy match to pending -- %s", e)


# ---------------------------------------------------------------------------
# Extractor dispatch
# ---------------------------------------------------------------------------

def _call_extractor(extractor_path: str, file_path: Path) -> DocumentRecord:
    module_path, func_name = extractor_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)(file_path)


def _get_secondary_extractor_path(result: RouteResult) -> str | None:
    if result.file_type == "pdf_text":
        return "auditai_data_normalization.extractors.ocr_extractor.extract"
    if result.file_type == "pdf_scanned":
        return "auditai_data_normalization.extractors.pdf_text_extractor.extract"
    return None


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

_NOISE_LABELS = {
    "instructions", "practical consideration", "comments",
    "yes", "no", "n/a", "date", "transaction_date",
}

_LABEL_VALUE_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9 /&\-\.\(\)]{1,60}):\s*(.+)$"
)

# Matches an embedded "Label: " pattern inside an extracted value, e.g.
# "06/30/2025 Engagement Date: 07/14/2025" → truncate at " Engagement Date: "
_EMBEDDED_LABEL_RE = re.compile(
    r"\s+[A-Z][A-Za-z0-9 /&\-\.\(\)]{1,60}:\s+"
)

# ---------------------------------------------------------------------------
# Checkbox table helpers
# ---------------------------------------------------------------------------

# Characters used as visual checkmarks in DOCX Yes/No form tables.
#  = Wingdings checkmark (PUA — appears as ✔ in Wingdings font).
# X / x  = manual fill.  ✓ ✔ ☑ ☒ = Unicode variants.
_CHECKBOX_CHARS: frozenset[str] = frozenset({
    "",            # Wingdings checkmark (Private Use Area)
    "X", "x",           # manual fill
    "✓", "✔",  # ✓ ✔
    "☑", "☒",  # ☑ ☒
})


def _has_checkmark(cell: str) -> bool:
    """
    Return True if the cell contains a checked checkbox sentinel.

    Handles both raw characters and forms normalized by _get_cell_text():
      "true" = normalize_text() output for \\uf061 (Wingdings) and Unicode boxes
      "x"    = manual capital-X fill used in some DOCX templates (not normalized)
    """
    if not cell:
        return False
    c = cell.strip().lower()
    if c in ("true", "x"):
        return True
    return any(ch in cell for ch in _CHECKBOX_CHARS)


def _question_to_short_label(question_text: str) -> str:
    """
    Extract the short question label from a long table cell.

    Long question cells include SOP guidance text after the core question.
    Strategy: take first line, strip trailing '?', strip parenthetical guidance.

    Examples:
        "Single Audit?\\nPractical Consideration: ..."  → "Single Audit"
        "Audit of organization's financial statements in accordance with GAAS?
         (Specify.) \\nPractical Considerations: ..."
             → "Audit of organization's financial statements in accordance with GAAS"
    """
    first_line = question_text.split("\n")[0].strip()
    before_paren = first_line.split("(")[0].strip()
    return before_paren.rstrip("?").rstrip(":").strip()


# Canonical field labels — used by inverted row detection to recognise
# when a cell that appears AFTER a value cell is actually the label.
# Built at module load from field_aliases.yaml via load_aliases().
# Populated lazily on first call to _extract_fields_from_record().
_CANONICAL_LABEL_SET: frozenset[str] | None = None

def _get_canonical_labels() -> frozenset[str]:
    """
    Return the set of all known label variants from field_aliases.yaml
    (lowercase, stripped). Used by the inverted-row pass.

    Cached after first load — load_aliases() is already lru_cached.
    Call reset_alias_cache() to invalidate both caches together.
    """
    global _CANONICAL_LABEL_SET
    if _CANONICAL_LABEL_SET is None:
        aliases = load_aliases()
        _CANONICAL_LABEL_SET = frozenset(aliases.keys())
    return _CANONICAL_LABEL_SET


def reset_alias_cache() -> None:
    """
    Invalidate both alias caches atomically.

    Must be called after alias_suggester.approve() writes a new mapping
    to field_aliases.yaml. Keeps load_aliases() (lru_cached) and
    _CANONICAL_LABEL_SET (module global) in sync.

    Called by the Streamlit UI after every alias approval.
    Safe to call multiple times — idempotent.
    """
    global _CANONICAL_LABEL_SET
    load_aliases.cache_clear()
    _CANONICAL_LABEL_SET = None
    logger.debug("reset_alias_cache: load_aliases + _CANONICAL_LABEL_SET cleared")
 
 
def _extract_fields_from_record(record: DocumentRecord) -> dict[str, list[str]]:
    """
    Extract label: value pairs from tables and section text.
 
    Phase 1: normalize_text() covers Unicode, checkboxes, underscores,
             inline markers. splitlines() inner loop catches appended lines.
    Phase 3: Inverted row detection — reverse pass over table rows to
             catch label-below-value patterns.
    """
    raw: dict[str, str] = {}
 
    # ── Table pass (Phase 1 logic, unchanged) ────────────────────────────
    for table in record.tables:
        all_rows = ([table.headers] + table.rows) if table.headers else table.rows
        for row in all_rows:
            for cell in row:
                if not cell or not cell.strip():
                    continue
                clean = normalize_text(cell)
                if len(clean) > 300:
                    continue
                for line in clean.splitlines():
                    if ":" not in line:
                        continue
                    label, _, value = line.partition(":")
                    label = label.strip()
                    value = value.strip().lstrip(":")
                    # Truncate at any embedded "Label: " pattern so that
                    # "06/30/2025 Engagement Date: 07/14/2025" → "06/30/2025"
                    _em = _EMBEDDED_LABEL_RE.search(value)
                    if _em:
                        value = value[:_em.start()].strip()
                    if label and value and len(label) < 80:
                        if label.lower() not in _NOISE_LABELS:
                            raw[label] = value
 
    # ── Section pass (Phase 1 logic, unchanged) ──────────────────────────
    for section in record.sections:
        for line in section.content.splitlines():
            clean = normalize_text(line)
            for norm_line in clean.splitlines():
                m = _LABEL_VALUE_RE.match(norm_line.strip())
                if m:
                    label = m.group(1).strip()
                    value = m.group(2).strip()
                    if " | " in value:
                        value = value.split(" | ")[0].strip()
                    value = value.lstrip(":").strip()
                    _em = _EMBEDDED_LABEL_RE.search(value)
                    if _em:
                        value = value[:_em.start()].strip()
                    if label and value and len(value) < 300:
                        if label.lower() not in _NOISE_LABELS:
                            raw[label] = value
 
    # ── Phase 3 — Inverted row pass ──────────────────────────────────────
    # Some PPC tables extract with the value row ABOVE the label row:
    #
    #   Row i:    "John Smith, CPA"       ← value (no colon, looks like data)
    #   Row i+1:  "Engagement Partner"    ← label (known alias)
    #
    # Strategy: walk each table's rows. When row[i+1]'s first cell
    # matches a known canonical label AND row[i]'s first cell looks
    # like a value (no colon, not a known label), treat row[i] as the
    # value for that label.
    #
    # Only fires for cells that were NOT already captured by the forward
    # pass (avoids overwriting good data with inverted speculation).
 
    canonical_labels = _get_canonical_labels()
 
    for table in record.tables:
        all_rows = ([table.headers] + table.rows) if table.headers else table.rows
        flat: list[list[str]] = [list(row) for row in all_rows if row]
 
        for i in range(len(flat) - 1):
            current_row = flat[i]
            next_row    = flat[i + 1]
 
            # First non-empty cell of each row
            current_cell = next(
                (normalize_text(c).strip() for c in current_row if c.strip()), ""
            )
            next_cell = next(
                (normalize_text(c).strip() for c in next_row if c.strip()), ""
            )
 
            if not current_cell or not next_cell:
                continue
 
            # next_cell must be a known label variant
            if next_cell.lower() not in canonical_labels:
                continue
 
            # current_cell must look like a value (not a label, no colon)
            if current_cell.endswith(":"):
                continue
            if current_cell.lower() in canonical_labels:
                continue
            if current_cell.lower() in _NOISE_LABELS:
                continue
            # Skip if current cell is a checkbox sentinel
            if current_cell.lower() in ("true", "false"):
                continue
 
            # Resolve next_cell to canonical name via aliases
            canonical = load_aliases().get(next_cell.lower())
            if canonical is None:
                continue
 
            # Only write if the forward pass didn't already find this field
            # (raw uses raw label keys — check both canonical and raw forms)
            already_found = (
                canonical in raw
                or next_cell in raw
                or any(
                    load_aliases().get(k.lower()) == canonical
                    for k in raw
                )
            )
            if not already_found:
                raw[next_cell] = current_cell
                logger.debug(
                    "_extract_fields_from_record: inverted row — '%s': '%s'",
                    next_cell, current_cell,
                )
 
    # ── Phase 4 — Header-column mapping ─────────────────────────────────
    # Some DOCX tables have canonical labels as column headers and their
    # values in the first data row, e.g.:
    #
    #   headers: ["Engagement Partner", "Preparer ID", ...]
    #   rows[0]: ["Sanwar Harshwal",    "SH-042",      ...]
    #
    # The existing passes miss this because no cell contains "Label: value".
    # Walk headers; for each known canonical label header, map it to the
    # same-column value from rows[0] (if that column exists and has a value).

    for table in record.tables:
        if not table.headers or not table.rows:
            continue
        first_row = table.rows[0]
        for col_idx, header in enumerate(table.headers):
            header_norm = normalize_text(header).strip()
            if not header_norm or header_norm.lower() in _NOISE_LABELS:
                continue
            if header_norm.lower() not in canonical_labels:
                continue
            if col_idx >= len(first_row):
                continue
            cell_value = normalize_text(first_row[col_idx]).strip()
            if not cell_value or cell_value.lower() in _NOISE_LABELS:
                continue
            if cell_value.lower() in canonical_labels:
                continue
            if cell_value.lower() in ("true", "false"):
                continue
            canonical = load_aliases().get(header_norm.lower())
            if canonical is None:
                continue
            already_found = (
                canonical in raw
                or header_norm in raw
                or any(load_aliases().get(k.lower()) == canonical for k in raw)
            )
            if not already_found:
                raw[header_norm] = cell_value
                logger.debug(
                    "_extract_fields_from_record: header-col — '%s': '%s'",
                    header_norm, cell_value,
                )

    # ── Phase 6 — Same-row value-label extraction ────────────────────────
    # Some PPC sign-off tables store value and label as adjacent columns
    # in the same row (no separate header row):
    #
    #   Row: ["Sanwar Harshwal",   "Engagement Partner"]
    #         value col=0          label col=1
    #
    # The forward and header-column passes miss this because the label is
    # in the same row as the value (not a separate row or a preceding header).
    #
    # Strategy: scan each adjacent (cell[j], cell[j+1]) pair in every row.
    # If cell[j+1] is a canonical label and cell[j] looks like a value,
    # emit the pair.

    for table in record.tables:
        all_rows = ([table.headers] + table.rows) if table.headers else table.rows
        for row in all_rows:
            if not row or len(row) < 2:
                continue
            norm_row = [normalize_text(str(c)).strip() for c in row]
            for j in range(len(norm_row) - 1):
                label_candidate = norm_row[j + 1]
                value_candidate = norm_row[j]
                if not label_candidate or not value_candidate:
                    continue
                label_lower = label_candidate.lower()
                if label_lower not in canonical_labels:
                    continue
                if value_candidate.endswith(":"):
                    continue
                if value_candidate.lower() in _NOISE_LABELS:
                    continue
                if value_candidate.lower() in canonical_labels:
                    continue
                canonical = load_aliases().get(label_lower)
                if canonical is None:
                    continue
                already_found = canonical in raw or any(
                    load_aliases().get(k.lower()) == canonical for k in raw
                )
                if not already_found:
                    raw[label_candidate] = value_candidate
                    logger.debug(
                        "_extract_fields_from_record: same-row — '%s': '%s'",
                        label_candidate, value_candidate,
                    )

    # ── Phase 5 — Checkbox Yes/No table extraction ───────────────────────
    # PPC checklists (NPO-CX-1.1 and similar) use a 5-column table:
    #   col 0 = question text (long, with SOP guidance in subsequent lines)
    #   col 1 = blank spacer
    #   col 2 = Yes column  ← checkmark here when answer is Yes
    #   col 3 = No column   ← checkmark here when answer is No
    #   col 4 = Comments
    #
    # The label-value passes above miss these entirely because the question text
    # and the checkmark live in separate cells with no "Label: value" pattern.
    # This pass detects the column structure by locating "Yes"/"No" header cells,
    # then reads raw cell content for checkbox sentinels on each data row.

    _aliases_rev = load_aliases()  # {alias_lower → canonical}

    for table in record.tables:
        all_rows = ([table.headers] + table.rows) if table.headers else table.rows
        if not all_rows:
            continue

        # Detect Yes/No column indices by scanning the first few rows.
        # The header row may be row 0 or buried within the first 5 rows.
        yes_col: int | None = None
        no_col:  int | None = None
        for scan_row in all_rows[:5]:
            norm_cells = [normalize_text(str(c)).strip().lower() for c in scan_row]
            if "yes" in norm_cells and "no" in norm_cells:
                yes_col = norm_cells.index("yes")
                no_col  = norm_cells.index("no")
                break

        if yes_col is None or no_col is None:
            continue  # table does not have a Yes/No column structure

        for row in all_rows:
            if not row or len(row) <= max(yes_col, no_col):
                continue

            question_cell = str(row[0]) if row[0] else ""
            if not question_cell.strip():
                continue

            # Check raw cell content for checkmarks BEFORE normalization.
            # normalize_text may strip PUA characters like .
            is_yes = _has_checkmark(str(row[yes_col])) if yes_col < len(row) else False
            is_no  = _has_checkmark(str(row[no_col]))  if no_col  < len(row) else False

            if not is_yes and not is_no:
                continue  # no response marked for this question row

            answer = "Yes" if is_yes else "No"

            # Normalise question text and extract a short label for alias lookup
            question_norm = normalize_text(question_cell).strip()
            short_label   = _question_to_short_label(question_norm)

            if len(short_label) < 3:
                continue

            # Three-tier alias resolution:
            #  1. Short label (text before first '(' and '?')
            #  2. Full first line (before any newline guidance text)
            #  3. Substring scan — multi-word aliases (≥ 2 words) only,
            #     scanned within the first 80 chars of the question stem.
            #     Two-word minimum prevents single-word aliases like "services"
            #     from matching unrelated rows (e.g. "Other attest services").
            #     80-char window anchors to the question stem and prevents
            #     long explanatory rows from matching via body text
            #     (e.g. "Does staff fail to meet CPE ... Government Auditing
            #     Standards" matching includes_gagas at char 83+).
            canonical = _aliases_rev.get(short_label.lower())
            if canonical is None:
                first_line_lower = (
                    question_norm.split("\n")[0].strip()
                    .rstrip("?").rstrip(":").strip().lower()
                )
                canonical = _aliases_rev.get(first_line_lower)
            if canonical is None:
                q_lower = question_norm[:60].lower()
                for variant, canon in sorted(
                    _aliases_rev.items(), key=lambda x: -len(x[0])
                ):
                    if len(variant.split()) >= 2 and variant in q_lower:
                        canonical = canon
                        break

            if canonical is None:
                logger.debug(
                    "_extract_fields_from_record: checkbox table — no alias "
                    "match for %r (answer=%s)", short_label[:60], answer,
                )
                continue

            # Only write if not already captured by an earlier extraction pass
            already_found = canonical in raw or any(
                _aliases_rev.get(k.lower()) == canonical for k in raw
            )
            if not already_found:
                raw[canonical] = answer
                logger.debug(
                    "_extract_fields_from_record: checkbox table — "
                    "'%s': '%s' (matched from %r)",
                    canonical, answer, short_label[:60],
                )

    resolved = resolve_aliases(raw)
    return {k: [v] for k, v in resolved.items() if v}


def _checkbox_completion(record: "DocumentRecord") -> tuple[int, int]:
    """
    Count (checked_rows, question_rows) across all Yes/No assessment tables.

    Used to detect blank templates — forms where the auditor never filled in
    the checkboxes. A Yes/No table is identified by locating 'yes' and 'no'
    cells in the first few rows (same detection as Phase 5).

    Returns (checked, eligible) where:
      checked  — rows with any checkmark in Yes or No column
      eligible — rows with a non-empty question cell (col 0)
    """
    checked_total  = 0
    eligible_total = 0

    for table in record.tables:
        all_rows = ([table.headers] + table.rows) if table.headers else table.rows
        if not all_rows:
            continue

        yes_col: int | None = None
        no_col:  int | None = None
        for scan_row in all_rows[:5]:
            norm_cells = [normalize_text(str(c)).strip().lower() for c in scan_row]
            if "yes" in norm_cells and "no" in norm_cells:
                yes_col = norm_cells.index("yes")
                no_col  = norm_cells.index("no")
                break

        if yes_col is None or no_col is None:
            continue

        for row in all_rows:
            if not row or len(row) <= max(yes_col, no_col):
                continue
            if not str(row[0]).strip():
                continue
            eligible_total += 1
            is_yes = _has_checkmark(str(row[yes_col])) if yes_col < len(row) else False
            is_no  = _has_checkmark(str(row[no_col]))  if no_col  < len(row) else False
            if is_yes or is_no:
                checked_total += 1

    return checked_total, eligible_total


# ---------------------------------------------------------------------------
# LLM tiebreaker field selection — tier-driven (Phase A4)
# ---------------------------------------------------------------------------

def _fields_needing_llm(
    fields_for_scoring: dict[str, list[Any]],
    tiers: TierConfig,
) -> list[str]:
    """
    Tier 1 + Tier 2 fields whose two-extractor confidence is < 0.70.
    Reads from TierConfig — stays in sync with field_tiers.yaml automatically.
    Previously was a hardcoded set.
    """
    candidates = tiers.tier1 | tiers.tier2
    return [
        fname
        for fname, vals in fields_for_scoring.items()
        if fname in candidates
        and score_confidence(fname, vals[0], vals[1], None) < 0.70
    ]


# ---------------------------------------------------------------------------
# Provenance helpers (Phase B4)
# ---------------------------------------------------------------------------

def _classify_extraction_method(
    record_b: DocumentRecord | None,
    llm_tiebreaker_ran: bool,
    llm_fallback_ran: bool,
    primary_method: str,
) -> str:
    if llm_fallback_ran:
        return "hybrid" if (record_b is not None or llm_tiebreaker_ran) else "llm_fallback"
    if llm_tiebreaker_ran:
        return "llm"
    return primary_method


def _build_flagged_fields(
    fields_for_scoring: dict[str, list[Any]],
    tiers: TierConfig,
    summary: ConfidenceSummary,
    llm_tiebreaker_fields: list[str],
    llm_fallback_fields: list[str],
) -> list[str]:
    """
    Fields that need auditor attention:
      - LLM-only (no deterministic value for either extractor)
      - Tier 1 with confidence < 0.70
      - LLM broke a tie between contradicting deterministic extractors
      - Filled by LLM fallback
    """
    flagged: set[str] = set()

    for fname, vals in fields_for_scoring.items():
        va, vb, vc = (vals + [None, None, None])[:3]
        if va is None and vb is None and vc is not None:
            flagged.add(fname)

    for fname in tiers.tier1:
        if summary.per_field_scores.get(fname, 0.0) < 0.70:
            flagged.add(fname)

    flagged.update(llm_tiebreaker_fields)
    flagged.update(llm_fallback_fields)
    return sorted(flagged)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def normalize_document(
    file_path: str | Path,
    run_parallel: bool = True,
) -> DocumentRecord:
    """
    Normalize any supported file into a DocumentRecord.

    Parameters
    ----------
    file_path : str | Path
    run_parallel : bool
        Run secondary PDF extractor in parallel for confidence. Default True.

    Returns
    -------
    DocumentRecord
        pii_scrubbed=True, extraction_gate set, quality_gate=False.
        auditor_approved=False — set after human review.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info("normalize_document: %s", path.name)
    tiers = load_tiers()

    # 1. Route
    result: RouteResult = route(path, phase="phase1")
    if result.should_skip:
        raise ValueError(f"{path.name}: file type always skipped. file_type='{result.file_type}'")
    if result.extractor is None:
        raise ValueError(f"{path.name}: no extractor in routing.yaml for file_type='{result.file_type}'")

    file_type = result.file_type

    # 1b. Document classification — sets document_category before extraction
    # so the LLM extractor and critical-fields tiebreaker can use the right
    # prompt context. Runs on filename first (fast), then content if needed.
    try:
        from auditai_data_normalization.doc_classifier import detect_category
        _doc_category = detect_category(path.name)
    except Exception as _dce:
        _doc_category = "unknown"
        logger.debug("normalize: doc_classifier failed for %s — %s", path.name, _dce)

    # 2. Primary extraction
    try:
        record_a: DocumentRecord = _call_extractor(result.extractor, path)
    except Exception as e:
        logger.error("Primary extractor failed for %s: %s", path.name, e)
        return DocumentRecord(
            source_path=str(path),
            file_name=path.name,
            file_type=file_type,
            extraction_method="unknown",
            extraction_status="failed",
            extraction_error=str(e),
            extraction_gate=False,
            quality_gate=False,
            needs_review=True,
        )

    primary_method = record_a.extraction_method

    # Stamp document_category into metadata so all downstream steps can use it.
    # If filename-only gave "unknown", retry with extracted text for content signals.
    if not record_a.metadata.get("document_category"):
        if _doc_category == "unknown" and record_a.cleaned_text:
            try:
                from auditai_data_normalization.doc_classifier import detect_category
                _doc_category = detect_category(
                    path.name,
                    cleaned_text=record_a.cleaned_text[:800],
                )
            except Exception:
                pass
        record_a.metadata["document_category"] = _doc_category

    # 3. Parallel secondary extraction
    record_b: DocumentRecord | None = None
    secondary_path = _get_secondary_extractor_path(result)

    if run_parallel and secondary_path:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(
                lambda: _call_extractor(secondary_path, path)
            )
            try:
                record_b = future.result(timeout=_PARALLEL_TIMEOUT)
            except Exception as e:
                logger.warning("Secondary extractor error for %s: %s", path.name, e)
                record_b = None

    # 4. PII stripping
    record_a = scrub_record(record_a)
    if record_b is not None:
        record_b = scrub_record(record_b)

    # 5. Field extraction + alias resolution
    fields_a = _extract_fields_from_record(record_a)
    fields_b = _extract_fields_from_record(record_b) if record_b else {}
    all_names = set(fields_a) | set(fields_b)
    fields_for_scoring: dict[str, list[Any]] = {
        fname: [
            fields_a.get(fname, [None])[0],
            fields_b.get(fname, [None])[0] if fields_b else None,
            None,
        ]
        for fname in all_names
    }

    # 5b. Structural extraction (Phase 4) — financial statement PDFs only
    # Runs after deterministic field extraction (step 5) so it only fills
    # gaps. Results go into slot B if slot B is empty (no secondary extractor
    # result for that field). Never overwrites slot A or slot B values.
    if record_a.metadata.get("document_category") == "financial_statement" \
            and file_type in ("pdf_text", "pdf_scanned"):
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as _pdf:
                _pages_text = [p.extract_text() or "" for p in _pdf.pages]

            structural_results = structural_extract(pages_text=_pages_text)

            _structural_fields_added = 0
            for fname, evidence in structural_results.items():
                # Map FieldEvidence.value into fields_for_scoring slot B
                # Slot B = secondary extractor result
                if fname in fields_for_scoring:
                    if fields_for_scoring[fname][1] is None:
                        fields_for_scoring[fname][1] = evidence.value
                        _structural_fields_added += 1
                else:
                    # Field not found by primary extractor at all
                    fields_for_scoring[fname] = [None, evidence.value, None]
                    _structural_fields_added += 1

            if _structural_fields_added:
                # Store structural evidence in metadata for auditor review
                record_a.metadata["structural_evidence"] = {
                    fname: {
                        "value":       ev.value,
                        "confidence":  ev.confidence,
                        "source_page": ev.source_page,
                        "method":      ev.method,
                        "anchor":      ev.anchor,
                    }
                    for fname, ev in structural_results.items()
                }
                # Tag extraction method
                if primary_method not in ("llm", "llm_fallback", "hybrid"):
                    record_a.extraction_method = "structural_heuristic"

                logger.info(
                    "normalize: structural_extractor added %d fields for %s",
                    _structural_fields_added, path.name,
                )
        except Exception as e:
            # Structural extraction is best-effort — never blocks the pipeline
            logger.warning(
                "normalize: structural_extractor failed for %s — %s",
                path.name, e,
            )
 

    # 6. LLM tiebreaker (tier-driven)
    llm_tiebreaker_ran = False
    llm_tiebreaker_fields: list[str] = []
    needing_llm = _fields_needing_llm(fields_for_scoring, tiers)

    if needing_llm:
        try:
            from auditai_data_normalization.extractors.llm_extractor import (
                extract_fields, is_available,
            )
            if is_available():
                llm_results = extract_fields(
                    record_a.cleaned_text,
                    fields_to_resolve=needing_llm,
                )
                for fname, llm_val in llm_results.items():
                    if fname in fields_for_scoring:
                        fields_for_scoring[fname][2] = llm_val or None
                    else:
                        fields_for_scoring[fname] = [None, None, llm_val or None]
                llm_tiebreaker_ran = bool(llm_results)
                llm_tiebreaker_fields = list(llm_results.keys())
                logger.debug("LLM tiebreaker resolved %d fields", len(llm_tiebreaker_fields))
        except ImportError:
            logger.debug("LLM extractor not importable — skipping tiebreaker")

    # 6b. Critical-fields completeness tiebreaker
    # The tier-driven tiebreaker (step 6) only fires for fields already in
    # fields_for_scoring with low confidence. It misses fields that were never
    # found by any deterministic extractor — common for docx table layouts where
    # "Organization:" and "Completed by:" cells don't produce key-value pairs.
    #
    # This step fires a targeted LLM call on any critical Tier 1 field that is
    # completely absent from fields_for_scoring. Runs for engagement_form doc
    # types where layout-driven extraction misses are most common.
    # Fires regardless of overall confidence score.
    _doc_type_meta = (
        record_a.metadata.get("document_category") or
        record_a.metadata.get("doc_type") or ""
    )
    _CRITICAL_T1_BY_DOCTYPE: dict[str, set[str]] = {
        "engagement_form":    {"client_name", "includes_gagas", "includes_single_audit",
                               "engagement_decision", "engagement_partner", "preparer_id",
                               "audit_type", "reporting_framework"},
        "planning_memo":      {"client_name", "engagement_partner", "includes_gagas",
                               "audit_type"},
    }
    # For known doc types: use the defined critical set.
    # For unknown/unclassified: fall back to all Tier 1 fields absent from
    # fields_for_scoring — catches any doc where the deterministic extractors
    # missed critical fields regardless of how the form is classified.
    _known_critical = _CRITICAL_T1_BY_DOCTYPE.get(_doc_type_meta)
    if _known_critical is not None:
        _critical_absent = _known_critical - set(fields_for_scoring)
    else:
        _critical_absent = (tiers.tier1 - set(fields_for_scoring))

    if _critical_absent:
        try:
            from auditai_data_normalization.extractors.llm_extractor import (
                extract_fields as _extract_fields_critical,
                is_available as _llm_available_critical,
            )
            if _llm_available_critical():
                _critical_results = _extract_fields_critical(
                    record_a.cleaned_text,
                    fields_to_resolve=sorted(_critical_absent),
                )
                for _fname, _val in _critical_results.items():
                    if _val and _val.lower() not in ("null", "none", ""):
                        fields_for_scoring[_fname] = [None, None, _val]
                        if _fname not in llm_tiebreaker_fields:
                            llm_tiebreaker_fields.append(_fname)
                llm_tiebreaker_ran = llm_tiebreaker_ran or bool(_critical_results)
                logger.info(
                    "normalize: critical-fields tiebreaker ran for %s (%s) — "
                    "absent=%s found=%s",
                    path.name, _doc_type_meta,
                    sorted(_critical_absent),
                    [k for k, v in _critical_results.items() if v and v.lower() not in ("null", "none", "")],
                )
        except ImportError:
            logger.debug("normalize: critical-fields tiebreaker — llm_extractor not importable")

    # 6c. audit_type synthesis from includes_* fields
    # NPO-CX-1.1 has no dedicated "audit type" checkbox row — the type is
    # inferred from the Q1 service-selection checkboxes (Q1a–Q1d). This pass
    # runs after all LLM passes so it has the full field picture. Only fires
    # when audit_type is completely absent from fields_for_scoring.
    def _fv(fname: str) -> str | None:
        """First non-None value across extractor slots for a field."""
        return next(
            (v for v in fields_for_scoring.get(fname, [None, None, None]) if v),
            None,
        )
    if not any(_fv("audit_type") for _ in [1]):
        _gagas = _fv("includes_gagas")
        _gaas  = _fv("includes_gaas_audit")
        _sa    = _fv("includes_single_audit")
        _synth: list[str] = []
        if str(_gagas).lower() in ("yes", "true"):
            _synth.append("GAGAS Audit")
        elif str(_gaas).lower() in ("yes", "true"):
            _synth.append("Financial Statement Audit")
        if str(_sa).lower() in ("yes", "true"):
            _synth.append("Single Audit")
        if _synth:
            fields_for_scoring["audit_type"] = [None, None, " / ".join(_synth)]
            logger.debug(
                "normalize: audit_type synthesized from includes_* — %r",
                fields_for_scoring["audit_type"][2],
            )

    # 7. Confidence scoring (Phase A2 tier-based)
    per_field_scores = score_fields(fields_for_scoring) if fields_for_scoring else {}
    summary: ConfidenceSummary = summarise(per_field_scores, tiers=tiers)

    if file_type == "pdf_scanned" and record_a.extraction_confidence > 0:
        ocr_conf = record_a.extraction_confidence
        record_a.extraction_confidence = (
            round(0.6 * summary.aggregate_score + 0.4 * ocr_conf, 4)
            if summary.aggregate_score > 0 else ocr_conf
        )
    else:
        record_a.extraction_confidence = summary.aggregate_score

    # 8. Set extraction_gate
    record_a.extraction_gate = record_a.extraction_confidence >= _EXTRACTION_GATE

    # 9. LLM fallback (Phase B2)
    # Fires when extraction_confidence < 0.50 after deterministic extractors ran.
    # Calls extract_all_fields() for the full Tier 1 + Tier 2 field set,
    # merges results into fields_for_scoring, re-scores, re-evaluates gate.
    llm_fallback_ran = False
    llm_fallback_fields: list[str] = []

    if not record_a.extraction_gate:
        logger.info(
            "normalize: extraction_gate=False for %s "
            "(conf=%.3f < %.2f) — running LLM fallback",
            path.name, record_a.extraction_confidence, _EXTRACTION_GATE,
        )
        try:
            from auditai_data_normalization.extractors.llm_extractor import (
                extract_all_fields,
                is_available,
                FieldResult,
            )
            if is_available():
                # Pass doc_type hint from metadata if the annotated sheet set it
                doc_type = record_a.metadata.get("document_category") or \
                           record_a.metadata.get("doc_type")

                fallback_results = extract_all_fields(
                    text=record_a.cleaned_text,
                    doc_type=doc_type,
                    tiers=tiers,
                )

                # Merge fallback into fields_for_scoring
                # Only fill slot C (LLM) — never overwrite deterministic values
                new_fallback_fields: list[str] = []
                for fname, fresult in fallback_results.items():
                    if not fresult.found:
                        continue
                    if fname in fields_for_scoring:
                        # Only write to slot C if not already filled
                        if fields_for_scoring[fname][2] is None:
                            fields_for_scoring[fname][2] = fresult.value
                    else:
                        # Field not found by any deterministic extractor at all
                        fields_for_scoring[fname] = [None, None, fresult.value]
                    new_fallback_fields.append(fname)

                if new_fallback_fields:
                    # Re-score with fallback values included
                    per_field_scores = score_fields(fields_for_scoring)

                    # B3 — calibrate LLM field scores before final summarise
                    from auditai_data_normalization.confidence import calibrate_llm_scores
                    per_field_scores, b3_flagged = calibrate_llm_scores(
                        per_field_scores=per_field_scores,
                        fields_for_scoring=fields_for_scoring,
                        fallback_results=fallback_results,
                        tiers=tiers,
                    )
                    # B3 contradictions added to flagged_fields downstream
                    llm_fallback_fields = new_fallback_fields + b3_flagged

                    summary = summarise(per_field_scores, tiers=tiers)

                    # Re-apply OCR blend if scanned PDF
                    if file_type == "pdf_scanned" and record_a.extraction_confidence > 0:
                        ocr_conf = record_a.metadata.get(
                            "confidence_summary", {}
                        ).get("ocr_confidence", record_a.extraction_confidence)
                        record_a.extraction_confidence = round(
                            0.6 * summary.aggregate_score + 0.4 * ocr_conf, 4
                        )
                    else:
                        record_a.extraction_confidence = summary.aggregate_score

                    # Re-evaluate extraction_gate with updated score
                    record_a.extraction_gate = \
                        record_a.extraction_confidence >= _EXTRACTION_GATE

                    llm_fallback_ran = True
                    # llm_fallback_fields already set on line 634 as
                    # new_fallback_fields + b3_flagged — do not reassign here

                    logger.info(
                        "normalize: LLM fallback added %d fields for %s — "
                        "new conf=%.3f ext_gate=%s",
                        len(new_fallback_fields), path.name,
                        record_a.extraction_confidence,
                        record_a.extraction_gate,
                    )
                else:
                    logger.info(
                        "normalize: LLM fallback found no additional fields for %s",
                        path.name,
                    )
            else:
                logger.warning(
                    "normalize: LLM fallback unavailable for %s — "
                    "Ollama not running or %s not pulled",
                    path.name, "gemma3:12b",
                )
        except ImportError:
            logger.debug("normalize: llm_extractor not importable — fallback skipped")

    # 10. Provenance flags (Phase B4)
    record_a.llm_assisted = llm_tiebreaker_ran or llm_fallback_ran
    record_a.flagged_fields = _build_flagged_fields(
        fields_for_scoring, tiers, summary,
        llm_tiebreaker_fields, llm_fallback_fields,
    )
    record_a.extraction_method = _classify_extraction_method(
        record_b, llm_tiebreaker_ran, llm_fallback_ran, primary_method,
    )

    # quality_gate stays False — completion_drafter.py owns it
    record_a.quality_gate = False

    # 11. needs_review + metadata
    record_a.needs_review = (
        not record_a.extraction_gate
        or record_a.extraction_status in ("failed", "partial")
    )

    record_a.metadata["confidence_summary"] = {
        "aggregate": summary.aggregate_score,
        "extraction_gate": record_a.extraction_gate,
        "tier1_found": summary.tier1_found,
        "tier1_total": summary.tier1_total,
        "tier2_found": summary.tier2_found,
        "tier2_total": summary.tier2_total,
        "tier1_missing": summary.tier1_missing,
        "floor_applied": summary.floor_applied,
        "fields_present": summary.fields_present,
        "fields_missing": summary.fields_missing,
        "low_confidence_fields": summary.low_confidence_fields,
        "per_field_scores": summary.per_field_scores,
        "llm_tiebreaker_fields": llm_tiebreaker_fields,
        "llm_fallback_fields": llm_fallback_fields,
    }

    # Incomplete template detection — engagement forms with zero checkbox responses.
    # Blank templates have no audit decisions recorded and must not generate pairs.
    # Runs after PII scrubbing so checkbox chars are still present in table cells.
    _checked, _eligible = _checkbox_completion(record_a)
    if _eligible > 0:
        _completion_ratio = _checked / _eligible
        record_a.metadata["checkbox_completion_ratio"] = round(_completion_ratio, 4)
        if _doc_category == "engagement_form" and _eligible >= 5 and _checked == 0:
            record_a.metadata["document_status"] = "incomplete_template"
            logger.info(
                "normalize_document: %s flagged as incomplete_template "
                "(0 checked / %d eligible rows in Yes/No table)",
                path.name, _eligible,
            )

    logger.info(
        "normalize_document: %s type=%s ext_conf=%.3f "
        "ext_gate=%s tier1=%d/%d flagged=%d needs_review=%s",
        path.name, file_type, record_a.extraction_confidence,
        record_a.extraction_gate, summary.tier1_found, summary.tier1_total,
        len(record_a.flagged_fields), record_a.needs_review,
    )

    return record_a