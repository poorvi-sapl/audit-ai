"""
auditai_data_normalization/alias_suggester.py
==============================================
Phase D1/D2 — alias learning system.

Every time normalize.py encounters a label it cannot map to a canonical
field, that label is logged here. Over time, this builds a catalogue of
unknown labels that the audit team can review and approve — growing the
alias dictionary without manual trawling through documents.

D1 — Logging (append-only CSV)
-------------------------------
log_unknown(raw_label, extracted_value, source_file)
    Appends one row to data/suggested_aliases.csv.
    Never modifies field_aliases.yaml automatically.
    Thread-safe via fcntl file locking.

D2 — LLM-assisted mapping
--------------------------
suggest_canonical(raw_label, canonical_fields) -> SuggestionResult
    Asks Gemma: "Is '[raw_label]' the same audit field as any of [list]?"
    Returns canonical_field + confidence (high/medium/low/none).
    Result stored in the CSV — never auto-applied.
    Only called when Ollama is available (graceful skip otherwise).

Approval flow
-------------
The Streamlit Section 0.5 UI reads the CSV, shows pending suggestions,
and calls approve() / reject() to update status.
approve() writes the approved mapping to field_aliases.yaml.
reject()  marks as rejected — stops suggesting the same label again.

CSV columns
-----------
    raw_label          str   — label as found in the document
    extracted_value    str   — sample value extracted alongside the label
    source_file        str   — filename where label was seen
    suggested_canonical str  — Gemma's best guess, or "" if no suggestion
    llm_confidence     str   — "high" | "medium" | "low" | "none" | "pending"
    seen_count         int   — how many times this label has been seen
    status             str   — "pending" | "approved" | "rejected" | "skipped"
    first_seen         str   — ISO datetime
    last_seen          str   — ISO datetime
    approved_by        str   — reviewer ID if approved

Public API
----------
    log_unknown(raw_label, extracted_value, source_file, run_llm)
    suggest_canonical(raw_label, canonical_fields) -> SuggestionResult
    load_suggestions(csv_path) -> list[dict]
    approve(raw_label, canonical_field, reviewer_id, csv_path, yaml_path)
    reject(raw_label, reviewer_id, csv_path)
    coverage_stats(csv_path, aliases_path) -> CoverageStats
"""

from __future__ import annotations

import csv
import fcntl
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PKG_DIR     = Path(__file__).parent
_PROJECT_DIR = _PKG_DIR.parent
_CSV_PATH    = _PROJECT_DIR / "data" / "suggested_aliases.csv"
_YAML_PATH   = _PKG_DIR / "field_aliases.yaml"

_CSV_COLUMNS = [
    "raw_label", "extracted_value", "source_file",
    "suggested_canonical", "llm_confidence",
    "seen_count", "status",
    "first_seen", "last_seen", "approved_by",
]

_MODEL = "gemma3:12b"


# ---------------------------------------------------------------------------
# SuggestionResult
# ---------------------------------------------------------------------------

@dataclass
class SuggestionResult:
    """Result of one LLM-assisted mapping attempt."""
    raw_label:           str
    suggested_canonical: str  = ""
    llm_confidence:      str  = "none"    # "high" | "medium" | "low" | "none"
    reasoning:           str  = ""

    @property
    def has_suggestion(self) -> bool:
        return bool(self.suggested_canonical)


# ---------------------------------------------------------------------------
# CoverageStats
# ---------------------------------------------------------------------------

@dataclass
class CoverageStats:
    """Alias coverage metrics for the Streamlit sidebar (Phase D4)."""
    total_seen:     int = 0   # unique raw labels encountered
    approved:       int = 0   # mapped to a canonical field
    pending:        int = 0   # awaiting review
    rejected:       int = 0   # confirmed as not mappable
    high_conf:      int = 0   # LLM said "high" confidence, still pending
    coverage_pct:   float = 0.0  # approved / total_seen * 100

    def __str__(self) -> str:
        return (
            f"coverage={self.coverage_pct:.1f}% "
            f"({self.approved}/{self.total_seen} mapped) "
            f"pending={self.pending} rejected={self.rejected}"
        )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _ensure_csv(csv_path: Path) -> None:
    """Create the CSV with headers if it doesn't exist."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()


def _load_csv(csv_path: Path) -> list[dict]:
    """Load all rows from the CSV. Returns [] if file doesn't exist."""
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_csv(rows: list[dict], csv_path: Path) -> None:
    """Rewrite the full CSV (used for updates to existing rows)."""
    _ensure_csv(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _append_row(row: dict, csv_path: Path) -> None:
    """
    Append one row to the CSV. Thread-safe via fcntl locking.
    Creates the file with headers if it doesn't exist.
    """
    _ensure_csv(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writerow({col: row.get(col, "") for col in _CSV_COLUMNS})
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# D1 — Log unknown labels
# ---------------------------------------------------------------------------

def log_unknown(
    raw_label: str,
    extracted_value: str = "",
    source_file: str = "",
    run_llm: bool = True,
    canonical_fields: list[str] | None = None,
    csv_path: Path | None = None,
) -> None:
    """
    Log an unmapped label to suggested_aliases.csv.

    If the label has been seen before, increments seen_count and updates
    last_seen. If new, appends a fresh row and optionally runs D2 LLM
    suggestion.

    Called by normalize.py resolve_aliases() for every unmapped key.

    Parameters
    ----------
    raw_label : str
        The label as it appeared in the document.
    extracted_value : str
        Sample value found alongside this label.
    source_file : str
        Filename where the label was seen.
    run_llm : bool
        If True and Ollama is available, run D2 suggestion automatically.
    canonical_fields : list[str] | None
        Fields to compare against. Loaded from field_aliases.yaml if None.
    csv_path : Path | None
        Override default path (useful in tests).
    """
    path = csv_path or _CSV_PATH
    label_clean = raw_label.strip()

    if not label_clean:
        return

    # Normalise — lowercase for dedup, preserve original for display
    label_key = label_clean.lower()

    rows = _load_csv(path)
    existing = next(
        (r for r in rows if r.get("raw_label", "").lower() == label_key), None
    )

    now = _now_iso()

    if existing:
        # Already seen — increment count, update last_seen
        if existing.get("status") in ("approved", "rejected"):
            # Already resolved — skip
            return
        try:
            existing["seen_count"] = str(int(existing.get("seen_count", 1)) + 1)
        except ValueError:
            existing["seen_count"] = "2"
        existing["last_seen"] = now
        # Update sample value if we have a better one
        if extracted_value and not existing.get("extracted_value"):
            existing["extracted_value"] = extracted_value[:200]
        _save_csv(rows, path)
        logger.debug("alias_suggester: seen_count++ for '%s'", label_clean)
        return

    # New label — get LLM suggestion if requested
    suggested_canonical = ""
    llm_confidence = "pending"

    if run_llm:
        try:
            fields = canonical_fields or _load_canonical_fields()
            result = suggest_canonical(label_clean, fields)
            suggested_canonical = result.suggested_canonical
            llm_confidence = result.llm_confidence
        except Exception as e:
            logger.debug("alias_suggester: LLM suggestion failed for '%s': %s", label_clean, e)
            llm_confidence = "pending"

    row = {
        "raw_label":           label_clean,
        "extracted_value":     str(extracted_value)[:200],
        "source_file":         source_file,
        "suggested_canonical": suggested_canonical,
        "llm_confidence":      llm_confidence,
        "seen_count":          "1",
        "status":              "pending",
        "first_seen":          now,
        "last_seen":           now,
        "approved_by":         "",
    }
    _append_row(row, path)
    logger.info(
        "alias_suggester: logged new label '%s' from %s (suggestion='%s' conf=%s)",
        label_clean, source_file, suggested_canonical, llm_confidence,
    )


def _load_canonical_fields() -> list[str]:
    """Load canonical field names from field_aliases.yaml."""
    if not _YAML_PATH.exists():
        # Fallback: hardcoded tier1+tier2 names
        return [
            "client_name", "fiscal_year_end", "engagement_decision",
            "engagement_partner", "audit_type", "includes_gagas",
            "includes_single_audit", "reporting_framework",
            "document_reference", "includes_gaas_audit", "includes_grant_compliance",
            "preparation_date", "partner_sign_date", "ein",
            "includes_nonattest_services", "financial_statement_use",
        ]
    with open(_YAML_PATH) as f:
        raw = yaml.safe_load(f) or {}
    return list(raw.keys())


# ---------------------------------------------------------------------------
# D2 — LLM-assisted mapping
# ---------------------------------------------------------------------------

_SUGGESTION_SYSTEM = (
    "You are an audit field mapping assistant. "
    "You only output valid JSON. No explanation, no markdown, no preamble."
)


def suggest_canonical(
    raw_label: str,
    canonical_fields: list[str],
) -> SuggestionResult:
    """
    Ask Gemma whether raw_label matches any canonical field.

    Never auto-applies the suggestion. Result is stored in CSV for
    human approval.

    Parameters
    ----------
    raw_label : str
        The unmapped label from the document.
    canonical_fields : list[str]
        The canonical field names to compare against.

    Returns
    -------
    SuggestionResult
        .suggested_canonical = best match or ""
        .llm_confidence = "high" | "medium" | "low" | "none"
    """
    try:
        import ollama
        models = ollama.list()
        if not any(_MODEL in m.model for m in models.models):
            return SuggestionResult(raw_label=raw_label, llm_confidence="pending")
    except Exception:
        return SuggestionResult(raw_label=raw_label, llm_confidence="pending")

    fields_list = "\n".join(f"- {f}" for f in canonical_fields[:40])
    prompt = (
        f"An audit workpaper contains the label: \"{raw_label}\"\n\n"
        f"Does this label refer to the same audit field as any of the following canonical fields?\n"
        f"{fields_list}\n\n"
        f"Return a JSON object:\n"
        f"{{\n"
        f'  "matched": true | false,\n'
        f'  "canonical_field": "<best matching field name, or null>",\n'
        f'  "confidence": "high" | "medium" | "low",\n'
        f'  "reasoning": "<one sentence>"\n'
        f"}}\n\n"
        f"Rules:\n"
        f"- matched=true only if you are confident this is the same field\n"
        f"- confidence=high: very clear match (e.g. 'EP:' → 'engagement_partner')\n"
        f"- confidence=medium: likely match but label is ambiguous\n"
        f"- confidence=low: possible match but uncertain\n"
        f"- If no match, set matched=false, canonical_field=null, confidence='low'\n"
        f"- Only match against fields in the list above\n"
        f"- Return ONLY the JSON object"
    )

    try:
        import ollama as _ollama
        response = _ollama.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SUGGESTION_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 256},
            format="json",
        )
        raw = response.message.content or ""
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        parsed = json.loads(m.group(0) if m else cleaned)

        matched   = bool(parsed.get("matched", False))
        canonical = parsed.get("canonical_field") or ""
        confidence = parsed.get("confidence", "low")
        reasoning  = parsed.get("reasoning", "")

        # Validate — only accept if canonical is in our list
        if matched and canonical and canonical in canonical_fields:
            return SuggestionResult(
                raw_label=raw_label,
                suggested_canonical=canonical,
                llm_confidence=confidence,
                reasoning=reasoning,
            )
        else:
            return SuggestionResult(
                raw_label=raw_label,
                suggested_canonical="",
                llm_confidence="none",
                reasoning=reasoning or "No confident match found.",
            )

    except Exception as e:
        logger.debug("alias_suggester: LLM suggestion error for '%s': %s", raw_label, e)
        return SuggestionResult(raw_label=raw_label, llm_confidence="pending")


# ---------------------------------------------------------------------------
# Approval / rejection
# ---------------------------------------------------------------------------

def approve(
    raw_label: str,
    canonical_field: str,
    reviewer_id: str,
    csv_path: Path | None = None,
    yaml_path: Path | None = None,
    confidence: dict | None = None,
    source_workpaper: str = "",
    notes: str = "",
) -> bool:
    """
    Approve a suggested mapping.

    Writes to pending_alias_updates.yaml (staging area) instead of directly
    to field_aliases.yaml. The mapping becomes live only after
    alias_merger.merge_pending() runs validation and creates a version snapshot.

    Updates the CSV row to status="approved" for UI tracking.
    Returns True if successful.
    """
    path = csv_path or _CSV_PATH

    rows = _load_csv(path)
    label_key = raw_label.strip().lower()
    row = next((r for r in rows if r.get("raw_label", "").lower() == label_key), None)

    if row is None:
        logger.warning("alias_suggester: approve() — label '%s' not in CSV", raw_label)
        return False

    # Update CSV status
    row["status"]              = "approved"
    row["suggested_canonical"] = canonical_field
    row["approved_by"]         = reviewer_id
    row["last_seen"]           = _now_iso()
    _save_csv(rows, path)

    # Write to pending_alias_updates.yaml — NOT directly to field_aliases.yaml
    _write_to_pending(
        raw_label=raw_label.strip(),
        canonical_field=canonical_field,
        reviewer_id=reviewer_id,
        confidence=confidence or {
            "score": float(row.get("llm_confidence") == "high") * 0.85 + 0.70,
            "source": "human",
            "method_weight": 1.0,
            "effective_score": float(row.get("llm_confidence") == "high") * 0.85 + 0.70,
        },
        source_workpaper=source_workpaper or row.get("source_file", ""),
        notes=notes,
    )

    logger.info(
        "alias_suggester: queued '%s' -> '%s' by %s (pending merge)",
        raw_label, canonical_field, reviewer_id,
    )
    return True


def reject(
    raw_label: str,
    reviewer_id: str,
    canonical_field: str = "",
    rejection_reason: str = "other",
    notes: str = "",
    source_workpaper: str = "",
    proposed_by: str = "human",
    proposed_score: float = 0.0,
    csv_path: Path | None = None,
) -> bool:
    """
    Mark a suggestion as rejected.

    Writes to rejected_mappings.yaml (permanent rejection memory) so the
    same (label, canonical) pair is never re-proposed by fuzzy or LLM.
    Also updates the CSV row to status="rejected".
    Returns True if successful.
    """
    path = csv_path or _CSV_PATH
    rows = _load_csv(path)
    label_key = raw_label.strip().lower()
    row = next((r for r in rows if r.get("raw_label", "").lower() == label_key), None)

    if row is None:
        return False

    # Resolve canonical_field from CSV if not supplied
    resolved_canonical = canonical_field or row.get("suggested_canonical", "")

    row["status"]      = "rejected"
    row["approved_by"] = reviewer_id
    row["last_seen"]   = _now_iso()
    _save_csv(rows, path)

    # Write to rejected_mappings.yaml — permanent rejection memory
    if resolved_canonical:
        _write_to_rejected(
            raw_label=raw_label.strip(),
            rejected_canonical=resolved_canonical,
            reviewer_id=reviewer_id,
            rejection_reason=rejection_reason,
            notes=notes,
            source_workpaper=source_workpaper or row.get("source_file", ""),
            proposed_by=proposed_by,
            proposed_score=proposed_score,
        )

    logger.info("alias_suggester: rejected '%s' -> '%s' by %s", raw_label, resolved_canonical, reviewer_id)
    return True


def _write_to_pending(
    raw_label: str,
    canonical_field: str,
    reviewer_id: str,
    confidence: dict,
    source_workpaper: str = "",
    notes: str = "",
) -> None:
    """
    Append an approved mapping to pending_alias_updates.yaml.
    alias_merger.merge_pending() processes this file to update field_aliases.yaml.
    """
    pending_path = _PKG_DIR / "alias_registry" / "pending_alias_updates.yaml"
    if not pending_path.exists():
        logger.warning("alias_suggester: pending_alias_updates.yaml not found — skipping")
        return

    with open(pending_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    entries = data.get("pending") or []

    # Deduplicate — skip if same (label, canonical) already pending
    normalized = raw_label.lower().strip()
    already = any(
        str(e.get("raw_label", "")).lower().strip() == normalized
        and e.get("canonical_field") == canonical_field
        and e.get("status") == "pending"
        for e in entries
    )
    if already:
        logger.debug("alias_suggester: '%s' -> '%s' already pending", raw_label, canonical_field)
        return

    entry = {
        "raw_label":        raw_label,
        "canonical_field":  canonical_field,
        "reviewer_id":      reviewer_id,
        "approval_type":    "human",
        "confidence":       confidence,
        "source_workpaper": source_workpaper,
        "timestamp":        _now_iso(),
        "notes":            notes,
        "status":           "pending",
    }
    entries.append(entry)

    with open(pending_path, "w", encoding="utf-8") as f:
        yaml.dump({"pending": entries}, f, default_flow_style=False, allow_unicode=True)

    logger.info("alias_suggester: '%s' -> '%s' written to pending", raw_label, canonical_field)


def _write_to_rejected(
    raw_label: str,
    rejected_canonical: str,
    reviewer_id: str,
    rejection_reason: str,
    notes: str = "",
    source_workpaper: str = "",
    proposed_by: str = "human",
    proposed_score: float = 0.0,
) -> None:
    """
    Append a rejection to rejected_mappings.yaml (permanent, append-only).
    """
    rejected_path = _PKG_DIR / "alias_registry" / "rejected_mappings.yaml"
    if not rejected_path.exists():
        logger.warning("alias_suggester: rejected_mappings.yaml not found — skipping")
        return

    with open(rejected_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rejections = data.get("rejections") or []

    # Deduplicate — same (label, canonical) rejection already recorded
    normalized = raw_label.lower().strip()
    already = any(
        str(e.get("raw_label", "")).lower().strip() == normalized
        and e.get("rejected_canonical") == rejected_canonical
        for e in rejections
    )
    if already:
        logger.debug(
            "alias_suggester: rejection '%s' -> '%s' already recorded",
            raw_label, rejected_canonical,
        )
        return

    entry = {
        "raw_label":          raw_label,
        "rejected_canonical": rejected_canonical,
        "rejected_by":        reviewer_id,
        "rejection_reason":   rejection_reason,
        "notes":              notes,
        "timestamp":          _now_iso(),
        "source_workpaper":   source_workpaper,
        "proposed_by":        proposed_by,
        "proposed_score":     proposed_score,
    }
    rejections.append(entry)

    with open(rejected_path, "w", encoding="utf-8") as f:
        yaml.dump({"rejections": rejections}, f, default_flow_style=False, allow_unicode=True)

    logger.info(
        "alias_suggester: rejection '%s' -> '%s' written to rejected_mappings",
        raw_label, rejected_canonical,
    )


def _write_alias_to_yaml(raw_label: str, canonical_field: str, yaml_path: Path) -> None:
    """
    DEPRECATED — retained for backward compatibility only.
    Direct writes to field_aliases.yaml are no longer permitted.
    Use approve() + alias_merger.merge_pending() instead.
    Logs a warning and returns without writing.
    """
    logger.warning(
        "alias_suggester: _write_alias_to_yaml() called directly for '%s' -> '%s'. "
        "This is deprecated. Use approve() + alias_merger.merge_pending() instead.",
        raw_label, canonical_field,
    )


# ---------------------------------------------------------------------------
# D4 — Coverage stats
# ---------------------------------------------------------------------------

def load_suggestions(csv_path: Path | None = None) -> list[dict]:
    """Load all suggestions. Returns [] if CSV doesn't exist."""
    return _load_csv(csv_path or _CSV_PATH)


def coverage_stats(
    csv_path: Path | None = None,
    aliases_path: Path | None = None,
) -> CoverageStats:
    """
    Compute alias coverage metrics for Streamlit sidebar (Phase D4).

    Returns
    -------
    CoverageStats
        Counts of total/approved/pending/rejected suggestions,
        plus coverage_pct = approved / total_seen * 100.
    """
    rows = _load_csv(csv_path or _CSV_PATH)

    if not rows:
        return CoverageStats()

    total    = len(rows)
    approved = sum(1 for r in rows if r.get("status") == "approved")
    rejected = sum(1 for r in rows if r.get("status") == "rejected")
    pending  = sum(1 for r in rows if r.get("status") == "pending")
    high_conf = sum(
        1 for r in rows
        if r.get("status") == "pending" and r.get("llm_confidence") == "high"
    )
    pct = round(approved / total * 100, 1) if total else 0.0

    return CoverageStats(
        total_seen=total,
        approved=approved,
        pending=pending,
        rejected=rejected,
        high_conf=high_conf,
        coverage_pct=pct,
    )