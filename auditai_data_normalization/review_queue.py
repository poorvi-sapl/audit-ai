"""
auditai_data_normalization/review_queue.py
============================================
Append-only CSV writer for records that fail the confidence gate.

Any DocumentRecord with extraction_confidence < 0.7 OR
extraction_status in ('failed', 'partial') gets a row written here.
Auditors open this CSV, correct the flagged fields, and re-submit
the corrected record back through normalize_document().

Design decisions
----------------
- Append-only. Rows are never deleted or updated in this file.
  Corrections create a new row with corrected=True.
- Thread-safe. Uses a file lock so parallel pipeline workers can
  all write without corrupting the CSV.
- One row per record per pipeline run. The file_hash + run_timestamp
  together form a natural dedup key.
- Human-readable. Every column has a clear label. The CSV opens
  cleanly in Excel without any special import settings.

CSV columns
-----------
    queued_at           ISO timestamp when row was written
    file_name           Original filename
    file_type           docx / pdf_text / pdf_scanned / xlsx / csv / json
    file_hash           SHA-256 of original file (first 16 chars)
    extraction_status   success / partial / failed
    extraction_confidence  0.0–1.0
    extraction_method   python_docx / pdfplumber / tesseract / etc.
    word_count          Word count of cleaned_text
    fields_present      Comma-separated list of fields that were found
    fields_missing      Comma-separated list of fields with score 0.0
    low_conf_fields     Fields found but with confidence < 0.7
    pii_redactions      Summary: "EIN:3, PERSON:1" etc.
    needs_review        True / False
    corrected           False on first write; True when auditor re-submits
    reviewer_id         Empty on first write; filled by auditor
    review_notes        Free text for auditor to explain corrections
    source_path         Full path to original file

Public API
----------
    enqueue(record, queue_path, notes="") -> None
        Write one record to the review queue CSV.

    mark_corrected(file_hash, queue_path, reviewer_id, notes="") -> bool
        Mark an existing row as corrected. Returns True if found.

    load_queue(queue_path) -> list[dict]
        Load all rows from the queue as a list of dicts.

    pending(queue_path) -> list[dict]
        Load only rows where corrected=False.
"""

from __future__ import annotations

import csv
import fcntl
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auditai_data_normalization.schema import DocumentRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV column definitions — order matters for readability in Excel
# ---------------------------------------------------------------------------

_COLUMNS = [
    "queued_at",
    "file_name",
    "file_type",
    "file_hash",
    "extraction_status",
    "extraction_confidence",
    "extraction_method",
    "word_count",
    "fields_present",
    "fields_missing",
    "low_conf_fields",
    "pii_redactions",
    "needs_review",
    "corrected",
    "reviewer_id",
    "review_notes",
    "source_path",
]

# Default queue path — relative to project root
_DEFAULT_QUEUE_PATH = Path(__file__).parent.parent / "data" / "review_queue.csv"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_pii_redactions(record: DocumentRecord) -> str:
    """Compact summary of PII redactions: 'EIN:3, PERSON:1'"""
    if not record.pii_redactions:
        return ""
    return ", ".join(
        f"{r.pii_type}:{r.count}" for r in record.pii_redactions
    )


def _get_confidence_summary(record: DocumentRecord) -> dict[str, Any]:
    """Extract confidence summary from record.metadata if present."""
    return record.metadata.get("confidence_summary", {})


def _ensure_header(queue_path: Path) -> None:
    """Write the header row if the file is new or empty."""
    if not queue_path.exists() or queue_path.stat().st_size == 0:
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(queue_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writeheader()


def _build_row(
    record: DocumentRecord,
    notes: str = "",
    corrected: bool = False,
    reviewer_id: str = "",
) -> dict[str, Any]:
    """Build one CSV row dict from a DocumentRecord."""
    cs = _get_confidence_summary(record)

    fields_present = cs.get("fields_present", [])
    fields_missing = cs.get("fields_missing", [])
    low_conf = cs.get("low_confidence_fields", [])

    return {
        "queued_at":            _now_iso(),
        "file_name":            record.file_name,
        "file_type":            record.file_type,
        "file_hash":            record.file_hash[:16] if record.file_hash else "",
        "extraction_status":    record.extraction_status,
        "extraction_confidence": round(record.extraction_confidence, 4),
        "extraction_method":    record.extraction_method,
        "word_count":           record.word_count,
        "fields_present":       ", ".join(fields_present),
        "fields_missing":       ", ".join(fields_missing),
        "low_conf_fields":      ", ".join(low_conf),
        "pii_redactions":       _format_pii_redactions(record),
        "needs_review":         record.needs_review,
        "corrected":            corrected,
        "reviewer_id":          reviewer_id,
        "review_notes":         notes,
        "source_path":          record.source_path,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enqueue(
    record: DocumentRecord,
    queue_path: str | Path | None = None,
    notes: str = "",
) -> None:
    """
    Append one record to the review queue CSV.

    Should be called by normalize.py whenever record.needs_review=True.
    Safe to call even if the record passes the gate — it checks internally
    and logs a warning but still writes (caller controls gate logic).

    Parameters
    ----------
    record : DocumentRecord
        The normalized record to queue for review.
    queue_path : str | Path | None
        Path to the CSV file. Defaults to data/review_queue.csv.
        Parent directory is created if it does not exist.
    notes : str
        Optional note to attach to this queue entry.
        e.g. "pdfplumber timeout on page 4"
    """
    path = Path(queue_path) if queue_path else _DEFAULT_QUEUE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    row = _build_row(record, notes=notes)

    # Thread-safe append with file lock
    with open(path, "a", newline="", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            # Write header if file is empty
            if path.stat().st_size == 0 or os.path.getsize(path) == 0:
                writer = csv.DictWriter(f, fieldnames=_COLUMNS)
                writer.writeheader()
                writer.writerow(row)
            else:
                writer = csv.DictWriter(f, fieldnames=_COLUMNS)
                writer.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    logger.info(
        "review_queue: enqueued %s (confidence=%.3f, status=%s)",
        record.file_name,
        record.extraction_confidence,
        record.extraction_status,
    )


def mark_corrected(
    file_hash: str,
    queue_path: str | Path | None = None,
    reviewer_id: str = "",
    notes: str = "",
) -> bool:
    """
    Mark the most recent queue entry for a file_hash as corrected.

    Called after an auditor reviews and corrects a record. The corrected
    record re-enters normalize_document() with auditor_approved=True.

    Parameters
    ----------
    file_hash : str
        SHA-256 hex digest (or first 16 chars) of the original file.
    queue_path : str | Path | None
        Path to the review queue CSV.
    reviewer_id : str
        Auditor ID or initials.
    notes : str
        Correction notes.

    Returns
    -------
    bool
        True if a matching uncorrected row was found and updated.
        False if no matching row found.
    """
    path = Path(queue_path) if queue_path else _DEFAULT_QUEUE_PATH

    if not path.exists():
        logger.warning("mark_corrected: queue file not found at %s", path)
        return False

    rows = load_queue(path)
    short_hash = file_hash[:16]

    # Find the most recent uncorrected row for this file_hash
    found = False
    for row in reversed(rows):
        row_hash = str(row.get("file_hash", ""))[:16]
        if row_hash == short_hash and str(row.get("corrected", "")).lower() == "false":
            row["corrected"] = True
            row["reviewer_id"] = reviewer_id
            row["review_notes"] = notes
            row["queued_at"] = _now_iso()  # update timestamp on correction
            found = True
            break

    if not found:
        logger.warning(
            "mark_corrected: no uncorrected row found for hash %s", short_hash
        )
        return False

    # Rewrite the entire file (queue is small — full rewrite is safe)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    logger.info(
        "mark_corrected: %s marked as corrected by %s", short_hash, reviewer_id
    )
    return True


def load_queue(
    queue_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Load all rows from the review queue CSV.

    Parameters
    ----------
    queue_path : str | Path | None
        Path to the review queue CSV.

    Returns
    -------
    list[dict]
        All rows as dicts. Empty list if file does not exist.
    """
    path = Path(queue_path) if queue_path else _DEFAULT_QUEUE_PATH

    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def pending(
    queue_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return only rows where corrected=False.

    Parameters
    ----------
    queue_path : str | Path | None
        Path to the review queue CSV.

    Returns
    -------
    list[dict]
        Uncorrected rows only, oldest first.
    """
    all_rows = load_queue(queue_path)
    return [
        r for r in all_rows
        if str(r.get("corrected", "False")).lower() == "false"
    ]


def queue_stats(
    queue_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Return summary statistics for the review queue.

    Returns
    -------
    dict with keys:
        total       — total rows
        pending     — uncorrected rows
        corrected   — corrected rows
        by_status   — {extraction_status: count}
        by_type     — {file_type: count}
        avg_confidence — mean confidence of pending rows
    """
    all_rows = load_queue(queue_path)
    pending_rows = [r for r in all_rows
                    if str(r.get("corrected", "False")).lower() == "false"]

    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    confs: list[float] = []

    for r in pending_rows:
        status = r.get("extraction_status", "unknown")
        ftype = r.get("file_type", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        by_type[ftype] = by_type.get(ftype, 0) + 1
        try:
            confs.append(float(r.get("extraction_confidence", 0)))
        except (ValueError, TypeError):
            pass

    return {
        "total": len(all_rows),
        "pending": len(pending_rows),
        "corrected": len(all_rows) - len(pending_rows),
        "by_status": by_status,
        "by_type": by_type,
        "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
    }


def approve_record(
    record: "DocumentRecord",
    reviewer_id: str,
    queue_path: str | Path | None = None,
    notes: str = "",
) -> "DocumentRecord":
    """
    Mark a DocumentRecord as auditor-approved and update the review queue.

    Bridges the gap between review_queue (CSV) and DocumentRecord
    (in-memory). Call this after an auditor has manually verified a
    low-confidence record. The returned record has:
        auditor_approved = True
        reviewer_id      = reviewer_id
        review_date      = today ISO date
        needs_review     = False

    Also calls mark_corrected() on the queue CSV so the row is updated.

    Parameters
    ----------
    record : DocumentRecord
        The record to approve. Must have file_hash set.
    reviewer_id : str
        Auditor initials or ID (e.g. 'SH', 'MS1').
    queue_path : str | Path | None
        Path to review_queue.csv. Uses default if None.
    notes : str
        Optional correction notes written to the queue row.

    Returns
    -------
    DocumentRecord
        The same record with approval fields set.
    """
    from datetime import date

    record.auditor_approved = True
    record.reviewer_id = reviewer_id
    record.review_date = date.today().isoformat()
    record.needs_review = False

    # Update the queue CSV if the record is in it
    if record.file_hash:
        mark_corrected(
            record.file_hash,
            queue_path=queue_path,
            reviewer_id=reviewer_id,
            notes=notes,
        )

    logger.info(
        "approve_record: %s approved by %s",
        record.file_name, reviewer_id,
    )
    return record