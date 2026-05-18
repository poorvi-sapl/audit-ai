"""
raw_to_training_pair/auditor_review.py
=======================================
Phase E3 additions — approval tiers.

Previously: approve() / reject() only.

Now four tiers:
    approve()             Full approve → goes straight to JSONL
    conditional_approve() Conditional approve → JSONL with reviewer note
    send_for_correction() Stays in queue, Gemma re-runs with auditor hint
    reject()              Discarded (stays in queue for audit trail)

Queue entry shape (unchanged):
{
    "pair":        { "messages": [...], "metadata": {...} },
    "status":      "pending" | "approved" | "conditional" | "correction" | "rejected",
    "reviewer":    str,
    "notes":       str,
    "hint":        str,   # auditor hint for correction re-run (E3)
    "queued_at":   str,
    "reviewed_at": str,
}

Public API
----------
    enqueue(pair, queue_path)
    load_all(queue_path)
    load_pending(queue_path)
    update_completion(pair_hash, new_content, queue_path)          -> bool
    approve(pair_hash, reviewer_id, notes, queue_path)             -> bool
    conditional_approve(pair_hash, reviewer_id, notes, queue_path) -> bool
    send_for_correction(pair_hash, reviewer_id, hint, queue_path)  -> bool
    reject(pair_hash, reviewer_id, notes, queue_path)              -> bool
    stats(queue_path)                                               -> dict
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_PATH = Path(__file__).parent.parent / "data" / "review_queue.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_lines(queue_path: Path) -> list[dict]:
    if not queue_path.exists():
        return []
    entries = []
    with open(queue_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed queue line: %s", e)
    return entries


def _write_lines(entries: list[dict], queue_path: Path) -> None:
    """Atomically rewrite the queue file via a temp file + os.replace().

    The queue file is never empty or partially written during the swap:
    - Write all content to a sibling .tmp file (same directory = same fs)
    - Lock the tmp file while writing
    - os.replace() is atomic on POSIX — readers always see a complete file
    """
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=queue_path.parent, prefix=".queue_write_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        os.replace(tmp_path, queue_path)
    except Exception:
        # Clean up the tmp file if anything went wrong before the rename
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _get_pair_hash(pair: dict) -> str:
    return pair.get("metadata", {}).get("pair_hash", "")


def _find_pending(entries: list[dict], pair_hash: str) -> dict | None:
    """Find a pending entry by pair_hash. Returns None if not found."""
    for entry in entries:
        if (
            _get_pair_hash(entry.get("pair", {})) == pair_hash
            and entry.get("status") == "pending"
        ):
            return entry
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enqueue(
    pair: dict,
    queue_path: str | Path | None = None,
    sop_text: str = "",
) -> None:
    """Add a pair to the review queue with status 'pending'."""
    path = Path(queue_path) if queue_path else _DEFAULT_QUEUE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "pair":        pair,
        "status":      "pending",
        "reviewer":    "",
        "notes":       "",
        "hint":        "",   # E3 — auditor correction hint
        "sop_text":    sop_text,  # stored for correction reprocess
        "queued_at":   _now_iso(),
        "reviewed_at": "",
    }

    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    logger.info(
        "auditor_review: enqueued %s for %s",
        _get_pair_hash(pair)[:8],
        pair.get("metadata", {}).get("file_name", "unknown"),
    )


def load_all(queue_path: str | Path | None = None) -> list[dict]:
    return _load_lines(Path(queue_path) if queue_path else _DEFAULT_QUEUE_PATH)


def load_pending(queue_path: str | Path | None = None) -> list[dict]:
    return [e for e in load_all(queue_path) if e.get("status") == "pending"]


def update_completion(
    pair_hash: str,
    new_content: str,
    queue_path: str | Path | None = None,
) -> bool:
    """
    Overwrite the assistant message content for a pending pair.

    Must be called before approve() / conditional_approve() when the
    reviewer has edited the completion in the UI — approve() reloads
    from disk, so in-memory edits are lost unless persisted first.

    Returns True if the entry was found and updated.
    """
    path = Path(queue_path) if queue_path else _DEFAULT_QUEUE_PATH
    entries = _load_lines(path)

    for entry in entries:
        if (
            _get_pair_hash(entry.get("pair", {})) == pair_hash
            and entry.get("status") == "pending"
        ):
            messages = entry.get("pair", {}).get("messages", [])
            for m in messages:
                if m.get("role") == "assistant":
                    m["content"] = new_content
            _write_lines(entries, path)
            logger.info(
                "auditor_review: updated completion for %s (%d chars)",
                pair_hash[:8], len(new_content),
            )
            return True

    logger.warning(
        "auditor_review: update_completion — %s not found or not pending",
        pair_hash[:8],
    )
    return False


def approve(
    pair_hash: str,
    reviewer_id: str,
    notes: str = "",
    queue_path: str | Path | None = None,
) -> bool:
    """
    Full approve — pair goes straight to JSONL on next export.
    Sets auditor_approved=True in pair metadata.
    """
    path = Path(queue_path) if queue_path else _DEFAULT_QUEUE_PATH
    entries = _load_lines(path)
    entry = _find_pending(entries, pair_hash)

    if entry is None:
        logger.warning("auditor_review: approve — %s not found or not pending", pair_hash[:8])
        return False

    entry["status"]      = "approved"
    entry["reviewer"]    = reviewer_id
    entry["notes"]       = notes
    entry["reviewed_at"] = _now_iso()
    if "metadata" in entry.get("pair", {}):
        entry["pair"]["metadata"]["auditor_approved"] = True

    _write_lines(entries, path)
    logger.info("auditor_review: approved %s by %s", pair_hash[:8], reviewer_id)
    return True


def conditional_approve(
    pair_hash: str,
    reviewer_id: str,
    notes: str = "",
    queue_path: str | Path | None = None,
) -> bool:
    """
    E3 — Conditional approve.

    Pair goes to JSONL but carries the reviewer note in metadata.
    Used when the completion is acceptable but has minor issues the
    reviewer wants to document (e.g. "Approved — finding 2 needs
    better SOP citation when updated SOP available").

    Sets auditor_approved=True and status='conditional'.
    The reviewer note is preserved in pair metadata.
    """
    path = Path(queue_path) if queue_path else _DEFAULT_QUEUE_PATH
    entries = _load_lines(path)
    entry = _find_pending(entries, pair_hash)

    if entry is None:
        logger.warning(
            "auditor_review: conditional_approve — %s not found", pair_hash[:8]
        )
        return False

    entry["status"]      = "conditional"
    entry["reviewer"]    = reviewer_id
    entry["notes"]       = notes
    entry["reviewed_at"] = _now_iso()

    if "metadata" in entry.get("pair", {}):
        entry["pair"]["metadata"]["auditor_approved"] = True
        entry["pair"]["metadata"]["reviewer_note"]   = notes
        entry["pair"]["metadata"]["approval_type"]   = "conditional"

    _write_lines(entries, path)
    logger.info(
        "auditor_review: conditional_approve %s by %s — note: %s",
        pair_hash[:8], reviewer_id, notes[:80],
    )
    return True


def send_for_correction(
    pair_hash: str,
    reviewer_id: str,
    hint: str,
    queue_path: str | Path | None = None,
) -> bool:
    """
    E3 — Send for correction.

    Pair stays in queue with status='correction'. The auditor hint is
    stored in entry['hint'] and will be passed to completion_drafter.py
    on the next re-run (via pipeline.py correction loop).

    Use when the completion is factually wrong or missing something
    specific the auditor can describe: e.g. "Finding 1 should cite
    SOP §3.2 not §2.1 — check the independence section."

    Parameters
    ----------
    hint : str
        Specific instruction for Gemma on the re-run.
        Should be concrete: what to fix, not just "improve this".
    """
    if not hint or not hint.strip():
        logger.warning(
            "auditor_review: send_for_correction requires a non-empty hint"
        )
        return False

    path = Path(queue_path) if queue_path else _DEFAULT_QUEUE_PATH
    entries = _load_lines(path)
    entry = _find_pending(entries, pair_hash)

    if entry is None:
        logger.warning(
            "auditor_review: send_for_correction — %s not found", pair_hash[:8]
        )
        return False

    entry["status"]      = "correction"
    entry["reviewer"]    = reviewer_id
    entry["hint"]        = hint.strip()
    entry["reviewed_at"] = _now_iso()

    if "metadata" in entry.get("pair", {}):
        entry["pair"]["metadata"]["correction_hint"]  = hint.strip()
        entry["pair"]["metadata"]["correction_by"]    = reviewer_id
        entry["pair"]["metadata"]["auditor_approved"] = False

    _write_lines(entries, path)
    logger.info(
        "auditor_review: sent for correction %s by %s — hint: %s",
        pair_hash[:8], reviewer_id, hint[:80],
    )
    return True


def reject(
    pair_hash: str,
    reviewer_id: str,
    notes: str = "",
    queue_path: str | Path | None = None,
) -> bool:
    """
    Reject a pair. Not written to JSONL. Stays in queue for audit trail.
    """
    path = Path(queue_path) if queue_path else _DEFAULT_QUEUE_PATH
    entries = _load_lines(path)
    entry = _find_pending(entries, pair_hash)

    if entry is None:
        logger.warning("auditor_review: reject — %s not found", pair_hash[:8])
        return False

    entry["status"]      = "rejected"
    entry["reviewer"]    = reviewer_id
    entry["notes"]       = notes
    entry["reviewed_at"] = _now_iso()

    _write_lines(entries, path)
    logger.info("auditor_review: rejected %s by %s", pair_hash[:8], reviewer_id)
    return True


def stats(queue_path: str | Path | None = None) -> dict[str, Any]:
    """
    Queue summary statistics for Streamlit sidebar.

    Returns
    -------
    dict with keys: total, pending, approved, conditional, correction,
                    rejected, by_stage, by_client
    """
    entries = load_all(queue_path)
    counts = {"pending": 0, "approved": 0, "conditional": 0,
              "correction": 0, "rejected": 0}
    by_stage:  dict[str, int] = {}
    by_client: dict[str, int] = {}

    for entry in entries:
        status = entry.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
        meta   = entry.get("pair", {}).get("metadata", {})
        stage  = meta.get("stage", "unknown")
        client = meta.get("client_type", "unknown")
        by_stage[stage]   = by_stage.get(stage, 0) + 1
        by_client[client] = by_client.get(client, 0) + 1

    return {
        "total":       len(entries),
        "pending":     counts["pending"],
        "approved":    counts["approved"],
        "conditional": counts["conditional"],
        "correction":  counts["correction"],
        "rejected":    counts["rejected"],
        "by_stage":    by_stage,
        "by_client":   by_client,
    }