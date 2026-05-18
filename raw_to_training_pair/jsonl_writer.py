"""
raw_to_training_pair/jsonl_writer.py
=====================================
Writes approved, gate-passed training pairs to JSONL files.

Rules (per roadmap)
--------------------
- json.dumps() per line — NOT json.dump() (no pretty printing)
- Append mode 'a' — never overwrites existing pairs
- sha256 dedup before every write — double safety net on top of quality_gates
- Stage isolation — stage2 → stage2_domain.jsonl, stage3 → stage3_firm.jsonl
- Thread-safe — fcntl file lock on every append

This module is the last step in the pipeline. It only writes pairs that
have already passed quality_gates.check(). It does a final sha256 check
as a safety net in case check() was bypassed.

Output files
------------
data/stage2_domain.jsonl   — domain fine-tune pairs
data/stage3_firm.jsonl     — firm-specific fine-tune pairs

Public API
----------
    append(pair, output_path)   -> WriteResult
    get_output_path(stage)      -> Path
    count(output_path)          -> int

    WriteResult
        .written    bool
        .reason     str
"""

from __future__ import annotations

import fcntl
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default output directory
_DATA_DIR = Path(__file__).parent.parent / "data"

_STAGE_FILES = {
    "stage2": "stage2_domain.jsonl",
    "stage3": "stage3_firm.jsonl",
}


# ---------------------------------------------------------------------------
# WriteResult
# ---------------------------------------------------------------------------

@dataclass
class WriteResult:
    """Result of an append() call."""

    written: bool
    """True if pair was written to the file."""

    reason: str = ""
    """Explanation if written=False."""

    def __str__(self) -> str:
        if self.written:
            return "WriteResult: WRITTEN"
        return f"WriteResult: SKIPPED — {self.reason}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _existing_hashes(output_path: Path) -> set[str]:
    """
    Read all pair_hash values already in the output file.
    Returns empty set if file does not exist.
    """
    hashes: set[str] = set()
    if not output_path.exists():
        return hashes
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                h = obj.get("metadata", {}).get("pair_hash", "")
                if h:
                    hashes.add(h)
            except json.JSONDecodeError:
                continue
    return hashes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_output_path(stage: str, data_dir: str | Path | None = None) -> Path:
    """
    Return the correct output file path for a given stage.

    Parameters
    ----------
    stage : str
        "stage2" or "stage3"
    data_dir : str | Path | None
        Override the default data/ directory. Useful in tests.

    Returns
    -------
    Path

    Raises
    ------
    ValueError
        If stage is not "stage2" or "stage3".
    """
    if stage not in _STAGE_FILES:
        raise ValueError(
            f"Unknown stage '{stage}'. Must be one of: {list(_STAGE_FILES.keys())}"
        )
    base = Path(data_dir) if data_dir else _DATA_DIR
    return base / _STAGE_FILES[stage]


def append(
    pair: dict,
    output_path: str | Path,
) -> WriteResult:
    """
    Append one training pair to the JSONL output file.

    Performs a final sha256 dedup check before writing — safety net
    in case quality_gates.check() was bypassed.

    Parameters
    ----------
    pair : dict
        Output of pair_builder.build() that has passed quality_gates.check().
        Must have 'messages' and 'metadata' keys.
    output_path : str | Path
        Path to the JSONL file. Parent directory created if needed.
        Must match the pair's stage (stage2_domain.jsonl or stage3_firm.jsonl).

    Returns
    -------
    WriteResult
        .written=True  → pair appended successfully
        .written=False → duplicate detected or write error
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pair_hash = pair.get("metadata", {}).get("pair_hash", "")

    if not pair_hash:
        return WriteResult(
            written=False,
            reason="pair_hash missing from metadata — cannot guarantee dedup.",
        )

    # Final dedup check
    existing = _existing_hashes(path)
    if pair_hash in existing:
        logger.warning(
            "jsonl_writer: duplicate pair_hash %s — skipping write to %s",
            pair_hash[:16], path.name,
        )
        return WriteResult(
            written=False,
            reason=f"Duplicate pair_hash {pair_hash[:16]}... already in {path.name}.",
        )

    # Serialize — json.dumps() per line, no pretty printing
    try:
        line = json.dumps(pair, ensure_ascii=False)
    except Exception as e:
        return WriteResult(
            written=False,
            reason=f"JSON serialization failed: {e}",
        )

    # Append with file lock
    try:
        with open(path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        return WriteResult(
            written=False,
            reason=f"File write failed: {e}",
        )

    logger.info(
        "jsonl_writer: wrote pair %s to %s | stage=%s client=%s",
        pair_hash[:16],
        path.name,
        pair.get("metadata", {}).get("stage", "unknown"),
        pair.get("metadata", {}).get("client_type", "unknown"),
    )

    return WriteResult(written=True)


def count(output_path: str | Path) -> int:
    """
    Count the number of pairs in a JSONL file.

    Parameters
    ----------
    output_path : str | Path

    Returns
    -------
    int
        Number of valid JSON lines. 0 if file does not exist.
    """
    path = Path(output_path)
    if not path.exists():
        return 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    json.loads(line)
                    total += 1
                except json.JSONDecodeError:
                    pass
    return total