"""
raw_to_training_pair/quality_gates.py
=======================================
Phase A4 changes:
  Gate 0 (new)     — extraction_gate=True (safety catch)
  Gate 1 (updated) — review_confidence >= 0.70  (was extraction_confidence >= 0.70)
  Gates 2-4        — unchanged (auditor_approved, no_dup, stage_isolation)
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_EXTRACTION_GATE_THRESHOLD   = 0.50
_REVIEW_CONFIDENCE_THRESHOLD = 0.70
_STAGE_FILES = {"stage2": "stage2_domain.jsonl", "stage3": "stage3_firm.jsonl"}


@dataclass
class GateResult:
    passed: bool
    failed_gate: str | None = None
    reason: str = ""

    def __str__(self) -> str:
        if self.passed:
            return "GateResult: PASS"
        return f"GateResult: FAIL gate='{self.failed_gate}' — {self.reason}"


def _gate_extraction(pair: dict) -> GateResult:
    """Gate 0 — extraction_gate=True (pipeline safety catch, Phase A4)."""
    if pair.get("metadata", {}).get("extraction_gate", False):
        return GateResult(passed=True)
    conf = pair.get("metadata", {}).get("extraction_confidence", 0.0)
    return GateResult(
        passed=False,
        failed_gate="extraction_gate",
        reason=(
            f"extraction_gate=False (extraction_confidence={conf:.3f} "
            f"< {_EXTRACTION_GATE_THRESHOLD}). "
            "Record should have gone through LLM fallback first (Phase B2)."
        ),
    )


def _gate_review_confidence(pair: dict) -> GateResult:
    """
    Gate 1 — review_confidence >= 0.70.
    Phase A4: replaces the old extraction_confidence >= 0.70 gate.
    review_confidence is set by completion_drafter.py, not normalize.py.
    """
    rev_conf = pair.get("metadata", {}).get("review_confidence", 0.0)
    if rev_conf >= _REVIEW_CONFIDENCE_THRESHOLD:
        return GateResult(passed=True)
    return GateResult(
        passed=False,
        failed_gate="review_confidence",
        reason=(
            f"review_confidence {rev_conf:.3f} < {_REVIEW_CONFIDENCE_THRESHOLD}. "
            "Completion quality below threshold. Send to review queue — "
            "auditor can edit and re-approve."
        ),
    )


def _gate_auditor_approved(pair: dict) -> GateResult:
    """Gate 2 — auditor_approved=True."""
    if pair.get("metadata", {}).get("auditor_approved", False):
        return GateResult(passed=True)
    return GateResult(
        passed=False,
        failed_gate="auditor_approved",
        reason=(
            "auditor_approved=False. Approve via auditor_review.approve() "
            "before writing to JSONL."
        ),
    )


def _gate_no_duplicate(pair: dict, output_path: Path) -> GateResult:
    """Gate 3 — pair_hash must not exist in output file."""
    pair_hash = pair.get("metadata", {}).get("pair_hash", "")
    if not pair_hash:
        return GateResult(
            passed=False,
            failed_gate="no_duplicate",
            reason="pair_hash missing from metadata.",
        )
    if not output_path.exists():
        return GateResult(passed=True)
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                existing_hash = json.loads(line).get("metadata", {}).get("pair_hash", "")
                if existing_hash == pair_hash:
                    return GateResult(
                        passed=False,
                        failed_gate="no_duplicate",
                        reason=f"pair_hash {pair_hash[:16]}... already exists in {output_path.name}.",
                    )
            except json.JSONDecodeError:
                continue
    return GateResult(passed=True)


def _gate_stage_isolation(pair: dict, output_path: Path) -> GateResult:
    """Gate 4 — stage2 pairs to stage2_domain.jsonl, stage3 to stage3_firm.jsonl."""
    stage = pair.get("metadata", {}).get("stage", "")
    if stage not in _STAGE_FILES:
        return GateResult(
            passed=False,
            failed_gate="stage_isolation",
            reason=f"Unknown stage '{stage}'. Must be one of: {list(_STAGE_FILES)}.",
        )
    expected, actual = _STAGE_FILES[stage], output_path.name
    if actual != expected:
        return GateResult(
            passed=False,
            failed_gate="stage_isolation",
            reason=f"Stage '{stage}' must go to '{expected}' but output_path is '{actual}'.",
        )
    return GateResult(passed=True)


def _gate_structural_duplicate(pair: dict) -> GateResult:
    """
    Gate 5 — structural fingerprint deduplication across workpapers.

    Computes a canonical fingerprint from (client_type, pair_type,
    sorted_missing_fields, sorted_present_fields) and checks it against
    the fingerprint index in data/pair_fingerprint_index.jsonl.

    Policy (from threshold_config.yaml deduplication.policy):
        "soft"   — duplicate found → passed=True but metadata flagged
                   as review_duplicate so reviewer is notified.
        "strict" — duplicate found → passed=False, pair blocked.

    Unlike the hash gate (exact content match), this catches near-duplicates
    where wording differs but the training signal is structurally identical.
    """
    try:
        from raw_to_training_pair.pair_index import check_duplicate
        result = check_duplicate(pair)

        if not result.is_duplicate:
            return GateResult(passed=True)

        if result.policy == "strict":
            return GateResult(
                passed=False,
                failed_gate="structural_duplicate",
                reason=result.message,
            )

        # Soft policy — pass but flag metadata for reviewer
        meta = pair.get("metadata", {})
        meta["review_duplicate"]       = True
        meta["duplicate_fingerprint"]  = result.fingerprint
        meta["duplicate_message"]      = result.message
        if result.matching_entry:
            meta["duplicate_prior_workpaper"] = result.matching_entry.get(
                "source_workpaper", ""
            )
        logger.info(
            "quality_gates: soft duplicate flagged — %s", result.message
        )
        return GateResult(passed=True)

    except Exception as e:
        # Never block a pair due to index errors — log and pass through
        logger.warning(
            "quality_gates: structural duplicate check failed (%s) — passing pair", e
        )
        return GateResult(passed=True)


def check(pair: dict, output_path: str | Path) -> GateResult:
    """
    Run all 6 quality gates against a training pair. Returns on first failure.

    Gate order:
        0. extraction_gate=True           (safety — Phase A4)
        1. review_confidence >= 0.70      (quality — Phase A4)
        2. auditor_approved=True
        3. no exact hash duplicate
        4. stage isolation
        5. structural fingerprint duplicate (cross-workpaper)
    """
    path = Path(output_path)
    gates = [
        lambda: _gate_extraction(pair),
        lambda: _gate_review_confidence(pair),
        lambda: _gate_auditor_approved(pair),
        lambda: _gate_no_duplicate(pair, path),
        lambda: _gate_stage_isolation(pair, path),
        lambda: _gate_structural_duplicate(pair),
    ]
    for gate_fn in gates:
        result = gate_fn()
        if not result.passed:
            logger.warning("quality_gates: FAIL %s", result)
            return result
    logger.info(
        "quality_gates: PASS %s rev_conf=%.3f",
        pair.get("metadata", {}).get("file_name", "?"),
        pair.get("metadata", {}).get("review_confidence", 0.0),
    )
    return GateResult(passed=True)