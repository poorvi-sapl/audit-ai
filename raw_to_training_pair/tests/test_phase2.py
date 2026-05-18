"""
raw_to_training_pair/tests/test_phase2.py
==========================================
Phase 2 exit criteria tests.

Two scenarios required before Phase 3 starts:

    Scenario 1 — one workpaper end-to-end produces a valid JSONL line
    Scenario 2 — all 4 quality gates verified by deliberate failure tests

Run with:
    pytest raw_to_training_pair/tests/test_phase2.py -v

All tests are self-contained — synthetic pairs built in memory.
Ollama/Gemma is mocked so tests run on any machine without GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Mock Ollama before any pipeline import
# ---------------------------------------------------------------------------

if "ollama" not in sys.modules:
    fake_ollama = types.ModuleType("ollama")

    class _FakeMessage:
        content = (
            "ENGAGEMENT TYPE: Single Audit — Nonprofit\n\n"
            "FINDINGS:\n"
            "1. Cash reconciliation not completed (SOP §3.1)\n"
            "2. Engagement partner signature missing (SOP §2.4)\n\n"
            "RECOMMENDATION:\n"
            "Complete reconciliation and obtain partner sign-off before issuance."
        )

    class _FakeResponse:
        message = _FakeMessage()

    class _FakeModel:
        def __init__(self, name):
            self.model = name

    class _FakeList:
        models = [_FakeModel("gemma3:12b")]

    fake_ollama.chat = lambda **kwargs: _FakeResponse()
    fake_ollama.list = lambda: _FakeList()
    sys.modules["ollama"] = fake_ollama


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair(
    file_name: str = "test.docx",
    file_type: str = "docx",
    client_type: str = "NPO",
    stage: str = "stage2",
    confidence: float = 0.85,
    auditor_approved: bool = True,
    pair_hash: str | None = None,
) -> dict:
    """Build a minimal valid training pair dict."""
    messages = [
        {"role": "system",    "content": "You are an audit assistant."},
        {"role": "user",      "content": f"Workpaper: {file_name}"},
        {"role": "assistant", "content": "ENGAGEMENT TYPE: GAAS\n\nFINDINGS:\n1. Finding (SOP §3.1)\n\nRECOMMENDATION:\nResolve finding."},
    ]
    content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    computed_hash = pair_hash or hashlib.sha256(content.encode()).hexdigest()

    return {
        "messages": messages,
        "metadata": {
            "file_name":             file_name,
            "file_type":             file_type,
            "client_type":           client_type,
            "is_gagas":              False,
            "has_single_audit":      False,
            "extraction_confidence": confidence,
            "auditor_approved":      auditor_approved,
            "pair_type":             "clean",
            "stage":                 stage,
            "fields_missing":        [],
            "sop_sections_used":     ["§3.1"],
            "file_hash":             "a" * 64,
            "pair_hash":             computed_hash,
        },
    }


def _make_document_record(tmp_path: Path):
    """Build a minimal DocumentRecord for pair_builder tests."""
    from auditai_data_normalization.schema import DocumentRecord, Section
    from auditai_data_normalization.pii import scrub_record

    rec = DocumentRecord(
        file_name="test_workpaper.docx",
        file_type="docx",
        file_hash="b" * 64,
        extraction_method="python_docx",
        extraction_status="success",
        extraction_confidence=0.85,
        cleaned_text="Organization: ECE STEP\nFiscal Year End: 2025-06-30\nEngagement Partner: Jane Smith",
        sections=[
            Section(index=0, heading="Engagement Form",
                    content="Organization: ECE STEP\nFiscal Year End: 2025-06-30")
        ],
        word_count=20,
        pii_scrubbed=False,
        auditor_approved=False,
        metadata={
            "confidence_summary": {
                "aggregate": 0.85,
                "passes_gate": True,
                "fields_present": ["client_name", "fiscal_year_end"],
                "fields_missing": ["engagement_partner"],
                "low_confidence_fields": [],
                "per_field_scores": {
                    "client_name": 0.9,
                    "fiscal_year_end": 0.9,
                },
            }
        },
    )
    rec = scrub_record(rec)
    rec.auditor_approved = True
    return rec


# ---------------------------------------------------------------------------
# SCENARIO 1 — end-to-end: workpaper → valid JSONL line
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Scenario 1: one workpaper produces a valid JSONL line."""

    def test_pair_builder_returns_dict(self, tmp_path):
        """pair_builder.build() returns a dict with messages and metadata."""
        from raw_to_training_pair.pair_builder import build

        rec = _make_document_record(tmp_path)
        pair = build(
            record=rec,
            sop_text="SOP §3.1 — Reconciliation must be completed monthly.",
            sop_sections=["§3.1"],
            client_type="NPO",
            is_gagas=False,
            has_single_audit=False,
            stage="stage2",
            pair_type="clean",
        )

        assert pair is not None
        assert "messages" in pair
        assert "metadata" in pair
        assert len(pair["messages"]) == 3
        assert pair["messages"][0]["role"] == "system"
        assert pair["messages"][1]["role"] == "user"
        assert pair["messages"][2]["role"] == "assistant"

    def test_messages_have_required_content(self, tmp_path):
        """All three messages are non-empty strings."""
        from raw_to_training_pair.pair_builder import build

        rec = _make_document_record(tmp_path)
        pair = build(
            record=rec,
            sop_text="SOP §3.1 — Cash reconciliation required.",
            sop_sections=["§3.1"],
            client_type="NPO",
            is_gagas=False,
        )

        assert pair is not None
        for msg in pair["messages"]:
            assert isinstance(msg["content"], str)
            assert len(msg["content"].strip()) > 0

    def test_assistant_content_has_required_sections(self, tmp_path):
        """Assistant completion contains ENGAGEMENT TYPE, FINDINGS, RECOMMENDATION."""
        from raw_to_training_pair.pair_builder import build

        rec = _make_document_record(tmp_path)
        pair = build(
            record=rec,
            sop_text="SOP §3.1 — Required procedures.",
            sop_sections=["§3.1"],
            client_type="NPO",
            is_gagas=False,
        )

        assert pair is not None
        assistant = pair["messages"][2]["content"]
        assert "ENGAGEMENT TYPE:" in assistant
        assert "FINDINGS:" in assistant
        assert "RECOMMENDATION:" in assistant

    def test_metadata_fields_present(self, tmp_path):
        """Pair metadata contains all required keys."""
        from raw_to_training_pair.pair_builder import build

        rec = _make_document_record(tmp_path)
        pair = build(
            record=rec,
            sop_text="SOP §3.1",
            sop_sections=["§3.1"],
            client_type="NPO",
            is_gagas=False,
            stage="stage2",
        )

        assert pair is not None
        meta = pair["metadata"]
        required_keys = [
            "file_name", "file_type", "client_type", "is_gagas",
            "extraction_confidence", "auditor_approved", "pair_type",
            "stage", "pair_hash", "file_hash",
        ]
        for key in required_keys:
            assert key in meta, f"Missing metadata key: {key}"

    def test_pair_hash_is_sha256(self, tmp_path):
        """pair_hash is a 64-char hex string."""
        from raw_to_training_pair.pair_builder import build

        rec = _make_document_record(tmp_path)
        pair = build(
            record=rec,
            sop_text="SOP §3.1",
            sop_sections=["§3.1"],
            client_type="NPO",
            is_gagas=False,
        )

        assert pair is not None
        pair_hash = pair["metadata"]["pair_hash"]
        assert len(pair_hash) == 64
        assert all(c in "0123456789abcdef" for c in pair_hash)

    def test_end_to_end_writes_valid_jsonl_line(self, tmp_path):
        """Full pipeline: build → gates → write → readable JSONL line."""
        from raw_to_training_pair.pair_builder import build
        from raw_to_training_pair.quality_gates import check
        from raw_to_training_pair.jsonl_writer import append, count

        rec = _make_document_record(tmp_path)
        pair = build(
            record=rec,
            sop_text="SOP §3.1 — Reconciliation required.",
            sop_sections=["§3.1"],
            client_type="NPO",
            is_gagas=False,
            stage="stage2",
        )
        assert pair is not None

        output_file = tmp_path / "stage2_domain.jsonl"
        gate_result = check(pair, output_file)
        assert gate_result.passed, f"Gates failed: {gate_result}"

        write_result = append(pair, output_file)
        assert write_result.written, f"Write failed: {write_result}"

        assert count(output_file) == 1

        # Verify the line is valid JSON with expected structure
        line = output_file.read_text().strip()
        parsed = json.loads(line)
        assert "messages" in parsed
        assert "metadata" in parsed
        assert len(parsed["messages"]) == 3

    def test_pii_scrubbed_required(self, tmp_path):
        """pair_builder raises ValueError if pii_scrubbed=False."""
        from raw_to_training_pair.pair_builder import build
        from auditai_data_normalization.schema import DocumentRecord

        rec = DocumentRecord(
            file_name="test.docx",
            file_type="docx",
            file_hash="c" * 64,
            extraction_status="success",
            extraction_confidence=0.85,
            pii_scrubbed=False,  # Not scrubbed
        )

        with pytest.raises(ValueError, match="pii_scrubbed=False"):
            build(
                record=rec,
                sop_text="SOP §3.1",
                sop_sections=["§3.1"],
                client_type="NPO",
                is_gagas=False,
            )


# ---------------------------------------------------------------------------
# SCENARIO 2 — all 4 gates verified by deliberate failure
# ---------------------------------------------------------------------------

class TestQualityGates:
    """Scenario 2: each gate fails when it should."""

    def test_gate1_fails_low_confidence(self, tmp_path):
        """Gate 1 fails when confidence < 0.7."""
        from raw_to_training_pair.quality_gates import check

        pair = _make_pair(confidence=0.5, stage="stage2")
        output = tmp_path / "stage2_domain.jsonl"
        result = check(pair, output)

        assert not result.passed
        assert result.failed_gate == "confidence"

    def test_gate1_passes_sufficient_confidence(self, tmp_path):
        """Gate 1 passes when confidence >= 0.7."""
        from raw_to_training_pair.quality_gates import check

        pair = _make_pair(confidence=0.7, stage="stage2", auditor_approved=True)
        output = tmp_path / "stage2_domain.jsonl"
        result = check(pair, output)

        # May fail on gate 2+ but not gate 1
        assert result.failed_gate != "confidence"

    def test_gate2_fails_not_approved(self, tmp_path):
        """Gate 2 fails when auditor_approved=False."""
        from raw_to_training_pair.quality_gates import check

        pair = _make_pair(confidence=0.9, auditor_approved=False, stage="stage2")
        output = tmp_path / "stage2_domain.jsonl"
        result = check(pair, output)

        assert not result.passed
        assert result.failed_gate == "auditor_approved"

    def test_gate2_passes_when_approved(self, tmp_path):
        """Gate 2 passes when auditor_approved=True."""
        from raw_to_training_pair.quality_gates import check

        pair = _make_pair(confidence=0.9, auditor_approved=True, stage="stage2")
        output = tmp_path / "stage2_domain.jsonl"
        result = check(pair, output)

        assert result.failed_gate != "auditor_approved"

    def test_gate3_fails_duplicate(self, tmp_path):
        """Gate 3 fails when pair_hash already exists in output file."""
        from raw_to_training_pair.quality_gates import check
        from raw_to_training_pair.jsonl_writer import append

        pair = _make_pair(confidence=0.9, auditor_approved=True, stage="stage2")
        output = tmp_path / "stage2_domain.jsonl"

        # Write it once
        append(pair, output)

        # Check again — should fail on duplicate
        result = check(pair, output)
        assert not result.passed
        assert result.failed_gate == "no_duplicate"

    def test_gate3_passes_unique_pair(self, tmp_path):
        """Gate 3 passes when pair_hash is not in output file."""
        from raw_to_training_pair.quality_gates import check

        pair = _make_pair(confidence=0.9, auditor_approved=True, stage="stage2")
        output = tmp_path / "stage2_domain.jsonl"

        result = check(pair, output)
        assert result.failed_gate != "no_duplicate"

    def test_gate4_fails_wrong_stage_file(self, tmp_path):
        """Gate 4 fails when stage2 pair is sent to stage3 file."""
        from raw_to_training_pair.quality_gates import check

        pair = _make_pair(
            confidence=0.9, auditor_approved=True, stage="stage2"
        )
        wrong_output = tmp_path / "stage3_firm.jsonl"  # Wrong file for stage2
        result = check(pair, wrong_output)

        assert not result.passed
        assert result.failed_gate == "stage_isolation"

    def test_gate4_fails_stage3_to_stage2_file(self, tmp_path):
        """Gate 4 fails when stage3 pair is sent to stage2 file."""
        from raw_to_training_pair.quality_gates import check

        pair = _make_pair(
            confidence=0.9, auditor_approved=True, stage="stage3"
        )
        wrong_output = tmp_path / "stage2_domain.jsonl"
        result = check(pair, wrong_output)

        assert not result.passed
        assert result.failed_gate == "stage_isolation"

    def test_gate4_passes_correct_stage_file(self, tmp_path):
        """Gate 4 passes when stage matches output filename."""
        from raw_to_training_pair.quality_gates import check

        pair = _make_pair(
            confidence=0.9, auditor_approved=True, stage="stage2"
        )
        correct_output = tmp_path / "stage2_domain.jsonl"
        result = check(pair, correct_output)

        assert result.passed, f"Expected all gates to pass: {result}"

    def test_all_gates_pass_valid_pair(self, tmp_path):
        """All 4 gates pass for a fully valid pair."""
        from raw_to_training_pair.quality_gates import check

        pair = _make_pair(
            confidence=0.9,
            auditor_approved=True,
            stage="stage2",
        )
        output = tmp_path / "stage2_domain.jsonl"
        result = check(pair, output)

        assert result.passed
        assert result.failed_gate is None


# ---------------------------------------------------------------------------
# JSONL writer tests
# ---------------------------------------------------------------------------

class TestJsonlWriter:
    """Spot-check jsonl_writer behaviour."""

    def test_append_writes_valid_json_line(self, tmp_path):
        from raw_to_training_pair.jsonl_writer import append

        pair = _make_pair(stage="stage2")
        output = tmp_path / "stage2_domain.jsonl"
        result = append(pair, output)

        assert result.written
        line = output.read_text().strip()
        parsed = json.loads(line)
        assert parsed["metadata"]["stage"] == "stage2"

    def test_append_dedup_prevents_second_write(self, tmp_path):
        from raw_to_training_pair.jsonl_writer import append, count

        pair = _make_pair(stage="stage2")
        output = tmp_path / "stage2_domain.jsonl"

        r1 = append(pair, output)
        r2 = append(pair, output)

        assert r1.written
        assert not r2.written
        assert count(output) == 1

    def test_count_returns_correct_number(self, tmp_path):
        from raw_to_training_pair.jsonl_writer import append, count

        output = tmp_path / "stage2_domain.jsonl"
        for i in range(3):
            pair = _make_pair(
                file_name=f"workpaper_{i}.docx",
                stage="stage2",
            )
            append(pair, output)

        assert count(output) == 3

    def test_get_output_path_stage2(self):
        from raw_to_training_pair.jsonl_writer import get_output_path

        path = get_output_path("stage2")
        assert path.name == "stage2_domain.jsonl"

    def test_get_output_path_stage3(self):
        from raw_to_training_pair.jsonl_writer import get_output_path

        path = get_output_path("stage3")
        assert path.name == "stage3_firm.jsonl"

    def test_get_output_path_invalid_stage(self):
        from raw_to_training_pair.jsonl_writer import get_output_path

        with pytest.raises(ValueError):
            get_output_path("stage99")


# ---------------------------------------------------------------------------
# Auditor review tests
# ---------------------------------------------------------------------------

class TestAuditorReview:
    """Spot-check auditor_review queue flow."""

    def test_enqueue_and_load_pending(self, tmp_path):
        from raw_to_training_pair.auditor_review import enqueue, load_pending

        queue = tmp_path / "review_queue.jsonl"
        pair = _make_pair(auditor_approved=False)
        enqueue(pair, queue)

        pending = load_pending(queue)
        assert len(pending) == 1
        assert pending[0]["status"] == "pending"

    def test_approve_sets_auditor_approved(self, tmp_path):
        from raw_to_training_pair.auditor_review import enqueue, approve, load_pending

        queue = tmp_path / "review_queue.jsonl"
        pair = _make_pair(auditor_approved=False)
        enqueue(pair, queue)

        pair_hash = pair["metadata"]["pair_hash"]
        result = approve(pair_hash, reviewer_id="SH", queue_path=queue)

        assert result is True
        assert len(load_pending(queue)) == 0

    def test_reject_removes_from_pending(self, tmp_path):
        from raw_to_training_pair.auditor_review import enqueue, reject, load_pending

        queue = tmp_path / "review_queue.jsonl"
        pair = _make_pair(auditor_approved=False)
        enqueue(pair, queue)

        pair_hash = pair["metadata"]["pair_hash"]
        reject(pair_hash, reviewer_id="SH", notes="Low quality", queue_path=queue)

        assert len(load_pending(queue)) == 0

    def test_stats_returns_correct_counts(self, tmp_path):
        from raw_to_training_pair.auditor_review import enqueue, approve, stats

        queue = tmp_path / "review_queue.jsonl"
        pair1 = _make_pair(file_name="a.docx", auditor_approved=False)
        pair2 = _make_pair(file_name="b.docx", auditor_approved=False)

        enqueue(pair1, queue)
        enqueue(pair2, queue)
        approve(pair1["metadata"]["pair_hash"], "SH", queue_path=queue)

        s = stats(queue)
        assert s["total"] == 2
        assert s["approved"] == 1
        assert s["pending"] == 1