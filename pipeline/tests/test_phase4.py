"""
pipeline/tests/test_phase4.py
==============================
Phase 4 exit criteria tests.

Two scenarios:
    Scenario 1 — 5 synthetic workpapers end-to-end produce valid queue entries
    Scenario 2 — write_approved() writes approved pairs to correct JSONL files

Ollama is mocked. Qdrant retrieval is mocked (empty SOP — valid pipeline path).
Tests run on any machine without GPU or services.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Mock Ollama
# ---------------------------------------------------------------------------

if "ollama" not in sys.modules:
    fake_ollama = types.ModuleType("ollama")

    class _Msg:
        content = (
            "ENGAGEMENT TYPE: GAAS Nonprofit\n\n"
            "FINDINGS:\n1. Missing reconciliation (SOP §3.1)\n\n"
            "RECOMMENDATION:\nComplete reconciliation."
        )
    class _Resp:
        message = _Msg()
    class _Model:
        def __init__(self, n): self.model = n
    class _List:
        models = [_Model("gemma3:12b")]

    fake_ollama.chat = lambda **kw: _Resp()
    fake_ollama.list = lambda: _List()
    sys.modules["ollama"] = fake_ollama

# Force regex PII tier
from auditai_data_normalization import pii as _pii
_pii._presidio_analyzer = False
_pii._presidio_available = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docx(path: Path, lines: list[str]) -> None:
    from docx import Document
    doc = Document()
    for line in lines:
        doc.add_paragraph(line)
    doc.save(str(path))


def _make_xlsx(path: Path) -> None:
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Trial Balance"
    ws.append(["Account Code", "Account Name", "Debit", "Credit"])
    ws.append(["1010", "Cash", "500000", ""])
    ws.append(["2010", "Accounts Payable", "", "100000"])
    wb.save(str(path))


def _make_csv(path: Path) -> None:
    path.write_text(
        "account_code,account_name,balance\n"
        "1010,Cash,505900\n"
        "1020,Accounts Receivable,380277\n"
    )


def _make_json(path: Path) -> None:
    import json
    path.write_text(json.dumps({
        "client_name": "ECE STEP",
        "fiscal_year_end": "2025-06-30",
        "audit_type": "GAAS",
    }))


def _mock_retrieve(record, top_k=5):
    from pipeline.qdrant_retriever import RetrievalResult
    return RetrievalResult(
        sop_text="SOP §3.1 — Bank Reconciliation: Complete within 30 days.",
        sop_sections=["§3.1"],
        chunks=[],
        strategy="mock",
    )


def _mock_normalize(file_path, run_parallel=False):
    from auditai_data_normalization.schema import DocumentRecord, Section
    from auditai_data_normalization.pii import scrub_record
    rec = DocumentRecord(
        file_name=Path(file_path).name,
        file_type="docx",
        file_hash="a" * 64,
        extraction_method="python_docx",
        extraction_status="success",
        extraction_confidence=0.85,
        cleaned_text="Organization: ECE STEP\nFiscal Year End: 2025-06-30",
        sections=[Section(index=0, heading="Form", content="Organization: ECE STEP")],
        word_count=20,
        pii_scrubbed=False,
        auditor_approved=True,
        metadata={"confidence_summary": {
            "aggregate": 0.85, "passes_gate": True,
            "fields_present": ["client_name", "fiscal_year_end"],
            "fields_missing": [], "low_confidence_fields": [],
            "per_field_scores": {"client_name": 0.9, "fiscal_year_end": 0.9},
        }},
    )
    return scrub_record(rec)


# ---------------------------------------------------------------------------
# SCENARIO 1 — 5 workpapers end-to-end
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Scenario 1: 5 workpaper types through full pipeline."""

    @patch("pipeline.pipeline.normalize_document", side_effect=_mock_normalize)
    @patch("pipeline.pipeline.retrieve", side_effect=_mock_retrieve)
    def test_docx_workpaper(self, mock_ret, mock_norm, tmp_path):
        from pipeline.pipeline import process_workpaper

        f = tmp_path / "engagement.docx"
        _make_docx(f, [
            "Organization: ECE STEP LLC",
            "Fiscal Year End: 2025-06-30",
            "Engagement Partner: Jane Smith",
            "Single Audit: Not Applicable",
        ])

        result = process_workpaper(
            str(f),
            queue_path=tmp_path / "queue.jsonl",
            data_dir=tmp_path,
            run_parallel=False,
            client_types=["NPO"],
        )

        assert not result.skipped, result.skip_reason
        assert result.pairs_built > 0
        assert result.record is not None
        assert result.record.pii_scrubbed is True

    @patch("pipeline.pipeline.normalize_document", side_effect=_mock_normalize)
    @patch("pipeline.pipeline.retrieve", side_effect=_mock_retrieve)
    def test_xlsx_workpaper(self, mock_ret, mock_norm, tmp_path):
        from pipeline.pipeline import process_workpaper

        f = tmp_path / "trial_balance.xlsx"
        _make_xlsx(f)

        result = process_workpaper(
            str(f),
            queue_path=tmp_path / "queue.jsonl",
            data_dir=tmp_path,
            run_parallel=False,
            client_types=["NPO"],
        )

        assert not result.skipped, result.skip_reason
        assert result.pairs_built > 0

    @patch("pipeline.pipeline.normalize_document", side_effect=_mock_normalize)
    @patch("pipeline.pipeline.retrieve", side_effect=_mock_retrieve)
    def test_csv_workpaper(self, mock_ret, mock_norm, tmp_path):
        from pipeline.pipeline import process_workpaper

        f = tmp_path / "data.csv"
        _make_csv(f)

        result = process_workpaper(
            str(f),
            queue_path=tmp_path / "queue.jsonl",
            data_dir=tmp_path,
            run_parallel=False,
            client_types=["NPO"],
        )

        assert not result.skipped, result.skip_reason
        assert result.pairs_built > 0

    @patch("pipeline.pipeline.normalize_document", side_effect=_mock_normalize)
    @patch("pipeline.pipeline.retrieve", side_effect=_mock_retrieve)
    def test_json_workpaper(self, mock_ret, mock_norm, tmp_path):
        from pipeline.pipeline import process_workpaper

        f = tmp_path / "metadata.json"
        _make_json(f)

        result = process_workpaper(
            str(f),
            queue_path=tmp_path / "queue.jsonl",
            data_dir=tmp_path,
            run_parallel=False,
            client_types=["NPO"],
        )

        assert not result.skipped, result.skip_reason
        assert result.pairs_built > 0

    @patch("pipeline.pipeline.normalize_document", side_effect=_mock_normalize)
    @patch("pipeline.pipeline.retrieve", side_effect=_mock_retrieve)
    def test_multiple_client_types(self, mock_ret, mock_norm, tmp_path):
        """Generates variants for all 4 client types."""
        from pipeline.pipeline import process_workpaper

        f = tmp_path / "form.docx"
        _make_docx(f, ["Organization: ABC Nonprofit Inc.", "FYE: 2025-06-30"])

        result = process_workpaper(
            str(f),
            queue_path=tmp_path / "queue.jsonl",
            data_dir=tmp_path,
            run_parallel=False,
            client_types=["NPO", "government", "for_profit", "tribal"],
        )

        assert not result.skipped, result.skip_reason
        # 4 client types × 3 variants (1 clean + 2 deficient) = 12
        assert result.pairs_built == 12

    @patch("pipeline.pipeline.normalize_document", side_effect=_mock_normalize)
    @patch("pipeline.pipeline.retrieve", side_effect=_mock_retrieve)
    def test_queue_file_populated(self, mock_ret, mock_norm, tmp_path):
        """Review queue file is created and contains valid JSONL."""
        from pipeline.pipeline import process_workpaper

        queue = tmp_path / "queue.jsonl"
        f = tmp_path / "form.docx"
        _make_docx(f, ["Organization: Test Org LLC", "FYE: 2025-06-30"])

        process_workpaper(
            str(f),
            queue_path=queue,
            data_dir=tmp_path,
            run_parallel=False,
            client_types=["NPO"],
        )

        assert queue.exists()
        lines = [l for l in queue.read_text().splitlines() if l.strip()]
        assert len(lines) > 0

        for line in lines:
            entry = json.loads(line)
            assert "pair" in entry
            assert "messages" in entry["pair"]
            assert "metadata" in entry["pair"]

    def test_file_not_found_returns_skipped(self, tmp_path):
        """Missing file returns skipped result, no exception."""
        from pipeline.pipeline import process_workpaper

        result = process_workpaper(
            str(tmp_path / "nonexistent.docx"),
            queue_path=tmp_path / "queue.jsonl",
            data_dir=tmp_path,
        )

        assert result.skipped is True
        assert len(result.errors) > 0

    def test_low_confidence_skipped(self, tmp_path):
        """Records below confidence gate are skipped."""
        from pipeline.pipeline import process_workpaper
        from unittest.mock import patch as p2
        from auditai_data_normalization.schema import DocumentRecord
        from auditai_data_normalization.pii import scrub_record

        f = tmp_path / "form.docx"
        _make_docx(f, ["Minimal content"])

        low_conf_record = DocumentRecord(
            file_name="form.docx",
            file_type="docx",
            file_hash="a" * 64,
            extraction_status="success",
            extraction_confidence=0.3,
            cleaned_text="Minimal content",
            pii_scrubbed=False,
        )
        low_conf_record = scrub_record(low_conf_record)

        with p2("pipeline.pipeline.normalize_document", return_value=low_conf_record):
            result = process_workpaper(
                str(f),
                queue_path=tmp_path / "queue.jsonl",
                data_dir=tmp_path,
                run_parallel=False,
            )

        assert result.skipped is True
        assert "confidence" in result.skip_reason


# ---------------------------------------------------------------------------
# SCENARIO 2 — write_approved writes to correct JSONL
# ---------------------------------------------------------------------------

class TestWriteApproved:
    """Scenario 2: approved pairs written to correct stage files."""

    def test_approved_pairs_written_to_stage2(self, tmp_path):
        from pipeline.pipeline import write_approved
        from raw_to_training_pair.auditor_review import enqueue, approve
        from raw_to_training_pair.pair_builder import SYSTEM_PROMPT
        import hashlib

        queue = tmp_path / "queue.jsonl"

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Test workpaper"},
            {"role": "assistant", "content": "ENGAGEMENT TYPE: GAAS\n\nFINDINGS:\n1. Finding (SOP §3.1)\n\nRECOMMENDATION:\nResolve."},
        ]
        content = json.dumps(messages, sort_keys=True)
        pair_hash = hashlib.sha256(content.encode()).hexdigest()

        pair = {
            "messages": messages,
            "metadata": {
                "file_name": "test.docx", "file_type": "docx",
                "client_type": "NPO", "is_gagas": False,
                "has_single_audit": False, "extraction_confidence": 0.85,
                "auditor_approved": False, "pair_type": "clean",
                "stage": "stage2", "fields_missing": [],
                "sop_sections_used": ["§3.1"],
                "file_hash": "b" * 64, "pair_hash": pair_hash,
            },
        }

        enqueue(pair, queue)
        approve(pair_hash, reviewer_id="SH", queue_path=queue)

        counts = write_approved(queue_path=queue, data_dir=tmp_path)

        assert counts["written"] >= 1
        stage2_file = tmp_path / "stage2_domain.jsonl"
        assert stage2_file.exists()
        line = json.loads(stage2_file.read_text().strip())
        assert line["metadata"]["stage"] == "stage2"

    def test_rejected_pairs_not_written(self, tmp_path):
        from pipeline.pipeline import write_approved
        from raw_to_training_pair.auditor_review import enqueue, reject
        import hashlib

        queue = tmp_path / "queue.jsonl"

        messages = [{"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"}]
        pair_hash = hashlib.sha256(json.dumps(messages).encode()).hexdigest()

        pair = {
            "messages": messages,
            "metadata": {
                "file_name": "r.docx", "file_type": "docx",
                "client_type": "NPO", "is_gagas": False,
                "has_single_audit": False, "extraction_confidence": 0.85,
                "auditor_approved": False, "pair_type": "clean",
                "stage": "stage2", "fields_missing": [],
                "sop_sections_used": [], "file_hash": "c" * 64,
                "pair_hash": pair_hash,
            },
        }

        enqueue(pair, queue)
        reject(pair_hash, reviewer_id="SH", notes="Low quality", queue_path=queue)

        counts = write_approved(queue_path=queue, data_dir=tmp_path)
        assert counts["written"] == 0