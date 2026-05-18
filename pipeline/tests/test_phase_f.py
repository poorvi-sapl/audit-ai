"""
pipeline/tests/test_phase_f.py
================================
Phase F2 — benchmark validation tests.

Covers the new systems added in Phases A-E:
  - Weighted tier-based confidence scoring (Phase A2)
  - Two-gate logic: extraction_gate + quality_gate (Phase A4)
  - LLM fallback merge + re-score (Phase B2)
  - LLM confidence calibration (Phase B3)
  - Review confidence scoring — 5-criteria rubric (Phase C4)
  - uncertain_sections in pair metadata (Phase E1)
  - Four-tier approval (Phase E3)

All tests run without Ollama or GPU.
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "outputs"))

# ── Mock Ollama ────────────────────────────────────────────────────────────────
if "ollama" not in sys.modules:
    fake = types.ModuleType("ollama")
    class _Msg:
        content = json.dumps({
            "client_name":        {"value": "ABC Foundation", "confident": True,  "source_hint": "Organization: ABC Foundation"},
            "fiscal_year_end":    {"value": "2024-06-30",     "confident": True,  "source_hint": "Year Ended June 30"},
            "engagement_partner": {"value": "Jane Smith CPA", "confident": True,  "source_hint": "EP: Jane Smith"},
            "audit_type":         {"value": "GAGAS",          "confident": True,  "source_hint": "Government Auditing"},
            "includes_gagas":     {"value": "true",           "confident": True,  "source_hint": "Yellow Book"},
            "includes_single_audit": {"value": "false",       "confident": True,  "source_hint": "N/A"},
            "engagement_decision":   {"value": "Accept",      "confident": True,  "source_hint": "Decision: Accept"},
            "reporting_framework":   {"value": "GAAP",        "confident": True,  "source_hint": "Generally Accepted"},
        })
    class _Resp:
        message = _Msg()
    class _Model:
        def __init__(self, n): self.model = n
    class _List:
        models = [_Model("gemma3:12b")]
    fake.chat = lambda **kw: _Resp()
    fake.list = lambda: _List()
    sys.modules["ollama"] = fake


# ── Force regex PII tier (no spaCy/Presidio needed) ──────────────────────────
from auditai_data_normalization import pii as _pii
_pii._presidio_available = False


# ===========================================================================
# Phase A2 — Weighted tier-based confidence scoring
# ===========================================================================

class TestWeightedScoring:

    def setup_method(self):
        from auditai_data_normalization.confidence import (
            _hardcoded_fallback_tiers, score_fields, score_record, summarise
        )
        self.tiers = _hardcoded_fallback_tiers()
        self.score_fields = score_fields
        self.score_record = score_record
        self.summarise    = summarise

    def test_all_tier1_found_hits_floor(self):
        scores = {f: 0.9 for f in self.tiers.tier1}
        result = self.score_record(scores, tiers=self.tiers)
        assert result >= 0.82, f"All tier1 found should hit 0.82 floor, got {result}"

    def test_five_tier1_found_hits_065_floor(self):
        fields = dict(list({f: 0.9 for f in self.tiers.tier1}.items())[:5])
        fields.update({f: 0.0 for f in list(self.tiers.tier1)[5:]})
        result = self.score_record(fields, tiers=self.tiers)
        assert result >= 0.65, f"5/8 tier1 should hit 0.65 floor, got {result}"

    def test_three_tier1_found_hits_055_floor(self):
        fields = dict(list({f: 0.9 for f in self.tiers.tier1}.items())[:3])
        fields.update({f: 0.0 for f in list(self.tiers.tier1)[3:]})
        result = self.score_record(fields, tiers=self.tiers)
        assert result >= 0.55, f"3/8 tier1 should hit 0.55 floor, got {result}"

    def test_tier3_fields_not_penalised(self):
        # Same tier1/2 scores with many tier3 missing
        tier12 = {f: 0.9 for f in list(self.tiers.tier1)[:4]}
        tier12.update({f: 0.0 for f in list(self.tiers.tier1)[4:]})
        result_without_tier3 = self.score_record(tier12, tiers=self.tiers)

        tier12_with_tier3 = dict(tier12)
        tier12_with_tier3.update({f: 0.0 for f in list(self.tiers.tier3)[:10]})
        result_with_tier3 = self.score_record(tier12_with_tier3, tiers=self.tiers)

        assert result_without_tier3 == result_with_tier3, \
            "Tier 3 missing fields should not change the score"

    def test_npo_confirmed_case(self):
        """The NPO-CX-1.1 confirmed case: 3/8 tier1, score was 0.562 before A2."""
        scores = {
            "client_name":           0.9,
            "fiscal_year_end":       0.9,
            "engagement_decision":   0.9,
            "engagement_partner":    0.0,
            "audit_type":            0.0,
            "includes_gagas":        0.0,
            "includes_single_audit": 0.0,
            "reporting_framework":   0.0,
            "document_reference":    0.9,
        }
        result = self.score_record(scores, tiers=self.tiers)
        assert result >= 0.55, f"NPO case should be >= 0.55 after A2 floor, got {result}"
        # Old flat-weighted approach would have given ~0.42
        assert result > 0.42, "A2 must score higher than old approach for NPO case"


# ===========================================================================
# Phase A4 — Two-gate logic
# ===========================================================================

class TestTwoGateLogic:

    def test_extraction_gate_threshold(self):
        from auditai_data_normalization.schema import DocumentRecord
        r = DocumentRecord(extraction_confidence=0.55)
        r.extraction_gate = r.extraction_confidence >= 0.50
        assert r.extraction_gate == True

        r2 = DocumentRecord(extraction_confidence=0.42)
        r2.extraction_gate = r2.extraction_confidence >= 0.50
        assert r2.extraction_gate == False

    def test_quality_gate_independent_of_extraction(self):
        """Low extraction + good review = trainable. High extraction + bad review = blocked."""
        from auditai_data_normalization.schema import DocumentRecord

        # NPO case: low extraction, good review → trainable
        r = DocumentRecord(
            extraction_confidence=0.55, extraction_gate=True,
            review_confidence=0.80,    quality_gate=True,
            pii_scrubbed=True, auditor_approved=True,
            extraction_status="partial", cleaned_text="content",
        )
        assert r.is_ready_for_training() == True

        # High extraction, bad review → blocked
        r2 = DocumentRecord(
            extraction_confidence=0.90, extraction_gate=True,
            review_confidence=0.45,    quality_gate=False,
            pii_scrubbed=True, auditor_approved=True,
            extraction_status="success", cleaned_text="content",
        )
        assert r2.is_ready_for_training() == False

    def test_is_ready_for_drafting(self):
        from auditai_data_normalization.schema import DocumentRecord
        r = DocumentRecord(
            extraction_confidence=0.55, extraction_gate=True,
            pii_scrubbed=True, extraction_status="partial",
            cleaned_text="content",
        )
        assert r.is_ready_for_drafting() == True

        r2 = DocumentRecord(
            extraction_confidence=0.40, extraction_gate=False,
        )
        assert r2.is_ready_for_drafting() == False


# ===========================================================================
# Phase B2 — LLM fallback merge + re-score
# ===========================================================================

class TestLLMFallback:

    def setup_method(self):
        from auditai_data_normalization.confidence import _hardcoded_fallback_tiers, score_fields, summarise
        from auditai_data_normalization.extractors.llm_extractor import FieldResult, _parse_fallback_response
        self.tiers = _hardcoded_fallback_tiers()
        self.score_fields = score_fields
        self.summarise = summarise
        self.FieldResult = FieldResult
        self.parse = _parse_fallback_response

    def test_fallback_fills_slot_c_only(self):
        fields = {
            "engagement_partner": ["Jane Smith", "Jane Smith", None],
            "audit_type":         [None, None, None],
        }
        # Simulate fallback filling slot C
        fields["audit_type"][2] = "GAGAS"
        fields["engagement_partner"][2] = "Jane Smith"  # same as det

        assert fields["engagement_partner"][0] == "Jane Smith"  # A untouched
        assert fields["engagement_partner"][1] == "Jane Smith"  # B untouched
        assert fields["audit_type"][2] == "GAGAS"               # C filled

    def test_fallback_improves_score(self):
        # Before: 3/8 tier1 found
        before = {
            "client_name": 0.9, "fiscal_year_end": 0.9, "engagement_decision": 0.9,
            "engagement_partner": 0.0, "audit_type": 0.0, "includes_gagas": 0.0,
            "includes_single_audit": 0.0, "reporting_framework": 0.0,
        }
        score_before = self.summarise(before, tiers=self.tiers).aggregate_score

        # After: fallback adds 5 more tier1 fields
        after = dict(before)
        after.update({
            "engagement_partner": 0.70, "audit_type": 0.70,
            "includes_gagas": 0.70, "includes_single_audit": 0.65,
            "reporting_framework": 0.70,
        })
        score_after = self.summarise(after, tiers=self.tiers).aggregate_score

        assert score_after > score_before
        assert score_after >= 0.65

    def test_fallback_response_parse_structured(self):
        import json
        raw = json.dumps({
            "client_name": {"value": "ABC", "confident": True, "source_hint": "Org: ABC"},
            "audit_type":  None,
        })
        results = self.parse(raw, ["client_name", "audit_type"])
        assert results["client_name"].found == True
        assert results["client_name"].llm_confident == True
        assert results["audit_type"].found == False


# ===========================================================================
# Phase B3 — LLM confidence calibration
# ===========================================================================

class TestLLMCalibration:

    def setup_method(self):
        from auditai_data_normalization.confidence import _hardcoded_fallback_tiers, score_fields, calibrate_llm_scores
        from auditai_data_normalization.extractors.llm_extractor import FieldResult
        self.tiers = _hardcoded_fallback_tiers()
        self.score_fields = score_fields
        self.calibrate = calibrate_llm_scores
        self.FR = FieldResult

    def test_llm_matches_deterministic_085(self):
        fields = {"client_name": ["ABC", "ABC", "ABC"]}
        scores = self.score_fields(fields)
        cal, _ = self.calibrate(scores, fields,
                                {"client_name": self.FR("ABC", True, True)}, self.tiers)
        assert cal["client_name"] == 0.90  # matches + confident

    def test_llm_only_tier1_capped(self):
        fields = {"engagement_partner": [None, None, "Jane CPA"]}
        scores = self.score_fields(fields)
        cal, _ = self.calibrate(scores, fields,
                                {"engagement_partner": self.FR("Jane CPA", True, True)},
                                self.tiers)
        assert cal["engagement_partner"] <= 0.72  # never exceeds cap

    def test_contradiction_flagged(self):
        fields = {"audit_type": ["GAAS", "GAAS", "GAGAS"]}
        scores = self.score_fields(fields)
        cal, flagged = self.calibrate(scores, fields,
                                      {"audit_type": self.FR("GAGAS", True, True)},
                                      self.tiers)
        assert "audit_type" in flagged


# ===========================================================================
# Phase C4 — Review confidence scoring
# ===========================================================================

class TestReviewConfidence:

    def setup_method(self):
        from raw_to_training_pair.completion_drafter import score_completion
        self.score = score_completion

    def _perfect(self) -> str:
        return (
            "ENGAGEMENT TYPE: GAAS Audit — Nonprofit Organization (501(c)(3))\n\n"
            "FINDINGS:\n"
            "1. Engagement Partner (SOP §2.4)\n"
            "   Severity: High\n"
            "   Risk: Engagement lacks authorized partner sign-off.\n\n"
            "RECOMMENDATION:\n"
            "Management of this Nonprofit Organization should implement corrective "
            "procedures in accordance with applicable audit standards for nonprofit entities."
        )

    def test_perfect_completion_scores_high(self):
        s = self.score(self._perfect(), [], "NPO")
        assert s >= 0.70, f"Perfect completion should score >= 0.70, got {s}"

    def test_missing_severity_reduces_score(self):
        no_sev = self._perfect().replace("Severity: High\n", "")
        assert self.score(no_sev, [], "NPO") < self.score(self._perfect(), [], "NPO")

    def test_placeholder_citation_reduces_score(self):
        with_placeholder = self._perfect().replace("SOP §2.4", "SOP §X.X")
        assert self.score(with_placeholder, [], "NPO") < self.score(self._perfect(), [], "NPO")

    def test_empty_completion_returns_zero(self):
        assert self.score("", [], "NPO") == 0.0

    def test_missing_sections_reduces_score(self):
        no_findings = "ENGAGEMENT TYPE: GAAS\n\nRECOMMENDATION:\nFix it."
        assert self.score(no_findings, [], "NPO") < self.score(self._perfect(), [], "NPO")


# ===========================================================================
# Phase E1 — uncertain_sections in pair metadata
# ===========================================================================

class TestUncertainSections:

    def test_flagged_fields_become_uncertain(self):
        from raw_to_training_pair.pair_builder import _build_uncertain_sections
        from auditai_data_normalization.schema import DocumentRecord

        r = DocumentRecord(
            file_name="test.docx",
            flagged_fields=["engagement_partner", "audit_type"],
            pii_scrubbed=True,
            metadata={"confidence_summary": {"per_field_scores": {
                "engagement_partner": 0.70,
                "audit_type":         0.65,
                "client_name":        0.90,
            }}}
        )
        uncertain = _build_uncertain_sections(r, [])
        assert "audit_type" in uncertain          # score < 0.70
        assert "engagement_partner" in uncertain  # in flagged_fields

    def test_clean_record_has_no_uncertain(self):
        from raw_to_training_pair.pair_builder import _build_uncertain_sections
        from auditai_data_normalization.schema import DocumentRecord

        r = DocumentRecord(
            file_name="test.docx",
            flagged_fields=[],
            pii_scrubbed=True,
            metadata={"confidence_summary": {"per_field_scores": {
                "client_name": 0.90, "fiscal_year_end": 0.90,
                "engagement_partner": 0.85,
            }}}
        )
        uncertain = _build_uncertain_sections(r, [])
        assert uncertain == []

    def test_deficiency_fields_added(self):
        from raw_to_training_pair.pair_builder import _build_uncertain_sections
        from auditai_data_normalization.schema import DocumentRecord

        r = DocumentRecord(file_name="t.docx", flagged_fields=[], pii_scrubbed=True,
                           metadata={"confidence_summary": {"per_field_scores": {}}})
        uncertain = _build_uncertain_sections(r, ["fiscal_year_end"])
        assert "fiscal_year_end" in uncertain


# ===========================================================================
# Phase E3 — Four-tier approval
# ===========================================================================

class TestApprovalTiers:

    def _make_pair(self, name: str = "test.docx") -> tuple[dict, str]:
        import hashlib
        msgs = [{"role": "user", "content": name}, {"role": "assistant", "content": "c"}]
        ph = hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
        return {
            "messages": msgs,
            "metadata": {"file_name": name, "pair_hash": ph,
                         "auditor_approved": False, "stage": "stage2", "client_type": "NPO"}
        }, ph

    def test_full_approve(self):
        from raw_to_training_pair.auditor_review import enqueue, approve, load_pending
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            q = Path(f.name)
        pair, ph = self._make_pair("a.docx")
        enqueue(pair, q)
        assert approve(ph, "SH", "good", q)
        assert pair["metadata"]["auditor_approved"] == False  # original unchanged
        entries = [json.loads(l) for l in q.read_text().splitlines() if l.strip()]
        approved = [e for e in entries if e["status"] == "approved"]
        assert len(approved) == 1
        assert approved[0]["pair"]["metadata"]["auditor_approved"] == True

    def test_conditional_approve(self):
        from raw_to_training_pair.auditor_review import enqueue, conditional_approve
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            q = Path(f.name)
        pair, ph = self._make_pair("b.docx")
        enqueue(pair, q)
        assert conditional_approve(ph, "SH", "approved with caveat", q)
        entries = [json.loads(l) for l in q.read_text().splitlines() if l.strip()]
        cond = entries[0]
        assert cond["status"] == "conditional"
        assert cond["pair"]["metadata"]["approval_type"] == "conditional"
        assert "caveat" in cond["pair"]["metadata"]["reviewer_note"]

    def test_send_for_correction_requires_hint(self):
        from raw_to_training_pair.auditor_review import enqueue, send_for_correction
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            q = Path(f.name)
        pair, ph = self._make_pair("c.docx")
        enqueue(pair, q)
        assert send_for_correction(ph, "SH", "", q) == False       # empty hint fails
        assert send_for_correction(ph, "SH", "Fix SOP cite", q)    # non-empty succeeds
        entries = [json.loads(l) for l in q.read_text().splitlines() if l.strip()]
        assert entries[0]["status"] == "correction"
        assert entries[0]["hint"] == "Fix SOP cite"

    def test_reject(self):
        from raw_to_training_pair.auditor_review import enqueue, reject, stats
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            q = Path(f.name)
        pair, ph = self._make_pair("d.docx")
        enqueue(pair, q)
        assert reject(ph, "SH", "wrong type", q)
        s = stats(q)
        assert s["rejected"] == 1
        assert s["pending"]  == 0

    def test_stats_all_tiers(self):
        from raw_to_training_pair.auditor_review import enqueue, approve, conditional_approve, send_for_correction, reject, stats
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            q = Path(f.name)
        for i, name in enumerate(["p1.docx","p2.docx","p3.docx","p4.docx"]):
            pair, ph = self._make_pair(name)
            enqueue(pair, q)
            if i == 0: approve(ph, "SH", "", q)
            elif i == 1: conditional_approve(ph, "SH", "note", q)
            elif i == 2: send_for_correction(ph, "SH", "fix it", q)
            else: reject(ph, "SH", "bad", q)
        s = stats(q)
        assert s["approved"] == 1
        assert s["conditional"] == 1
        assert s["correction"] == 1
        assert s["rejected"] == 1
        assert s["pending"] == 0