"""
tests/test_doc_classifier.py
=============================
Unit tests for auditai_data_normalization/doc_classifier.py (Phase 2).

Run with:
    pytest tests/test_doc_classifier.py -v

All tests are deterministic — no LLM, no file I/O.
Section objects are stubbed inline.
"""

import pytest
from dataclasses import dataclass
from auditai_data_normalization.doc_classifier import detect_category


# ---------------------------------------------------------------------------
# Minimal Section stub — avoids importing DocumentRecord
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    heading: str = ""
    content: str = ""


# ---------------------------------------------------------------------------
# Filename signal tests — real filenames from the data folder
# ---------------------------------------------------------------------------

class TestFilenameSignals:

    # engagement_form
    def test_npo_cx_form(self):
        assert detect_category(
            "NPO-CX-1_1 Engagement Accept and Cont Form.docx"
        ) == "engagement_form"

    def test_npo_cx_prepared(self):
        assert detect_category(
            "NPO-CX-1_1 Prepared  Engagement Accept and Cont Form.docx"
        ) == "engagement_form"

    def test_gov_cx_variant(self):
        assert detect_category("GOV-CX-2_1 Engagement Form.docx") == "engagement_form"

    def test_engagement_accept_in_name(self):
        assert detect_category("2024_Engagement_Acceptance_Form.docx") == "engagement_form"

    def test_continuance_in_name(self):
        assert detect_category("Client_Continuance_2024.docx") == "engagement_form"

    # financial_statement
    def test_final_audit_report_rwanda(self):
        assert detect_category(
            "Final Audit Report The Rwanda School Project FY 2024.pdf"
        ) == "financial_statement"

    def test_final_audit_report_ocgi(self):
        assert detect_category(
            "Final Audit Report_OCGI_06.30.2023 Old 1.pdf"
        ) == "financial_statement"

    def test_final_audit_report_imm(self):
        assert detect_category(
            "FinalAuditReport-IMM06-30-2024.pdf"
        ) == "financial_statement"

    def test_draft_financial_statements(self):
        assert detect_category(
            "DraftFinancialStatements_2024_IITKF[1].pdf"
        ) == "financial_statement"

    def test_ecestep_financial_statements(self):
        assert detect_category(
            "ECEStep_Final_Financial_Statements_2024[111.pdf"
        ) == "financial_statement"

    def test_heffernan_final_report(self):
        # "Final Report" alone — should match financial_statement via "final audit report"
        # This filename has "Final Report" but not "Audit" — expect unknown or financial
        # depending on content. Without content, filename alone is ambiguous.
        result = detect_category("Heffernan_Foundation_12-31-2024_Final Report. 2.pdf")
        # Acceptable: unknown (no "audit" in name) or financial_statement
        assert result in ("unknown", "financial_statement")

    # planning_memo
    def test_planning_memo_filename(self):
        assert detect_category("Audit_Planning_Memo_2024.docx") == "planning_memo"

    def test_risk_assessment_filename(self):
        assert detect_category("Risk_Assessment_Matrix_Q3.docx") == "planning_memo"

    # SOP — should be unknown (not a workpaper)
    def test_sop_document_is_unknown(self):
        assert detect_category("SOP_NPO_CX_1_1_Final_v10 (1).docx") == "unknown"

    def test_generic_filename_is_unknown(self):
        assert detect_category("Client_Workpaper_2024.docx") == "unknown"

    def test_empty_filename(self):
        assert detect_category("") == "unknown"


# ---------------------------------------------------------------------------
# Content signal tests — filename is ambiguous, content decides
# ---------------------------------------------------------------------------

class TestContentSignals:

    def test_engagement_form_by_content(self):
        """Ambiguous filename — engagement form detected from section headings."""
        sections = [
            _Section(heading="Engagement Acceptance and Continuance"),
            _Section(heading="Independence Assessment"),
            _Section(heading="Engagement Partner Sign-Off"),
        ]
        cleaned = (
            "This engagement form documents our decision to accept or continue "
            "the engagement. Independence has been evaluated. GAGAS applies. "
            "Single Audit requirements reviewed."
        )
        result = detect_category("workpaper_2024.docx", sections=sections, cleaned_text=cleaned)
        assert result == "engagement_form"

    def test_financial_statement_by_content(self):
        """PDF with generic name — detected as financial statement from body."""
        sections = [
            _Section(heading="Independent Auditor's Report"),
            _Section(heading="Statement of Financial Position"),
        ]
        cleaned = (
            "Independent auditor's report on financial statements. "
            "Opinion on the financial statements. "
            "Total assets: $1,234,567. Net assets: $890,000. "
            "Management is responsible for the preparation of these financial statements."
        )
        result = detect_category("report_2024.pdf", sections=sections, cleaned_text=cleaned)
        assert result == "financial_statement"

    def test_planning_memo_by_content(self):
        sections = [
            _Section(heading="Audit Plan — FY 2024"),
            _Section(heading="Risk Assessment"),
        ]
        cleaned = (
            "Audit plan for fiscal year 2024. Risk assessment completed. "
            "Materiality set at $50,000. Significant risk identified in revenue. "
            "Inherent risk: medium. Control risk: low."
        )
        result = detect_category("memo_draft.docx", sections=sections, cleaned_text=cleaned)
        assert result == "planning_memo"

    def test_content_signal_insufficient_optional_hits(self):
        """Required keyword present but not enough optional hits → unknown."""
        sections = [_Section(heading="Engagement Letter")]
        cleaned = "This engagement letter confirms our engagement."
        # "engagement" is present but only 1 optional hit ("engagement" itself)
        # min_optional is 3 — should not fire
        result = detect_category("engagement_letter.docx", sections=sections, cleaned_text=cleaned)
        # Filename "engagement_letter.docx" doesn't match filename patterns either
        assert result == "unknown"

    def test_empty_content_falls_back_to_unknown(self):
        result = detect_category("mystery.docx", sections=[], cleaned_text="")
        assert result == "unknown"

    def test_none_sections_handled(self):
        """Passing None for sections should not raise."""
        result = detect_category("mystery.docx", sections=None, cleaned_text="")
        assert result == "unknown"


# ---------------------------------------------------------------------------
# Priority: filename wins over content
# ---------------------------------------------------------------------------

class TestFilenameBeatsContent:

    def test_filename_engagement_form_even_with_financial_content(self):
        """
        Edge case: filename clearly says engagement form,
        content looks like financial statements.
        Filename should win.
        """
        sections = [_Section(heading="Independent Auditor's Report")]
        cleaned = (
            "financial statement opinion total assets net assets "
            "management is responsible balance sheet"
        )
        result = detect_category(
            "NPO-CX-1_1 Engagement Accept Form.docx",
            sections=sections,
            cleaned_text=cleaned,
        )
        assert result == "engagement_form"