"""
tests/test_structural_extractor.py
====================================
Unit tests for Phase 4 structural_extractor.py.

Tests are self-contained — no PDFs, no file I/O.
Page text is constructed inline to mirror real audit report patterns
validated across 6 PDFs.

Run with:
    pytest tests/test_structural_extractor.py -v
"""

import pytest
from auditai_data_normalization.extractors.structural_extractor import (
    extract,
    FieldEvidence,
    METHOD_SALUTATION_BLOCK,
    METHOD_STANDALONE_DATE,
    METHOD_SIGNOFF_DATE,
    METHOD_PROSE_FLAG,
    _find_segments,
    _infer_fiscal_year_end,
    _MIN_CONFIDENCE,
)


# ---------------------------------------------------------------------------
# Helpers — realistic page text builders
# ---------------------------------------------------------------------------

def _cover_page(entity: str, date: str) -> str:
    return f"{entity}\nAUDITED FINANCIAL STATEMENTS\n{date}\n(With summarized comparative totals)\n"


def _toc_page() -> str:
    return (
        "TABLE OF CONTENTS\n"
        "Independent Auditor's Report.....................................................................1\n"
        "Financial Statements\n"
        ".....Statements of Financial Position................................................................3\n"
        "Independent Auditor's Report on Internal Control\n"
        ".....Performed in Accordance with Government Auditing Standards......................................22\n"
    )


def _auditor_report_page(entity: str, city: str) -> str:
    return (
        "INDEPENDENT AUDITOR'S REPORT\n"
        "To the Board of Directors\n"
        f"{entity}\n"
        f"{city}\n"
        "Report on the Audit of the Financial Statements\n"
        "Opinion\n"
        "We have audited the accompanying financial statements of "
        f"{entity}, a nonprofit organization, which comprise the "
        "statement of financial position.\n"
        "In our opinion, the financial statements present fairly, in all "
        "material respects, in accordance with accounting principles "
        "generally accepted in the United States of America.\n"
        "We conducted our audit in accordance with auditing standards "
        "generally accepted in the United States of America.\n"
    )


def _auditor_report_close(sign_date: str) -> str:
    return (
        "We are required to communicate with those charged with governance.\n"
        "Report on Summarized Comparative Information\n"
        "We previously expressed an unmodified opinion on the prior year.\n"
        f"{sign_date.split(',')[0].split()[-1]}, California\n"
        f"{sign_date}\n"
        "2\n"
    )


def _financial_stmt_page(entity: str, date: str) -> str:
    return (
        f"{entity.upper()}\n"
        "STATEMENTS OF FINANCIAL POSITION\n"
        f"{date.upper()}\n"
        "ASSETS\n"
        "Cash and investments $ 505,900\n"
        "Total assets 1,158,399\n"
        "LIABILITIES AND NET ASSETS\n"
        "Total liabilities 432,295\n"
        "Total net assets 726,104\n"
    )


def _gagas_prose_page() -> str:
    return (
        "We conducted our audit in accordance with auditing standards "
        "generally accepted in the United States of America and the standards "
        "applicable to financial audits contained in Government Auditing "
        "Standards, issued by the Comptroller General of the United States. "
        "Those standards require that we plan and perform the audit to obtain "
        "reasonable assurance about whether the financial statements are free "
        "from material misstatement. The Yellow Book requires independence.\n"
    )


def _single_audit_prose_page() -> str:
    return (
        "We conducted our audit of compliance in accordance with auditing "
        "standards generally accepted in the United States of America; the "
        "standards applicable to financial audits contained in Government "
        "Auditing Standards; and the audit requirements of Title 2 U.S. Code "
        "of Federal Regulations Part 200, Uniform Administrative Requirements, "
        "Cost Principles, and Audit Requirements for Federal Awards "
        "(Uniform Guidance). Single Audit requirements apply.\n"
    )


# ---------------------------------------------------------------------------
# Layer 1 — Segment detection
# ---------------------------------------------------------------------------

class TestFindSegments:

    def test_finds_auditor_report(self):
        pages = [
            _cover_page("Acme NPO", "JUNE 30, 2024"),
            _auditor_report_page("Acme NPO", "Oakland, California"),
            _auditor_report_close("December 13, 2024"),
        ]
        segs = _find_segments(pages)
        names = [s.name for s in segs]
        assert "auditor_report" in names

    def test_toc_page_not_detected_as_report(self):
        pages = [
            _toc_page(),
            _auditor_report_page("Acme NPO", "Oakland, California"),
        ]
        segs = _find_segments(pages)
        # Only one auditor_report segment — the real one, not the TOC
        report_segs = [s for s in segs if s.name == "auditor_report"]
        assert len(report_segs) == 1
        assert report_segs[0].start_page == 2

    def test_supplement_block_filtered(self):
        """GAGAS supplement block with all-caps entity line should be filtered."""
        supplement_page = (
            "INDEPENDENT AUDITOR'S REPORT\n"
            "To the Board of Directors\n"
            "REPORTING AND ON COMPLIANCE AND OTHER MATTERS BASED ON AN\n"
            "AUDIT OF FINANCIAL STATEMENTS PERFORMED IN ACCORDANCE WITH\n"
            "GOVERNMENT AUDITING STANDARDS\n"
        )
        pages = [
            supplement_page,
            _auditor_report_page("Acme NPO", "Oakland, California"),
        ]
        segs = _find_segments(pages)
        report_segs = [s for s in segs if s.name == "auditor_report"]
        assert len(report_segs) == 1
        assert report_segs[0].start_page == 2

    def test_no_city_line_not_detected(self):
        """Block without city line should not be detected as primary report."""
        no_city_page = (
            "INDEPENDENT AUDITOR'S REPORT\n"
            "To the Board of Directors\n"
            "Acme NPO\n"
            "Report on the Audit of the Financial Statements\n"
        )
        pages = [no_city_page]
        segs = _find_segments(pages)
        report_segs = [s for s in segs if s.name == "auditor_report"]
        assert len(report_segs) == 0

    def test_financial_statements_segment_detected(self):
        pages = [
            _cover_page("Acme NPO", "JUNE 30, 2024"),
            _auditor_report_page("Acme NPO", "Oakland, California"),
            "FINANCIAL STATEMENTS\n",
        ]
        segs = _find_segments(pages)
        names = [s.name for s in segs]
        assert "financial_statements" in names

    def test_salutation_variants(self):
        """'The Board of Directors' (no 'To') should also fire."""
        page = (
            "INDEPENDENT AUDITOR'S REPORT\n"
            "The Board of Directors\n"
            "IIT Kanpur Foundation\n"
            "Palo Alto, California\n"
            "Report on the Audit of the Financial Statements\n"
        )
        segs = _find_segments([page])
        assert any(s.name == "auditor_report" for s in segs)

    def test_empty_pages_returns_empty(self):
        assert _find_segments([]) == []
        assert _find_segments([""]) == []


# ---------------------------------------------------------------------------
# Layer 2 — Fiscal year end inference
# ---------------------------------------------------------------------------

class TestInferFiscalYearEnd:

    def test_allcaps_date_highest_confidence(self):
        pages = ["JUNE 30, 2024\n"]
        ev = _infer_fiscal_year_end(pages)
        assert ev is not None
        assert ev.confidence >= 0.90
        assert "2024" in ev.value

    def test_mixed_case_date_medium_confidence(self):
        pages = ["Early Childhood Education STEP\nJune 30, 2024\n"]
        ev = _infer_fiscal_year_end(pages)
        assert ev is not None
        assert ev.confidence >= 0.55

    def test_prose_embedded_date_lower_confidence(self):
        pages = ["We audited the financial statements for the year ended June 30, 2024.\n"]
        ev = _infer_fiscal_year_end(pages)
        # Should be found but with lower confidence
        if ev:
            assert ev.confidence < 0.80

    def test_no_date_returns_none(self):
        pages = ["No dates here at all.\n"]
        assert _infer_fiscal_year_end(pages) is None

    def test_only_first_3_pages_scanned(self):
        """Fiscal year from page 10 should not be returned."""
        pages = [""] * 9 + ["JUNE 30, 2024\n"]
        ev = _infer_fiscal_year_end(pages)
        assert ev is None

    def test_december_31_fiscal_year(self):
        pages = ["DECEMBER 31, 2024\n"]
        ev = _infer_fiscal_year_end(pages)
        assert ev is not None
        assert "2024" in ev.value


# ---------------------------------------------------------------------------
# Full extract() — end-to-end tests
# ---------------------------------------------------------------------------

class TestExtract:

    def _standard_pages(
        self,
        entity="Acme Nonprofit",
        city="Oakland, California",
        fye_date="JUNE 30, 2024",
        sign_date="December 13, 2024",
    ):
        return [
            _cover_page(entity, fye_date),
            _toc_page(),
            _auditor_report_page(entity, city),
            _auditor_report_close(sign_date),
            "FINANCIAL STATEMENTS\n",
            _financial_stmt_page(entity, fye_date),
        ]

    def test_client_name_extracted(self):
        pages = self._standard_pages()
        results = extract(pages)
        assert "client_name" in results
        assert results["client_name"].value == "Acme Nonprofit"
        assert results["client_name"].method == METHOD_SALUTATION_BLOCK

    def test_client_address_extracted(self):
        pages = self._standard_pages()
        results = extract(pages)
        assert "client_address" in results
        assert results["client_address"].value == "Oakland, California"

    def test_fiscal_year_end_extracted(self):
        pages = self._standard_pages()
        results = extract(pages)
        assert "fiscal_year_end" in results
        assert "2024" in results["fiscal_year_end"].value
        assert results["fiscal_year_end"].method == METHOD_STANDALONE_DATE

    def test_partner_sign_date_extracted(self):
        pages = self._standard_pages(sign_date="December 13, 2024")
        results = extract(pages)
        assert "partner_sign_date" in results
        assert "2024" in results["partner_sign_date"].value
        assert results["partner_sign_date"].method == METHOD_SIGNOFF_DATE

    def test_gaas_prose_flag_detected(self):
        pages = self._standard_pages()
        # auditor_report_page includes GAAS prose
        results = extract(pages)
        assert "includes_gaas_audit" in results
        assert results["includes_gaas_audit"].value == "true"
        assert results["includes_gaas_audit"].method == METHOD_PROSE_FLAG

    def test_reporting_framework_gaap(self):
        pages = self._standard_pages()
        results = extract(pages)
        assert "reporting_framework" in results
        assert results["reporting_framework"].value == "GAAP"

    def test_gagas_prose_flag(self):
        pages = self._standard_pages()
        pages[2] += _gagas_prose_page()
        results = extract(pages)
        assert "includes_gagas" in results
        assert results["includes_gagas"].value == "true"
        assert results["includes_gagas"].confidence >= 0.70

    def test_single_audit_prose_flag(self):
        pages = self._standard_pages()
        pages[2] += _single_audit_prose_page()
        results = extract(pages)
        assert "includes_single_audit" in results
        assert results["includes_single_audit"].value == "true"

    def test_all_results_above_min_confidence(self):
        """Evidence gating: no result below _MIN_CONFIDENCE threshold."""
        pages = self._standard_pages()
        results = extract(pages)
        for field, ev in results.items():
            assert ev.confidence >= _MIN_CONFIDENCE, (
                f"{field} has confidence {ev.confidence} below minimum {_MIN_CONFIDENCE}"
            )

    def test_empty_pages_returns_empty(self):
        assert extract([]) == {}
        assert extract([""]) == {}

    def test_engagement_partner_not_extracted(self):
        """engagement_partner must never be emitted — it's registry-only."""
        pages = self._standard_pages()
        results = extract(pages)
        assert "engagement_partner" not in results

    def test_toc_only_no_false_positives(self):
        """TOC page alone should not extract any fields."""
        results = extract([_toc_page()])
        # No auditor_report segment found → no salutation block fields
        assert "client_name" not in results

    def test_all_field_evidences_have_source_page(self):
        pages = self._standard_pages()
        results = extract(pages)
        for field, ev in results.items():
            assert ev.source_page >= 1, f"{field} has invalid source_page {ev.source_page}"

    def test_real_pattern_rwanda(self):
        """Mirror the Rwanda School Project PDF structure."""
        pages = [
            _cover_page("The Rwanda School Project", "DECEMBER 31, 2024"),
            _toc_page(),
            _auditor_report_page("The Rwanda School Project", "Santa Rosa, California"),
            _auditor_report_close("May 23, 2025"),
            "FINANCIAL STATEMENTS\n",
        ]
        results = extract(pages)
        assert results["client_name"].value == "The Rwanda School Project"
        assert "2024" in results["fiscal_year_end"].value
        assert "2025" in results["partner_sign_date"].value

    def test_real_pattern_iitkf(self):
        """Mirror the IITKF Draft Financial Statements structure."""
        pages = [
            _cover_page("IIT Kanpur Foundation", "DECEMBER 31, 2024"),
            _toc_page(),
            (
                "INDEPENDENT AUDITOR'S REPORT\n"
                "The Board of Directors\n"   # 'The' not 'To the'
                "IIT Kanpur Foundation\n"
                "Palo Alto, California\n"
                "Report on the Audit of the Financial Statements\n"
                "Opinion\n"
                "In our opinion, the financial statements present fairly in "
                "accordance with accounting principles generally accepted in "
                "the United States of America.\n"
            ),
            _auditor_report_close("August 19, 2025"),
        ]
        results = extract(pages)
        assert results["client_name"].value == "IIT Kanpur Foundation"
        assert "2025" in results["partner_sign_date"].value