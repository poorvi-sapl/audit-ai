"""
tests/test_phase3_structural.py
================================
Unit tests for Phase 3 structural parsing additions:
  A. Vertical continuation  (_extract_table in docx_extractor)
  B. Checkbox column tracking (_resolve_checkbox_columns)
  C. Inverted row detection  (_extract_fields_from_record in normalize)

Run with:
    pytest tests/test_phase3_structural.py -v

All tests are deterministic — no LLM, no file I/O, no fixtures.
Where DocumentRecord or ExtractedTable are needed they are constructed
directly from the dataclass.
"""

import pytest
from auditai_data_normalization.extractors.docx_extractor import (
    _looks_like_value,
    _resolve_checkbox_columns,
)


# ---------------------------------------------------------------------------
# A. _looks_like_value helper
# ---------------------------------------------------------------------------

class TestLooksLikeValue:

    def test_plain_name_is_value(self):
        assert _looks_like_value("John Smith, CPA") is True

    def test_date_string_is_value(self):
        assert _looks_like_value("June 30, 2025") is True

    def test_colon_suffix_is_not_value(self):
        """A label ending with colon should never be promoted as value."""
        assert _looks_like_value("Engagement Partner:") is False

    def test_empty_string_is_not_value(self):
        assert _looks_like_value("") is False

    def test_whitespace_only_is_not_value(self):
        assert _looks_like_value("   ") is False

    def test_known_header_yes_is_not_value(self):
        assert _looks_like_value("yes") is False

    def test_known_header_no_is_not_value(self):
        assert _looks_like_value("no") is False

    def test_known_header_na_is_not_value(self):
        assert _looks_like_value("n/a") is False

    def test_checkbox_sentinel_true_is_not_value(self):
        """Checkbox sentinels are handled by Phase 3B, not continuation."""
        assert _looks_like_value("true") is False

    def test_checkbox_sentinel_false_is_not_value(self):
        assert _looks_like_value("false") is False

    def test_instructions_is_not_value(self):
        assert _looks_like_value("instructions") is False

    def test_dollar_amount_is_value(self):
        assert _looks_like_value("$1,234,567") is True

    def test_yes_no_mixed_text_is_value(self):
        """'Yes, with caveats' is a data value, not a header."""
        assert _looks_like_value("Yes, with caveats") is True


# ---------------------------------------------------------------------------
# B. _resolve_checkbox_columns
# ---------------------------------------------------------------------------

class TestResolveCheckboxColumns:

    def test_single_audit_yes_checked(self):
        headers   = ["Audit Scope Item", "Yes", "No"]
        data_rows = [
            ["Single Audit", "true",  "false"],
            ["GAGAS",        "false", "true" ],
        ]
        pairs = _resolve_checkbox_columns(headers, data_rows)
        pair_dict = dict(pairs)
        assert pair_dict.get("Single Audit") == "true"
        assert pair_dict.get("GAGAS") == "false"

    def test_no_column_checked_inverts(self):
        """When the 'No' column is checked, value should be false."""
        headers   = ["Item", "Yes", "No"]
        data_rows = [["Non-attest Services", "false", "true"]]
        pairs = _resolve_checkbox_columns(headers, data_rows)
        pair_dict = dict(pairs)
        assert pair_dict.get("Non-attest Services") == "false"

    def test_both_unchecked_no_pair(self):
        """If neither Yes nor No is checked, no pair emitted for that row."""
        headers   = ["Item", "Yes", "No"]
        data_rows = [["Mystery Item", "false", "false"]]
        pairs = _resolve_checkbox_columns(headers, data_rows)
        assert len(pairs) == 0

    def test_no_checkbox_columns_returns_empty(self):
        """Table with no Yes/No headers — nothing to resolve."""
        headers   = ["Label", "Value"]
        data_rows = [["Engagement Partner", "John Smith"]]
        pairs = _resolve_checkbox_columns(headers, data_rows)
        assert pairs == []

    def test_empty_headers(self):
        assert _resolve_checkbox_columns([], []) == []

    def test_empty_data_rows(self):
        headers = ["Item", "Yes", "No"]
        assert _resolve_checkbox_columns(headers, []) == []

    def test_known_header_label_skipped(self):
        """Rows whose label is a known header word are skipped."""
        headers   = ["Item", "Yes", "No"]
        data_rows = [
            ["yes", "true", "false"],    # "yes" is a known header — skip
            ["GAGAS", "true", "false"],  # genuine audit scope item
        ]
        pairs = _resolve_checkbox_columns(headers, data_rows)
        pair_dict = dict(pairs)
        assert "yes" not in pair_dict
        assert "GAGAS" in pair_dict

    def test_multiple_yes_cols_first_match_wins(self):
        """Edge case: two Yes columns — first checked one wins."""
        headers   = ["Item", "Yes", "Also Yes"]
        data_rows = [["Single Audit", "true", "false"]]
        pairs = _resolve_checkbox_columns(headers, data_rows)
        assert dict(pairs).get("Single Audit") == "true"

    def test_row_shorter_than_header(self):
        """Gracefully handle rows with fewer cells than headers."""
        headers   = ["Item", "Yes", "No"]
        data_rows = [["GAGAS"]]   # only 1 cell, no Yes/No cells
        # Should not raise — should return empty (no checkbox values found)
        pairs = _resolve_checkbox_columns(headers, data_rows)
        assert isinstance(pairs, list)

    def test_full_engagement_form_table(self):
        """
        Realistic PPC engagement form scope table.
        All three Tier 1 boolean fields resolved in one pass.
        """
        headers = [
            "Audit Services to be Provided",
            "Yes",
            "No",
        ]
        data_rows = [
            ["Audit of Financial Statements (GAAS)",     "true",  "false"],
            ["Government Auditing Standards (GAGAS)",    "true",  "false"],
            ["Single Audit (2 CFR 200)",                 "false", "true" ],
            ["Grant Compliance Audit",                   "true",  "false"],
            ["Non-Attest Services",                      "false", "true" ],
        ]
        pairs = _resolve_checkbox_columns(headers, data_rows)
        pair_dict = dict(pairs)

        assert pair_dict["Audit of Financial Statements (GAAS)"] == "true"
        assert pair_dict["Government Auditing Standards (GAGAS)"] == "true"
        assert pair_dict["Single Audit (2 CFR 200)"] == "false"
        assert pair_dict["Grant Compliance Audit"] == "true"
        assert pair_dict["Non-Attest Services"] == "false"


# ---------------------------------------------------------------------------
# C. Inverted row detection — tested via _extract_fields_from_record
#    using a minimal DocumentRecord constructed directly
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as dc_field
from auditai_data_normalization.schema import DocumentRecord, ExtractedTable, Section
from auditai_data_normalization.normalize import _extract_fields_from_record


class TestInvertedRowDetection:

    def _make_record(
        self,
        table_rows: list[list[str]],
        headers: list[str] | None = None,
    ) -> DocumentRecord:
        """Helper — build a minimal DocumentRecord with one table."""
        tbl = ExtractedTable(
            index=0,
            source="Table 1",
            headers=headers or [],
            rows=table_rows,
            raw_text="",
        )
        return DocumentRecord(
            source_path="test.docx",
            file_name="test.docx",
            file_type="docx",
            tables=[tbl],
            sections=[],
        )

    def test_inverted_engagement_partner(self):
        """
        Value row appears BEFORE label row — should still extract correctly.
        Row 0: "John Smith, CPA"   ← value
        Row 1: "Engagement Partner" ← label (known alias)
        """
        record = self._make_record(
            table_rows=[
                ["John Smith, CPA"],
                ["Engagement Partner"],
            ]
        )
        result = _extract_fields_from_record(record)
        assert "engagement_partner" in result
        assert result["engagement_partner"][0] == "John Smith, CPA"

    def test_inverted_client_name(self):
        record = self._make_record(
            table_rows=[
                ["Acme Nonprofit Organization"],
                ["Client Name"],
            ]
        )
        result = _extract_fields_from_record(record)
        assert "client_name" in result
        assert "Acme Nonprofit" in result["client_name"][0]

    def test_forward_pass_not_overwritten(self):
        """
        If the forward pass already found engagement_partner,
        the inverted pass must NOT overwrite it.
        """
        record = self._make_record(
            table_rows=[
                ["Engagement Partner: Jane Doe, CPA"],  # forward pass finds this
                ["Some Other Value"],
                ["Engagement Partner"],                  # inverted — should be ignored
            ]
        )
        result = _extract_fields_from_record(record)
        assert "engagement_partner" in result
        assert result["engagement_partner"][0] == "Jane Doe, CPA"

    def test_checkbox_sentinel_not_promoted_as_value(self):
        """
        'true' from Phase 1 checkbox normalization must not be
        promoted as the value for a label in the next row.
        """
        record = self._make_record(
            table_rows=[
                ["true"],
                ["Engagement Partner"],
            ]
        )
        result = _extract_fields_from_record(record)
        # engagement_partner should NOT be "true"
        if "engagement_partner" in result:
            assert result["engagement_partner"][0] != "true"

    def test_unknown_label_below_value_not_extracted(self):
        """
        Cell in label position that's NOT in canonical_labels
        should not trigger inverted extraction.
        """
        record = self._make_record(
            table_rows=[
                ["Some Random Value"],
                ["Not A Known Field Label"],
            ]
        )
        result = _extract_fields_from_record(record)
        assert "Not A Known Field Label".lower() not in {
            k.lower() for k in result
        }

    def test_forward_and_inverted_combined(self):
        """
        Table with mix of forward and inverted rows — both resolved.
        """
        record = self._make_record(
            table_rows=[
                ["Client Name: Riverside NPO"],    # forward
                ["June 30, 2025"],                  # inverted value
                ["Fiscal Year End"],                # inverted label
            ]
        )
        result = _extract_fields_from_record(record)
        assert "client_name" in result
        assert result["client_name"][0] == "Riverside NPO"
        assert "fiscal_year_end" in result
        assert result["fiscal_year_end"][0] == "June 30, 2025"