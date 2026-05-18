"""
tests/test_text_normalizer.py
==============================
Unit tests for auditai_data_normalization/text_normalizer.py (Phase 1).

Run with:
    pytest tests/test_text_normalizer.py -v

All tests are deterministic — no LLM, no file I/O, no fixtures.
"""

import pytest
from auditai_data_normalization.text_normalizer import (
    normalize_checkbox,
    normalize_underscores,
    normalize_unicode_whitespace,
    normalize_inline_markers,
    normalize_text,
)


# ---------------------------------------------------------------------------
# normalize_checkbox
# ---------------------------------------------------------------------------

class TestNormalizeCheckbox:

    def test_wingding_checked_box(self):
        """Most common Wingdings PPC form symbol."""
        assert normalize_checkbox("\uf0fe") == "true"

    def test_wingding_square(self):
        assert normalize_checkbox("\uf0a3") == "true"

    def test_unicode_check_mark(self):
        assert normalize_checkbox("\u2713") == "true"   # ✓

    def test_unicode_heavy_check(self):
        assert normalize_checkbox("\u2714") == "true"   # ✔

    def test_ballot_box_with_check(self):
        assert normalize_checkbox("\u2611") == "true"   # ☑

    def test_ballot_box_with_x(self):
        assert normalize_checkbox("\u2612") == "true"   # ☒

    def test_black_square(self):
        assert normalize_checkbox("\u25a0") == "true"   # ■

    def test_unchecked_ballot_box(self):
        assert normalize_checkbox("\u2610") == "false"  # ☐

    def test_white_square(self):
        assert normalize_checkbox("\u25a1") == "false"  # □

    def test_mixed_row(self):
        """Simulates a Yes/No column row from an engagement form table."""
        result = normalize_checkbox("\u2612 Yes  \u2610 No")
        assert result == "true Yes  false No"

    def test_accept_decline_row(self):
        """Engagement decision row from PPC NPO-CX form."""
        result = normalize_checkbox("☒ Accept  ☐ Decline")
        assert "true" in result
        assert "false" in result

    def test_no_symbols_unchanged(self):
        """Text without checkbox symbols passes through unchanged."""
        text = "John Smith, CPA"
        assert normalize_checkbox(text) == text

    def test_empty_string(self):
        assert normalize_checkbox("") == ""

    def test_none_like_empty(self):
        # Function signature takes str — empty string edge case
        assert normalize_checkbox("") == ""

    def test_preserves_surrounding_text(self):
        result = normalize_checkbox("Partner: \uf0fe John Smith")
        assert result == "Partner: true John Smith"


# ---------------------------------------------------------------------------
# normalize_underscores
# ---------------------------------------------------------------------------

class TestNormalizeUnderscores:

    def test_date_with_surrounding_underscores(self):
        """Classic OCR date artifact."""
        result = normalize_underscores("_08_/ _07_/ _2025__")
        # Should produce something parseable as a date — underscores removed
        assert "_" not in result or result.count("_") < 2
        assert "08" in result
        assert "2025" in result

    def test_pure_underscores_become_space(self):
        """Blank field rendered as underscores."""
        result = normalize_underscores("___")
        assert result.strip() == ""

    def test_no_underscores_unchanged(self):
        text = "June 30, 2025"
        assert normalize_underscores(text) == text

    def test_empty_string(self):
        assert normalize_underscores("") == ""

    def test_long_underscore_run(self):
        """Long blank fill line."""
        result = normalize_underscores("Name: ____________")
        assert result.count("_") < 3

    def test_preserves_short_single_underscore(self):
        """Single underscore in a field name — not an artifact."""
        # normalize_underscores only targets runs of 2+
        text = "fiscal_year"
        assert normalize_underscores(text) == text


# ---------------------------------------------------------------------------
# normalize_unicode_whitespace
# ---------------------------------------------------------------------------

class TestNormalizeUnicodeWhitespace:

    def test_en_space_becomes_ascii(self):
        assert normalize_unicode_whitespace("a\u2002b") == "a b"

    def test_em_space_becomes_ascii(self):
        assert normalize_unicode_whitespace("a\u2003b") == "a b"

    def test_no_break_space(self):
        assert normalize_unicode_whitespace("a\u00a0b") == "a b"

    def test_zero_width_space_removed(self):
        assert normalize_unicode_whitespace("a\u200bb") == "ab"

    def test_bom_removed(self):
        assert normalize_unicode_whitespace("\ufeffHello") == "Hello"

    def test_soft_hyphen_removed(self):
        assert normalize_unicode_whitespace("en\u00adgage") == "engage"

    def test_plain_ascii_unchanged(self):
        text = "Engagement Partner: John Smith"
        assert normalize_unicode_whitespace(text) == text

    def test_empty_string(self):
        assert normalize_unicode_whitespace("") == ""

    def test_multiple_variants_in_one_string(self):
        """Realistic cell with mixed unicode spaces from PDF extraction."""
        raw = "Fiscal\u2002Year\u2003End:\u00a0June 30, 2025"
        result = normalize_unicode_whitespace(raw)
        assert "\u2002" not in result
        assert "\u2003" not in result
        assert "\u00a0" not in result
        assert "Fiscal Year End: June 30, 2025" == result


# ---------------------------------------------------------------------------
# normalize_inline_markers
# ---------------------------------------------------------------------------

class TestNormalizeInlineMarkers:

    # engagement_decision — forward pattern
    def test_accept_forward(self):
        text = "Accept (X)  Decline ( )"
        result = normalize_inline_markers(text)
        assert "engagement_decision: Accept" in result

    def test_decline_forward(self):
        text = "Accept ( )  Decline (X)"
        result = normalize_inline_markers(text)
        assert "engagement_decision: Decline" in result

    # engagement_decision — backward pattern
    def test_accept_backward(self):
        text = "(X) Accept   ( ) Decline"
        result = normalize_inline_markers(text)
        assert "engagement_decision: Accept" in result

    def test_continue_backward(self):
        text = "(x) Continue engagement"
        result = normalize_inline_markers(text)
        assert "engagement_decision: Accept" in result

    # GAGAS
    def test_gagas_forward(self):
        text = "Government Auditing Standards (X)   GAAS Only ( )"
        result = normalize_inline_markers(text)
        assert "includes_gagas: true" in result

    def test_yellow_book_backward(self):
        text = "(X) Yellow Book applies"
        result = normalize_inline_markers(text)
        assert "includes_gagas: true" in result

    # Single audit
    def test_single_audit_forward(self):
        text = "Single Audit (X)"
        result = normalize_inline_markers(text)
        assert "includes_single_audit: true" in result

    def test_uniform_guidance_backward(self):
        text = "(X) 2 CFR 200 / Uniform Guidance"
        result = normalize_inline_markers(text)
        assert "includes_single_audit: true" in result

    # No match — original text unchanged
    def test_no_marker_unchanged(self):
        text = "Engagement Partner: John Smith"
        result = normalize_inline_markers(text)
        assert result == text

    def test_unchecked_box_no_decision(self):
        """Unchecked boxes with no checked counterpart — no decision appended."""
        text = "Accept ( )  Decline ( )"
        result = normalize_inline_markers(text)
        assert "engagement_decision" not in result

    def test_original_text_preserved(self):
        """Appended lines are additions — original text must survive."""
        text = "(X) Accept"
        result = normalize_inline_markers(text)
        assert result.startswith("(X) Accept")

    def test_empty_string(self):
        assert normalize_inline_markers("") == ""


# ---------------------------------------------------------------------------
# normalize_text — full pipeline integration
# ---------------------------------------------------------------------------

class TestNormalizeText:

    def test_checkbox_then_inline_marker(self):
        """
        Wingdings checkbox converted to 'true' BEFORE inline marker scan.
        The inline marker regex must NOT rely on seeing literal checkbox symbols.
        This tests that the pipeline order is correct.
        """
        # After checkbox normalization: "true Accept  false Decline"
        # After inline marker: no (X) pattern — markers already consumed by checkbox step
        # That's correct — the checkbox step already told us "Accept" is checked.
        text = "\uf0fe Accept  \uf0a1 Decline"
        result = normalize_text(text)
        # Checkbox symbols gone
        assert "\uf0fe" not in result
        assert "\uf0a1" not in result
        # "true" present where checked box was
        assert "true" in result

    def test_date_extraction_end_to_end(self):
        """OCR date artifact fully resolved."""
        text = "Fiscal Year End: _06_/ _30_/ _2024__"
        result = normalize_text(text)
        assert "06" in result
        assert "30" in result
        assert "2024" in result
        assert result.count("_") == 0

    def test_unicode_space_in_label(self):
        """Unicode space in label resolved — field extractable."""
        text = "Engagement\u2002Partner:\u2002John Smith"
        result = normalize_text(text)
        assert "Engagement Partner: John Smith" == result

    def test_full_engagement_form_cell(self):
        """
        Realistic PPC engagement form row combining multiple issues.
        'Engagement decision: ☒ Accept  ☐ Decline'
        """
        text = "Engagement decision: \u2612 Accept  \u2610 Decline"
        result = normalize_text(text)
        # Checkbox symbols normalized
        assert "true" in result
        assert "false" in result
        # Original structure preserved for label extraction
        assert "Engagement decision:" in result

    def test_empty_string_pipeline(self):
        assert normalize_text("") == ""

    def test_no_normalization_needed(self):
        """Clean ASCII text passes through unchanged (modulo whitespace collapse)."""
        text = "Client Name: Acme Nonprofit"
        result = normalize_text(text)
        assert result == text

    def test_multiple_inline_markers_detected(self):
        """
        Form line containing GAGAS and Single Audit markers together.
        Both structured lines should be appended.
        """
        text = "(X) Government Auditing Standards  (X) Single Audit (2 CFR 200)"
        result = normalize_text(text)
        assert "includes_gagas: true" in result
        assert "includes_single_audit: true" in result

    def test_whitespace_collapse(self):
        """Multiple spaces collapsed to one."""
        text = "Engagement   Partner:   John   Smith"
        result = normalize_text(text)
        assert "  " not in result