"""
raw_to_training_pair/generation/tests/test_phase_2_2_ui.py
============================================================
Phase 2.2 Component 4 tests — pair-builder metadata extension
(extracted_facts_summary) and pure-function helpers from
streamlit_app.generation_pair_renderer.

The Streamlit rendering itself is tested by hand in the browser
(st.* calls have side effects on a live runtime); we test the
pure helpers that build the comparison rows and compute statuses.

Run with:
    pytest raw_to_training_pair/generation/tests/test_phase_2_2_ui.py -v
"""

from __future__ import annotations

import pytest

from auditai_data_normalization.generation_contract import (
    ExtractedFact,
    GenerationInput,
    SourceCitation,
)
from raw_to_training_pair.generation.pair_builder import (
    _facts_summary,
    build_generation_pair,
)
from streamlit_app.generation_pair_renderer import (
    _BOTH_NULL,
    _FACT_ONLY,
    _GOLD_ONLY,
    _MATCH,
    _MISMATCH,
    _build_comparison_rows,
    _compare_status,
    _summary_counts,
    _value_for_display,
)

# Pull the existing synthetic fixtures
from raw_to_training_pair.generation.tests.test_generation_pipeline import (
    _synthetic_gen_input,
    _synthetic_gold_minimal_npo,
)

WORKPAPER = "NPO-CX-1.1"


# ---------------------------------------------------------------------
# _facts_summary helper in pair_builder
# ---------------------------------------------------------------------

class TestFactsSummary:

    def test_summary_excludes_absent_facts(self):
        gi = _synthetic_gen_input()  # has only organization_name present
        summary = _facts_summary(gi)
        assert "organization_name" in summary
        # template fields without facts should not appear
        assert "completed_by" not in summary
        assert "q1_a" not in summary

    def test_summary_includes_value_and_confidence(self):
        gi = _synthetic_gen_input()
        summary = _facts_summary(gi)
        entry = summary["organization_name"]
        assert entry["value"] == "Sample Foundation, Inc."
        assert entry["confidence"] == 0.92

    def test_summary_includes_extractor_method(self):
        gi = _synthetic_gen_input()
        summary = _facts_summary(gi)
        assert summary["organization_name"]["extractor_method"] == (
            "structural_heuristic"
        )

    def test_summary_includes_sources_with_compact_shape(self):
        gi = _synthetic_gen_input()
        summary = _facts_summary(gi)
        sources = summary["organization_name"]["sources"]
        assert isinstance(sources, list)
        assert len(sources) >= 1
        src = sources[0]
        assert "document_type" in src
        assert "page" in src
        assert "quoted_text" in src

    def test_summary_clips_quoted_text_to_200_chars(self):
        long_quote = "x" * 500
        citation = SourceCitation(
            document_path="/x.pdf", document_type="engagement_letter",
            page=1, char_start=0, char_end=500, quoted_text=long_quote,
        )
        gi = GenerationInput(
            workpaper_type=WORKPAPER, engagement_id="ENG-X",
            sop_chunks=[], template_field_ids=["organization_name"],
            extracted_facts={
                "organization_name": ExtractedFact(
                    field_id="organization_name", field_type="text",
                    value="X", confidence=0.9, sources=[citation],
                    extractor_method="structural_heuristic",
                )
            },
        )
        summary = _facts_summary(gi)
        assert len(summary["organization_name"]["sources"][0]["quoted_text"]) <= 200

    def test_summary_caps_sources_at_three(self):
        # Build 5 citations, expect only 3 in summary
        citations = [
            SourceCitation(
                document_path=f"/{i}.pdf",
                document_type="engagement_letter",
                page=i + 1, char_start=0, char_end=10,
                quoted_text=f"src{i}",
            )
            for i in range(5)
        ]
        gi = GenerationInput(
            workpaper_type=WORKPAPER, engagement_id="ENG-X",
            sop_chunks=[], template_field_ids=["organization_name"],
            extracted_facts={
                "organization_name": ExtractedFact(
                    field_id="organization_name", field_type="text",
                    value="X", confidence=0.9, sources=citations,
                    extractor_method="structural_heuristic",
                )
            },
        )
        summary = _facts_summary(gi)
        assert len(summary["organization_name"]["sources"]) == 3


# ---------------------------------------------------------------------
# build_generation_pair: metadata now carries extracted_facts_summary
# ---------------------------------------------------------------------

class TestPairMetadataFactsSummary:

    def test_metadata_includes_extracted_facts_summary(self):
        pair = build_generation_pair(
            _synthetic_gen_input(), _synthetic_gold_minimal_npo(),
        )
        assert "extracted_facts_summary" in pair["metadata"]

    def test_summary_in_metadata_matches_present_facts(self):
        pair = build_generation_pair(
            _synthetic_gen_input(), _synthetic_gold_minimal_npo(),
        )
        summary = pair["metadata"]["extracted_facts_summary"]
        # synthetic gen_input has exactly one present fact
        assert "organization_name" in summary
        assert summary["organization_name"]["value"] == "Sample Foundation, Inc."

    def test_summary_in_metadata_is_json_serializable(self):
        import json
        pair = build_generation_pair(
            _synthetic_gen_input(), _synthetic_gold_minimal_npo(),
        )
        # Should round-trip through JSON without error
        s = json.dumps(pair["metadata"]["extracted_facts_summary"])
        restored = json.loads(s)
        assert restored == pair["metadata"]["extracted_facts_summary"]


# ---------------------------------------------------------------------
# _compare_status — the core comparison logic
# ---------------------------------------------------------------------

class TestCompareStatus:

    def test_both_null_returns_both_null(self):
        assert _compare_status(None, None) == _BOTH_NULL

    def test_fact_only_when_gold_null(self):
        assert _compare_status("Sample Foundation", None) == _FACT_ONLY

    def test_gold_only_when_fact_null(self):
        assert _compare_status(None, "Sample Foundation") == _GOLD_ONLY

    def test_match_when_values_equal(self):
        assert _compare_status("Sample Foundation", "Sample Foundation") == _MATCH

    def test_match_strips_whitespace(self):
        assert _compare_status("  Sample  ", "Sample") == _MATCH

    def test_mismatch_when_values_differ(self):
        assert _compare_status(
            "Sample Foundation", "Different Org",
        ) == _MISMATCH

    def test_boolean_match(self):
        assert _compare_status(True, True) == _MATCH
        assert _compare_status(False, False) == _MATCH

    def test_boolean_mismatch(self):
        assert _compare_status(True, False) == _MISMATCH

    def test_empty_string_treated_as_null_on_gold_side(self):
        # gold "" should NOT count as present
        assert _compare_status("X", "") == _FACT_ONLY


# ---------------------------------------------------------------------
# _build_comparison_rows
# ---------------------------------------------------------------------

class TestBuildComparisonRows:

    def test_rows_union_facts_and_gold(self):
        facts = {"a": {"value": "x", "sources": []}}
        gold = {"b": {"value": "y", "citations": []}}
        rows = _build_comparison_rows(facts, gold)
        ids = {r["field_id"] for r in rows}
        assert ids == {"a", "b"}

    def test_rows_sorted_by_field_id(self):
        facts = {"zzz": {"value": "x", "sources": []}}
        gold = {"aaa": {"value": "y", "citations": []}}
        rows = _build_comparison_rows(facts, gold)
        assert [r["field_id"] for r in rows] == ["aaa", "zzz"]

    def test_row_carries_status(self):
        facts = {"a": {"value": "x", "sources": []}}
        gold = {"a": {"value": "x", "citations": []}}
        rows = _build_comparison_rows(facts, gold)
        assert rows[0]["status"] == _MATCH

    def test_row_carries_source_and_citation_counts(self):
        facts = {
            "a": {
                "value": "x",
                "sources": [{"document_type": "engagement_letter", "page": 1, "quoted_text": "x"}],
            }
        }
        gold = {"a": {"value": "x", "citations": [{"document": "engagement_letter", "page": 1, "quoted_text": "x"}]}}
        rows = _build_comparison_rows(facts, gold)
        assert rows[0]["sources_count"] == 1
        assert rows[0]["citations_count"] == 1

    def test_row_extracts_first_source(self):
        facts = {
            "a": {
                "value": "x",
                "sources": [
                    {"document_type": "engagement_letter", "page": 2, "quoted_text": "first"},
                    {"document_type": "prior_year_file", "page": 5, "quoted_text": "second"},
                ],
            }
        }
        gold = {}
        rows = _build_comparison_rows(facts, gold)
        assert rows[0]["first_source"]["quoted_text"] == "first"


# ---------------------------------------------------------------------
# _summary_counts
# ---------------------------------------------------------------------

class TestSummaryCounts:

    def test_counts_each_status(self):
        rows = [
            {"status": _MATCH},
            {"status": _MATCH},
            {"status": _MISMATCH},
            {"status": _GOLD_ONLY},
        ]
        counts = _summary_counts(rows)
        assert counts[_MATCH] == 2
        assert counts[_MISMATCH] == 1
        assert counts[_GOLD_ONLY] == 1
        assert counts[_FACT_ONLY] == 0
        assert counts[_BOTH_NULL] == 0

    def test_empty_rows(self):
        counts = _summary_counts([])
        assert counts[_MATCH] == 0


# ---------------------------------------------------------------------
# _value_for_display
# ---------------------------------------------------------------------

class TestValueForDisplay:

    def test_none_shown_as_dash(self):
        assert _value_for_display(None) == "—"

    def test_bool_shown_as_yes_no(self):
        assert _value_for_display(True) == "Yes"
        assert _value_for_display(False) == "No"

    def test_long_string_truncated(self):
        long = "x" * 200
        out = _value_for_display(long)
        assert len(out) <= 80
        assert out.endswith("...")

    def test_short_string_unchanged(self):
        assert _value_for_display("Sample Foundation, Inc.") == "Sample Foundation, Inc."
