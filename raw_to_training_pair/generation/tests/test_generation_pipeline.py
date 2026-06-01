"""
raw_to_training_pair/generation/tests/test_generation_pipeline.py
==================================================================
Phase 2.1 walking-skeleton tests — locks in the generation pair
production pipeline end-to-end with synthetic data.

Coverage:
    target_schema   — GeneratedFieldValue/Workpaper structure,
                       registry validation, JSON round-trip
    prompt           — SYSTEM_PROMPT shape, render_user_message
                       layout, allowed_values surfaced for categoricals
    pair_builder     — 3-message structure, hash determinism, metadata
                       completeness, workpaper-type mismatch raises,
                       block_on_schema_issues behavior
    orchestrator     — synthetic and from-extractions entry points

All tests use synthetic data; no live Qdrant, no Ollama, no models.

Run with:
    pytest raw_to_training_pair/generation/tests/test_generation_pipeline.py -v
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from auditai_data_normalization.assembly_layer import SourceDocumentExtraction
from auditai_data_normalization.extractors.structural_extractor import (
    FieldEvidence,
)
from auditai_data_normalization.generation_contract import (
    CHAR_OFFSET_UNAVAILABLE,
    ExtractedFact,
    GenerationInput,
    SourceCitation,
)
from raw_to_training_pair.generation.orchestrator import (
    build_generation_pair_from_extractions,
    build_generation_pair_synthetic,
)
from raw_to_training_pair.generation.pair_builder import (
    build_generation_pair,
    pair_hash,
)
from raw_to_training_pair.generation.prompt import (
    SYSTEM_PROMPT,
    TASK_IDENTIFIER,
    render_user_message,
)
from raw_to_training_pair.generation.target_schema import (
    GeneratedCitation,
    GeneratedFieldValue,
    GeneratedWorkpaper,
    from_json_string,
    to_json_string,
    validate_against_registry,
)

WORKPAPER = "NPO-CX-1.1"


# ---------------------------------------------------------------------
# Helpers — synthetic data factories
# ---------------------------------------------------------------------

def _synthetic_citation(doc: str = "engagement_letter", page: int = 2,
                        text: str = "Sample Foundation, Inc.") -> GeneratedCitation:
    return GeneratedCitation(document=doc, page=page, quoted_text=text)


def _synthetic_gold_minimal_npo() -> GeneratedWorkpaper:
    """Build a complete-by-registry synthetic gold for NPO-CX-1.1.
    Every registry field gets a GeneratedFieldValue (mostly None
    except a handful with realistic values + citations)."""
    from auditai_data_normalization.field_type_registry import load_registry
    reg = load_registry(WORKPAPER)

    fields: dict[str, GeneratedFieldValue] = {}
    for fid, spec in reg.items():
        # Default: None (unfilled), no citations
        fields[fid] = GeneratedFieldValue(value=None, citations=[])

    # Override a few representative fields with real values
    fields["organization_name"] = GeneratedFieldValue(
        value="Sample Foundation, Inc.",
        citations=[_synthetic_citation()],
    )
    fields["financial_position_date"] = GeneratedFieldValue(
        value="2024-06-30",
        citations=[_synthetic_citation(doc="cover_page", page=1,
                                       text="June 30, 2024")],
    )
    fields["q1_a"] = GeneratedFieldValue(
        value="Accrual / GAAP",
        citations=[_synthetic_citation(doc="audit_report", page=4,
                                       text="financial statements were prepared in accordance with GAAP")],
    )
    fields["q1_c"] = GeneratedFieldValue(value=False, citations=[])
    fields["q4"] = GeneratedFieldValue(value=False, citations=[])

    return GeneratedWorkpaper(
        workpaper_type=WORKPAPER,
        engagement_id="ENG-2024-SYN-001",
        fields=fields,
    )


def _synthetic_gen_input(workpaper_type: str = WORKPAPER) -> GenerationInput:
    """Build a synthetic GenerationInput with a few facts + 2 SOPs."""
    from auditai_data_normalization.field_type_registry import load_registry
    reg = load_registry(workpaper_type)
    template_ids = sorted(reg.keys())

    citation = SourceCitation(
        document_path="/data/engagement_letter.pdf",
        document_type="engagement_letter",
        page=2,
        char_start=120, char_end=143,
        quoted_text="Sample Foundation, Inc.",
    )
    facts: dict[str, ExtractedFact] = {
        "organization_name": ExtractedFact(
            field_id="organization_name", field_type="text",
            value="Sample Foundation, Inc.", confidence=0.92,
            sources=[citation], extractor_method="structural_heuristic",
        ),
    }
    return GenerationInput(
        workpaper_type=workpaper_type,
        engagement_id="ENG-2024-SYN-001",
        sop_chunks=[
            "SOP §3.1 — Engagement acceptance criteria for nonprofits...",
            "SOP Table 2, Q1(a) — Basis of accounting may be GAAP, cash, or other.",
        ],
        extracted_facts=facts,
        template_field_ids=template_ids,
    )


# ---------------------------------------------------------------------
# target_schema
# ---------------------------------------------------------------------

class TestTargetSchemaValidation:

    def test_complete_gold_validates_clean(self):
        gold = _synthetic_gold_minimal_npo()
        issues = validate_against_registry(gold)
        assert issues == []

    def test_missing_field_flagged(self):
        gold = _synthetic_gold_minimal_npo()
        del gold.fields["q1_a"]
        issues = validate_against_registry(gold)
        assert any("q1_a" in i and "missing" in i for i in issues)

    def test_extra_field_flagged(self):
        gold = _synthetic_gold_minimal_npo()
        gold.fields["totally_made_up_field"] = GeneratedFieldValue(value="x")
        issues = validate_against_registry(gold)
        assert any("totally_made_up_field" in i for i in issues)

    def test_categorical_out_of_vocab_flagged(self):
        gold = _synthetic_gold_minimal_npo()
        gold.fields["q1_a"] = GeneratedFieldValue(value="Bogus Basis")
        issues = validate_against_registry(gold)
        assert any("q1_a" in i and "allowed_values" in i for i in issues)

    def test_boolean_with_string_value_flagged(self):
        gold = _synthetic_gold_minimal_npo()
        gold.fields["q1_c"] = GeneratedFieldValue(value="yes")  # should be bool
        issues = validate_against_registry(gold)
        assert any("q1_c" in i and "boolean" in i for i in issues)

    def test_unknown_workpaper_returns_registry_issue(self):
        gw = GeneratedWorkpaper(workpaper_type="NOPE-9.9", engagement_id="x", fields={})
        issues = validate_against_registry(gw)
        assert any("registry missing" in i for i in issues)


class TestTargetSchemaJSONRoundTrip:

    def test_round_trip_preserves_values_and_citations(self):
        gold = _synthetic_gold_minimal_npo()
        s = to_json_string(gold)
        restored = from_json_string(s)
        assert restored.workpaper_type == gold.workpaper_type
        assert restored.engagement_id == gold.engagement_id
        assert (
            restored.fields["organization_name"].value
            == "Sample Foundation, Inc."
        )
        assert len(restored.fields["organization_name"].citations) == 1
        assert (
            restored.fields["organization_name"].citations[0].document
            == "engagement_letter"
        )

    def test_field_order_deterministic(self):
        gold1 = _synthetic_gold_minimal_npo()
        gold2 = _synthetic_gold_minimal_npo()
        assert to_json_string(gold1) == to_json_string(gold2)

    def test_to_json_is_valid_json(self):
        gold = _synthetic_gold_minimal_npo()
        s = to_json_string(gold)
        parsed = json.loads(s)
        assert parsed["workpaper_type"] == WORKPAPER
        assert "fields" in parsed


# ---------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------

class TestSystemPrompt:

    def test_contains_task_identifier(self):
        assert TASK_IDENTIFIER in SYSTEM_PROMPT
        assert "GENERATE_WORKPAPER" == TASK_IDENTIFIER

    def test_mentions_citation_requirement(self):
        assert "citations" in SYSTEM_PROMPT.lower()

    def test_forbids_inventing_numerics(self):
        assert "do not invent" in SYSTEM_PROMPT.lower()


class TestRenderUserMessage:

    def test_includes_workpaper_and_engagement_headers(self):
        msg = render_user_message(_synthetic_gen_input())
        assert "WORKPAPER: NPO-CX-1.1" in msg
        assert "ENGAGEMENT: ENG-2024-SYN-001" in msg

    def test_extracted_facts_block_includes_known_value(self):
        msg = render_user_message(_synthetic_gen_input())
        assert "Sample Foundation, Inc." in msg

    def test_missing_fact_renders_as_null_with_marker(self):
        msg = render_user_message(_synthetic_gen_input())
        # All fields except organization_name are missing in the synthetic
        # gen_input → most field_ids should render as null + missing marker
        assert "null,  // missing" in msg

    def test_categorical_allowed_values_surfaced(self):
        msg = render_user_message(_synthetic_gen_input())
        # q1_a is categorical with allowed_values
        assert 'q1_a (categorical, allowed: "Accrual / GAAP"' in msg

    def test_sop_block_renders_chunks(self):
        msg = render_user_message(_synthetic_gen_input())
        assert "SOP §3.1" in msg
        assert "SOP Table 2, Q1(a)" in msg

    def test_empty_sop_chunks_renders_explanation(self):
        gi = _synthetic_gen_input()
        gi.sop_chunks = []
        msg = render_user_message(gi)
        assert "no SOP chunks retrieved" in msg


# ---------------------------------------------------------------------
# pair_builder
# ---------------------------------------------------------------------

class TestPairBuilder:

    def test_returns_three_messages_in_order(self):
        pair = build_generation_pair(_synthetic_gen_input(), _synthetic_gold_minimal_npo())
        roles = [m["role"] for m in pair["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_system_message_matches_constant(self):
        pair = build_generation_pair(_synthetic_gen_input(), _synthetic_gold_minimal_npo())
        assert pair["messages"][0]["content"] == SYSTEM_PROMPT

    def test_assistant_content_is_valid_json(self):
        pair = build_generation_pair(_synthetic_gen_input(), _synthetic_gold_minimal_npo())
        parsed = json.loads(pair["messages"][2]["content"])
        assert parsed["workpaper_type"] == WORKPAPER

    def test_metadata_marks_pair_type_generation(self):
        pair = build_generation_pair(_synthetic_gen_input(), _synthetic_gold_minimal_npo())
        assert pair["metadata"]["pair_type"] == "generation"
        assert pair["metadata"]["task"] == TASK_IDENTIFIER

    def test_metadata_has_required_keys(self):
        pair = build_generation_pair(_synthetic_gen_input(), _synthetic_gold_minimal_npo())
        meta = pair["metadata"]
        for k in (
            "pair_type", "task", "workpaper_type", "engagement_id",
            "file_hash", "pair_hash", "schema_issues",
            "fields_present_in_facts", "fields_missing_in_facts",
            "sop_chunks_count", "built_at",
        ):
            assert k in meta, f"metadata missing key: {k}"

    def test_pair_hash_deterministic_for_identical_inputs(self):
        gi = _synthetic_gen_input()
        gold = _synthetic_gold_minimal_npo()
        p1 = build_generation_pair(gi, gold)
        p2 = build_generation_pair(gi, gold)
        assert p1["metadata"]["pair_hash"] == p2["metadata"]["pair_hash"]

    def test_pair_hash_changes_when_gold_changes(self):
        gi = _synthetic_gen_input()
        gold_a = _synthetic_gold_minimal_npo()
        gold_b = _synthetic_gold_minimal_npo()
        gold_b.fields["organization_name"] = GeneratedFieldValue(
            value="Different Org, LLC",
            citations=[_synthetic_citation(text="Different Org, LLC")],
        )
        p1 = build_generation_pair(gi, gold_a)
        p2 = build_generation_pair(gi, gold_b)
        assert p1["metadata"]["pair_hash"] != p2["metadata"]["pair_hash"]

    def test_workpaper_type_mismatch_raises(self):
        gi = _synthetic_gen_input()
        gold = _synthetic_gold_minimal_npo()
        gold.workpaper_type = "GOV-CX-1.1"
        with pytest.raises(ValueError, match="mismatch"):
            build_generation_pair(gi, gold)

    def test_soft_schema_issue_recorded_in_metadata_when_not_blocking(self):
        gi = _synthetic_gen_input()
        gold = _synthetic_gold_minimal_npo()
        gold.fields["q1_a"] = GeneratedFieldValue(value="Bogus Basis")
        pair = build_generation_pair(gi, gold, block_on_schema_issues=False)
        assert pair["metadata"]["schema_issues"]
        assert any("q1_a" in i for i in pair["metadata"]["schema_issues"])

    def test_soft_schema_issue_does_NOT_block_even_when_strict(self):
        # Out-of-vocab categorical is a SOFT issue — real auditors write
        # free text in dropdown fields. Should not block pair build even
        # when block_on_schema_issues=True.
        gi = _synthetic_gen_input()
        gold = _synthetic_gold_minimal_npo()
        gold.fields["q1_a"] = GeneratedFieldValue(value="Bogus Basis")
        pair = build_generation_pair(gi, gold, block_on_schema_issues=True)
        # The pair builds; issue is recorded as soft
        assert any("q1_a" in i for i in pair["metadata"]["schema_issues"])

    def test_hard_schema_issue_raises_when_blocking(self):
        # Adding an extra-field-not-in-registry is a HARD issue.
        gi = _synthetic_gen_input()
        gold = _synthetic_gold_minimal_npo()
        gold.fields["totally_made_up_field"] = GeneratedFieldValue(value="x")
        with pytest.raises(ValueError, match="HARD schema issue"):
            build_generation_pair(gi, gold, block_on_schema_issues=True)

    def test_extra_metadata_merged(self):
        pair = build_generation_pair(
            _synthetic_gen_input(), _synthetic_gold_minimal_npo(),
            extra_metadata={"reviewer_id": "test_reviewer"},
        )
        assert pair["metadata"]["reviewer_id"] == "test_reviewer"


class TestPairHash:

    def test_hash_is_hex_digest(self):
        h = pair_hash([{"role": "system", "content": "x"}])
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex length
        int(h, 16)  # raises if not hex

    def test_hash_changes_on_content_change(self):
        a = pair_hash([{"role": "user", "content": "a"}])
        b = pair_hash([{"role": "user", "content": "b"}])
        assert a != b


# ---------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------

class TestOrchestratorSynthetic:

    def test_synthetic_entry_point_returns_full_pair(self):
        pair = build_generation_pair_synthetic(
            _synthetic_gen_input(),
            _synthetic_gold_minimal_npo(),
        )
        assert pair["metadata"]["pair_type"] == "generation"
        assert len(pair["messages"]) == 3


class TestOrchestratorFromExtractions:
    """The from-extractions entry point assembles a GenerationInput and
    then builds the pair. Phase 1A assembly path is exercised here."""

    def test_no_extractions_still_produces_valid_pair(self):
        # Empty source_extractions → GenerationInput with no facts; pair
        # is still buildable because the gold is provided separately
        # (Phase 2.1 walking skeleton — gold is synthetic)
        pair = build_generation_pair_from_extractions(
            workpaper_type=WORKPAPER,
            engagement_id="ENG-TEST",
            source_extractions=[],
            gold=_synthetic_gold_minimal_npo(),
        )
        assert pair["metadata"]["pair_type"] == "generation"
        assert pair["metadata"]["fields_present_in_facts"] == 0

    def test_with_extractions_populates_facts(self):
        ev = FieldEvidence(
            value="Sample Foundation, Inc.",
            confidence=0.85,
            source_page=2,
            method="salutation_block",
            anchor="To the Board of Sample Foundation, Inc.",
            char_start=10, char_end=33,
            full_quoted_text="To the Board of Sample Foundation, Inc.",
        )
        extraction = SourceDocumentExtraction(
            document_path="/data/letter.pdf",
            document_type="engagement_letter",
            field_evidence={"organization_name": ev},
        )
        pair = build_generation_pair_from_extractions(
            workpaper_type=WORKPAPER,
            engagement_id="ENG-TEST",
            source_extractions=[extraction],
            gold=_synthetic_gold_minimal_npo(),
        )
        assert pair["metadata"]["fields_present_in_facts"] == 1

    @patch("engineering_benchmark.sop_retriever.retrieve_sop_chunks")
    def test_with_sop_retrieval_populates_sop_count(self, mock_retrieve):
        mock_retrieve.return_value = ["SOP §3.1 ...", "SOP §3.2 ..."]
        pair = build_generation_pair_from_extractions(
            workpaper_type=WORKPAPER,
            engagement_id="ENG-TEST",
            source_extractions=[],
            gold=_synthetic_gold_minimal_npo(),
            with_sop_retrieval=True,
        )
        assert pair["metadata"]["sop_chunks_count"] == 2

    def test_passes_sop_chunks_through_when_not_retrieving(self):
        chunks = ["manually-provided SOP §X ...", "another chunk"]
        pair = build_generation_pair_from_extractions(
            workpaper_type=WORKPAPER,
            engagement_id="ENG-TEST",
            source_extractions=[],
            gold=_synthetic_gold_minimal_npo(),
            sop_chunks=chunks,
            with_sop_retrieval=False,
        )
        assert pair["metadata"]["sop_chunks_count"] == 2
