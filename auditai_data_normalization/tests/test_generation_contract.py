"""
auditai_data_normalization/tests/test_generation_contract.py
=============================================================
Phase 1A foundation tests — locks in the contract, registry, adapter,
and assembly layer as regression-protected before invasive edits to
existing extractors begin.

Coverage:
    contract     — ExtractedFact validation, no-LLM-numbers rule,
                   SourceCitation sentinel handling
    registry     — load, validate, lookup, LLM allowance check
    adapter      — FieldEvidence → ExtractedFact conversion, method
                   mapping, bulk conversion, unknown-field skip
    assembly     — single doc, multi-doc agreement, disagreement +
                   penalty, missing fields, empty input, sop pass-through

Run with:
    pytest auditai_data_normalization/tests/test_generation_contract.py -v
"""

from __future__ import annotations

import pytest

from auditai_data_normalization.assembly_layer import (
    SourceDocumentExtraction,
    assemble_generation_input,
    merge_facts,
)
from auditai_data_normalization.extractors.structural_extractor import (
    FieldEvidence,
)
from auditai_data_normalization.field_evidence_adapter import (
    field_evidence_map_to_facts,
    field_evidence_to_extracted_fact,
)
from auditai_data_normalization.field_type_registry import (
    FieldSpec,
    get_field_spec,
    is_llm_allowed,
    load_registry,
)
from auditai_data_normalization.generation_contract import (
    CHAR_OFFSET_UNAVAILABLE,
    LLM_FORBIDDEN_FIELD_TYPES,
    ExtractedFact,
    GenerationInput,
    SourceCitation,
)

WORKPAPER = "NPO-CX-1.1"


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def citation_text():
    return SourceCitation(
        document_path="/data/letter.pdf",
        document_type="engagement_letter",
        page=1,
        char_start=0,
        char_end=22,
        quoted_text="Sample Foundation, Inc.",
    )


@pytest.fixture
def citation_page_only():
    return SourceCitation(
        document_path="/data/letter.pdf",
        document_type="engagement_letter",
        page=1,
        char_start=CHAR_OFFSET_UNAVAILABLE,
        char_end=CHAR_OFFSET_UNAVAILABLE,
        quoted_text="Sample Foundation",
    )


@pytest.fixture
def fe_org_name():
    return FieldEvidence(
        value="Sample Foundation, Inc.",
        confidence=0.92,
        source_page=1,
        method="salutation_block",
        anchor="To the Board of Directors of Sample Foundation, Inc.",
    )


@pytest.fixture
def fe_fye_date():
    return FieldEvidence(
        value="2024-06-30",
        confidence=0.95,
        source_page=1,
        method="standalone_date",
        anchor="June 30, 2024",
    )


# ---------------------------------------------------------------------
# Contract — SourceCitation
# ---------------------------------------------------------------------

class TestSourceCitation:

    def test_has_char_offsets_true_when_real_offsets(self, citation_text):
        assert citation_text.has_char_offsets is True

    def test_has_char_offsets_false_when_sentinel(self, citation_page_only):
        assert citation_page_only.has_char_offsets is False

    def test_render_citation_format(self, citation_text):
        assert citation_text.render_citation() == "[engagement_letter p.1]"


# ---------------------------------------------------------------------
# Contract — ExtractedFact validation
# ---------------------------------------------------------------------

class TestExtractedFactValidation:

    def test_text_field_with_value_requires_sources(self, citation_text):
        with pytest.raises(ValueError, match="no sources"):
            ExtractedFact(
                field_id="organization_name",
                field_type="text",
                value="Sample Foundation",
                confidence=0.9,
                sources=[],  # ← invalid: value present but no provenance
                extractor_method="structural_heuristic",
            )

    def test_none_value_allowed_without_sources(self):
        fact = ExtractedFact(
            field_id="organization_name",
            field_type="text",
            value=None,
            confidence=0.0,
            sources=[],
            extractor_method="structural_heuristic",
        )
        assert fact.is_present is False

    def test_confidence_out_of_range_raises(self, citation_text):
        with pytest.raises(ValueError, match="outside"):
            ExtractedFact(
                field_id="organization_name",
                field_type="text",
                value="X",
                confidence=1.5,  # invalid
                sources=[citation_text],
                extractor_method="structural_heuristic",
            )

    def test_negative_confidence_raises(self, citation_text):
        with pytest.raises(ValueError, match="outside"):
            ExtractedFact(
                field_id="organization_name",
                field_type="text",
                value="X",
                confidence=-0.1,
                sources=[citation_text],
                extractor_method="structural_heuristic",
            )

    def test_is_high_confidence_threshold(self, citation_text):
        low = ExtractedFact(
            field_id="organization_name", field_type="text",
            value="X", confidence=0.80,
            sources=[citation_text], extractor_method="structural_heuristic",
        )
        high = ExtractedFact(
            field_id="organization_name", field_type="text",
            value="X", confidence=0.90,
            sources=[citation_text], extractor_method="structural_heuristic",
        )
        assert low.is_high_confidence is False
        assert high.is_high_confidence is True


# ---------------------------------------------------------------------
# Contract — no-LLM-numbers rule (existential per Phase 0)
# ---------------------------------------------------------------------

class TestNoLLMNumbersRule:

    def test_llm_forbidden_set_matches_contract(self):
        # If this changes, downstream code that depends on the set
        # composition must be updated.
        assert LLM_FORBIDDEN_FIELD_TYPES == frozenset({"numeric", "date", "id"})

    @pytest.mark.parametrize("forbidden_type", ["numeric", "date", "id"])
    def test_llm_extraction_forbidden_for_forbidden_types(
        self, forbidden_type, citation_text,
    ):
        with pytest.raises(ValueError, match="Numerical accuracy rule"):
            ExtractedFact(
                field_id="some_field",
                field_type=forbidden_type,
                value="something",
                confidence=0.9,
                sources=[citation_text],
                extractor_method="llm_extraction",
            )

    @pytest.mark.parametrize("allowed_type", ["text", "boolean", "categorical"])
    def test_llm_extraction_allowed_for_other_types(
        self, allowed_type, citation_text,
    ):
        fact = ExtractedFact(
            field_id="some_field",
            field_type=allowed_type,
            value="Yes" if allowed_type == "boolean" else "x",
            confidence=0.7,
            sources=[citation_text],
            extractor_method="llm_extraction",
        )
        assert fact.extractor_method == "llm_extraction"


# ---------------------------------------------------------------------
# Registry — load + validate + lookup
# ---------------------------------------------------------------------

class TestFieldTypeRegistry:

    def test_load_npo_cx_1_1(self):
        registry = load_registry(WORKPAPER)
        assert len(registry) == 57

    def test_distribution_matches_expected(self):
        registry = load_registry(WORKPAPER)
        types = {}
        for spec in registry.values():
            types[spec.field_type] = types.get(spec.field_type, 0) + 1
        # 38 boolean + 11 text + 3 date + 5 categorical = 57
        # (11 text: prior 10 + q12_remark added for yes_no_with_remark_on_yes)
        assert types == {"boolean": 38, "text": 11, "date": 3, "categorical": 5}

    def test_acceptance_decision_excluded(self):
        with pytest.raises(KeyError):
            get_field_spec(WORKPAPER, "acceptance_decision")

    def test_unknown_workpaper_raises(self):
        with pytest.raises(FileNotFoundError):
            load_registry("DOES-NOT-EXIST")

    def test_categorical_fields_carry_allowed_values(self):
        registry = load_registry(WORKPAPER)
        for fid in ("engagement_type", "q1_a", "q1_b",
                    "q1_f_specification", "q2_reference"):
            spec = registry[fid]
            assert spec.field_type == "categorical"
            assert spec.allowed_values is not None
            assert len(spec.allowed_values) >= 2

    def test_q1_a_allowed_values_exact(self):
        spec = get_field_spec(WORKPAPER, "q1_a")
        assert spec.allowed_values == ("Accrual / GAAP", "Cash Basis", "Other")

    def test_is_llm_allowed_for_date(self):
        assert is_llm_allowed(WORKPAPER, "sign_off_date") is False

    def test_is_llm_allowed_for_text(self):
        assert is_llm_allowed(WORKPAPER, "q3") is True

    def test_is_llm_allowed_for_categorical(self):
        # categorical is NOT in the forbidden set
        assert is_llm_allowed(WORKPAPER, "q1_a") is True

    def test_field_spec_is_immutable(self):
        spec = get_field_spec(WORKPAPER, "q1_a")
        with pytest.raises(Exception):  # frozen=True → FrozenInstanceError
            spec.field_type = "text"  # type: ignore


# ---------------------------------------------------------------------
# Adapter — FieldEvidence → ExtractedFact
# ---------------------------------------------------------------------

class TestFieldEvidenceAdapter:

    def test_normal_text_conversion(self, fe_org_name):
        fact = field_evidence_to_extracted_fact(
            evidence=fe_org_name,
            field_id="organization_name",
            workpaper_type=WORKPAPER,
            document_path="/data/audit_report.pdf",
            document_type="audit_report",
            extractor_version="structural_extractor@1.0",
        )
        assert fact.value == "Sample Foundation, Inc."
        assert fact.field_type == "text"
        assert fact.confidence == 0.92
        assert fact.extractor_method == "structural_heuristic"
        assert fact.extractor_version == "structural_extractor@1.0"
        assert len(fact.sources) == 1
        src = fact.sources[0]
        assert src.page == 1
        assert src.has_char_offsets is False
        assert "Sample Foundation" in src.quoted_text

    def test_date_field_method_maps_correctly(self, fe_fye_date):
        fact = field_evidence_to_extracted_fact(
            evidence=fe_fye_date,
            field_id="financial_position_date",
            workpaper_type=WORKPAPER,
            document_path="/data/cover.pdf",
            document_type="audit_report",
        )
        assert fact.field_type == "date"
        assert fact.extractor_method == "regex_pattern"

    def test_unknown_method_defaults_to_structural_heuristic(self, caplog):
        ev = FieldEvidence(
            value="Yes", confidence=0.7, source_page=2,
            method="novel_heuristic_xyz", anchor="confirmed yes",
        )
        with caplog.at_level("WARNING"):
            fact = field_evidence_to_extracted_fact(
                evidence=ev,
                field_id="q1_c",
                workpaper_type=WORKPAPER,
                document_path="/data/doc.pdf",
                document_type="engagement_letter",
            )
        assert fact.extractor_method == "structural_heuristic"
        assert any("unknown legacy method" in r.message for r in caplog.records)

    def test_bulk_conversion_skips_unknown_field(
        self, caplog, fe_org_name, fe_fye_date,
    ):
        evidences = {
            "organization_name": fe_org_name,
            "financial_position_date": fe_fye_date,
            "unknown_field_xyz": fe_org_name,  # not in registry
        }
        with caplog.at_level("WARNING"):
            facts = field_evidence_map_to_facts(
                evidences=evidences,
                workpaper_type=WORKPAPER,
                document_path="/data/doc.pdf",
                document_type="audit_report",
            )
        assert "organization_name" in facts
        assert "financial_position_date" in facts
        assert "unknown_field_xyz" not in facts
        assert any("not in registry" in r.message for r in caplog.records)

    def test_notes_flag_page_level_provenance(self, fe_org_name):
        fact = field_evidence_to_extracted_fact(
            evidence=fe_org_name,
            field_id="organization_name",
            workpaper_type=WORKPAPER,
            document_path="/data/doc.pdf",
            document_type="audit_report",
        )
        assert "page-level" in fact.notes

    def test_notes_flag_missing_quoted_text(self):
        ev = FieldEvidence(
            value="X", confidence=0.5, source_page=1,
            method="salutation_block", anchor="",  # no anchor text
        )
        fact = field_evidence_to_extracted_fact(
            evidence=ev,
            field_id="organization_name",
            workpaper_type=WORKPAPER,
            document_path="/data/x.pdf",
            document_type="audit_report",
        )
        assert "no quoted text" in fact.notes


# ---------------------------------------------------------------------
# Assembly layer — merge + GenerationInput
# ---------------------------------------------------------------------

class TestAssemblyLayer:

    def test_single_document(self, fe_org_name, fe_fye_date):
        extraction = SourceDocumentExtraction(
            document_path="/data/letter.pdf",
            document_type="engagement_letter",
            field_evidence={
                "organization_name": fe_org_name,
                "financial_position_date": fe_fye_date,
            },
            extractor_version="struct@1.0",
        )
        gen_input = assemble_generation_input(
            workpaper_type=WORKPAPER,
            engagement_id="ENG-2024-001",
            source_extractions=[extraction],
        )
        assert gen_input.workpaper_type == WORKPAPER
        assert gen_input.engagement_id == "ENG-2024-001"
        assert len(gen_input.template_field_ids) == 57
        assert set(gen_input.fields_present()) == {
            "organization_name", "financial_position_date",
        }
        assert len(gen_input.fields_missing()) == 55

    def test_multi_document_agreement_combines_sources(self):
        ev_a = FieldEvidence(
            value="Sample Foundation, Inc.", confidence=0.92,
            source_page=1, method="salutation_block",
            anchor="Sample Foundation",
        )
        ev_b = FieldEvidence(
            value="Sample Foundation, Inc.", confidence=0.88,
            source_page=3, method="salutation_block",
            anchor="Sample Foundation, Inc.",
        )
        doc_a = SourceDocumentExtraction(
            document_path="/data/letter.pdf",
            document_type="engagement_letter",
            field_evidence={"organization_name": ev_a},
        )
        doc_b = SourceDocumentExtraction(
            document_path="/data/prior.pdf",
            document_type="prior_year_file",
            field_evidence={"organization_name": ev_b},
        )
        gen_input = assemble_generation_input(
            workpaper_type=WORKPAPER,
            engagement_id="ENG-2024-002",
            source_extractions=[doc_a, doc_b],
        )
        merged = gen_input.extracted_facts["organization_name"]
        assert merged.value == "Sample Foundation, Inc."
        assert len(merged.sources) == 2
        assert merged.extractor_method == "multi_extractor_agreement"
        assert merged.confidence == 0.92  # max of inputs
        assert "agree" in merged.notes

    def test_multi_document_disagreement_picks_highest_and_penalizes(self):
        ev_high = FieldEvidence(
            value="Sample Foundation, Inc.", confidence=0.92,
            source_page=1, method="salutation_block",
            anchor="Sample Foundation",
        )
        ev_low = FieldEvidence(
            value="Sample Foundation",  # different value
            confidence=0.75, source_page=2, method="salutation_block",
            anchor="Sample Foundation",
        )
        gen_input = assemble_generation_input(
            workpaper_type=WORKPAPER,
            engagement_id="ENG-2024-003",
            source_extractions=[
                SourceDocumentExtraction(
                    document_path="/data/letter.pdf",
                    document_type="engagement_letter",
                    field_evidence={"organization_name": ev_high},
                ),
                SourceDocumentExtraction(
                    document_path="/data/old.pdf",
                    document_type="prior_year_file",
                    field_evidence={"organization_name": ev_low},
                ),
            ],
        )
        merged = gen_input.extracted_facts["organization_name"]
        assert merged.value == "Sample Foundation, Inc."
        # 20% penalty: 0.92 * 0.8 = 0.736
        assert abs(merged.confidence - 0.92 * 0.8) < 1e-6
        assert "CONFLICT" in merged.notes

    def test_low_confidence_fields_helper(self):
        ev_low = FieldEvidence(
            value="X Co.", confidence=0.55, source_page=1,
            method="salutation_block", anchor="X Co.",
        )
        ev_high = FieldEvidence(
            value="Jane Auditor", confidence=0.95, source_page=2,
            method="salutation_block", anchor="Jane Auditor",
        )
        gen_input = assemble_generation_input(
            workpaper_type=WORKPAPER,
            engagement_id="ENG-2024-004",
            source_extractions=[
                SourceDocumentExtraction(
                    document_path="/data/x.pdf",
                    document_type="engagement_letter",
                    field_evidence={
                        "organization_name": ev_low,
                        "completed_by": ev_high,
                    },
                ),
            ],
        )
        low = gen_input.low_confidence_fields(threshold=0.70)
        assert "organization_name" in low
        assert "completed_by" not in low

    def test_empty_source_documents(self):
        gen_input = assemble_generation_input(
            workpaper_type=WORKPAPER,
            engagement_id="ENG-2024-005",
            source_extractions=[],
        )
        assert gen_input.extracted_facts == {}
        assert len(gen_input.fields_missing()) == 57

    def test_sop_chunks_pass_through(self, fe_org_name):
        chunks = [
            "SOP Table 2, Q1(a): basis of accounting...",
            "SOP Table 6: predecessor inquiries...",
        ]
        gen_input = assemble_generation_input(
            workpaper_type=WORKPAPER,
            engagement_id="ENG-2024-006",
            source_extractions=[
                SourceDocumentExtraction(
                    document_path="/data/letter.pdf",
                    document_type="engagement_letter",
                    field_evidence={"organization_name": fe_org_name},
                ),
            ],
            sop_chunks=chunks,
        )
        assert gen_input.sop_chunks == chunks


# ---------------------------------------------------------------------
# merge_facts — direct unit tests
# ---------------------------------------------------------------------

class TestMergeFacts:

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            merge_facts([])

    def test_single_fact_returned_as_is(self, citation_text):
        fact = ExtractedFact(
            field_id="organization_name", field_type="text",
            value="X", confidence=0.8,
            sources=[citation_text], extractor_method="structural_heuristic",
        )
        merged = merge_facts([fact])
        assert merged is fact

    def test_mixed_field_ids_raises(self, citation_text):
        f1 = ExtractedFact(
            field_id="a", field_type="text", value="x", confidence=0.5,
            sources=[citation_text], extractor_method="structural_heuristic",
        )
        f2 = ExtractedFact(
            field_id="b", field_type="text", value="y", confidence=0.5,
            sources=[citation_text], extractor_method="structural_heuristic",
        )
        with pytest.raises(ValueError, match="mixed field_ids"):
            merge_facts([f1, f2])

    def test_all_none_values_returns_first(self):
        f1 = ExtractedFact(
            field_id="a", field_type="text", value=None, confidence=0.0,
            sources=[], extractor_method="structural_heuristic",
        )
        f2 = ExtractedFact(
            field_id="a", field_type="text", value=None, confidence=0.0,
            sources=[], extractor_method="structural_heuristic",
        )
        merged = merge_facts([f1, f2])
        assert merged.value is None


# ---------------------------------------------------------------------
# LLM extractor unification — contract-shape API with mocked Ollama
# ---------------------------------------------------------------------

from unittest.mock import patch

from auditai_data_normalization.extractors import llm_extractor
from auditai_data_normalization.extractors.llm_extractor import (
    _filter_llm_allowed,
    extract_all_fields_as_facts,
    extract_fields_as_facts,
)
from auditai_data_normalization.generation_contract import PAGE_UNKNOWN


class TestLLMNoLLMRuleFilter:
    """The pre-filter that drops forbidden field types before the LLM
    is consulted. Pure unit test — no Ollama needed."""

    def test_filters_out_date_fields(self):
        allowed, dropped = _filter_llm_allowed(
            ["organization_name", "sign_off_date", "financial_position_date"],
            WORKPAPER,
        )
        assert "organization_name" in allowed
        assert "sign_off_date" not in allowed
        assert "financial_position_date" not in allowed
        assert ("sign_off_date", "date") in dropped
        assert ("financial_position_date", "date") in dropped

    def test_passes_text_and_boolean_through(self):
        allowed, dropped = _filter_llm_allowed(
            ["organization_name", "q1_c", "q3"], WORKPAPER,
        )
        assert set(allowed) == {"organization_name", "q1_c", "q3"}
        assert dropped == []

    def test_passes_categorical_through(self):
        # categorical is NOT in LLM_FORBIDDEN_FIELD_TYPES
        allowed, dropped = _filter_llm_allowed(["q1_a", "engagement_type"], WORKPAPER)
        assert set(allowed) == {"q1_a", "engagement_type"}
        assert dropped == []

    def test_unknown_fields_pass_through(self):
        # Fields not in registry are passed through; downstream catches them.
        allowed, _ = _filter_llm_allowed(["unknown_field_xyz"], WORKPAPER)
        assert allowed == ["unknown_field_xyz"]

    def test_filter_logs_dropped_with_rule_reason(self, caplog):
        with caplog.at_level("WARNING"):
            _filter_llm_allowed(["sign_off_date"], WORKPAPER)
        msgs = [r.message for r in caplog.records]
        assert any("no-LLM-numbers rule" in m for m in msgs)
        assert any("sign_off_date" in m for m in msgs)

    def test_unknown_workpaper_passes_all_through_with_warning(self, caplog):
        # If the registry is missing, we cannot filter — pass all through
        # but log a warning. __post_init__ still enforces the rule at
        # construction time if a forbidden type is later built.
        with caplog.at_level("WARNING"):
            allowed, dropped = _filter_llm_allowed(
                ["sign_off_date", "organization_name"], "NONEXISTENT-WP",
            )
        assert set(allowed) == {"sign_off_date", "organization_name"}
        assert dropped == []
        assert any("no registry" in r.message for r in caplog.records)


class TestExtractAllFieldsAsFacts:
    """The contract-shape fallback function with mocked Ollama."""

    # Synthetic LLM response: returns a valid value for organization_name,
    # null for the others. We never send forbidden types to the LLM because
    # the filter runs first, so the mock only sees allowed fields.
    _MOCK_OLLAMA_RESPONSE = (
        '{"organization_name": {'
        '"value": "Sample Foundation, Inc.", '
        '"confident": true, '
        '"source_hint": "Sample Foundation, Inc."}, '
        '"q1_c": {"value": null, "confident": false, "source_hint": ""}}'
    )

    @patch.object(llm_extractor, "is_available", return_value=True)
    @patch.object(llm_extractor, "_llm_chat_response_json")
    def test_returns_extracted_facts_for_found_fields(self, mock_chat, _avail):
        mock_chat.return_value = self._MOCK_OLLAMA_RESPONSE
        facts = extract_all_fields_as_facts(
            text="Document text mentioning Sample Foundation, Inc.",
            workpaper_type=WORKPAPER,
            field_ids=["organization_name", "q1_c"],
            document_path="/data/letter.pdf",
            document_type="engagement_letter",
        )
        assert "organization_name" in facts
        assert "q1_c" not in facts  # null in response → absent
        assert facts["organization_name"].value == "Sample Foundation, Inc."

    @patch.object(llm_extractor, "is_available", return_value=True)
    @patch.object(llm_extractor, "_llm_chat_response_json")
    def test_returned_facts_have_llm_extraction_method(self, mock_chat, _avail):
        mock_chat.return_value = self._MOCK_OLLAMA_RESPONSE
        facts = extract_all_fields_as_facts(
            text="x", workpaper_type=WORKPAPER,
            field_ids=["organization_name"],
            document_path="/x.pdf", document_type="engagement_letter",
        )
        assert facts["organization_name"].extractor_method == "llm_extraction"

    @patch.object(llm_extractor, "is_available", return_value=True)
    @patch.object(llm_extractor, "_llm_chat_response_json")
    def test_source_citation_uses_page_unknown_sentinel(self, mock_chat, _avail):
        mock_chat.return_value = self._MOCK_OLLAMA_RESPONSE
        facts = extract_all_fields_as_facts(
            text="x", workpaper_type=WORKPAPER,
            field_ids=["organization_name"],
            document_path="/x.pdf", document_type="engagement_letter",
        )
        src = facts["organization_name"].sources[0]
        assert src.page == PAGE_UNKNOWN
        assert src.has_char_offsets is False

    @patch.object(llm_extractor, "is_available", return_value=True)
    @patch.object(llm_extractor, "_llm_chat_response_json")
    def test_quoted_text_carries_source_hint(self, mock_chat, _avail):
        mock_chat.return_value = self._MOCK_OLLAMA_RESPONSE
        facts = extract_all_fields_as_facts(
            text="x", workpaper_type=WORKPAPER,
            field_ids=["organization_name"],
            document_path="/x.pdf", document_type="engagement_letter",
        )
        assert facts["organization_name"].sources[0].quoted_text == "Sample Foundation, Inc."

    @patch.object(llm_extractor, "is_available", return_value=True)
    @patch.object(llm_extractor, "_llm_chat_response_json")
    def test_forbidden_field_types_never_reach_llm(self, mock_chat, _avail):
        # Caller asks for a date field; pre-filter must drop it so the
        # prompt never contains it and the mock's response is irrelevant.
        mock_chat.return_value = (
            '{"organization_name": {"value": "Foo Org", "confident": true, '
            '"source_hint": "Foo Org"}}'
        )
        facts = extract_all_fields_as_facts(
            text="x", workpaper_type=WORKPAPER,
            field_ids=["organization_name", "sign_off_date"],
            document_path="/x.pdf", document_type="engagement_letter",
        )
        # sign_off_date must NOT appear in returned facts
        assert "sign_off_date" not in facts
        # Verify the prompt that was actually sent excludes sign_off_date
        sent_prompt = mock_chat.call_args[0][0]  # first positional arg
        assert "sign_off_date" not in sent_prompt
        assert "organization_name" in sent_prompt

    @patch.object(llm_extractor, "is_available", return_value=True)
    @patch.object(llm_extractor, "_llm_chat_response_json")
    def test_empty_after_filter_returns_empty_dict(self, mock_chat, _avail):
        # All requested fields are forbidden → no LLM call at all
        facts = extract_all_fields_as_facts(
            text="x", workpaper_type=WORKPAPER,
            field_ids=["sign_off_date", "financial_position_date"],
            document_path="/x.pdf", document_type="engagement_letter",
        )
        assert facts == {}
        mock_chat.assert_not_called()

    @patch.object(llm_extractor, "is_available", return_value=False)
    def test_ollama_unavailable_returns_empty(self, _avail):
        facts = extract_all_fields_as_facts(
            text="x", workpaper_type=WORKPAPER,
            field_ids=["organization_name"],
            document_path="/x.pdf", document_type="engagement_letter",
        )
        assert facts == {}

    def test_empty_text_returns_empty(self):
        # Should short-circuit before any Ollama check
        facts = extract_all_fields_as_facts(
            text="",
            workpaper_type=WORKPAPER,
            field_ids=["organization_name"],
            document_path="/x.pdf", document_type="engagement_letter",
        )
        assert facts == {}

    @patch.object(llm_extractor, "is_available", return_value=True)
    @patch.object(llm_extractor, "_llm_chat_response_json")
    def test_value_with_no_source_hint_still_constructs_fact(
        self, mock_chat, _avail,
    ):
        mock_chat.return_value = (
            '{"organization_name": {"value": "X Co", "confident": true, '
            '"source_hint": ""}}'
        )
        facts = extract_all_fields_as_facts(
            text="x", workpaper_type=WORKPAPER,
            field_ids=["organization_name"],
            document_path="/x.pdf", document_type="engagement_letter",
        )
        assert facts["organization_name"].value == "X Co"
        assert facts["organization_name"].sources[0].quoted_text == ""
        assert "no source_hint" in facts["organization_name"].notes


class TestExtractFieldsAsFactsTiebreaker:
    """The tiebreaker variant — same rule enforcement, smaller field set."""

    @patch.object(llm_extractor, "is_available", return_value=True)
    @patch.object(llm_extractor, "_llm_chat_response_json")
    def test_tiebreaker_respects_no_llm_rule(self, mock_chat, _avail):
        mock_chat.return_value = (
            '{"q1_c": {"value": "true", "confident": true, '
            '"source_hint": "Single Audit"}}'
        )
        facts = extract_fields_as_facts(
            text="Single Audit applies.",
            workpaper_type=WORKPAPER,
            fields_to_resolve=["sign_off_date", "q1_c"],
            document_path="/x.pdf", document_type="engagement_letter",
        )
        assert "sign_off_date" not in facts  # filtered out
        assert "q1_c" in facts
        sent_prompt = mock_chat.call_args[0][0]
        assert "sign_off_date" not in sent_prompt

    def test_empty_fields_returns_empty(self):
        facts = extract_fields_as_facts(
            text="x", workpaper_type=WORKPAPER,
            fields_to_resolve=[],
            document_path="/x.pdf", document_type="engagement_letter",
        )
        assert facts == {}


class TestNoLLMRuleDefenseInDepth:
    """If a forbidden type somehow bypasses the pre-filter, ExtractedFact
    construction must still catch it. This is the structural enforcement
    that makes the rule existential rather than defensive."""

    def test_forging_a_forbidden_fact_raises(self):
        from auditai_data_normalization.generation_contract import (
            CHAR_OFFSET_UNAVAILABLE, PAGE_UNKNOWN, ExtractedFact, SourceCitation,
        )
        citation = SourceCitation(
            document_path="/x.pdf", document_type="engagement_letter",
            page=PAGE_UNKNOWN,
            char_start=CHAR_OFFSET_UNAVAILABLE, char_end=CHAR_OFFSET_UNAVAILABLE,
            quoted_text="2024-06-30",
        )
        with pytest.raises(ValueError, match="Numerical accuracy rule"):
            ExtractedFact(
                field_id="sign_off_date",
                field_type="date",  # forbidden for LLM
                value="2024-06-30",
                confidence=0.7,
                sources=[citation],
                extractor_method="llm_extraction",  # ← rule violation
            )


# ---------------------------------------------------------------------
# Char-level provenance — structural_extractor + adapter propagation
# ---------------------------------------------------------------------

from auditai_data_normalization.extractors.structural_extractor import (
    _locate_value_in_page,
)


class TestLocateValueInPage:
    """Unit tests for the char-level locator helper."""

    def test_finds_value_with_context(self):
        page = (
            "INDEPENDENT AUDITOR'S REPORT\n"
            "To the Board of Directors of Sample Foundation, Inc.\n"
            "San Francisco, California\n"
        )
        start, end, quoted = _locate_value_in_page(page, "Sample Foundation, Inc.")
        assert start > 0
        assert end == start + len("Sample Foundation, Inc.")
        assert page[start:end] == "Sample Foundation, Inc."
        assert "Sample Foundation, Inc." in quoted
        # Context: "To the Board of Directors of" before, nothing after on this line
        assert "Board" in quoted or "Directors" in quoted

    def test_returns_sentinel_when_not_found(self):
        from auditai_data_normalization.extractors.structural_extractor import (
            CHAR_OFFSET_UNAVAILABLE,
        )
        start, end, quoted = _locate_value_in_page(
            "Some unrelated page text", "Nonexistent Value",
        )
        assert start == CHAR_OFFSET_UNAVAILABLE
        assert end == CHAR_OFFSET_UNAVAILABLE
        assert quoted == "Nonexistent Value"

    def test_clips_context_at_line_boundary(self):
        # Value at start of line — should not pull context from previous line
        page = "First line that is quite long\nSecond line: Sample Org Inc.\nThird line"
        start, end, quoted = _locate_value_in_page(page, "Sample Org Inc.")
        # The context should NOT include "First line..." (previous line) or "Third line"
        assert "First line" not in quoted
        assert "Third line" not in quoted
        assert "Sample Org Inc." in quoted

    def test_empty_inputs_handled(self):
        from auditai_data_normalization.extractors.structural_extractor import (
            CHAR_OFFSET_UNAVAILABLE,
        )
        # Empty value
        s, e, q = _locate_value_in_page("page text", "")
        assert s == CHAR_OFFSET_UNAVAILABLE
        # Empty page
        s, e, q = _locate_value_in_page("", "value")
        assert s == CHAR_OFFSET_UNAVAILABLE

    def test_first_occurrence_wins_on_duplicates(self):
        page = "Sample appears here. And Sample appears again here."
        start, end, _ = _locate_value_in_page(page, "Sample")
        # Should be the first occurrence (position 0)
        assert start == 0
        assert end == len("Sample")


class TestFieldEvidenceCharFields:
    """FieldEvidence now carries optional char-level provenance."""

    def test_default_construction_uses_sentinel(self):
        ev = FieldEvidence(
            value="X", confidence=0.5, source_page=1, method="salutation_block",
        )
        assert ev.has_char_offsets is False
        assert ev.char_start == -1
        assert ev.char_end == -1
        assert ev.full_quoted_text == ""

    def test_construction_with_char_fields(self):
        ev = FieldEvidence(
            value="Sample Foundation, Inc.",
            confidence=0.85,
            source_page=2,
            method="salutation_block",
            anchor="INDEPENDENT AUDITOR'S REPORT",
            char_start=42,
            char_end=66,
            full_quoted_text="Directors of Sample Foundation, Inc.",
        )
        assert ev.has_char_offsets is True
        assert ev.char_start == 42
        assert ev.char_end == 66
        assert "Sample Foundation" in ev.full_quoted_text

    def test_repr_shows_char_offsets_when_present(self):
        ev = FieldEvidence(
            value="X", confidence=0.5, source_page=1, method="salutation_block",
            char_start=10, char_end=11, full_quoted_text="X",
        )
        r = repr(ev)
        assert "chars[10:11]" in r

    def test_repr_shows_page_only_when_sentinel(self):
        ev = FieldEvidence(
            value="X", confidence=0.5, source_page=1, method="salutation_block",
        )
        assert "page-only" in repr(ev)


class TestAdapterCharLevelPropagation:
    """The adapter must propagate real char offsets when present and
    fall back to sentinels (with anchor as quoted_text) otherwise."""

    def test_char_offsets_propagate_through_adapter(self):
        ev = FieldEvidence(
            value="Sample Foundation, Inc.",
            confidence=0.85,
            source_page=2,
            method="salutation_block",
            anchor="INDEPENDENT AUDITOR'S REPORT",
            char_start=42,
            char_end=66,
            full_quoted_text="Directors of Sample Foundation, Inc.",
        )
        fact = field_evidence_to_extracted_fact(
            evidence=ev,
            field_id="organization_name",
            workpaper_type=WORKPAPER,
            document_path="/data/audit_report.pdf",
            document_type="audit_report",
        )
        src = fact.sources[0]
        assert src.has_char_offsets is True
        assert src.char_start == 42
        assert src.char_end == 66
        assert src.quoted_text == "Directors of Sample Foundation, Inc."
        assert "char-level provenance" in fact.notes

    def test_sentinel_offsets_fall_back_to_anchor(self):
        # FieldEvidence WITHOUT char_start/char_end (legacy path)
        ev = FieldEvidence(
            value="Sample Org",
            confidence=0.7,
            source_page=1,
            method="salutation_block",
            anchor="INDEPENDENT AUDITOR'S REPORT line",
        )
        fact = field_evidence_to_extracted_fact(
            evidence=ev,
            field_id="organization_name",
            workpaper_type=WORKPAPER,
            document_path="/data/x.pdf",
            document_type="audit_report",
        )
        src = fact.sources[0]
        assert src.has_char_offsets is False
        assert src.quoted_text == "INDEPENDENT AUDITOR'S REPORT line"
        assert "page-level" in fact.notes

    def test_empty_full_quoted_text_falls_back_to_anchor(self):
        # has_char_offsets=True but full_quoted_text is empty —
        # adapter should fall back to anchor as quoted_text
        ev = FieldEvidence(
            value="X", confidence=0.6, source_page=1, method="salutation_block",
            anchor="anchor text",
            char_start=5, char_end=6, full_quoted_text="",
        )
        fact = field_evidence_to_extracted_fact(
            evidence=ev,
            field_id="organization_name",
            workpaper_type=WORKPAPER,
            document_path="/x.pdf",
            document_type="audit_report",
        )
        # char offsets stay real; quoted_text falls back to anchor
        assert fact.sources[0].has_char_offsets is True
        assert fact.sources[0].quoted_text == "anchor text"
