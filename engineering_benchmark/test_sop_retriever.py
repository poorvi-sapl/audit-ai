"""
engineering_benchmark/test_sop_retriever.py
=============================================
Phase 1B foundation tests — locks in SOP retrieval, query embedding,
and the assembly-layer wrapper as regression-protected before any
real Qdrant data lands.

All tests mock the Qdrant client and the embedding model so they
run offline (no GPU, no Qdrant service required).

Run with:
    pytest engineering_benchmark/test_sop_retriever.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engineering_benchmark import sop_retriever
from engineering_benchmark.sop_chunker import SOPChunk
from engineering_benchmark.sop_retriever import (
    WORKPAPER_TYPE_TO_WORKFLOW,
    _build_filter,
    retrieve_sop_chunks,
    workflow_for,
)


# ---------------------------------------------------------------------
# Workpaper-type → workflow lookup
# ---------------------------------------------------------------------

class TestWorkflowLookup:

    def test_npo_cx_1_1_maps_to_engagement_acceptance(self):
        assert workflow_for("NPO-CX-1.1") == "engagement_acceptance"

    def test_all_four_engagement_variants_share_workflow(self):
        wfs = {
            workflow_for("NPO-CX-1.1"),
            workflow_for("GOV-CX-1.1"),
            workflow_for("FP-CX-1.1"),
            workflow_for("TRB-CX-1.1"),
        }
        assert wfs == {"engagement_acceptance"}

    def test_unknown_workpaper_type_raises_keyerror(self):
        with pytest.raises(KeyError, match="WORKPAPER_TYPE_TO_WORKFLOW"):
            workflow_for("UNKNOWN-WP-9.9")


# ---------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------

class TestBuildFilter:

    def test_filter_has_workpaper_type_must_condition(self):
        f = _build_filter("engagement_acceptance", "NPO-CX-1.1")
        # must clause includes workpaper_type
        keys_in_must = [c.key for c in f.must if hasattr(c, "key")]
        assert "workpaper_type" in keys_in_must

    def test_filter_adds_version_when_provided(self):
        f = _build_filter("engagement_acceptance", "NPO-CX-1.1",
                          sop_version="2024-Q1")
        keys_in_must = [c.key for c in f.must if hasattr(c, "key")]
        assert "sop_version" in keys_in_must

    def test_filter_omits_version_when_none(self):
        f = _build_filter("engagement_acceptance", "NPO-CX-1.1",
                          sop_version=None)
        keys_in_must = [c.key for c in f.must if hasattr(c, "key")]
        assert "sop_version" not in keys_in_must

    def test_filter_should_clause_handles_workpaper_ids_match_or_empty(self):
        f = _build_filter("engagement_acceptance", "NPO-CX-1.1")
        # should clause has 2 entries: IsEmpty(workpaper_ids) and
        # MatchAny on workpaper_ids
        assert len(f.should) == 2


# ---------------------------------------------------------------------
# retrieve_sop_chunks — happy path + edge cases (Qdrant mocked)
# ---------------------------------------------------------------------

def _mock_hit(content: str, score: float = 0.9):
    hit = MagicMock()
    hit.payload = {"content": content, "score": score}
    return hit


class TestRetrieveSOPChunks:

    @patch("engineering_benchmark.sop_retriever.embed_query")
    def test_returns_chunk_contents_in_order(self, mock_embed):
        mock_embed.return_value = [0.1] * 4096
        mock_client = MagicMock()
        mock_client.search.return_value = [
            _mock_hit("SOP §3.1 — Engagement acceptance criteria…"),
            _mock_hit("SOP §3.2 — Non-attest services restrictions…"),
        ]
        result = retrieve_sop_chunks(
            workpaper_type="NPO-CX-1.1",
            query="engagement acceptance for nonprofits",
            top_k=5,
            qdrant_client=mock_client,
        )
        assert len(result) == 2
        assert result[0].startswith("SOP §3.1")
        assert result[1].startswith("SOP §3.2")

    @patch("engineering_benchmark.sop_retriever.embed_query")
    def test_calls_qdrant_with_correct_collection_and_top_k(self, mock_embed):
        mock_embed.return_value = [0.1] * 4096
        mock_client = MagicMock()
        mock_client.search.return_value = []
        retrieve_sop_chunks(
            workpaper_type="NPO-CX-1.1",
            query="test",
            top_k=7,
            qdrant_client=mock_client,
        )
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["limit"] == 7
        assert call_kwargs["collection_name"]  # uses settings

    @patch("engineering_benchmark.sop_retriever.embed_query")
    def test_filter_carries_workflow_not_workpaper_id(self, mock_embed):
        mock_embed.return_value = [0.1] * 4096
        mock_client = MagicMock()
        mock_client.search.return_value = []
        retrieve_sop_chunks(
            workpaper_type="NPO-CX-1.1",
            query="x",
            qdrant_client=mock_client,
        )
        filter_obj = mock_client.search.call_args.kwargs["query_filter"]
        # The must clause should reference workflow ("engagement_acceptance")
        # not workpaper_id ("NPO-CX-1.1") as the workpaper_type filter value
        wp_type_conditions = [
            c for c in filter_obj.must
            if hasattr(c, "key") and c.key == "workpaper_type"
        ]
        assert len(wp_type_conditions) == 1
        assert wp_type_conditions[0].match.value == "engagement_acceptance"

    @patch("engineering_benchmark.sop_retriever.embed_query")
    def test_unknown_workpaper_type_raises(self, mock_embed):
        with pytest.raises(KeyError):
            retrieve_sop_chunks(
                workpaper_type="UNKNOWN-XYZ",
                query="test",
                qdrant_client=MagicMock(),
            )

    @patch("engineering_benchmark.sop_retriever.embed_query")
    def test_skips_hits_with_empty_content(self, mock_embed):
        mock_embed.return_value = [0.1] * 4096
        mock_client = MagicMock()
        mock_client.search.return_value = [
            _mock_hit("real content here"),
            _mock_hit(""),  # empty payload content — should be skipped
        ]
        result = retrieve_sop_chunks(
            workpaper_type="NPO-CX-1.1",
            query="x",
            qdrant_client=mock_client,
        )
        assert result == ["real content here"]

    @patch("engineering_benchmark.sop_retriever.embed_query")
    def test_embed_query_failure_returns_empty(self, mock_embed):
        mock_embed.side_effect = RuntimeError("model load failed")
        mock_client = MagicMock()
        result = retrieve_sop_chunks(
            workpaper_type="NPO-CX-1.1",
            query="x",
            qdrant_client=mock_client,
        )
        assert result == []
        # Qdrant should NOT have been called if query embedding failed
        mock_client.search.assert_not_called()

    @patch("engineering_benchmark.sop_retriever.embed_query")
    def test_qdrant_search_failure_returns_empty(self, mock_embed):
        mock_embed.return_value = [0.1] * 4096
        mock_client = MagicMock()
        mock_client.search.side_effect = ConnectionError("qdrant down")
        result = retrieve_sop_chunks(
            workpaper_type="NPO-CX-1.1",
            query="x",
            qdrant_client=mock_client,
        )
        assert result == []


# ---------------------------------------------------------------------
# embed_query — uses the e5-mistral instruction prefix
# ---------------------------------------------------------------------

class TestEmbedQueryPrefix:

    def test_prefix_constant_starts_with_instruct(self):
        from engineering_benchmark.embedder import E5_QUERY_INSTRUCTION
        assert E5_QUERY_INSTRUCTION.startswith("Instruct:")
        assert E5_QUERY_INSTRUCTION.endswith("Query: ")

    def test_embed_query_rejects_empty(self):
        from engineering_benchmark.embedder import embed_query
        with pytest.raises(ValueError, match="empty"):
            embed_query("")
        with pytest.raises(ValueError, match="empty"):
            embed_query("   ")

    @patch("engineering_benchmark.embedder._embed_batch")
    def test_embed_query_prepends_instruction_prefix(self, mock_batch):
        from engineering_benchmark.embedder import (
            E5_QUERY_INSTRUCTION, embed_query,
        )
        mock_batch.return_value = [[0.1] * 4096]
        embed_query("How do I assess engagement acceptance?")
        sent = mock_batch.call_args[0][0]
        assert len(sent) == 1
        assert sent[0].startswith(E5_QUERY_INSTRUCTION)
        assert "How do I assess" in sent[0]


# ---------------------------------------------------------------------
# SOPChunk gains workpaper_ids field
# ---------------------------------------------------------------------

class TestSOPChunkWorkpaperIds:

    def test_default_is_empty_list(self):
        c = SOPChunk(
            chunk_id="abc", source_doc="x.pdf", sop_version="v1",
            content="...", section_prefix="", char_start=0, char_end=10,
            token_count=2,
        )
        assert c.workpaper_ids == []

    def test_can_set_workpaper_ids(self):
        c = SOPChunk(
            chunk_id="abc", source_doc="x.pdf", sop_version="v1",
            content="...", section_prefix="", char_start=0, char_end=10,
            token_count=2, workpaper_type="engagement_acceptance",
            workpaper_ids=["NPO-CX-1.1"],
        )
        assert c.workpaper_ids == ["NPO-CX-1.1"]

    def test_workpaper_ids_independent_per_chunk(self):
        c1 = SOPChunk(
            chunk_id="a", source_doc="x", sop_version="v1",
            content="...", section_prefix="", char_start=0,
            char_end=10, token_count=1,
        )
        c2 = SOPChunk(
            chunk_id="b", source_doc="x", sop_version="v1",
            content="...", section_prefix="", char_start=10,
            char_end=20, token_count=1,
        )
        c1.workpaper_ids.append("NPO-CX-1.1")
        assert c2.workpaper_ids == []  # not aliased


# ---------------------------------------------------------------------
# Assembly-layer wrapper integrates retrieval cleanly
# ---------------------------------------------------------------------

class TestAssemblyLayerSOPRetrievalWrapper:

    @patch("engineering_benchmark.sop_retriever.retrieve_sop_chunks")
    def test_wrapper_retrieves_and_populates_sop_chunks(self, mock_retrieve):
        from auditai_data_normalization.assembly_layer import (
            assemble_generation_input_with_sop_retrieval,
        )
        mock_retrieve.return_value = [
            "SOP §3.1 …", "SOP §3.2 …", "SOP §3.3 …",
        ]
        gen_input = assemble_generation_input_with_sop_retrieval(
            workpaper_type="NPO-CX-1.1",
            engagement_id="ENG-TEST-001",
            source_extractions=[],
        )
        assert gen_input.workpaper_type == "NPO-CX-1.1"
        assert gen_input.engagement_id == "ENG-TEST-001"
        assert len(gen_input.sop_chunks) == 3
        assert gen_input.sop_chunks[0].startswith("SOP §3.1")

    @patch("engineering_benchmark.sop_retriever.retrieve_sop_chunks")
    def test_wrapper_uses_default_query_when_none_provided(self, mock_retrieve):
        from auditai_data_normalization.assembly_layer import (
            assemble_generation_input_with_sop_retrieval,
        )
        mock_retrieve.return_value = []
        assemble_generation_input_with_sop_retrieval(
            workpaper_type="NPO-CX-1.1",
            engagement_id="ENG-TEST-002",
            source_extractions=[],
        )
        query_arg = mock_retrieve.call_args.kwargs["query"]
        assert "NPO-CX-1.1" in query_arg
        assert "Fields to reason about" in query_arg

    @patch("engineering_benchmark.sop_retriever.retrieve_sop_chunks")
    def test_wrapper_passes_explicit_query_through(self, mock_retrieve):
        from auditai_data_normalization.assembly_layer import (
            assemble_generation_input_with_sop_retrieval,
        )
        mock_retrieve.return_value = []
        assemble_generation_input_with_sop_retrieval(
            workpaper_type="NPO-CX-1.1",
            engagement_id="ENG-TEST-003",
            source_extractions=[],
            sop_query="custom query about Yellow Book independence",
        )
        assert mock_retrieve.call_args.kwargs["query"] == (
            "custom query about Yellow Book independence"
        )

    @patch("engineering_benchmark.sop_retriever.retrieve_sop_chunks")
    def test_wrapper_degrades_gracefully_on_unknown_workpaper(
        self, mock_retrieve,
    ):
        # If the retriever raises KeyError (unknown workpaper),
        # the wrapper should return GenerationInput with empty sop_chunks
        # rather than crashing.
        from auditai_data_normalization.assembly_layer import (
            assemble_generation_input_with_sop_retrieval,
        )
        mock_retrieve.side_effect = KeyError("unknown wp")
        gen_input = assemble_generation_input_with_sop_retrieval(
            workpaper_type="NPO-CX-1.1",
            engagement_id="ENG-TEST-004",
            source_extractions=[],
        )
        assert gen_input.sop_chunks == []

    @patch("engineering_benchmark.sop_retriever.retrieve_sop_chunks")
    def test_wrapper_passes_sop_version_through(self, mock_retrieve):
        from auditai_data_normalization.assembly_layer import (
            assemble_generation_input_with_sop_retrieval,
        )
        mock_retrieve.return_value = []
        assemble_generation_input_with_sop_retrieval(
            workpaper_type="NPO-CX-1.1",
            engagement_id="ENG-TEST-005",
            source_extractions=[],
            sop_version="2024-Q1",
        )
        assert mock_retrieve.call_args.kwargs["sop_version"] == "2024-Q1"
