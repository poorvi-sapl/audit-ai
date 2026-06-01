"""
raw_to_training_pair/generation/batch_orchestrator.py
=======================================================
Walk a tree of engagement folders, produce one generation pair per
engagement, and append each to the training-corpus JSONL.

Folder layout (default convention)
----------------------------------
    engagements_root/
        ENG-001/
            engagement_letter.pdf
            prior_year.pdf
            filled_npo_cx_1_1.docx       ← gold (matches default pattern)
        ENG-002/
            ...

The gold workpaper is identified by a filename glob pattern
(`gold_filename_pattern`, default "*filled*.docx"). All other files
in the engagement folder are treated as source documents.

Per-engagement failures (extractor errors, loader errors, gold not
found, etc.) are caught, logged, and recorded in BatchResult.errors
so a single bad engagement doesn't tank the whole batch.

Public API
----------
    run_batch_from_folder(engagements_root, output_path, ...)
        → BatchResult
    BatchResult                  — counts + per-engagement errors
    EngagementBuildResult        — per-engagement outcome record
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from raw_to_training_pair.generation.citation_linker import (
    auto_link_citations,
)
from raw_to_training_pair.generation.engagement_ingest import (
    ingest_engagement_folder,
)
from raw_to_training_pair.generation.gold_loader import (
    load_filled_workpaper,
)
from raw_to_training_pair.generation.orchestrator import (
    build_generation_pair_from_extractions,
)
from raw_to_training_pair.jsonl_writer import append as jsonl_append

logger = logging.getLogger(__name__)


@dataclass
class EngagementBuildResult:
    """Outcome of building one engagement's generation pair."""
    engagement_id: str
    success: bool
    pair_hash: str = ""
    written: bool = False     # False if dedup skipped the write
    error: str = ""


@dataclass
class BatchResult:
    """Aggregate outcome of a run_batch_from_folder call."""
    total_engagements: int = 0
    pairs_built: int = 0           # build_generation_pair succeeded
    pairs_written: int = 0         # jsonl_append wrote (not deduplicated)
    pairs_deduplicated: int = 0    # build succeeded but write was a dup
    errors: list[str] = field(default_factory=list)
    per_engagement: list[EngagementBuildResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"BatchResult: total={self.total_engagements} "
            f"built={self.pairs_built} written={self.pairs_written} "
            f"deduplicated={self.pairs_deduplicated} "
            f"errors={len(self.errors)}"
        )


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _find_gold_workpaper(
    engagement_folder: Path,
    gold_filename_pattern: str,
) -> Path | None:
    """Find the gold workpaper file in an engagement folder.

    Looks for a single .docx matching `gold_filename_pattern`. Returns
    None if zero matches found; raises ValueError if more than one
    matches (ambiguous engagement layout).
    """
    candidates = sorted(engagement_folder.glob(gold_filename_pattern))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            f"_find_gold_workpaper: multiple files matched "
            f"{gold_filename_pattern!r} in {engagement_folder}: "
            f"{[c.name for c in candidates]}. Tighten the pattern or "
            "rearrange the folder so exactly one gold .docx is present."
        )
    return candidates[0]


def _ingest_sources_excluding(
    engagement_folder: Path,
    exclude: Path,
    extractor_version: str,
) -> list:
    """Run ingest_engagement_folder on the engagement folder, then
    filter out the gold workpaper (which would otherwise be ingested
    as a source document)."""
    extractions = ingest_engagement_folder(
        engagement_folder, extractor_version=extractor_version,
    )
    return [e for e in extractions if Path(e.document_path) != exclude]


def _build_one_engagement(
    engagement_folder: Path,
    workpaper_type: str,
    output_path: Path,
    gold_filename_pattern: str,
    with_sop_retrieval: bool,
    pii_strict: bool,
    block_on_schema_issues: bool,
    extractor_version: str,
    sop_top_k: int,
    sop_version: str | None,
    engagement_id_override: str | None,
) -> EngagementBuildResult:
    """End-to-end build for one engagement. Returns the outcome record."""
    engagement_id = engagement_id_override or engagement_folder.name

    try:
        gold_path = _find_gold_workpaper(engagement_folder, gold_filename_pattern)
        if gold_path is None:
            return EngagementBuildResult(
                engagement_id=engagement_id,
                success=False,
                error=f"no gold workpaper found matching {gold_filename_pattern!r}",
            )

        source_extractions = _ingest_sources_excluding(
            engagement_folder, gold_path, extractor_version,
        )

        gold = load_filled_workpaper(
            gold_path, workpaper_type=workpaper_type,
            engagement_id=engagement_id,
        )

        # Quick-and-cheap linker pass: we need an ExtractedFact dict
        # keyed by field_id. assembly_layer.merge_facts handles this
        # internally, but for linker convenience we re-run a lightweight
        # version here. Simplest: build the GenerationInput first via the
        # orchestrator, then read back its facts dict for the linker.
        pair = build_generation_pair_from_extractions(
            workpaper_type=workpaper_type,
            engagement_id=engagement_id,
            source_extractions=source_extractions,
            gold=gold,                       # un-linked yet
            with_sop_retrieval=with_sop_retrieval,
            sop_top_k=sop_top_k,
            sop_version=sop_version,
            block_on_schema_issues=block_on_schema_issues,
            extra_metadata={"batch_run": True},
            # PII enforcement applied separately below so we can
            # auto-link first and re-check.
            # (build_generation_pair defaults pii_strict=False.)
        )

        # Auto-link citations: re-render the assistant content with
        # citations populated from matching extracted facts. We do this
        # AFTER the first build so we have access to the assembled
        # gen_input's facts dict (via the pair metadata).
        # Reconstruct extracted_facts from metadata.extracted_facts_summary
        # — close enough for citation-value matching.
        from auditai_data_normalization.generation_contract import (
            CHAR_OFFSET_UNAVAILABLE, ExtractedFact, SourceCitation,
        )
        from auditai_data_normalization.field_type_registry import (
            get_field_spec,
        )
        from raw_to_training_pair.generation.pair_builder import (
            build_generation_pair,
        )
        from raw_to_training_pair.generation.target_schema import (
            to_json_string,
        )
        import json

        summary = pair["metadata"].get("extracted_facts_summary", {})
        facts_for_link: dict[str, ExtractedFact] = {}
        for fid, entry in summary.items():
            try:
                spec = get_field_spec(workpaper_type, fid)
            except KeyError:
                continue
            sources = [
                SourceCitation(
                    document_path="",                # not preserved in summary
                    document_type=s.get("document_type", "unknown"),
                    page=int(s.get("page", 0)),
                    char_start=CHAR_OFFSET_UNAVAILABLE,
                    char_end=CHAR_OFFSET_UNAVAILABLE,
                    quoted_text=s.get("quoted_text", ""),
                )
                for s in entry.get("sources", [])
            ]
            facts_for_link[fid] = ExtractedFact(
                field_id=fid,
                field_type=spec.field_type,
                value=entry.get("value"),
                confidence=float(entry.get("confidence", 0.0)),
                sources=sources,
                extractor_method=entry.get("extractor_method", "structural_heuristic"),
            )

        linked_gold = auto_link_citations(gold, facts_for_link)

        # Rebuild the pair with the linked gold, this time with PII
        # enforcement honoring the caller's pii_strict setting and
        # any block_on_schema_issues.
        # Reconstruct the same GenerationInput via the orchestrator
        # for consistency with the linker's view of facts.
        pair = build_generation_pair_from_extractions(
            workpaper_type=workpaper_type,
            engagement_id=engagement_id,
            source_extractions=source_extractions,
            gold=linked_gold,
            with_sop_retrieval=with_sop_retrieval,
            sop_top_k=sop_top_k,
            sop_version=sop_version,
            block_on_schema_issues=block_on_schema_issues,
            extra_metadata={
                "batch_run": True,
                "engagement_folder": str(engagement_folder),
                "gold_workpaper_path": str(gold_path),
            },
        )

        # Final PII strict pass if requested — defer to build_generation_pair
        # which raises ValueError on PII when pii_strict=True.
        # We've already built the pair without strict checking, so do a
        # final check by re-invoking the builder with pii_strict.
        # NOTE: This is intentionally separate from build above so the
        # citation-linking pass can use the un-strict path.
        if pii_strict:
            from auditai_data_normalization.assembly_layer import (
                assemble_generation_input,
            )
            # Recompute gen_input cheaply (no SOP retrieval re-run)
            sop_chunks = pair["metadata"].get("sop_chunks_count")
            # Use the assembled chunks from the first pass
            assistant_msg = next(
                (m["content"] for m in pair["messages"] if m["role"] == "assistant"),
                "",
            )
            # We don't have direct access to the sop_chunks list from
            # metadata (it stores count only). For pii_strict we
            # re-assemble without SOP retrieval (worst case: empty SOPs
            # in the strict-check pair). The check is what matters; the
            # final written pair came from the first build above.
            gen_input_check = assemble_generation_input(
                workpaper_type=workpaper_type,
                engagement_id=engagement_id,
                source_extractions=source_extractions,
                sop_chunks=[],
            )
            try:
                build_generation_pair(
                    gen_input=gen_input_check,
                    gold=linked_gold,
                    pii_strict=True,
                    block_on_schema_issues=False,
                )
            except ValueError as e:
                return EngagementBuildResult(
                    engagement_id=engagement_id,
                    success=False,
                    error=f"pii_strict check failed: {e}",
                )

        # Write
        result = jsonl_append(pair, output_path)
        pair_hash = pair["metadata"].get("pair_hash", "")
        return EngagementBuildResult(
            engagement_id=engagement_id,
            success=True,
            pair_hash=pair_hash,
            written=result.written,
            error="" if result.written else (result.reason or ""),
        )

    except Exception as e:
        logger.warning(
            "batch_orchestrator: engagement %s failed — %s",
            engagement_id, e,
        )
        return EngagementBuildResult(
            engagement_id=engagement_id,
            success=False,
            error=str(e),
        )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_batch_from_folder(
    engagements_root: str | Path,
    output_path: str | Path,
    workpaper_type: str = "NPO-CX-1.1",
    gold_filename_pattern: str = "*filled*.docx",
    with_sop_retrieval: bool = False,
    pii_strict: bool = False,
    block_on_schema_issues: bool = False,
    extractor_version: str = "",
    sop_top_k: int = 10,
    sop_version: str | None = None,
) -> BatchResult:
    """Walk an engagements root folder and produce one generation pair
    per subdirectory.

    Args:
        engagements_root: parent folder containing engagement subfolders
        output_path: JSONL file to append produced pairs to
        workpaper_type: registry workpaper type (e.g., "NPO-CX-1.1")
        gold_filename_pattern: glob pattern identifying the gold .docx
            within each engagement folder (default "*filled*.docx")
        with_sop_retrieval: pass through to the orchestrator
        pii_strict: if True, refuse to write pairs with detected PII
        block_on_schema_issues: if True, refuse pairs failing
            target-schema validation against the registry
        extractor_version, sop_top_k, sop_version: forwarded to the
            assembly + retrieval stack

    Returns:
        BatchResult with aggregate counts and per-engagement outcomes.

    Notes:
        - Engagement IDs are taken from the subfolder name by default.
        - Subfolders starting with "." or "_" are skipped (treat as
          scratch / non-engagement).
        - Files at the top level of `engagements_root` (not in a
          subdirectory) are ignored.
    """
    root = Path(engagements_root)
    if not root.exists():
        raise FileNotFoundError(
            f"run_batch_from_folder: {root} does not exist"
        )
    if not root.is_dir():
        raise NotADirectoryError(
            f"run_batch_from_folder: {root} is not a directory"
        )

    out = Path(output_path)
    result = BatchResult()

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.startswith("_"):
            continue

        result.total_engagements += 1

        per_eng = _build_one_engagement(
            engagement_folder=child,
            workpaper_type=workpaper_type,
            output_path=out,
            gold_filename_pattern=gold_filename_pattern,
            with_sop_retrieval=with_sop_retrieval,
            pii_strict=pii_strict,
            block_on_schema_issues=block_on_schema_issues,
            extractor_version=extractor_version,
            sop_top_k=sop_top_k,
            sop_version=sop_version,
            engagement_id_override=None,
        )
        result.per_engagement.append(per_eng)

        if per_eng.success:
            result.pairs_built += 1
            if per_eng.written:
                result.pairs_written += 1
            else:
                result.pairs_deduplicated += 1
        else:
            result.errors.append(f"{per_eng.engagement_id}: {per_eng.error}")

    logger.info("batch_orchestrator: %s", result.summary())
    return result
