"""
pipeline/pipeline.py
=====================
Main orchestration entry point for the AuditAI training pair pipeline.

Wires Phase 1 → Phase 3 → Phase 2 into a single call:

    from pipeline.pipeline import process_workpaper
    result = process_workpaper("path/to/workpaper.docx")

Full flow
---------
1. Phase 1: normalize_document()       — extract, PII scrub, confidence score
2. Confidence gate                     — skip if below threshold
3. Phase 3: qdrant_retriever.retrieve() — hybrid SOP retrieval
4. Phase 2: pair_builder.build()       — assemble training pair (× variants)
5. Phase 2: quality_gates.check()      — enforce 4 gates
6. Phase 2: auditor_review.enqueue()   — send to review queue
7. (post-approval) jsonl_writer.append() — write approved pairs to JSONL

Variant generation
------------------
Per roadmap: 1 clean + 2 deficient variants × 4 client_type combos.
process_workpaper() generates all variants in one call.
Each variant goes through quality gates independently.

Public API
----------
    process_workpaper(file_path, sop_version, client_types,
                      queue_path, data_dir) -> PipelineResult

    PipelineResult
        .record          DocumentRecord
        .pairs_built     int
        .pairs_queued    int
        .pairs_failed    int
        .gate_failures   list[str]
        .errors          list[str]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from auditai_data_normalization.normalize import normalize_document
from auditai_data_normalization.schema import DocumentRecord
from pipeline.qdrant_retriever import retrieve, RetrievalResult
from raw_to_training_pair.pair_builder import build, SYSTEM_PROMPT as _PAIR_SYSTEM_PROMPT
from raw_to_training_pair.quality_gates import check
from raw_to_training_pair.auditor_review import enqueue
from raw_to_training_pair.jsonl_writer import append, get_output_path
from auditai_data_normalization.confidence import load_tiers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIDENCE_GATE = 0.50   # Phase A4: matches _EXTRACTION_GATE in normalize.py — quality gate (0.70) enforced by quality_gates.check() downstream

# Default client types to generate variants for (per roadmap × 4 combos)
_DEFAULT_CLIENT_TYPES = ["NPO", "Government", "For-Profit", "Tribal"]   # Bug #3: aligned with completion_drafter._CLIENT_CONTEXT keys

# Deficiency field sets replaced by deficiency_sampler.sample() —
# randomised per-document combinations instead of fixed sets.
# Kept as fallback only if sampler returns empty.
_DEFICIENCY_SETS_FALLBACK = [
    ["engagement_partner"],
    ["fiscal_year_end", "engagement_decision"],
]


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Result of processing one workpaper through the full pipeline."""

    file_name: str = ""
    record: DocumentRecord | None = None

    pairs_built:  int = 0
    pairs_queued: int = 0
    pairs_failed: int = 0

    gate_failures: list[str] = field(default_factory=list)
    errors:        list[str] = field(default_factory=list)

    skipped: bool = False
    skip_reason: str = ""

    def __str__(self) -> str:
        if self.skipped:
            return f"PipelineResult: SKIPPED — {self.skip_reason}"
        return (
            f"PipelineResult: {self.file_name} | "
            f"built={self.pairs_built} queued={self.pairs_queued} "
            f"failed={self.pairs_failed} errors={len(self.errors)}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_gagas(record: DocumentRecord) -> bool:
    """Detect if this is a GAGAS engagement from record fields."""
    fields_present = set(
        record.metadata.get("confidence_summary", {}).get("fields_present", [])
    )
    if "includes_gagas" in fields_present:
        return True
    # Fallback: keyword scan
    text = (record.cleaned_text or "").lower()
    return any(kw in text for kw in ["yellow book", "gagas", "government auditing standards"])


def _detect_single_audit(record: DocumentRecord) -> bool:
    """Detect if this is a Single Audit engagement."""
    fields_present = set(
        record.metadata.get("confidence_summary", {}).get("fields_present", [])
    )
    if "includes_single_audit" in fields_present:
        return True
    text = (record.cleaned_text or "").lower()
    return any(kw in text for kw in ["single audit", "2 cfr 200", "uniform guidance"])


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _process_single_variant(
    record: DocumentRecord,
    retrieval: RetrievalResult,
    client_type: str,
    is_gagas: bool,
    has_single_audit: bool,
    pair_type: str,
    deficiency_fields: list[str],
    queue_path: Path,
    data_dir: Path,
    use_mock: bool = False
) -> tuple[bool, str]:
    """
    Build, gate-check, and enqueue one training pair variant.

    Returns (success, failure_reason).
    """
    # Build pair
    pair = build(
        record=record,
        sop_text=retrieval.sop_text,
        sop_sections=retrieval.sop_sections,
        client_type=client_type,
        is_gagas=is_gagas,
        has_single_audit=has_single_audit,
        pair_type=pair_type,
        deficiency_fields=deficiency_fields,
        use_mock=use_mock,
        tiers=load_tiers(),
        sop_chunks=retrieval.chunks,
    )

    if pair is None:
        return False, "completion_drafter returned None (Ollama unavailable?)"

    # Determine output path for gate check
    stage = pair["metadata"].get("stage", "stage2")
    try:
        output_path = get_output_path(stage, data_dir)
    except ValueError as e:
        return False, str(e)

    # Quality gate check
    gate_result = check(pair, output_path)
    if not gate_result.passed:
        # Enqueue for review even if gates fail — auditor can correct
        enqueue(pair, queue_path)
        return False, f"gate={gate_result.failed_gate}: {gate_result.reason}"

    # Enqueue for auditor review
    enqueue(pair, queue_path)

    logger.info(
        "pipeline: queued %s pair | client=%s stage=%s gagas=%s",
        pair_type, client_type, stage, is_gagas,
    )

    return True, ""


def _process_single_variant_r7(
    record: DocumentRecord,
    retrieval: RetrievalResult,
    client_type: str,
    is_gagas: bool,
    has_single_audit: bool,
    pair_type: str,
    deficiency_fields: list[str],
    queue_path: Path,
    data_dir: Path,
) -> tuple[bool, str]:
    """
    R7 variant of _process_single_variant.

    Uses the classifier→validator→renderer pipeline instead of Gemma free-text
    generation. All narrative is rendered deterministically from field states.

    Returns (success, failure_reason).
    """
    import hashlib
    from raw_to_training_pair import field_classifier, claim_mapper, completion_renderer
    from raw_to_training_pair import completion_drafter
    from raw_to_training_pair.field_classifier import FieldClassification

    workpaper_text = record.cleaned_text or ""
    if not workpaper_text.strip():
        return False, "cleaned_text is empty — cannot classify"

    # Pass 1: constrained field classification
    classification = field_classifier.classify(workpaper_text, client_type=client_type)
    if classification is None:
        return False, "field_classifier.classify() returned None (Ollama unavailable?)"

    # For deficient pairs: force the designated fields to "absent" before Pass 2.
    # This teaches the model to flag those fields as findings even when the
    # workpaper text might contain partial evidence for them.
    if deficiency_fields:
        overridden = dict(classification.field_states)
        for f in deficiency_fields:
            if f in overridden:
                overridden[f] = "absent"
        classification = FieldClassification(
            field_states=overridden,
            canonical_fields=classification.canonical_fields,
            model=classification.model,
            absent_fields=[f for f, s in overridden.items() if s == "absent"],
            present_fields=[f for f, s in overridden.items() if s == "present"],
            uncertain_fields=[f for f, s in overridden.items() if s == "uncertain"],
        )

    # Pass 2: validate classification (keyword + embedding + Llama spot-check)
    validated, signals = claim_mapper.validate_classification(
        classification, workpaper_text, client_type
    )

    # Build compile-time SOP mapping table (cached per client_type)
    sop_table = completion_renderer.build_sop_mapping_table(client_type)
    mapping_version = completion_renderer.sop_mapping_version()

    # Deterministic render — no LLM inference
    render_result = completion_renderer.render_completion(
        validated=validated,
        sop_table=sop_table,
        client_type=client_type,
        is_gagas=is_gagas,
        has_single_audit=has_single_audit,
        mapping_version=mapping_version,
    )

    # Score and set review gate on the record
    completion_drafter.set_review_gate_r7(
        record=record,
        render_result=render_result,
        validated=validated,
        signals=signals,
        client_type=client_type,
        sop_sections=retrieval.sop_sections,
    )

    # Assemble training pair dict
    user_lines = [
        f"Client Type: {client_type}",
        f"GAGAS: {'Yes' if is_gagas else 'No'}",
        f"Single Audit: {'Yes' if has_single_audit else 'No'}",
        "",
    ]
    if pair_type == "deficient" and deficiency_fields:
        user_lines.append(
            f"[TRAINING NOTE — The following fields are absent from this workpaper: "
            f"{', '.join(deficiency_fields)}]"
        )
        user_lines.append("")
    user_lines.append(workpaper_text)
    user_content = "\n".join(user_lines)

    stage = "stage2" if record.review_confidence >= 0.70 else "stage1"

    pair_hash = hashlib.sha256(
        f"{record.file_hash}|{client_type}|{pair_type}|r7|"
        f"{','.join(sorted(deficiency_fields))}".encode()
    ).hexdigest()

    pair = {
        "messages": [
            {"role": "system",    "content": _PAIR_SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": render_result.completion},
        ],
        "metadata": {
            "file_name":             record.file_name,
            "file_type":             record.file_type,
            "client_type":           client_type,
            "is_gagas":              is_gagas,
            "has_single_audit":      has_single_audit,
            "extraction_confidence": record.extraction_confidence,
            "review_confidence":     record.review_confidence,
            "quality_gate":          record.quality_gate,
            "auditor_approved":      False,
            "pair_type":             pair_type,
            "stage":                 stage,
            "fields_missing":        render_result.absent_fields,
            "fields_present":        validated.present_fields,
            "fields_uncertain":      render_result.uncertain_fields,
            "deficiency_fields":     deficiency_fields,
            "sop_sections_used":     retrieval.sop_sections,
            "sop_unverified":        render_result.sop_unverified,
            "file_hash":             record.file_hash,
            "pair_hash":             pair_hash,
            "r7":                    True,
            "sop_mapping_version":   render_result.sop_mapping_version,
            "provisional_fields":    render_result.provisional_fields,
            "classification_signals": {
                "uncertain_rate":         signals.uncertain_rate,
                "uncertain_count":        signals.uncertain_count,
                "total_canonical":        signals.total_canonical,
                "structural_valid":       signals.structural_valid,
                "drift_count":            signals.drift_count,
                "schema_violation_count": signals.schema_violation_count,
                "pass2_rejection_rate":   signals.pass2_rejection_rate,
                "pass2_downgrades":       signals.pass2_downgrades,
                "pass1_present_count":    signals.pass1_present_count,
            },
        },
    }

    # Determine output path and run quality gate
    try:
        output_path = get_output_path(stage, data_dir)
    except ValueError as e:
        return False, str(e)

    gate_result = check(pair, output_path)
    if not gate_result.passed:
        enqueue(pair, queue_path)
        return False, f"gate={gate_result.failed_gate}: {gate_result.reason}"

    enqueue(pair, queue_path)

    logger.info(
        "pipeline: queued R7 %s pair | client=%s stage=%s gagas=%s "
        "uncertain_rate=%.2f pass2_downgrades=%d",
        pair_type, client_type, stage, is_gagas,
        signals.uncertain_rate, signals.pass2_downgrades,
    )

    return True, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_workpaper(
    file_path: str | Path,
    sop_version: str = "latest",
    client_types: list[str] | None = None,
    queue_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    run_parallel: bool = True,
    use_mock: bool = False,
    use_r7: bool = False,
) -> PipelineResult:
    """
    Process one workpaper file through the full AuditAI pipeline.

    Generates 1 clean + 2 deficient variants × len(client_types) combos.
    All variants go to the review queue — approved pairs are written
    to JSONL by calling write_approved() separately.

    Parameters
    ----------
    file_path : str | Path
        Path to any supported workpaper file.
    sop_version : str
        SOP version to tag chunks with. Default 'latest'.
    client_types : list[str] | None
        Client types to generate variants for.
        Default: ["NPO", "government", "for_profit", "tribal"]
    queue_path : str | Path | None
        Review queue file. Default: data/review_queue.jsonl
    data_dir : str | Path | None
        Output directory for JSONL files. Default: data/
    run_parallel : bool
        Whether to run parallel extractors in Phase 1. Default True.
    use_r7 : bool
        Use the R7 classifier→renderer pipeline instead of Gemma free-text
        generation. Constrained decoding + deterministic rendering, no LLM
        narrative inference. Default False (legacy path).

    Returns
    -------
    PipelineResult
    """
    path = Path(file_path)
    result = PipelineResult(file_name=path.name)

    _client_types = client_types or _DEFAULT_CLIENT_TYPES
    _queue_path = Path(queue_path) if queue_path else Path("data/review_queue.jsonl")
    _data_dir = Path(data_dir) if data_dir else Path("data")

    # ------------------------------------------------------------------
    # Step 1 — Phase 1: normalize
    # ------------------------------------------------------------------
    try:
        record = normalize_document(file_path, run_parallel=run_parallel)
        result.record = record
    except (FileNotFoundError, ValueError) as e:
        result.errors.append(f"normalize_document failed: {e}")
        result.skipped = True
        result.skip_reason = str(e)
        return result
    except Exception as e:
        result.errors.append(f"Unexpected error in normalize_document: {e}")
        result.skipped = True
        result.skip_reason = str(e)
        return result

    # ------------------------------------------------------------------
    # Step 2 — Confidence gate
    # ------------------------------------------------------------------
    if record.extraction_confidence < _CONFIDENCE_GATE:
        result.skipped = True
        result.skip_reason = (
            f"extraction_confidence {record.extraction_confidence:.3f} < "
            f"{_CONFIDENCE_GATE} — sent to review queue for manual correction"
        )
        logger.warning("pipeline: low confidence %s — queuing for review", path.name)
        # Still enqueue a placeholder pair for human review
        placeholder_pair = {
            "messages": [
                {"role": "system", "content": "Low confidence extraction — requires manual review"},
                {"role": "user", "content": f"File: {record.file_name}\nConfidence: {record.extraction_confidence:.3f}\nFields found: {record.metadata.get('confidence_summary', {}).get('fields_present', [])}"},
                {"role": "assistant", "content": "REQUIRES MANUAL COMPLETION BY AUDITOR"},
            ],
            "metadata": {
                "file_name": record.file_name,
                "file_type": record.file_type,
                "client_type": "unknown",
                "is_gagas": False,
                "has_single_audit": False,
                "extraction_confidence": record.extraction_confidence,
                "auditor_approved": False,
                "pair_type": "low_confidence",
                "stage": "stage2",
                "fields_missing": record.metadata.get("confidence_summary", {}).get("fields_missing", []),
                "sop_sections_used": [],
                "file_hash": record.file_hash,
                "pair_hash": __import__('hashlib').sha256(record.file_hash.encode()).hexdigest(),
            }
        }
        enqueue(placeholder_pair, _queue_path)
        return result

    # ------------------------------------------------------------------
    # Step 3 — Phase 3: retrieve SOP context
    # ------------------------------------------------------------------
    try:
        retrieval = retrieve(record, top_k=5)
        logger.info(
            "pipeline: retrieved %d SOP chunks via strategy=%s",
            len(retrieval.chunks), retrieval.strategy,
        )
    except Exception as e:
        error_msg = f"SOP retrieval failed: {e}"
        logger.error("pipeline: %s", error_msg)
        result.errors.append(error_msg)
        from pipeline.qdrant_retriever import RetrievalResult
        retrieval = RetrievalResult(sop_text="", sop_sections=[], chunks=[], strategy="error")

    # Retrieval quality gate — check before spending Gemma tokens on empty SOP
    _retrieval_healthy = (
        len(retrieval.chunks) >= 1
        and retrieval.sop_text.strip() != ""
        and retrieval.strategy not in ("error", "fallback_empty")
    )

    if not _retrieval_healthy:
        # Load fail policy from threshold config
        try:
            import yaml as _yaml
            from pathlib import Path as _Path
            _thr = _yaml.safe_load(
                open(_Path(__file__).parent.parent /
                     "auditai_data_normalization/alias_registry/threshold_config.yaml")
            ) or {}
            _fail_policy = _thr.get("sop_retrieval", {}).get("fail_policy", "flag")
        except Exception:
            _fail_policy = "flag"

        logger.warning(
            "pipeline: SOP retrieval quality gate failed for %s "
            "(chunks=%d strategy=%s) — policy=%s",
            record.file_name, len(retrieval.chunks),
            retrieval.strategy, _fail_policy,
        )

        if _fail_policy == "skip":
            result.skipped    = True
            result.skip_reason = (
                f"SOP retrieval returned no usable chunks "
                f"(strategy={retrieval.strategy}). "
                "Run embedder to populate Qdrant collection."
            )
            return result

        # "flag" — continue but mark all pairs as sop_unverified
        record.metadata["sop_unverified"] = True
        record.metadata["sop_retrieval_strategy"] = retrieval.strategy

    is_gagas_base = _detect_gagas(record)
    has_single_audit_base = _detect_single_audit(record)

    # ------------------------------------------------------------------
    # Step 4 — Phase 2: generate variants
    # ------------------------------------------------------------------
    for client_type in _client_types:
        # Override GAGAS/Single Audit flags based on client type when the
        # workpaper doesn't have those checkboxes filled in.
        # Government and Tribal engagements always require GAGAS.
        # Tribal engagements typically require Single Audit (federal funding).
        if client_type == "Government":
            is_gagas = True
            has_single_audit = has_single_audit_base
        elif client_type == "Tribal":
            is_gagas = True
            has_single_audit = True
        elif client_type == "NPO":
            is_gagas = is_gagas_base
            has_single_audit = has_single_audit_base
        else:  # For-Profit
            is_gagas = False
            has_single_audit = False
        # 1 clean variant
        if use_r7:
            success, reason = _process_single_variant_r7(
                record=record,
                retrieval=retrieval,
                client_type=client_type,
                is_gagas=is_gagas,
                has_single_audit=has_single_audit,
                pair_type="clean",
                deficiency_fields=[],
                queue_path=_queue_path,
                data_dir=_data_dir,
            )
        else:
            success, reason = _process_single_variant(
                record=record,
                retrieval=retrieval,
                client_type=client_type,
                is_gagas=is_gagas,
                has_single_audit=has_single_audit,
                pair_type="clean",
                deficiency_fields=[],
                queue_path=_queue_path,
                data_dir=_data_dir,
                use_mock=use_mock,
            )
        result.pairs_built += 1
        if success:
            result.pairs_queued += 1
        else:
            result.pairs_failed += 1
            result.gate_failures.append(
                f"{client_type}/clean: {reason}"
            )

        # Deficient variants — randomised combinations via deficiency_sampler
        _fields_present = list(
            record.metadata.get("confidence_summary", {}).get("fields_present", [])
        )

        try:
            from pipeline.deficiency_sampler import sample as _sample_deficiencies
            _sample_result = _sample_deficiencies(
                present_fields=_fields_present,
                file_name=record.file_name,
                client_type=client_type,
            )
            deficiency_combinations = _sample_result.combinations
            if _sample_result.pool_coverage_warning:
                logger.warning(
                    "pipeline: low SOP pool coverage %.3f for %s/%s — "
                    "excluded fields: %s",
                    _sample_result.pool_coverage, client_type, record.file_name,
                    [e.field for e in _sample_result.excluded_fields],
                )
        except Exception as _e:
            logger.warning(
                "pipeline: deficiency_sampler failed (%s) — using fallback sets", _e
            )
            deficiency_combinations = [
                [f for f in s if f in set(_fields_present)]
                for s in _DEFICIENCY_SETS_FALLBACK
            ]
            deficiency_combinations = [c for c in deficiency_combinations if c]

        for i, effective_deficiency in enumerate(deficiency_combinations):
            if not effective_deficiency:
                result.pairs_built += 1
                result.pairs_failed += 1
                result.gate_failures.append(
                    f"{client_type}/deficient_{i+1}: skipped — no eligible fields"
                )
                continue

            if use_r7:
                success, reason = _process_single_variant_r7(
                    record=record,
                    retrieval=retrieval,
                    client_type=client_type,
                    is_gagas=is_gagas,
                    has_single_audit=has_single_audit,
                    pair_type="deficient",
                    deficiency_fields=effective_deficiency,
                    queue_path=_queue_path,
                    data_dir=_data_dir,
                )
            else:
                success, reason = _process_single_variant(
                    record=record,
                    retrieval=retrieval,
                    client_type=client_type,
                    is_gagas=is_gagas,
                    has_single_audit=has_single_audit,
                    pair_type="deficient",
                    deficiency_fields=effective_deficiency,
                    queue_path=_queue_path,
                    data_dir=_data_dir,
                    use_mock=use_mock,
                )
            result.pairs_built += 1
            if success:
                result.pairs_queued += 1
            else:
                result.pairs_failed += 1
                result.gate_failures.append(
                    f"{client_type}/deficient_{i+1}: {reason}"
                )

    logger.info("pipeline: %s", result)
    return result


def write_approved(
    queue_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, int]:
    """
    Write all approved pairs from the review queue to JSONL files.

    Called after auditors have approved pairs via auditor_review.approve().
    Reads the queue, finds approved pairs, writes them to the correct
    stage JSONL file, skips duplicates.

    Parameters
    ----------
    queue_path : str | Path | None
    data_dir : str | Path | None

    Returns
    -------
    dict with keys: written, skipped, errors
    """
    from raw_to_training_pair.auditor_review import load_all
    from raw_to_training_pair.jsonl_writer import append, get_output_path

    _queue_path = Path(queue_path) if queue_path else Path("data/review_queue.jsonl")
    _data_dir = Path(data_dir) if data_dir else Path("data")

    all_entries = load_all(_queue_path)
    approved = [e for e in all_entries if e.get("status") == "approved"]

    written = skipped = errors = 0

    for entry in approved:
        pair = entry.get("pair", {})
        stage = pair.get("metadata", {}).get("stage", "stage2")

        try:
            output_path = get_output_path(stage, _data_dir)
        except ValueError as e:
            logger.error("write_approved: %s", e)
            errors += 1
            continue

        write_result = append(pair, output_path)
        if write_result.written:
            written += 1
            # Update fingerprint index after successful JSONL write
            try:
                from raw_to_training_pair.pair_index import add_to_index
                from auditai_data_normalization.alias_versioning import current_version
                add_to_index(pair, alias_version=current_version())
            except Exception as _idx_e:
                logger.debug("write_approved: pair_index update failed — %s", _idx_e)
        else:
            skipped += 1

    logger.info(
        "write_approved: written=%d skipped=%d errors=%d",
        written, skipped, errors,
    )

    # Dataset observability — print report after every batch write
    if written > 0:
        try:
            from raw_to_training_pair.dataset_observer import snapshot, drift_report, print_report
            snap  = snapshot()
            drift = drift_report()
            print_report(snap, drift)
        except Exception as _obs_e:
            logger.debug("write_approved: observability report failed — %s", _obs_e)

    return {"written": written, "skipped": skipped, "errors": errors}