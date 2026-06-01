"""
raw_to_training_pair/generation/pair_builder.py
=================================================
Build a generation-task training pair from a (GenerationInput, gold)
tuple. Returns a JSONL-ready dict matching the shape produced by the
existing review-task pair_builder so the downstream review queue and
JSONL export reuse the same schema.

Pair shape
----------
    {
      "messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": <rendered user message>},
        {"role": "assistant", "content": <gold workpaper JSON>}
      ],
      "metadata": {
        "pair_type":         "generation",
        "workpaper_type":    "NPO-CX-1.1",
        "engagement_id":     "ENG-...",
        "file_hash":         "<sha256 of inputs>",
        "pair_hash":         "<sha256 of messages>",
        "task":              "GENERATE_WORKPAPER",
        "schema_issues":     [...]   # populated if validate_against_registry found problems
      }
    }

Public API
----------
    build_generation_pair(gen_input, gold, ...)   → dict
    pair_hash(messages)                           → str
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

from auditai_data_normalization.generation_contract import GenerationInput
from raw_to_training_pair.generation.prompt import (
    SYSTEM_PROMPT,
    TASK_IDENTIFIER,
    render_user_message,
)
from raw_to_training_pair.generation.target_schema import (
    GeneratedWorkpaper,
    hard_issues,
    to_json_string,
    validate_against_registry,
)

logger = logging.getLogger(__name__)


def pair_hash(messages: list[dict]) -> str:
    """Deterministic sha256 over the canonical-JSON messages list.

    Same algorithm as raw_to_training_pair.pair_builder._pair_hash so
    generation pairs are deduplicated identically to review pairs.
    """
    content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _facts_summary(gen_input: GenerationInput) -> dict[str, dict]:
    """Compact, JSON-serializable summary of the GenerationInput's
    ExtractedFacts. Stored in metadata so the review UI can render a
    side-by-side facts-vs-gold comparison without re-parsing the
    user message text.

    Only present (non-null) facts are included. Sources are
    truncated to the first 3 per field; quoted_text is clipped to
    200 chars per source to bound metadata size.
    """
    summary: dict[str, dict] = {}
    for fid, fact in gen_input.extracted_facts.items():
        if not fact.is_present:
            continue
        sources = [
            {
                "document_type": src.document_type,
                "page": src.page,
                "quoted_text": (src.quoted_text or "")[:200],
            }
            for src in fact.sources[:3]
        ]
        summary[fid] = {
            "value": fact.value,
            "confidence": round(fact.confidence, 3),
            "extractor_method": fact.extractor_method,
            "sources": sources,
        }
    return summary


def _inputs_hash(gen_input: GenerationInput, gold: GeneratedWorkpaper) -> str:
    """Hash over the (input facts + gold) so two pairs from the same
    raw materials collide on this hash even if the rendered messages
    drift slightly between versions of the prompt assembler."""
    facts_canonical = sorted(
        (
            fid,
            fact.value,
            fact.confidence,
            tuple(
                (s.document_type, s.page, s.char_start, s.char_end)
                for s in fact.sources
            ),
        )
        for fid, fact in gen_input.extracted_facts.items()
    )
    gold_canonical = sorted(
        (fid, fv.value, len(fv.citations))
        for fid, fv in gold.fields.items()
    )
    payload = {
        "workpaper_type": gen_input.workpaper_type,
        "engagement_id": gen_input.engagement_id,
        "facts": facts_canonical,
        "gold": gold_canonical,
        "sop_count": len(gen_input.sop_chunks),
    }
    raw = json.dumps(payload, default=str, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _check_pii_in_content(content: str) -> tuple[int, list[str]]:
    """Run the Presidio-backed scrubber on `content` to detect any PII
    that wasn't scrubbed upstream. Returns (replacement_count,
    pii_types_found). A non-zero count means the content carries
    detectable PII and should NOT flow to training data without
    further scrubbing.
    """
    try:
        from auditai_data_normalization.pii import scrub
    except ImportError:
        logger.warning(
            "build_generation_pair: pii.scrub unavailable — skipping "
            "PII enforcement check. PII may flow downstream unchecked."
        )
        return (0, [])
    try:
        result = scrub(content)
    except Exception as e:
        logger.warning(
            "build_generation_pair: pii.scrub raised %s — treating as "
            "no-PII for this build but flagging in metadata", e,
        )
        return (-1, [str(e)])
    # total_replacements and types_found are methods on ScrubResult, not properties
    return (result.total_replacements(), list(result.types_found()))


def build_generation_pair(
    gen_input: GenerationInput,
    gold: GeneratedWorkpaper,
    block_on_schema_issues: bool = False,
    pii_strict: bool = False,
    extra_metadata: dict | None = None,
) -> dict[str, Any]:
    """Build a JSONL-ready generation training pair.

    Args:
        gen_input: Phase 1A/1B GenerationInput (extracted facts + SOPs)
        gold: The gold-label GeneratedWorkpaper (assistant target)
        block_on_schema_issues: If True, raise ValueError when the gold
            fails registry validation (extra fields, missing fields,
            bad categorical values, wrong-typed booleans). If False
            (default), the pair is built anyway and issues are recorded
            in metadata.schema_issues so reviewers can decide.
        pii_strict: PII enforcement mode (Decision Y2). Default False:
            scan both messages for PII, log a warning if any detected,
            record findings in metadata.pii_issues, but build the pair
            anyway. True: raise ValueError if any PII detected in either
            message. Production / training-data-flow callers should
            pass True.
        extra_metadata: Optional dict merged into the pair's metadata.

    Returns:
        Pair dict ready for JSONL append.

    Raises:
        ValueError: If gen_input.workpaper_type != gold.workpaper_type,
                    or (block_on_schema_issues=True and validation fails),
                    or (pii_strict=True and PII detected).
    """
    if gen_input.workpaper_type != gold.workpaper_type:
        raise ValueError(
            f"build_generation_pair: workpaper_type mismatch — "
            f"gen_input={gen_input.workpaper_type!r} vs "
            f"gold={gold.workpaper_type!r}"
        )

    issues = validate_against_registry(gold)
    blocking = hard_issues(issues)
    if blocking and block_on_schema_issues:
        raise ValueError(
            f"build_generation_pair: gold has {len(blocking)} HARD "
            f"schema issue(s) and block_on_schema_issues=True. "
            f"First few: {blocking[:3]}"
        )
    if issues:
        logger.warning(
            "build_generation_pair: %d schema issue(s) for %s/%s "
            "(%d hard, %d soft) — recording in metadata.schema_issues",
            len(issues),
            gen_input.workpaper_type, gen_input.engagement_id,
            len(blocking), len(issues) - len(blocking),
        )

    user_message = render_user_message(gen_input)
    assistant_message = to_json_string(gold, indent=2)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]

    # PII enforcement (Decision Y2)
    user_pii_count, user_pii_types = _check_pii_in_content(user_message)
    asst_pii_count, asst_pii_types = _check_pii_in_content(assistant_message)
    pii_issues = {
        "user_pii_count": user_pii_count,
        "user_pii_types": user_pii_types,
        "assistant_pii_count": asst_pii_count,
        "assistant_pii_types": asst_pii_types,
    }
    total_pii_detected = max(0, user_pii_count) + max(0, asst_pii_count)
    if total_pii_detected > 0:
        msg = (
            f"build_generation_pair: PII detected in pair for "
            f"{gen_input.workpaper_type}/{gen_input.engagement_id} "
            f"(user={user_pii_count}, assistant={asst_pii_count}). "
            f"Types: user={user_pii_types}, assistant={asst_pii_types}."
        )
        if pii_strict:
            raise ValueError(msg + " pii_strict=True → refusing to build.")
        logger.warning(msg + " pii_strict=False → recording and continuing.")

    metadata: dict[str, Any] = {
        "pair_type": "generation",
        "task": TASK_IDENTIFIER,
        "workpaper_type": gen_input.workpaper_type,
        "engagement_id": gen_input.engagement_id,
        "file_hash": _inputs_hash(gen_input, gold),
        "pair_hash": pair_hash(messages),
        "schema_issues": issues,
        "pii_issues": pii_issues,
        "fields_present_in_facts": len(gen_input.fields_present()),
        "fields_missing_in_facts": len(gen_input.fields_missing()),
        "sop_chunks_count": len(gen_input.sop_chunks),
        # Compact facts summary for the review UI's side-by-side
        # facts-vs-gold view. See _facts_summary docstring.
        "extracted_facts_summary": _facts_summary(gen_input),
        "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {"messages": messages, "metadata": metadata}
