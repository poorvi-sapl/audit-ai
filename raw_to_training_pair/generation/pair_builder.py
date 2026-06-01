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


def build_generation_pair(
    gen_input: GenerationInput,
    gold: GeneratedWorkpaper,
    block_on_schema_issues: bool = False,
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
        extra_metadata: Optional dict merged into the pair's metadata.

    Returns:
        Pair dict ready for JSONL append.

    Raises:
        ValueError: If gen_input.workpaper_type != gold.workpaper_type
                    or (block_on_schema_issues=True and validation fails).
    """
    if gen_input.workpaper_type != gold.workpaper_type:
        raise ValueError(
            f"build_generation_pair: workpaper_type mismatch — "
            f"gen_input={gen_input.workpaper_type!r} vs "
            f"gold={gold.workpaper_type!r}"
        )

    issues = validate_against_registry(gold)
    if issues and block_on_schema_issues:
        raise ValueError(
            f"build_generation_pair: gold has {len(issues)} schema issue(s) "
            f"and block_on_schema_issues=True. First few: {issues[:3]}"
        )
    if issues:
        logger.warning(
            "build_generation_pair: %d schema issue(s) for %s/%s — "
            "recording in metadata.schema_issues",
            len(issues), gen_input.workpaper_type, gen_input.engagement_id,
        )

    user_message = render_user_message(gen_input)
    assistant_message = to_json_string(gold, indent=2)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]

    metadata: dict[str, Any] = {
        "pair_type": "generation",
        "task": TASK_IDENTIFIER,
        "workpaper_type": gen_input.workpaper_type,
        "engagement_id": gen_input.engagement_id,
        "file_hash": _inputs_hash(gen_input, gold),
        "pair_hash": pair_hash(messages),
        "schema_issues": issues,
        "fields_present_in_facts": len(gen_input.fields_present()),
        "fields_missing_in_facts": len(gen_input.fields_missing()),
        "sop_chunks_count": len(gen_input.sop_chunks),
        "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {"messages": messages, "metadata": metadata}
