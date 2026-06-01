"""
raw_to_training_pair/generation/prompt.py
===========================================
Prompt assembly for generation training pairs.

- SYSTEM_PROMPT carries an explicit task identifier (Decision C2)
  so one fine-tuned model can serve multiple tasks (generate, review,
  Q&A) by switching the TASK line.
- render_user_message() produces a structured + prose hybrid
  (Decision B2): facts and template fields as JSON-ish structure,
  SOPs as named blocks, instructions as natural language.

Public API
----------
    SYSTEM_PROMPT                            — the generation-task system prompt
    render_user_message(gen_input)           → str
    TASK_IDENTIFIER                          — "GENERATE_WORKPAPER"
"""

from __future__ import annotations

import json

from auditai_data_normalization.generation_contract import (
    ExtractedFact,
    GenerationInput,
)

TASK_IDENTIFIER: str = "GENERATE_WORKPAPER"

SYSTEM_PROMPT: str = f"""\
You are an HCLLP audit assistant. TASK: {TASK_IDENTIFIER}.

For this task you produce filled workpaper JSON for the engagement
described by the user. Strict rules:

- Every narrative field MUST cite at least one source document via
  the citations array. Format:
      "citations": [{{"document": "<doc>", "page": <int>,
                      "quoted_text": "<excerpt>"}}]
- Numerical values, dates, and entity identifiers MUST come from the
  EXTRACTED FACTS provided. Do not invent or alter them.
- Categorical fields MUST be one of the allowed values listed under
  TEMPLATE FIELDS. Reject the request rather than emit an
  out-of-vocabulary value.
- If the EXTRACTED FACTS do not support a field, return value: null
  with an empty citations array. Do not guess.
- Ground every narrative claim in firm SOPs (provided under
  RELEVANT SOP SECTIONS) or in the EXTRACTED FACTS.
- Return ONLY the JSON object — no surrounding prose, no markdown
  code fences."""


def _fact_inline_summary(fact: ExtractedFact) -> str:
    """One-line summary of an ExtractedFact for the user-message JSON
    block. Includes the value and a compact citation tag."""
    if not fact.is_present:
        return "null"
    cite_parts = []
    for src in fact.sources[:2]:  # first two citations to keep it short
        cite_parts.append(f"{src.document_type} p.{src.page}")
    cite_str = "; ".join(cite_parts) if cite_parts else "no-source"
    return f"{json.dumps(fact.value, ensure_ascii=False)}  // [{cite_str}]"


def _render_extracted_facts_block(gen_input: GenerationInput) -> str:
    """Render the EXTRACTED FACTS block: JSON-like keyed listing of
    every fact, with inline citation comments. Fields not extracted
    appear as null with a // missing comment."""
    lines: list[str] = ["{"]
    for fid in sorted(gen_input.template_field_ids):
        fact = gen_input.extracted_facts.get(fid)
        if fact is None or not fact.is_present:
            lines.append(f'  "{fid}": null,  // missing — model must reason or leave null')
            continue
        lines.append(f'  "{fid}": {_fact_inline_summary(fact)},')
    # Drop trailing comma on last line for valid-ish appearance
    if lines[-1].endswith(","):
        lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def _render_template_fields_block(gen_input: GenerationInput) -> str:
    """Render TEMPLATE FIELDS — all field IDs the workpaper requires,
    with allowed_values for categoricals shown inline."""
    from auditai_data_normalization.field_type_registry import load_registry

    try:
        registry = load_registry(gen_input.workpaper_type)
    except FileNotFoundError:
        # Defensive: shouldn't happen at this layer, but degrade gracefully
        return ", ".join(sorted(gen_input.template_field_ids))

    lines: list[str] = []
    for fid in sorted(gen_input.template_field_ids):
        spec = registry.get(fid)
        if spec is None:
            lines.append(f"  - {fid}")
            continue
        if spec.field_type == "categorical" and spec.allowed_values:
            allowed = ", ".join(f'"{v}"' for v in spec.allowed_values)
            lines.append(f"  - {fid} (categorical, allowed: {allowed})")
        else:
            lines.append(f"  - {fid} ({spec.field_type})")
    return "\n".join(lines)


def _render_sop_block(gen_input: GenerationInput) -> str:
    """Render RELEVANT SOP SECTIONS. Each chunk on its own line block,
    separated by blank lines."""
    if not gen_input.sop_chunks:
        return "(no SOP chunks retrieved — model must reason from facts alone)"
    return "\n\n".join(gen_input.sop_chunks)


def render_user_message(gen_input: GenerationInput) -> str:
    """Render a full user message from a GenerationInput per Decision B2.

    Layout:
      WORKPAPER: <type>
      ENGAGEMENT: <id>

      EXTRACTED FACTS (from source documents):
      { JSON-keyed listing with inline citation comments }

      TEMPLATE FIELDS TO FILL:
        - field_id (type, [allowed: ...])
        - ...

      RELEVANT SOP SECTIONS:
      <chunks>

      Produce the filled workpaper as JSON per the SYSTEM message rules.
    """
    facts_block = _render_extracted_facts_block(gen_input)
    fields_block = _render_template_fields_block(gen_input)
    sop_block = _render_sop_block(gen_input)

    return (
        f"WORKPAPER: {gen_input.workpaper_type}\n"
        f"ENGAGEMENT: {gen_input.engagement_id}\n"
        f"\n"
        f"EXTRACTED FACTS (from source documents):\n"
        f"{facts_block}\n"
        f"\n"
        f"TEMPLATE FIELDS TO FILL:\n"
        f"{fields_block}\n"
        f"\n"
        f"RELEVANT SOP SECTIONS:\n"
        f"{sop_block}\n"
        f"\n"
        f"Produce the filled workpaper as JSON per the SYSTEM message rules."
    )
