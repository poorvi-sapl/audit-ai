"""
raw_to_training_pair/generation/target_schema.py
==================================================
The shape of the *assistant* message in a generation training pair.

This is the gold-label target the fine-tuned model learns to produce.
Per Phase 2.1 Decision A2 (grounded outputs), every field value
carries its own source citations — teaching the model to ground
every claim, not just emit values.

Structural rules
----------------
1. Every field in the workpaper's field-type registry must be present
   in the GeneratedWorkpaper (use value=None for fields the workpaper
   leaves blank, e.g., a "No" answer with no comment).
2. No fields outside the registry are allowed.
3. For categorical fields, value must be in the registry's
   allowed_values (or None).
4. Every non-null narrative/text field SHOULD carry at least one
   citation. Lack of citations is allowed for fields the model can
   defensibly produce from SOPs alone (e.g., sop_fixed boolean
   answers like Q4 "No") but is flagged.

Public API
----------
    GeneratedFieldValue   — dataclass: value + citations
    GeneratedWorkpaper    — dataclass: workpaper_type + fields dict
    validate_against_registry(gw)  → list[str]   (issues, empty = valid)
    to_json_string(gw)            → str
    from_json_string(json_str, workpaper_type) → GeneratedWorkpaper
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from auditai_data_normalization.field_type_registry import load_registry


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratedCitation:
    """One source citation supporting a generated field value.

    Lighter than the Phase 1A SourceCitation — the assistant target
    only needs document, page, and quoted text. Full char-level
    offsets live on the input side (the extraction provenance), not
    on the model's output.
    """
    document: str           # 'engagement_letter' | 'prior_year_file' | ...
    page: int               # 1-based; PAGE_UNKNOWN (0) allowed when source had no page
    quoted_text: str        # short excerpt supporting the value


@dataclass
class GeneratedFieldValue:
    """One field's value in the assistant output, with grounding."""
    value: str | bool | None
    citations: list[GeneratedCitation] = field(default_factory=list)

    @property
    def is_present(self) -> bool:
        return self.value is not None and self.value != ""


@dataclass
class GeneratedWorkpaper:
    """The full assistant target — one workpaper with all fields filled.

    `fields` is keyed by registry field_id. Every registered field
    should be present (use a GeneratedFieldValue with value=None for
    fields the workpaper leaves blank).
    """
    workpaper_type: str                           # e.g. "NPO-CX-1.1"
    engagement_id: str
    fields: dict[str, GeneratedFieldValue]


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_against_registry(gw: GeneratedWorkpaper) -> list[str]:
    """Validate a GeneratedWorkpaper against its workpaper's registry.

    Returns a list of issue strings. Empty list means valid.
    Does NOT raise — validation is informational; callers decide
    whether issues block pair construction.
    """
    issues: list[str] = []
    try:
        registry = load_registry(gw.workpaper_type)
    except FileNotFoundError as e:
        return [f"registry missing for workpaper_type={gw.workpaper_type!r}: {e}"]

    registry_ids = set(registry.keys())
    gold_ids = set(gw.fields.keys())

    # 1. Extra fields not in registry
    extras = gold_ids - registry_ids
    for fid in sorted(extras):
        issues.append(f"field {fid!r} not in registry for {gw.workpaper_type}")

    # 2. Missing fields (in registry but absent from gold)
    missing = registry_ids - gold_ids
    for fid in sorted(missing):
        issues.append(f"field {fid!r} missing from gold (registry requires it)")

    # 3. Categorical values must match allowed_values
    for fid, fv in gw.fields.items():
        if fid not in registry:
            continue
        spec = registry[fid]
        if spec.field_type == "categorical" and fv.value is not None:
            if fv.value not in (spec.allowed_values or ()):
                issues.append(
                    f"field {fid!r} value {fv.value!r} not in allowed_values "
                    f"{list(spec.allowed_values or ())}"
                )
        if spec.field_type == "boolean" and fv.value is not None:
            if not isinstance(fv.value, bool):
                issues.append(
                    f"field {fid!r} is boolean but got value of type "
                    f"{type(fv.value).__name__}: {fv.value!r}"
                )

    return issues


# ---------------------------------------------------------------------
# JSON (de)serialization
# ---------------------------------------------------------------------

def _field_to_json_obj(fv: GeneratedFieldValue) -> dict:
    return {
        "value": fv.value,
        "citations": [
            {"document": c.document, "page": c.page, "quoted_text": c.quoted_text}
            for c in fv.citations
        ],
    }


def to_json_string(gw: GeneratedWorkpaper, indent: int | None = 2) -> str:
    """Render the workpaper as a JSON string suitable for the assistant
    message slot of a training pair. Field order is sorted for
    determinism (so pair_hash is stable across runs)."""
    payload = {
        "workpaper_type": gw.workpaper_type,
        "engagement_id": gw.engagement_id,
        "fields": {
            fid: _field_to_json_obj(gw.fields[fid])
            for fid in sorted(gw.fields.keys())
        },
    }
    return json.dumps(payload, indent=indent, sort_keys=False, ensure_ascii=False)


def from_json_string(
    json_str: str, workpaper_type: str | None = None,
) -> GeneratedWorkpaper:
    """Inverse of to_json_string. workpaper_type override is for the
    case where the source JSON omits it (caller knows the context)."""
    payload = json.loads(json_str)
    wp_type = payload.get("workpaper_type") or workpaper_type
    if not wp_type:
        raise ValueError(
            "from_json_string: cannot determine workpaper_type — "
            "absent from JSON and no override provided"
        )
    fields: dict[str, GeneratedFieldValue] = {}
    for fid, fobj in payload.get("fields", {}).items():
        citations = [
            GeneratedCitation(
                document=c.get("document", ""),
                page=int(c.get("page", 0)),
                quoted_text=c.get("quoted_text", ""),
            )
            for c in fobj.get("citations", [])
        ]
        fields[fid] = GeneratedFieldValue(
            value=fobj.get("value"),
            citations=citations,
        )
    return GeneratedWorkpaper(
        workpaper_type=wp_type,
        engagement_id=payload.get("engagement_id", ""),
        fields=fields,
    )
