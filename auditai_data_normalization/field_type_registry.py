"""
Field-type registry loader.
============================

Loads per-workpaper YAML registry files and validates them against
the FieldType set declared in generation_contract.py.

Provides field_id → FieldSpec lookup for the extraction → generation
contract assembly layer. Each registry file lives at
config/field_type_registry/<workpaper_type_slug>.yaml and is keyed
by field_id, with `type` and optional `allowed_values`.

The slug for a workpaper_type is derived by lowercasing and
replacing '-' and '.' with '_': e.g. "NPO-CX-1.1" → "npo_cx_1_1".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import get_args

import yaml

from auditai_data_normalization.generation_contract import (
    LLM_FORBIDDEN_FIELD_TYPES,
    FieldType,
)

_REGISTRY_DIR = Path(__file__).parent.parent / "config" / "field_type_registry"
_VALID_FIELD_TYPES: frozenset[str] = frozenset(get_args(FieldType))


@dataclass(frozen=True)
class FieldSpec:
    """One field's type contract, loaded from the registry YAML."""
    field_id: str
    field_type: FieldType
    allowed_values: tuple[str, ...] | None = None  # only for categorical

    @property
    def is_llm_allowed(self) -> bool:
        """True if the LLM extractor is permitted to produce this field."""
        return self.field_type not in LLM_FORBIDDEN_FIELD_TYPES


def _slug(workpaper_type: str) -> str:
    return workpaper_type.lower().replace("-", "_").replace(".", "_")


def _validate_spec(field_id: str, raw: dict) -> FieldSpec:
    """Validate a raw YAML entry and convert to a FieldSpec."""
    if "type" not in raw:
        raise ValueError(f"Field {field_id!r} missing 'type' in registry")
    ftype = raw["type"]
    if ftype not in _VALID_FIELD_TYPES:
        raise ValueError(
            f"Field {field_id!r} has invalid type {ftype!r}. "
            f"Must be one of {sorted(_VALID_FIELD_TYPES)}"
        )
    if ftype == "categorical":
        allowed = raw.get("allowed_values")
        if not allowed or not isinstance(allowed, list):
            raise ValueError(
                f"Field {field_id!r} is categorical but has no "
                "'allowed_values' list in registry"
            )
        return FieldSpec(
            field_id=field_id,
            field_type=ftype,
            allowed_values=tuple(str(v) for v in allowed),
        )
    if "allowed_values" in raw:
        raise ValueError(
            f"Field {field_id!r} has 'allowed_values' but type "
            f"{ftype!r} is not categorical. Only categorical fields "
            "may declare allowed_values."
        )
    return FieldSpec(field_id=field_id, field_type=ftype)


@lru_cache(maxsize=8)
def load_registry(workpaper_type: str) -> dict[str, FieldSpec]:
    """Load the field-type registry for a workpaper type.

    Result is cached per workpaper_type — call `load_registry.cache_clear()`
    after editing a registry YAML during development.

    Raises:
        FileNotFoundError if no registry exists for the workpaper type.
        ValueError if the YAML is malformed or contains invalid types.
    """
    path = _REGISTRY_DIR / f"{_slug(workpaper_type)}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No field-type registry found for workpaper type "
            f"{workpaper_type!r} at {path}"
        )
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Registry at {path} is not a YAML mapping")
    fields_raw = raw.get("fields", {})
    if not fields_raw:
        raise ValueError(
            f"Registry at {path} has empty or missing 'fields' section"
        )
    specs: dict[str, FieldSpec] = {}
    for fid, spec_raw in fields_raw.items():
        specs[fid] = _validate_spec(fid, spec_raw)
    return specs


def get_field_spec(workpaper_type: str, field_id: str) -> FieldSpec:
    """Get the FieldSpec for one field_id. Raises KeyError if unknown."""
    registry = load_registry(workpaper_type)
    if field_id not in registry:
        raise KeyError(
            f"Field {field_id!r} not in registry for workpaper "
            f"{workpaper_type!r}. Known fields: "
            f"{sorted(registry.keys())[:5]}... ({len(registry)} total)"
        )
    return registry[field_id]


def is_llm_allowed(workpaper_type: str, field_id: str) -> bool:
    """Check if the LLM extractor is permitted to produce this field.

    Returns False for numeric/date/id types (the LLM-forbidden set).
    """
    return get_field_spec(workpaper_type, field_id).is_llm_allowed
