"""
raw_to_training_pair/field_classifier.py
=========================================
R7-A: Constrained field classifier (Pass 1).

Contract (see docs/architecture/09_r7_classifier_invariants.md)
-------
The classifier is a function, not a generator. It has no narrative capability.

    classify(workpaper_text, canonical_fields, client_type)
        → FieldClassification

    |field_states| == |canonical_fields|   (full keyspace, no omission)
    values ∈ {"absent", "present", "uncertain"}   (closed enum)
    temperature = 0.0   (deterministic)

Enforcement: outlines wraps ollama's chat API with a JSON schema passed via
the `format` parameter. Ollama enforces the schema using llama.cpp's
grammar-based constrained sampling — the model cannot emit a field name
outside canonical_fields or a value outside the three-state enum.
Enforcement happens at decoding time, not post-hoc.

Canonical fields
----------------
Only tier1 and tier2 fields from field_tiers.yaml are classified.
Tier3 fields (financial data, extraction metadata) are not engagement-form
fields and are excluded from classification.

Public API
----------
    classify(workpaper_text, canonical_fields=None, client_type="")
        -> FieldClassification

    load_canonical_fields() -> list[str]
        Returns sorted tier1+tier2 field names from field_tiers.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

_PKG_DIR     = Path(__file__).parent
_PROJECT_DIR = _PKG_DIR.parent
_TIERS_PATH  = _PROJECT_DIR / "config" / "field_tiers.yaml"

_MODEL       = "gemma3:12b"
_TEMPERATURE = 0.0
_MAX_TOKENS  = 512   # classification output is small; cap prevents runaway

# Workpaper text is capped at this many words before being sent to the model.
# Keeps prompt size bounded without losing the most relevant content (top of doc).
_MAX_WORKPAPER_WORDS = 1500

FieldState = Literal["absent", "present", "uncertain"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FieldClassification:
    field_states:     dict[str, str]   # canonical_field → absent/present/uncertain
    canonical_fields: list[str]        # exact field list the schema was built from
    model:            str              # model used (for reproducibility)
    absent_fields:    list[str]        # convenience — fields classified "absent"
    present_fields:   list[str]        # convenience — fields classified "present"
    uncertain_fields: list[str]        # convenience — fields classified "uncertain"


# ---------------------------------------------------------------------------
# Canonical field loader (tier1 + tier2 only)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_canonical_fields() -> list[str]:
    """
    Load tier1 + tier2 canonical field names from field_tiers.yaml.

    Tier3 fields (financial data, extraction metadata) are excluded —
    they are not engagement-form fields and should not be classified.
    Returns a sorted list for deterministic schema construction.
    """
    if not _TIERS_PATH.exists():
        logger.error("field_classifier: field_tiers.yaml not found at %s", _TIERS_PATH)
        return []
    with open(_TIERS_PATH) as f:
        tiers = yaml.safe_load(f) or {}
    fields: list[str] = []
    for tier_key in ("tier1", "tier2"):
        for entry in (tiers.get(tier_key) or []):
            if isinstance(entry, dict) and "field" in entry:
                fields.append(entry["field"])
    return sorted(fields)


# ---------------------------------------------------------------------------
# Dynamic schema builder
# ---------------------------------------------------------------------------

def _build_output_schema(canonical_fields: list[str]):
    """
    Build a Pydantic model whose JSON schema outlines passes to ollama as
    the `format` constraint. The schema enforces:

      - exactly canonical_fields as required keys (no more, no fewer)
      - values constrained to the {"absent","present","uncertain"} enum

    Ollama's llama.cpp grammar backend ensures the model cannot emit anything
    outside this schema at decoding time.
    """
    from pydantic import create_model

    field_spec = {f: (FieldState, ...) for f in canonical_fields}
    FieldStates  = create_model("FieldStates",  **field_spec)
    OutputSchema = create_model("OutputSchema", field_states=(FieldStates, ...))
    return OutputSchema


# ---------------------------------------------------------------------------
# Prompt builder — NO narrative instructions
# ---------------------------------------------------------------------------

def _build_prompt(
    workpaper_text: str,
    canonical_fields: list[str],
    client_type: str,
) -> str:
    words = workpaper_text.split()
    if len(words) > _MAX_WORKPAPER_WORDS:
        workpaper_text = " ".join(words[:_MAX_WORKPAPER_WORDS]) + "\n[... text truncated ...]"

    field_list = "\n".join(f"  - {f}" for f in canonical_fields)
    client_hint = f" (client type: {client_type})" if client_type else ""

    return (
        f"Audit workpaper text{client_hint}:\n"
        f"---\n{workpaper_text}\n---\n\n"
        f"Classify each field as present, absent, or uncertain based solely on the text above.\n\n"
        f"Classification rules:\n"
        f"  absent    — field is not present or is clearly incomplete\n"
        f"  present   — field is documented with sufficient evidence\n"
        f"  uncertain — field may exist but evidence is ambiguous\n\n"
        f"Fields to classify:\n{field_list}"
    )


# ---------------------------------------------------------------------------
# Model singleton (lazy init)
# ---------------------------------------------------------------------------

_outlines_model = None


def _get_model():
    global _outlines_model
    if _outlines_model is None:
        try:
            import ollama as _ollama
            import outlines as _outlines
            client = _ollama.Client()
            _outlines_model = _outlines.from_ollama(client, model_name=_MODEL)
            logger.debug("field_classifier: outlines+ollama model loaded (%s)", _MODEL)
        except Exception as e:
            logger.error("field_classifier: failed to load model — %s", e)
            _outlines_model = False  # sentinel: tried and failed
    return _outlines_model if _outlines_model is not False else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(
    workpaper_text: str,
    canonical_fields: list[str] | None = None,
    client_type: str = "",
) -> FieldClassification | None:
    """
    Classify each canonical field as absent / present / uncertain.

    Parameters
    ----------
    workpaper_text : str
        Extracted text of the workpaper to classify.
    canonical_fields : list[str] | None
        Field list to classify. If None, loads tier1+tier2 from
        field_tiers.yaml. Must be valid Python identifiers (all
        canonical fields in field_tiers.yaml satisfy this).
    client_type : str
        Client type hint included in the prompt for context.

    Returns
    -------
    FieldClassification | None
        None if the model is unavailable or inference fails.

    Invariants enforced
    -------------------
    - |field_states| == |canonical_fields| (full keyspace — no omission)
    - values ∈ {"absent", "present", "uncertain"} (closed enum)
    - temperature = 0.0 (deterministic)
    - No narrative in output (schema permits only the three-state values)
    """
    if canonical_fields is None:
        canonical_fields = load_canonical_fields()

    if not canonical_fields:
        logger.error("field_classifier: empty canonical field list — cannot classify")
        return None

    model = _get_model()
    if model is None:
        logger.error(
            "field_classifier: model unavailable. "
            "Ensure ollama is running: ollama serve && ollama pull %s", _MODEL,
        )
        return None

    output_schema = _build_output_schema(canonical_fields)
    prompt = _build_prompt(workpaper_text, canonical_fields, client_type)

    try:
        import outlines as _outlines

        generator = _outlines.Generator(model, output_schema)
        # outlines 1.3.0 Ollama backend has a debug print() in generate().
        # Suppress it during inference so pipeline logs stay clean.
        import io, sys as _sys
        _stdout, _sys.stdout = _sys.stdout, io.StringIO()
        try:
            raw: str = generator(
                prompt,
                options={"temperature": _TEMPERATURE, "num_predict": _MAX_TOKENS},
            )
        finally:
            _sys.stdout = _stdout

        # outlines returns the model output as a string; parse it back to dict
        import json
        parsed = json.loads(raw) if isinstance(raw, str) else raw

        # Handle both raw dict and pydantic-validated object
        if hasattr(parsed, "field_states"):
            states = {k: v for k, v in parsed.field_states.__dict__.items()
                      if not k.startswith("_")}
        elif isinstance(parsed, dict) and "field_states" in parsed:
            states = parsed["field_states"]
        else:
            states = parsed  # outlines may return the inner dict directly

        # Validate full keyspace (invariant 1)
        missing_keys = [f for f in canonical_fields if f not in states]
        extra_keys   = [k for k in states if k not in canonical_fields]
        if missing_keys:
            logger.error(
                "field_classifier: schema violation — missing keys: %s "
                "(outlines should prevent this — check model/schema compatibility)",
                missing_keys,
            )
            return None
        if extra_keys:
            logger.warning(
                "field_classifier: unexpected keys in output (pruned): %s",
                extra_keys,
            )
            for k in extra_keys:
                del states[k]

        # Partition into convenience lists
        absent   = [f for f in canonical_fields if states.get(f) == "absent"]
        present  = [f for f in canonical_fields if states.get(f) == "present"]
        uncertain = [f for f in canonical_fields if states.get(f) == "uncertain"]

        logger.info(
            "field_classifier: %s — absent=%d present=%d uncertain=%d",
            client_type or "unknown", len(absent), len(present), len(uncertain),
        )

        return FieldClassification(
            field_states     = {f: states[f] for f in canonical_fields},
            canonical_fields = list(canonical_fields),
            model            = _MODEL,
            absent_fields    = absent,
            present_fields   = present,
            uncertain_fields = uncertain,
        )

    except Exception as e:
        logger.error("field_classifier: inference failed — %s", e)
        return None


def reset_model_cache() -> None:
    """Force model reload on next classify() call. Useful in tests."""
    global _outlines_model
    _outlines_model = None
    load_canonical_fields.cache_clear()
