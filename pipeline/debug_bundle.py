"""
pipeline/debug_bundle.py
========================
Debug artifact writer for failed training pair generation.

For each failed generation, writes four files to a timestamped subdirectory
of the configured debug_dir:

    prompt.txt          — field classifier prompt sent to Gemma (Pass 1)
    raw_output.txt      — raw model response before parsing
    parsed_output.json  — FieldClassification + ValidatedClassification +
                          ClassificationSignals + RenderResult
    gate_failures.json  — failure_reason, review_confidence, R7 score signals

Purpose: iteration speed.  Prompt debugging without these artifacts requires
re-running the pipeline with logging cranked up and mentally reconstructing
what Gemma saw.  With them, you open the folder and read.

Public API
----------
    write_bundle(debug_dir, label, ...)  — write one bundle; returns bundle path
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def write_bundle(
    debug_dir:          Path,
    label:              str,
    failure_reason:     str,
    classification=None,       # FieldClassification | None
    validated=None,            # ValidatedClassification | None
    signals=None,              # ClassificationSignals | None
    render_result=None,        # RenderResult | None
    pair:               dict | None = None,
    review_confidence:  float = 0.0,
    quality_gate:       bool  = False,
) -> Path:
    """
    Write debug artifacts for one failed generation to debug_dir/label/.

    The label should be short and unique per run:
        "{file_stem}__{client_type}__{pair_type}"
    A UTC timestamp is appended to avoid collisions across runs.

    Returns the bundle directory path.
    """
    ts = datetime.now(timezone.utc).strftime("%H%M%S")
    bundle_dir = debug_dir / f"{label}__{ts}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # prompt.txt
    # ------------------------------------------------------------------
    prompt_text = getattr(classification, "prompt_text", "") or ""
    if prompt_text:
        (bundle_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    else:
        (bundle_dir / "prompt.txt").write_text(
            "[prompt not captured — classify() returned None or prompt_text empty]\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # raw_output.txt
    # ------------------------------------------------------------------
    raw_output = getattr(classification, "raw_output", "") or ""
    if raw_output:
        (bundle_dir / "raw_output.txt").write_text(raw_output, encoding="utf-8")
    else:
        (bundle_dir / "raw_output.txt").write_text(
            "[raw output not captured — classify() returned None or raw_output empty]\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # parsed_output.json
    # ------------------------------------------------------------------
    parsed: dict = {}

    if classification is not None:
        parsed["field_classification"] = {
            "model":            classification.model,
            "field_states":     classification.field_states,
            "absent_fields":    classification.absent_fields,
            "present_fields":   classification.present_fields,
            "uncertain_fields": classification.uncertain_fields,
            "unknown_keys":     classification.unknown_keys,
            "derived_fields":   classification.derived_fields,
        }

    if validated is not None:
        parsed["validated_classification"] = {
            "field_states":       validated.field_states,
            "absent_fields":      validated.absent_fields,
            "present_fields":     validated.present_fields,
            "uncertain_fields":   validated.uncertain_fields,
            "provisional_fields": validated.provisional_fields,
        }

    if signals is not None:
        parsed["classification_signals"] = {
            "uncertain_rate":         signals.uncertain_rate,
            "uncertain_count":        signals.uncertain_count,
            "total_canonical":        signals.total_canonical,
            "structural_valid":       signals.structural_valid,
            "drift_count":            signals.drift_count,
            "schema_violation_count": signals.schema_violation_count,
            "pass2_rejection_rate":   signals.pass2_rejection_rate,
            "pass2_downgrades":       signals.pass2_downgrades,
            "pass1_present_count":    signals.pass1_present_count,
        }

    if render_result is not None:
        parsed["render_result"] = {
            "completion":         render_result.completion,
            "absent_fields":      render_result.absent_fields,
            "uncertain_fields":   render_result.uncertain_fields,
            "provisional_fields": render_result.provisional_fields,
            "sop_unverified":     render_result.sop_unverified,
        }

    (bundle_dir / "parsed_output.json").write_text(
        json.dumps(parsed, indent=2, default=str),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # gate_failures.json
    # ------------------------------------------------------------------
    gate: dict = {
        "failure_reason":    failure_reason,
        "review_confidence": review_confidence,
        "quality_gate":      quality_gate,
    }

    if pair:
        meta = pair.get("metadata", {})
        gate["pair_type"]   = meta.get("pair_type")
        gate["client_type"] = meta.get("client_type")
        gate["pair_hash"]   = meta.get("pair_hash")
        gate["deficiency_fields"] = meta.get("deficiency_fields", [])

    if signals is not None:
        gate["signals"] = {
            "structural_valid":     signals.structural_valid,
            "uncertain_rate":       signals.uncertain_rate,
            "pass2_rejection_rate": signals.pass2_rejection_rate,
        }

    if render_result is not None:
        gate["completion_preview"] = render_result.completion[:400]

    (bundle_dir / "gate_failures.json").write_text(
        json.dumps(gate, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info(
        "debug_bundle: wrote failure artifacts → %s  reason=%r",
        bundle_dir, failure_reason[:80],
    )
    return bundle_dir
