"""
workpaper_generator/rule_engine.py
====================================
Applies the NPO-CX-1.1 field manifest rules to (client PDF, engagement
type, auditor inputs) and produces resolved field values with citations.

Each manifest field declares a `source` that determines how it is resolved:

  sop_fixed              rule.value (deterministic; supports Part II
                         recurring/initial branching)
  sop_fixed_plus_lookup  rule.value plus a reference resolved from a PDF
                         section lookup (Q2)
  sop_fixed_plus_manual  rule.value plus a remark from auditor_inputs (Q2j)
  py_audit_report_lookup if_found / if_not_found based on PDF section
                         detection (Q1c, Q1d)
  auditor_selection      taken from auditor_inputs, else marked needs_input
  manual_entry           taken from auditor_inputs, else marked needs_input
  derived                computed from other resolved field values
                         (acceptance_decision per SOP Table 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

from workpaper_generator.pdf_section_detector import PDFDetectionReport, detect

_DEFAULT_MANIFEST_PATH = Path(__file__).parent.parent / "config" / "npo_cx_1_1_manifest.yaml"

ENGAGEMENT_INITIAL = "Initial / 1st Year"
ENGAGEMENT_RECURRING = "Recurring / 2nd Year or Subsequent"

# Maps manifest lookup_target strings to detector section keys.
_LOOKUP_TARGET_TO_KEY = {
    "Schedule of Expenditures of Federal Awards": "sefa",
    "Supplementary Information": "supplementary",
    "Compliance Section": "compliance",
}

# Per SOP Table 3, a Yes on any of these blocks acceptance.
_ACCEPTANCE_BLOCKERS = ["q4", "q5_a", "q5_b", "q5_c", "q9", "q9_a", "q9_b", "q11"]


@dataclass
class ResolvedField:
    field_id: str
    value: Optional[str]
    source: str
    status: str  # 'resolved' | 'needs_input' | 'na'
    citation: dict[str, Optional[str]] = field(default_factory=dict)
    rule_applied: Optional[str] = None


def _load_manifest(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _iter_part_i_fields(part_i: dict) -> Iterator[dict]:
    """Flatten Part I: yield every leaf field, parents before sub_fields."""
    for q_key, q_def in part_i.items():
        if "field_id" in q_def:
            yield q_def
            for sub in q_def.get("sub_fields", []):
                yield sub
        elif "sub_fields" in q_def:
            for sub in q_def["sub_fields"]:
                yield sub


def _resolve_lookup(
    field_def: dict, detection: PDFDetectionReport, engagement_type: str
) -> tuple[str, str]:
    rule = field_def["rule"]
    target = rule["lookup_target"]
    key = _LOOKUP_TARGET_TO_KEY.get(target)

    if engagement_type == ENGAGEMENT_INITIAL and "initial_audit_default" in field_def:
        return field_def["initial_audit_default"], "Initial engagement — no PY report"

    section = detection.sections.get(key) if key else None
    if section and section.found:
        loc = section.location or "doc"
        return rule["if_found"], f"Found '{section.match_text}' in PDF {loc}"
    return rule["if_not_found"], f"'{target}' not present in PDF"


def _resolve_sop_fixed_plus_lookup(
    field_def: dict, detection: PDFDetectionReport, engagement_type: str
) -> tuple[str, str, str]:
    rule = field_def["rule"]
    value = rule["value"]
    ref = rule["reference_lookup"]
    target = ref["lookup_target"]
    key = _LOOKUP_TARGET_TO_KEY.get(target)

    if engagement_type == ENGAGEMENT_INITIAL:
        default_ref = field_def.get("initial_audit_default_reference", ref["if_not_found_reference"])
        return value, default_ref, "Initial engagement — no PY report"

    section = detection.sections.get(key) if key else None
    if section and section.found:
        return value, ref["if_found_reference"], f"Found '{section.match_text}' in PDF"
    return value, ref["if_not_found_reference"], f"'{target}' not present in PDF"


def _resolve_acceptance_decision(
    resolved: dict[str, ResolvedField],
) -> tuple[str, str]:
    for fid in _ACCEPTANCE_BLOCKERS:
        rf = resolved.get(fid)
        if rf and rf.value == "Yes":
            return "DO NOT ACCEPT / CONTINUE", f"Blocked: {fid} == Yes"
    return "ACCEPT / CONTINUE", "All blocker fields evaluate to No"


def _resolve_field(
    field_def: dict,
    detection: PDFDetectionReport,
    engagement_type: str,
    auditor_inputs: dict[str, Any],
    resolved_so_far: dict[str, ResolvedField],
) -> ResolvedField:
    fid = field_def["field_id"]
    source = field_def.get("source")
    rule = field_def.get("rule", {})
    citation = {"sop": field_def.get("sop_reference"), "pdf": None}

    if source == "sop_fixed":
        # Part II branching: rule may carry recurring/initial variants.
        if isinstance(rule, dict) and "recurring" in rule:
            if engagement_type == ENGAGEMENT_RECURRING:
                return ResolvedField(fid, rule["recurring"], source, "na", citation)
            initial = rule["initial"]
            if isinstance(initial, dict):
                return ResolvedField(fid, initial.get("value"), source, "resolved", citation)
            return ResolvedField(fid, initial, source, "resolved", citation)
        value = rule.get("value") or rule.get("prefill_text")
        if value is not None and rule.get("prefill_text") and rule.get("value"):
            value = f"{rule['value']} — {rule['prefill_text']}"
        return ResolvedField(fid, value, source, "resolved", citation)

    if source == "py_audit_report_lookup":
        value, pdf_cite = _resolve_lookup(field_def, detection, engagement_type)
        citation["pdf"] = pdf_cite
        return ResolvedField(
            fid, value, source, "resolved", citation,
            rule_applied=f"{rule['lookup_target']} → {value}",
        )

    if source == "sop_fixed_plus_lookup":
        value, reference, pdf_cite = _resolve_sop_fixed_plus_lookup(
            field_def, detection, engagement_type
        )
        citation["pdf"] = pdf_cite
        return ResolvedField(
            fid, f"Refer: {reference}", source, "resolved", citation,
            rule_applied=f"Compliance lookup → {reference}",
        )

    if source == "sop_fixed_plus_manual":
        if engagement_type == ENGAGEMENT_RECURRING:
            return ResolvedField(fid, "N/A", source, "na", citation)
        remark = auditor_inputs.get(fid)
        if remark:
            return ResolvedField(fid, f"Yes — {remark}", source, "resolved", citation)
        return ResolvedField(fid, "Yes (remark required)", source, "needs_input", citation)

    if source in ("auditor_selection", "manual_entry"):
        provided = auditor_inputs.get(fid)
        if provided not in (None, ""):
            return ResolvedField(fid, str(provided), source, "resolved", citation)
        return ResolvedField(fid, None, source, "needs_input", citation)

    if source == "derived":
        if fid == "acceptance_decision":
            value, applied = _resolve_acceptance_decision(resolved_so_far)
            return ResolvedField(fid, value, source, "resolved", citation, rule_applied=applied)

    return ResolvedField(fid, None, source or "unknown", "needs_input", citation)


def resolve_workpaper(
    pdf_path: str | Path,
    engagement_type: str = ENGAGEMENT_RECURRING,
    auditor_inputs: Optional[dict[str, Any]] = None,
    manifest_path: str | Path = _DEFAULT_MANIFEST_PATH,
) -> dict[str, ResolvedField]:
    """
    Resolve every field in NPO-CX-1.1 for one engagement.

    Returns an ordered dict of field_id → ResolvedField, with fields evaluated
    in the order: header → Part I (Q1–Q16) → acceptance_decision → sign_off
    → engagement_type → Part II.
    """
    auditor_inputs = dict(auditor_inputs or {})
    manifest = _load_manifest(manifest_path)
    detection = detect(pdf_path)

    resolved: dict[str, ResolvedField] = {}

    def _process(field_def: dict) -> None:
        rf = _resolve_field(field_def, detection, engagement_type, auditor_inputs, resolved)
        resolved[rf.field_id] = rf

    for f in manifest["header_fields"]:
        _process(f)

    for f in _iter_part_i_fields(manifest["part_i"]):
        _process(f)

    _process(manifest["acceptance_decision"])

    for f in manifest["sign_off"]:
        _process(f)

    et = manifest["engagement_type"]
    resolved[et["field_id"]] = ResolvedField(
        et["field_id"], engagement_type, "auditor_selection", "resolved",
        {"sop": et["sop_reference"], "pdf": None},
    )

    for f in manifest["part_ii"]:
        _process(f)

    return resolved


def summarize(resolved: dict[str, ResolvedField]) -> dict[str, int]:
    counts = {"resolved": 0, "needs_input": 0, "na": 0}
    for rf in resolved.values():
        counts[rf.status] = counts.get(rf.status, 0) + 1
    return counts
