"""
raw_to_training_pair/pair_builder.py
=====================================
Phase E1 additions + Fix 2 (variant differentiation).

Fix 2: _build_user_content() now prepends a VARIANT TYPE header that
explicitly tells the model whether this is a clean or deficient pair.
Without this, when a field is already missing from the document,
clean and deficient variants produce identical completions because
the model sees no difference in the input.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from auditai_data_normalization.schema import DocumentRecord
from raw_to_training_pair.completion_drafter import draft, is_available
from raw_to_training_pair.findings_extractor import extract_findings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert audit assistant for Harshwal & Company LLP (HCLLP), \
a US-based CPA firm specializing in nonprofit, governmental, and single audit \
engagements under GAAS, GAGAS (Yellow Book), and the Uniform Guidance \
(2 CFR Part 200).

Your role is to analyze audit workpapers and produce structured audit findings \
and recommendations that:
- Cite specific SOP sections for every finding
- Apply the correct audit standard based on client type and engagement scope
- Flag missing required fields as documentation deficiencies
- Use professional, concise, audit-standard language throughout

Always follow the required output format exactly."""

_CLIENT_TYPE_DESCRIPTIONS = {
    "NPO":        "nonprofit organization (IRS 501(c)(3) or similar)",
    "Government": "state or local government entity",
    "For-Profit": "for-profit commercial entity",
    "Tribal":     "federally recognized tribal government or entity",
}

_FIRM_SPECIFIC_FIELDS = {
    "engagement_partner", "preparer_id", "reviewer_id",
    "engagement_code", "document_reference", "partner_sign_date",
}


def _pair_hash(messages: list[dict]) -> str:
    content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _assign_stage(record: DocumentRecord) -> str:
    confidence_summary = record.metadata.get("confidence_summary", {})
    fields_present = set(confidence_summary.get("fields_present", []))
    return "stage3" if fields_present & _FIRM_SPECIFIC_FIELDS else "stage2"


def _build_user_content(
    record: DocumentRecord,
    client_type: str,
    is_gagas: bool,
    has_single_audit: bool,
    deficiency_fields: list[str],
) -> str:
    confidence_summary = record.metadata.get("confidence_summary", {})
    fields_present = confidence_summary.get("fields_present", [])
    fields_missing = confidence_summary.get("fields_missing", [])
    all_missing = list(set(fields_missing + deficiency_fields))

    text_excerpt = record.cleaned_text or ""
    words = text_excerpt.split()
    if len(words) > 1500:
        text_excerpt = " ".join(words[:1500]) + "\n[... excerpt truncated ...]"

    client_desc = _CLIENT_TYPE_DESCRIPTIONS.get(client_type, client_type or "unknown")

    # Fix 2 — variant type header
    # Explicitly signals to the model whether this is a clean or deficient pair.
    # Without this, when a field is already missing from the document,
    # the model sees identical input for clean and deficient variants.
    is_deficient = bool(deficiency_fields)
    if is_deficient:
        variant_header = (
            "VARIANT TYPE: DEFICIENT\n"
            "The following required field(s) have been intentionally removed to "
            "simulate a documentation deficiency. Your findings MUST include each "
            "removed field as a High severity deficiency with its specific audit "
            f"risk and SOP citation.\n"
            f"REMOVED FIELDS: {', '.join(deficiency_fields)}\n"
        )
    else:
        # For clean pairs: list genuinely missing fields explicitly so Gemma
        # generates specific SOP citations rather than broad generic references.
        genuinely_missing = [
            f for f in fields_missing
            if f not in deficiency_fields
        ]
        missing_hint = (
            f"FIELDS GENUINELY ABSENT FROM THIS DOCUMENT: {', '.join(genuinely_missing)}\n"
            "For each absent field, cite the specific SOP section (e.g. §Q1, §Q9(a)) "
            "that requires it.\n"
            if genuinely_missing else ""
        )
        variant_header = (
            "VARIANT TYPE: CLEAN\n"
            "All present fields are properly documented. Generate findings only "
            "for fields that are genuinely absent from this workpaper — do NOT "
            "flag fields listed under FIELDS PRESENT as deficiencies.\n"
            + missing_hint
        )

    lines = [
        variant_header,
        f"CLIENT TYPE: {client_desc}",
        f"GAGAS ENGAGEMENT: {'Yes' if is_gagas else 'No'}",
        f"SINGLE AUDIT: {'Yes' if has_single_audit else 'No'}",
        f"FILE: {record.file_name}",
        f"FILE TYPE: {record.file_type}",
        f"EXTRACTION CONFIDENCE: {record.extraction_confidence:.2f}",
        "",
        "FIELDS PRESENT:",
    ]
    for f in fields_present:
        if f not in deficiency_fields:
            lines.append(f"  {f}: [extracted]")

    if all_missing:
        lines += ["", "FIELDS MISSING (must be flagged as findings):"]
        for f in all_missing:
            if f in deficiency_fields:
                lines.append(f"  - {f}  [INTENTIONALLY REMOVED — flag as High severity deficiency]")
            else:
                lines.append(f"  - {f}")

    lines += ["", "WORKPAPER TEXT EXCERPT:", text_excerpt]
    return "\n".join(lines)


def _build_uncertain_sections(
    record: DocumentRecord,
    deficiency_fields: list[str],
) -> list[str]:
    uncertain: set[str] = set()
    uncertain.update(record.flagged_fields or [])
    per_field_scores = record.metadata.get(
        "confidence_summary", {}
    ).get("per_field_scores", {})
    for fname, score in per_field_scores.items():
        if 0.0 < score < 0.70:
            uncertain.add(fname)
    uncertain.update(deficiency_fields)
    return sorted(uncertain)


def build(
    record: DocumentRecord,
    sop_text: str,
    sop_sections: list[str],
    client_type: str,
    is_gagas: bool,
    has_single_audit: bool = False,
    stage: str | None = None,
    pair_type: str = "clean",
    deficiency_fields: list[str] | None = None,
    sop_chunks: list[dict] | None = None,
    tiers: Any | None = None,
    use_mock: bool = False,
    correction_hint: str = "",
) -> dict | None:
    if not record.pii_scrubbed:
        raise ValueError(
            f"pair_builder: {record.file_name} has pii_scrubbed=False."
        )

    deficiency_fields = deficiency_fields or []
    assigned_stage = stage or _assign_stage(record)

    if tiers is None:
        try:
            from auditai_data_normalization.confidence import load_tiers
            tiers = load_tiers()
        except Exception:
            tiers = None

    findings: list[dict] = []
    if tiers is not None:
        try:
            findings = extract_findings(
                record=record,
                sop_chunks=sop_chunks or [],
                tiers=tiers,
                flagged_fields=record.flagged_fields,
                deficiency_fields=deficiency_fields,
                client_type=client_type,
            )
        except Exception as e:
            logger.warning("pair_builder: findings_extractor failed — %s", e)

    user_content = _build_user_content(
        record=record,
        client_type=client_type,
        is_gagas=is_gagas,
        has_single_audit=has_single_audit,
        deficiency_fields=deficiency_fields,
    )

    confidence_summary = record.metadata.get("confidence_summary", {})
    fields_present = confidence_summary.get("fields_present", [])
    fields_missing = confidence_summary.get("fields_missing", [])

    fields_dict: dict[str, Any] = {f: "extracted" for f in fields_present}
    fields_dict.update({
        "client_type":      client_type,
        "is_gagas":         is_gagas,
        "has_single_audit": has_single_audit,
    })

    # Exclude deficiency_fields from fields_present so claim_mapper doesn't mark
    # intentionally-removed fields as MISANCHORED when they appear as findings.
    _deficiency_set = set(deficiency_fields)
    effective_fields_present = [f for f in fields_present if f not in _deficiency_set]

    # For clean pairs, strip llm_only/informational findings from the Gemma prompt.
    # llm_only findings flag *present* fields as AI-inferred and cause Gemma to
    # generate numbered findings about those fields — all of which are then
    # MISANCHORED by claim_mapper (the fields ARE present), dragging grounding_score
    # to 0 and capping the final score at 0.56 even when the completion is correct.
    # Deficient pairs keep llm_only findings because the grounding from genuine
    # deficiency coverage offsets the MISANCHORED penalty.
    prompt_findings = (
        [f for f in findings if f.get("status") != "llm_only"]
        if not deficiency_fields else findings
    )

    assistant_content = draft(
        record=record,
        findings=prompt_findings,
        sop_text=sop_text,
        client_type=client_type,
        is_gagas=is_gagas,
        has_single_audit=has_single_audit,
        fields=fields_dict,
        use_mock=use_mock,
        correction_hint=correction_hint,
        sop_sections=sop_sections,
        fields_missing=list(set(fields_missing + deficiency_fields)),
        fields_present=effective_fields_present,
    )

    if assistant_content is None:
        logger.warning(
            "pair_builder: draft() returned None for %s — routing to review queue",
            record.file_name,
        )
        return None

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    all_missing = list(set(fields_missing + deficiency_fields))
    uncertain_sections = _build_uncertain_sections(record, deficiency_fields)

    pair = {
        "messages": messages,
        "metadata": {
            "file_name":             record.file_name,
            "file_type":             record.file_type,
            "client_type":           client_type,
            "is_gagas":              is_gagas,
            "has_single_audit":      has_single_audit,
            "extraction_confidence": record.extraction_confidence,
            "auditor_approved":      record.auditor_approved,
            "pair_type":             pair_type,
            "stage":                 assigned_stage,
            "fields_missing":        all_missing,
            "sop_sections_used":     sop_sections,
            "file_hash":             record.file_hash,
            "pair_hash":             _pair_hash(messages),
            "review_confidence":     record.review_confidence,
            "extraction_gate":       record.extraction_gate,
            "llm_assisted":          record.llm_assisted,
            "uncertain_sections":    uncertain_sections,
        },
    }

    logger.info(
        "pair_builder: built %s pair for %s | stage=%s client=%s "
        "rev_conf=%.2f uncertain=%d",
        pair_type, record.file_name, assigned_stage, client_type,
        record.review_confidence, len(uncertain_sections),
    )

    return pair