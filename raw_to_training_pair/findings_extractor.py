"""
raw_to_training_pair/findings_extractor.py
===========================================
Phase C2 — structured findings extraction step.
[same module docstring as before]
"""

from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from auditai_data_normalization.confidence import TierConfig
    from auditai_data_normalization.schema import DocumentRecord

logger = logging.getLogger(__name__)

_RISK_REGISTRY: dict[str, dict] = {
    "client_name": {
        "label":    "Client Name",
        "risk":     "Workpaper cannot be linked to an engagement — violates documentation standards.",
        "severity": "High",
        "sop_hint": "engagement acceptance",
    },
    "fiscal_year_end": {
        "label":    "Fiscal Year End",
        "risk":     "Audit period is undefined — evidence cannot be evaluated for timeliness or completeness.",
        "severity": "High",
        "sop_hint": "period under audit",
    },
    "engagement_decision": {
        "label":    "Engagement Decision (Accept/Continue)",
        "risk":     "Engagement was initiated without documented acceptance — violates firm independence and client acceptance procedures.",
        "severity": "High",
        "sop_hint": "acceptance continuance",
    },
    "engagement_partner": {
        "label":    "Engagement Partner",
        "risk":     "Engagement lacks authorized partner sign-off — violates independence requirements and firm authorization procedures.",
        "severity": "High",
        "sop_hint": "partner authorization",
    },
    "audit_type": {
        "label":    "Audit Type",
        "risk":     "Applicable auditing standards cannot be determined — GAAS, GAGAS, and Single Audit requirements may not be met.",
        "severity": "High",
        "sop_hint": "audit type standards",
    },
    "includes_gagas": {
        "label":    "Government Auditing Standards (GAGAS)",
        "risk":     "Yellow Book independence, CPE, and reporting requirements may have been overlooked.",
        "severity": "High",
        "sop_hint": "government auditing standards yellow book",
    },
    "includes_single_audit": {
        "label":    "Single Audit (2 CFR 200)",
        "risk":     "Uniform Guidance compliance requirements and SEFA reporting may not have been addressed.",
        "severity": "High",
        "sop_hint": "single audit uniform guidance",
    },
    "reporting_framework": {
        "label":    "Reporting Framework",
        "risk":     "Applicable accounting standards (GAAP/GASB/special purpose) are undocumented — financial statement presentation may be non-compliant.",
        "severity": "High",
        "sop_hint": "basis of accounting reporting framework",
    },
    "document_reference": {
        "label":    "Document Reference",
        "risk":     "Workpaper cannot be cross-referenced within the engagement file.",
        "severity": "Low",
        "sop_hint": "workpaper documentation",
    },
    "includes_gaas_audit": {
        "label":    "GAAS Financial Statement Audit",
        "risk":     "Scope of financial statement audit is not documented.",
        "severity": "Medium",
        "sop_hint": "financial statement audit scope",
    },
    "includes_grant_compliance": {
        "label":    "Grant Compliance Audit",
        "risk":     "Federal program compliance requirements may not have been addressed.",
        "severity": "Medium",
        "sop_hint": "grant compliance federal programs",
    },
    "preparation_date": {
        "label":    "Preparation Date",
        "risk":     "Timeliness of workpaper completion cannot be assessed.",
        "severity": "Low",
        "sop_hint": "documentation timeliness",
    },
    "partner_sign_date": {
        "label":    "Partner Sign Date",
        "risk":     "Authorization timestamp is missing — independence dating cannot be confirmed.",
        "severity": "Medium",
        "sop_hint": "partner sign off date",
    },
    "ein": {
        "label":    "EIN / Federal Tax ID",
        "risk":     "Entity identity cannot be independently confirmed against federal records.",
        "severity": "High",
        "sop_hint": "entity identification",
    },
    "includes_nonattest_services": {
        "label":    "Non-Attest Services",
        "risk":     "Independence threat assessment for non-attest services is incomplete.",
        "severity": "Medium",
        "sop_hint": "nonattest services independence",
    },
    "financial_statement_use": {
        "label":    "Intended Use of Financial Statements",
        "risk":     "Report distribution restrictions may not have been applied correctly.",
        "severity": "Low",
        "sop_hint": "financial statement use distribution",
    },
}

_LLM_ONLY_RISK = "Value was extracted by AI inference — requires auditor verification before reliance."
_LLM_ONLY_SEVERITY = "Informational"

# ---------------------------------------------------------------------------
# Score thresholds
# ---------------------------------------------------------------------------
# A single-extractor document always scores 0.6 per field (score_confidence
# with ran_count=1, has_value=True). This is NOT low confidence — it is the
# maximum achievable score without a secondary extractor. Generating a finding
# for a 0.6 field would flag every field in every DOCX as uncertain, making
# clean pairs indistinguishable from deficient ones.
#
# Rules:
#   score == 0.0  → field is genuinely missing → generate finding
#   0.0 < score < _LOW_CONF_THRESHOLD → extracted but uncertain → low_confidence finding
#   score >= _LOW_CONF_THRESHOLD → present and sufficiently confident → no finding
#
# _LOW_CONF_THRESHOLD is set BELOW the single-extractor floor (0.6) so that
# single-extractor fields are treated as "present" not "uncertain".
# Only fields with genuinely low scores (e.g. LLM contradiction at 0.2,
# or partial extraction) generate low_confidence findings.

_LOW_CONF_THRESHOLD = 0.50   # below single-extractor floor of 0.6


# ---------------------------------------------------------------------------
# SOP section resolver
# ---------------------------------------------------------------------------

_SOP_SECTION_RE = re.compile(
    r"SOP\s*§\s*([\dA-Za-z][A-Za-z0-9]*(?:[.\-][\dA-Za-z0-9]+)*(?:\([^)]*\))?)",
    re.IGNORECASE,
)


def _resolve_sop_section(
    sop_chunks: list[dict],
    hint: str,
) -> tuple[str, str]:
    if not sop_chunks or not hint:
        return "", ""

    hint_words = set(hint.lower().split())
    best_score = 0
    best_chunk = None

    for chunk in sop_chunks:
        chunk_text = (
            chunk.get("text", "") + " " +
            chunk.get("heading", "") + " " +
            chunk.get("section", "")
        ).lower()
        score = sum(1 for w in hint_words if w in chunk_text)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None or best_score == 0:
        return "", ""

    chunk_text = best_chunk.get("text", "")
    section_ref = best_chunk.get("section", "")

    if section_ref and not section_ref.startswith("SOP"):
        section_ref = f"SOP {section_ref}"

    # Fallback 1: scan chunk text for SOP §X.X pattern
    if not section_ref:
        m = _SOP_SECTION_RE.search(chunk_text)
        if m:
            section_ref = f"SOP §{m.group(1)}"

    # Fallback 2: extract Q# reference from chunk text (e.g. Q1(a), Q9(b))
    # This covers SOP documents where section labels are Q-numbers not §-numbers
    if not section_ref:
        q_match = re.search(r"\bQ(\d+(?:\([a-z]\))?)", chunk_text, re.IGNORECASE)
        if q_match:
            section_ref = f"SOP-QM-001 §{q_match.group(0)}"

    # Final fallback: use the SOP document ID without section
    if not section_ref:
        # Try to get SOP ID from heading or chunk metadata
        heading = best_chunk.get("heading", "")
        sop_id_match = re.search(r"SOP[-\s]?[A-Z]{2,}-\d+", heading + " " + chunk_text)
        section_ref = sop_id_match.group(0) if sop_id_match else "SOP-QM-001"

    snippet = chunk_text[:200].strip()
    if len(chunk_text) > 200:
        snippet += "..."

    return section_ref, snippet


def label_for_field(field_name: str) -> str:
    entry = _RISK_REGISTRY.get(field_name)
    if entry:
        return entry["label"]
    return field_name.replace("_", " ").title()


def extract_findings(
    record: "DocumentRecord",
    sop_chunks: list[dict],
    tiers: "TierConfig",
    flagged_fields: list[str] | None = None,
    deficiency_fields: list[str] | None = None,
    client_type: str | None = None,
) -> list[dict]:
    """
    Extract structured findings from a DocumentRecord.

    Fix 1 (clean pair correctness): fields with score >= _LOW_CONF_THRESHOLD
    (0.50) are treated as present — no finding generated. Single-extractor
    score of 0.6 is above this threshold so extracted fields are never
    flagged as deficiencies on clean pairs.

    Fix 2 (deficient pair isolation): deficiency_fields override only
    generates a finding for the specific missing field. Other present
    fields are not flagged unless genuinely missing (score == 0.0) or
    below the low-confidence threshold.

    Fix 3 (tier1_missing source of truth): tier1_missing from
    confidence_summary is cross-checked against per_field_scores to avoid
    stale values from before PII scrubbing. A field is only treated as
    missing if BOTH tier1_missing lists it AND its score == 0.0.

    R3 (sop_field_classes guard): fields classified as fixed_administrative
    or informational_only in sop_field_classes.yaml are excluded from finding
    generation regardless of their extraction score. Client-type-aware —
    e.g. includes_gagas is excluded for For-Profit clients.
    """
    conf_summary = record.metadata.get("confidence_summary", {})
    per_field_scores: dict[str, float] = conf_summary.get("per_field_scores", {})

    # Fix 3: use per_field_scores as source of truth for missing fields
    # tier1_missing from confidence_summary can be stale after PII scrubbing
    tier1_genuinely_missing = {
        f for f in tiers.tier1
        if per_field_scores.get(f, 0.0) == 0.0
    }

    flagged = set(flagged_fields or record.flagged_fields or [])
    deficient = set(deficiency_fields or [])

    # R3 — SOP field class guard
    # Fields classified as fixed_administrative or informational_only for this
    # client type must never generate findings, regardless of extraction score.
    try:
        from config.settings import load_sop_field_classes
        _non_deficient: frozenset[str] = (
            load_sop_field_classes().non_deficient_canonical_fields(client_type)
        )
    except Exception as _e:
        logger.debug("findings_extractor: sop_field_classes unavailable — %s", _e)
        _non_deficient = frozenset()

    findings: list[dict] = []

    def _make_finding(
        field: str,
        status: str,
        tier: str,
        deficient_override: bool = False,
    ) -> dict | None:
        entry = _RISK_REGISTRY.get(field)
        if entry is None:
            return None

        sop_section, sop_snippet = _resolve_sop_section(
            sop_chunks, entry["sop_hint"]
        )

        return {
            "field":       field,
            "tier":        tier,
            "status":      status,
            "label":       entry["label"],
            "risk":        entry["risk"],
            "severity":    entry["severity"],
            "sop_section": sop_section,
            "sop_snippet": sop_snippet,
            "deficient":   deficient_override or (status in ("missing", "low_confidence")),
        }

    # ── Tier 1 findings ───────────────────────────────────────────────
    for field in sorted(tiers.tier1):
        if field in _non_deficient:
            logger.debug(
                "findings_extractor: skipping %s — non-deficient for client_type=%s",
                field, client_type,
            )
            continue

        score = per_field_scores.get(field, 0.0)
        is_deficient_variant = field in deficient

        if is_deficient_variant:
            # Deficient variant: generate finding only for THIS specific field
            finding = _make_finding(field, "missing", "tier1", True)

        elif field in tier1_genuinely_missing:
            # Field is genuinely absent from the document
            finding = _make_finding(field, "missing", "tier1")

        elif score < _LOW_CONF_THRESHOLD:
            # Fix 1: only flag if below 0.50 — single extractor (0.6) is NOT low confidence
            finding = _make_finding(field, "low_confidence", "tier1")

        else:
            # Present and sufficiently confident — no finding for clean pair
            continue

        if finding:
            findings.append(finding)

    # ── Tier 2 findings ───────────────────────────────────────────────
    for field in sorted(tiers.tier2):
        if field in _non_deficient:
            logger.debug(
                "findings_extractor: skipping %s — non-deficient for client_type=%s",
                field, client_type,
            )
            continue

        score = per_field_scores.get(field, 0.0)
        is_deficient_variant = field in deficient

        if is_deficient_variant or score == 0.0:
            finding = _make_finding(field, "missing", "tier2", is_deficient_variant)
            if finding:
                if finding["severity"] == "High":
                    finding["severity"] = "Medium"
                findings.append(finding)

        elif score < _LOW_CONF_THRESHOLD:
            # Fix 1: same threshold applied to tier2
            finding = _make_finding(field, "low_confidence", "tier2")
            if finding:
                finding["severity"] = "Low"
                findings.append(finding)

    # ── Informational flags (LLM-only / flagged fields) ───────────────
    # Only append if the field doesn't already have a finding above.
    # Without this guard, fields like document_reference appear twice:
    # once as a real Tier 2 finding and again as an Informational flag.
    already_in_findings = {f["field"] for f in findings}

    for field in sorted(flagged):
        score = per_field_scores.get(field, 0.0)
        if score > 0.0 and field not in deficient and field not in already_in_findings:
            entry = _RISK_REGISTRY.get(field)
            label = entry["label"] if entry else label_for_field(field)
            findings.append({
                "field":       field,
                "tier":        tiers.tier_of(field),
                "status":      "llm_only",
                "label":       label,
                "risk":        _LLM_ONLY_RISK,
                "severity":    _LLM_ONLY_SEVERITY,
                "sop_section": "",
                "sop_snippet": "",
                "deficient":   False,
            })

    logger.debug(
        "findings_extractor: %d findings (%d deficient) for %s",
        len(findings),
        sum(1 for f in findings if f["deficient"]),
        record.file_name,
    )

    return findings