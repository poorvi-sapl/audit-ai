"""
auditai_data_normalization/extractors/llm_extractor.py
=========================================================
LLM-based field extractor using Gemma 3 12B via Ollama.

Two extraction modes (Phase B1)
--------------------------------
Mode 1 — TIEBREAKER (original, unchanged)
    Called when deterministic extractors disagree on specific fields.
    entry: extract_fields(text, fields_to_resolve)

Mode 2 — FALLBACK (Phase B1/B2, new)
    Called when extraction_confidence < 0.50.
    Attempts all Tier 1 + Tier 2 fields in one shot.
    Includes doc type hint + inline label variant hints.
    Returns FieldResult per field for B3 confidence calibration.
    entry: extract_all_fields(text, doc_type, tiers)

Public API
----------
    extract_fields(text, fields_to_resolve)    -> dict[str, str]
    extract_all_fields(text, doc_type, tiers)  -> dict[str, FieldResult]
    is_available()                             -> bool
    extract_fields_from_record(record, ...)    -> dict[str, str]
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_MODEL       = "gemma3:12b"
_TEMPERATURE = 0.0
_OLLAMA_HOST = "http://localhost:11434"
_MAX_WORDS_TIEBREAKER = 3000
_MAX_WORDS_FALLBACK   = 6000


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    """Per-field output for fallback mode. Used by B3 for confidence calibration."""
    value: str = ""
    found: bool = False
    llm_confident: bool = False
    source_hint: str = ""

    def __bool__(self) -> bool:
        return self.found


# ---------------------------------------------------------------------------
# Document type context for fallback prompt
# ---------------------------------------------------------------------------

_DOC_TYPE_CONTEXT: dict[str, dict] = {
    "engagement_form": {
        "description": "an audit engagement acceptance or continuance form",
        "emphasis": (
            "This form decides whether the firm accepts or continues the engagement. "
            "Priority fields: engagement_decision, engagement_partner, audit_type, "
            "includes_gagas, includes_single_audit, client_name, fiscal_year_end."
        ),
        "label_hints": {
            "engagement_partner":    "EP: / Engagement Partner / Partner Name / EP Approval / Partner Authorization / Responsible CPA / Signing Partner",
            "engagement_decision":   "Accept / Continue / Decline / Go / No-Go / Acceptance or Continuance / We Should / Should Not Accept",
            "audit_type":            "Type of Audit / Engagement Type / Services to be Provided / Type of Services",
            "includes_gagas":        "Government Auditing Standards / Yellow Book / GAGAS / GAS Audit",
            "includes_single_audit": "Single Audit / 2 CFR 200 / Uniform Guidance / A-133 / Federal Program Audit / SEFA Required",
            "reporting_framework":   "Basis of Accounting / GAAP / GASB / Special Purpose Framework",
            "fiscal_year_end":       "Year Ended / FYE / Period End / As of Date / Balance Sheet Date / Period Covered",
            "client_name":           "Organization / Client / Entity Name / Name of Entity",
            "includes_gaas_audit":   "Audit of Financial Statements / GAAS Audit / Financial Statement Audit",
            "includes_grant_compliance": "Grant Compliance / Grant Audit / Program Compliance / Federal Compliance",
            "includes_nonattest_services": "Nonattest Services / Nonaudit Services / Bookkeeping / Tax Services",
            "ein":                   "EIN / Federal ID / Tax ID / FEIN / Federal Employer ID",
            "preparation_date":      "Date / Completed Date / Date Prepared / Date of Form",
            "partner_sign_date":     "Partner Date / Sign Date / Date Signed / EP Date",
            "document_reference":    "Ref / W/P No. / Form No. / Index / Workpaper Number",
        },
    },
    "financial_statement": {
        "description": "an audit report or set of financial statements",
        "emphasis": (
            "Look for entity name and fiscal year in the heading. "
            "The auditor report section contains engagement_partner, audit_type, opinion_type. "
            "Balance sheet has total_assets, liabilities, net_assets. "
            "Income statement has total_revenue, total_expenses."
        ),
        "label_hints": {
            "engagement_partner":  "CPA firm signatory / Partner / Engagement Partner",
            "audit_type":          "Type of engagement in the auditor report",
            "reporting_framework": "GAAP / GASB / Basis of Accounting",
            "total_assets":        "Total Assets / Assets Total / TOTAL ASSETS",
            "total_revenue":       "Total Revenues and Support / Total Revenue / Total Income",
            "net_assets":          "Net Assets / Fund Balance / Net Position / Equity",
        },
    },
    "planning_memo": {
        "description": "an audit planning memorandum or risk assessment",
        "emphasis": (
            "Narrative document. Client identity and engagement scope are usually in the header. "
            "Engagement partner named in the sign-off section."
        ),
        "label_hints": {
            "engagement_partner":    "Engagement Partner / Signing Partner / EP",
            "audit_type":            "Scope / Type of Engagement",
            "includes_gagas":        "Government Auditing Standards / Yellow Book",
            "includes_single_audit": "Single Audit / Uniform Guidance / 2 CFR 200",
        },
    },
    "unknown": {
        "description": "an audit-related document",
        "emphasis": (
            "Extract whatever fields you can find. Prioritize: "
            "client_name, fiscal_year_end, engagement_partner, audit_type, engagement_decision."
        ),
        "label_hints": {},
    },
}

_DOC_TYPE_MAP = {
    "Engagement Form": "engagement_form",
    "engagement_form": "engagement_form",
    "Financial Statement": "financial_statement",
    "financial_statement": "financial_statement",
    "Planning Memo": "planning_memo",
    "Risk Assessment": "planning_memo",
    "planning_memo": "planning_memo",
}

def _resolve_doc_type(doc_type: str | None) -> str:
    if not doc_type:
        return "unknown"
    return _DOC_TYPE_MAP.get(doc_type, "unknown")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a JSON data extraction API for audit workpapers. Output ONLY valid JSON.\n"
    "RULES:\n"
    "- First character must be {, last must be }\n"
    "- No markdown, no explanation, no preamble\n"
    "- Use null for fields not found\n"
    "- All values must be strings\n"
    "- Dates in YYYY-MM-DD format\n"
    "- Yes/Applicable/Checked -> \"true\", No/Not Applicable -> \"false\"\n"
    "- Do NOT guess or infer values not explicitly in the document"
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_tiebreaker_prompt(text: str, fields: list[str]) -> str:
    words = text.split()
    if len(words) > _MAX_WORDS_TIEBREAKER:
        text = " ".join(words[:_MAX_WORDS_TIEBREAKER]) + "\n[... truncated ...]"
    fields_json = json.dumps({f: None for f in fields}, indent=2)
    return (
        f"Extract these fields from the audit document below.\n"
        f"Return a JSON object with exactly these keys:\n{fields_json}\n\n"
        f"DOCUMENT TEXT:\n{text}\n\n"
        f"Return ONLY the JSON object. Use null for any field not found."
    )


def _build_fallback_prompt(
    text: str,
    tier1_fields: list[str],
    tier2_fields: list[str],
    doc_type_key: str,
) -> str:
    ctx = _DOC_TYPE_CONTEXT[doc_type_key]
    hints = ctx["label_hints"]

    words = text.split()
    if len(words) > _MAX_WORDS_FALLBACK:
        text = " ".join(words[:_MAX_WORDS_FALLBACK]) + "\n[... truncated ...]"

    def field_line(fname: str) -> str:
        hint = hints.get(fname, "")
        comment = f"  // labels: {hint}" if hint else ""
        return f'  "{fname}": null{comment}'

    t1_lines = "\n".join(field_line(f) for f in tier1_fields)
    t2_lines = "\n".join(field_line(f) for f in tier2_fields)

    example = json.dumps({
        "_example": {
            "value": "extracted string or null",
            "confident": True,
            "source_hint": "3-10 words from the document"
        }
    }, indent=2)

    return (
        f"You are extracting fields from {ctx['description']}.\n\n"
        f"{ctx['emphasis']}\n\n"
        f"TIER 1 - CRITICAL FIELDS (extract first):\n{t1_lines}\n\n"
        f"TIER 2 - IMPORTANT FIELDS:\n{t2_lines}\n\n"
        f"RESPONSE FORMAT for each field:\n{example}\n\n"
        f"Rules:\n"
        f"- value: the extracted string, or null if not found\n"
        f"- confident: true if clearly in the document, false if uncertain\n"
        f"- source_hint: 3-10 words from the document containing this value\n"
        f"- Do NOT guess. If not in the document, use null.\n\n"
        f"DOCUMENT TEXT:\n{text}\n\n"
        f"Return ONLY the JSON object."
    )


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _clean_json(raw: str) -> str:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return m.group(0) if m else cleaned


def _parse_response(raw: str) -> dict[str, str]:
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(_clean_json(raw))
    except json.JSONDecodeError as e:
        logger.warning("llm_extractor: JSON parse failed — %s | raw: %.200s", e, raw)
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {k: "" if v is None else str(v).strip() for k, v in parsed.items()}


def _parse_fallback_response(
    raw: str,
    expected_fields: list[str],
) -> dict[str, FieldResult]:
    empty = {f: FieldResult() for f in expected_fields}
    if not raw or not raw.strip():
        return empty
    try:
        parsed = json.loads(_clean_json(raw))
    except json.JSONDecodeError as e:
        logger.warning("llm_extractor fallback: JSON parse failed — %s | raw: %.200s", e, raw)
        return empty
    if not isinstance(parsed, dict):
        return empty

    results: dict[str, FieldResult] = {}
    for fname in expected_fields:
        raw_val = parsed.get(fname)
        if raw_val is None:
            results[fname] = FieldResult()
        elif isinstance(raw_val, dict):
            val = raw_val.get("value")
            val_str = "" if val is None else str(val).strip()
            conf_raw = raw_val.get("confident", False)
            confident = (conf_raw.lower() == "true") if isinstance(conf_raw, str) else bool(conf_raw)
            results[fname] = FieldResult(
                value=val_str,
                found=bool(val_str),
                llm_confident=confident,
                source_hint=str(raw_val.get("source_hint", "")).strip()[:200],
            )
        else:
            val_str = str(raw_val).strip()
            results[fname] = FieldResult(value=val_str, found=bool(val_str))
    return results


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def is_available() -> bool:
    try:
        import ollama
        models = ollama.list()
        names = [m.model for m in models.models]
        available = any(_MODEL in n for n in names)
        if not available:
            logger.debug("llm_extractor: %s not in ollama models: %s", _MODEL, names)
        return available
    except Exception as e:
        logger.debug("llm_extractor: Ollama not reachable — %s", e)
        return False


# ---------------------------------------------------------------------------
# Mode 1 — Tiebreaker
# ---------------------------------------------------------------------------

def extract_fields(
    text: str,
    fields_to_resolve: list[str] | None = None,
) -> dict[str, str]:
    """Tiebreaker: extract specific disagreed fields. Returns dict[field, value_str]."""
    if not text or not text.strip():
        return {}

    _ALL = [
        "client_name", "fiscal_year_end", "engagement_partner",
        "preparation_date", "partner_sign_date", "audit_type",
        "includes_gaas_audit", "includes_single_audit", "includes_gagas",
        "includes_nonattest_services", "engagement_decision",
        "reporting_framework", "document_reference",
        "includes_grant_compliance", "ein", "financial_statement_use",
    ]
    fields = fields_to_resolve if fields_to_resolve else _ALL

    if not is_available():
        logger.warning("llm_extractor: Ollama unavailable — skipping tiebreaker")
        return {}

    try:
        import ollama
        resp = ollama.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_tiebreaker_prompt(text, fields)},
            ],
            options={"temperature": _TEMPERATURE, "num_predict": 1024},
            format="json",
        )
        result = {k: v for k, v in _parse_response(resp.message.content or "").items() if k in fields}
        logger.info("llm_extractor tiebreaker: %d/%d resolved", sum(1 for v in result.values() if v), len(fields))
        return result
    except Exception as e:
        logger.warning("llm_extractor tiebreaker failed — %s", e)
        return {}


# ---------------------------------------------------------------------------
# Mode 2 — Fallback  (Phase B1)
# ---------------------------------------------------------------------------

def extract_all_fields(
    text: str,
    doc_type: str | None = None,
    tiers: Any | None = None,
) -> dict[str, FieldResult]:
    """
    Fallback: extract all Tier 1 + Tier 2 fields in one LLM call.

    Called when extraction_confidence < 0.50 (Phase B2).
    Returns FieldResult per field for B3 confidence calibration.

    Parameters
    ----------
    text     : cleaned_text from DocumentRecord (pii_scrubbed=True)
    doc_type : document_category from record.metadata
    tiers    : TierConfig from confidence.load_tiers()
    """
    if not text or not text.strip():
        return {}

    doc_type_key = _resolve_doc_type(doc_type)

    if tiers is not None:
        tier1 = sorted(tiers.tier1)
        tier2 = sorted(tiers.tier2)
    else:
        tier1 = [
            "client_name", "fiscal_year_end", "engagement_decision",
            "engagement_partner", "audit_type", "includes_gagas",
            "includes_single_audit", "reporting_framework",
        ]
        tier2 = [
            "document_reference", "includes_gaas_audit", "includes_grant_compliance",
            "preparation_date", "partner_sign_date", "ein",
            "includes_nonattest_services", "financial_statement_use",
        ]

    all_fields = tier1 + tier2

    if not is_available():
        logger.warning("llm_extractor: Ollama unavailable — fallback skipped")
        return {}

    try:
        import ollama
        resp = ollama.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_fallback_prompt(text, tier1, tier2, doc_type_key)},
            ],
            options={"temperature": _TEMPERATURE, "num_predict": 2048},
            format="json",
        )
        results = _parse_fallback_response(resp.message.content or "", all_fields)
        found = sum(1 for r in results.values() if r.found)
        confident = sum(1 for r in results.values() if r.llm_confident)
        logger.info(
            "llm_extractor fallback: %d/%d found (%d confident) doc_type=%s",
            found, len(all_fields), confident, doc_type_key,
        )
        return results
    except Exception as e:
        logger.warning("llm_extractor fallback failed — %s", e)
        return {}


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def extract_fields_from_record(
    record: Any,
    fields_to_resolve: list[str] | None = None,
) -> dict[str, str]:
    """Tiebreaker mode on a DocumentRecord. Refuses if pii_scrubbed=False."""
    if not record.pii_scrubbed:
        logger.warning("llm_extractor: %s pii_scrubbed=False — refusing", record.file_name)
        return {}
    return extract_fields(record.cleaned_text, fields_to_resolve)