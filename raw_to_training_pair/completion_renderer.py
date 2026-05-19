"""
raw_to_training_pair/completion_renderer.py
============================================
R7-C: Deterministic completion renderer.

Converts a ValidatedClassification (R7-B output) into a formatted audit
completion string without any LLM inference. Every decision — SOP citation,
severity, finding label, risk text — comes from precompiled tables derived
from field_tiers.yaml and sop_field_classes.yaml. The renderer is a pure
function: same classification + same config version → same completion text.

FIELD_TO_SOP table (see docs/architecture/09_r7_classifier_invariants.md)
--------------------------------------------------------------------------
Built once at pipeline startup via build_sop_mapping_table(). Never computed
at render time. Every training pair's metadata records the sop_mapping_version
so the exact config snapshot that produced each pair is reproducible.

Three-state handling
--------------------
    absent    → generate finding with SOP citation and severity
    present   → no finding generated
    uncertain → no finding generated; field name appended to uncertain_fields
                in pair metadata (Tier 1 uncertain → blocks JSONL write)

SOP citation for absent fields
-------------------------------
Primary source: sop_field_classes.yaml slug → §section derived from slug.
Fallback:       field_tiers.yaml reason text (scanned for §N.N pattern).
If neither:     sop_section = null, sop_unverified = true in metadata.
Context chunks are NOT consulted during rendering. The table is precompiled.

Public API
----------
    build_sop_mapping_table(client_type) -> SopTable
        Call once at pipeline startup. Pass the result to render_completion().

    render_completion(validated, sop_table, client_type, is_gagas,
                      has_single_audit, sop_mapping_version) -> RenderResult
        Deterministic. Returns RenderResult with completion text + metadata.

    sop_mapping_version() -> str
        Stable version key derived from config file content hashes.
        Include in every training pair's metadata.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PKG_DIR      = Path(__file__).parent
_PROJECT_DIR  = _PKG_DIR.parent
_TIERS_PATH   = _PROJECT_DIR / "config" / "field_tiers.yaml"
_SFC_PATH     = _PROJECT_DIR / "config" / "sop_field_classes.yaml"
_STDS_CTX_PATH = _PROJECT_DIR / "config" / "field_standards_context.yaml"
_FIRM         = "Harshwal & Company LLP (HCLLP)"

# SOP section pattern in free text (e.g. "SOP §2.1", "SOP §2.4")
_SOP_REF_RE = re.compile(r"§\s*([A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*)")

# Slug → §Section conversion: q1a → §Q1(a), q2 → §Q2, part_ii_q2j → §Part II Q2(j)
_SLUG_RE = re.compile(
    r"^(?:(part_[ivx]+)_)?q(\d+[a-z]?)$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SopMapping:
    """Compile-time SOP metadata for one canonical field."""
    canonical_field: str
    sop_section:     str | None   # None → sop_unverified: true on findings
    severity:        str          # High | Medium | Low | Informational
    label:           str          # human-readable finding label


@dataclass
class RenderResult:
    """Output of render_completion()."""
    completion:         str            # formatted audit completion text
    absent_fields:      list[str]      # fields that generated findings
    uncertain_fields:   list[str]      # fields with no finding + flagged
    provisional_fields: list[str]      # present fields confirmed by weak evidence
    sop_unverified:     bool           # True if any finding has no SOP section
    sop_mapping_version: str           # version of the FIELD_TO_SOP table used
    metadata:           dict[str, Any] # full metadata dict for JSONL pair


SopTable = dict[str, SopMapping]      # canonical_field → SopMapping


# ---------------------------------------------------------------------------
# Slug → §Section conversion
# ---------------------------------------------------------------------------

def _slug_to_section(slug: str) -> str | None:
    """
    Convert a sop_field_classes.yaml slug to a formatted §Section reference.

    Examples:
        q1a        → §Q1(a)
        q2         → §Q2
        q1b        → §Q1(b)
        part_ii_q2j → §Part II Q2(j)
    """
    m = _SLUG_RE.match(slug.strip().lower())
    if not m:
        return None
    part_prefix = m.group(1)   # e.g. "part_ii" or None
    q_body      = m.group(2)   # e.g. "1a", "2", "9b"

    # Split numeric part from letter suffix
    num_match = re.match(r"(\d+)([a-z]?)$", q_body)
    if not num_match:
        return None
    num    = num_match.group(1)
    letter = num_match.group(2)

    section = f"§Q{num}" + (f"({letter})" if letter else "")
    if part_prefix:
        part_label = part_prefix.replace("_", " ").title()
        section = f"§{part_label} Q{num}" + (f"({letter})" if letter else "")
    return section


def _extract_sop_ref_from_reason(reason: str) -> str | None:
    """Scan a field_tiers.yaml reason string for an explicit §N.N reference."""
    m = _SOP_REF_RE.search(reason or "")
    return f"§{m.group(1)}" if m else None


# ---------------------------------------------------------------------------
# Severity defaults
# ---------------------------------------------------------------------------

_SEVERITY_TIER = {
    "tier1": "High",
    "tier2": "Medium",
}
_SEVERITY_VALID = frozenset({"High", "Medium", "Low", "Informational"})


def _coerce_severity(raw: Any, tier_key: str) -> str:
    if raw and str(raw).title() in _SEVERITY_VALID:
        return str(raw).title()
    return _SEVERITY_TIER.get(tier_key, "Medium")


# ---------------------------------------------------------------------------
# Consequence text registry (no embedded citations)
# ---------------------------------------------------------------------------
# Pure consequence statements — what goes wrong operationally if this field
# is absent. NO §section or standard references embedded here. Citations are
# assembled at render time by _resolve_standards() from field_standards_context.yaml.
# This ensures SopMapping.sop_section is the single authoritative citation source.

_FIELD_CONSEQUENCE_TEXT: dict[str, str] = {
    "engagement_decision": (
        "Absence of the engagement acceptance or continuance decision means the "
        "engagement lacks formal authorization."
    ),
    "engagement_partner": (
        "Missing engagement partner identification prevents independence verification "
        "and removes the required partner sign-off."
    ),
    "audit_type": (
        "Without a documented audit type, the applicable standards (GAAS, GAGAS, "
        "or Single Audit) cannot be confirmed, risking scope misalignment."
    ),
    "client_name": (
        "The workpaper cannot be linked to a specific engagement without a client "
        "name, undermining the audit file's evidentiary value."
    ),
    "fiscal_year_end": (
        "An undated engagement period makes the workpaper unusable for continuance "
        "decisions and timeline verification by supervisory review."
    ),
    "reporting_framework": (
        "Without a documented reporting framework (GAAP / GASB / special purpose), "
        "the basis of accounting cannot be confirmed or disclosed appropriately."
    ),
    "includes_gagas": (
        "Failure to document GAGAS applicability may cause the engagement to omit "
        "Yellow Book independence, CPE, and reporting requirements."
    ),
    "includes_single_audit": (
        "Undocumented Single Audit scope creates a risk of non-compliance with "
        "Uniform Guidance reporting and major program testing requirements."
    ),
    "includes_gaas_audit": (
        "Without explicit financial statement audit scope documentation, "
        "the engagement's basis in auditing standards is unverifiable."
    ),
    "includes_grant_compliance": (
        "Unrecorded grant compliance scope may result in omitted federal award "
        "testing required under the Uniform Guidance."
    ),
    "includes_nonattest_services": (
        "Undocumented non-attest services create independence risk that cannot "
        "be assessed without explicit disclosure."
    ),
    "document_reference": (
        "A missing workpaper reference number impedes cross-referencing within "
        "the engagement file and complicates supervisory review."
    ),
    "preparation_date": (
        "Without a preparation date, the timeliness of the engagement form "
        "completion cannot be verified against engagement deadlines."
    ),
    "partner_sign_date": (
        "The absence of a partner sign date prevents independence dating "
        "verification required under the applicable engagement standards."
    ),
    "ein": (
        "An undocumented EIN creates entity identification risk and may impede "
        "tax-exempt status verification for nonprofit engagements."
    ),
}

_DEFAULT_CONSEQUENCE_TEXT = (
    "This documentation deficiency weakens the evidentiary basis of the engagement "
    "workpaper and should be corrected prior to report issuance."
)


# ---------------------------------------------------------------------------
# Standards context loader and resolver
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_standards_ctx() -> dict:
    """Load field_standards_context.yaml → {field: {base, gagas_additive, single_audit_additive}}."""
    if not _STDS_CTX_PATH.exists():
        logger.warning(
            "completion_renderer: field_standards_context.yaml not found at %s — "
            "risk text will be consequence-only (no standards citations)",
            _STDS_CTX_PATH,
        )
        return {}
    with open(_STDS_CTX_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_standards(
    field_name:       str,
    is_gagas:         bool,
    has_single_audit: bool,
) -> str | None:
    """
    Build the standards citation string for a field finding.

    Returns a semicolon-joined string like:
        "AU-C 220; Government Auditing Standards (Chapter 3)"
    or None if no applicable standards are configured for this field/context.

    Standards are additive: base always applies; context flags append layers.
    The internal SOP section is NOT included here — it already appears in the
    finding headline as "(§4)". Including it here would create two independent
    citation sources that could drift.
    """
    ctx  = _load_standards_ctx().get(field_name, {})
    cits: list[str] = list(ctx.get("base") or [])

    if is_gagas:
        cits.extend(ctx.get("gagas_additive") or [])
    if has_single_audit:
        cits.extend(ctx.get("single_audit_additive") or [])

    if not cits:
        return None

    from raw_to_training_pair.citation_resolver import resolve as _resolve_cits
    resolved = _resolve_cits(cits)
    return "; ".join(resolved) if resolved else None


# ---------------------------------------------------------------------------
# Finding label registry (audit finding headlines, not SOP section headings)
# ---------------------------------------------------------------------------

# Per-field finding labels used as the finding headline in the completion.
# These describe the deficiency, not the SOP section. Keys are canonical field
# names. Fallback is mapping.label from the FIELD_TO_SOP table.
_FIELD_FINDING_LABEL: dict[str, str] = {
    "audit_type":                "Audit Type — Service Scope Not Documented",
    "client_name":               "Client Entity — Not Identified",
    "document_reference":        "Document Reference — Missing",
    "ein":                       "EIN — Not Documented",
    "engagement_decision":       "Engagement Decision — Not Documented",
    "engagement_partner":        "Engagement Partner — Not Identified",
    "fiscal_year_end":           "Fiscal Year End — Not Documented",
    "includes_gaas_audit":       "GAAS Financial Statement Audit — Scope Not Documented",
    "includes_gagas":            "GAGAS / Yellow Book Applicability — Not Documented",
    "includes_grant_compliance":  "Grant Compliance Audit — Scope Not Documented",
    "includes_nonattest_services": "Non-Attest Services Disclosure — Not Documented",
    "includes_single_audit":     "Single Audit Requirement — Not Documented",
    "partner_sign_date":         "Engagement Partner Sign-Off Date — Missing",
    "preparation_date":          "Form Preparation Date — Not Documented",
    "reporting_framework":       "Reporting Framework — Not Documented",
}


# ---------------------------------------------------------------------------
# FIELD_TO_SOP table builder (compile-time)
# ---------------------------------------------------------------------------

def build_sop_mapping_table(client_type: str = "") -> SopTable:
    """
    Build the FIELD_TO_SOP compile-time table for one client type.

    Called once at pipeline startup. The result is passed to render_completion()
    for every workpaper — no file I/O or model inference at render time.

    Build order (first match wins):
        1. sop_field_classes.yaml: canonical_field → slug → §section + severity
        2. field_tiers.yaml reason text: §N.N pattern extraction
        3. Tier-based severity default (High for tier1, Medium for tier2)
        4. Unmapped fields: sop_section=None, sop_unverified implied

    Parameters
    ----------
    client_type : str
        Used to apply client_type_overrides from sop_field_classes.yaml
        when present. Empty string → use base class for all fields.
    """
    table: SopTable = {}

    # Load both config files
    tiers: dict = {}
    if _TIERS_PATH.exists():
        with open(_TIERS_PATH) as f:
            tiers = yaml.safe_load(f) or {}

    sfc: dict = {}
    if _SFC_PATH.exists():
        with open(_SFC_PATH) as f:
            sfc = yaml.safe_load(f) or {}

    # Build tier lookup: canonical_field → (tier_key, reason)
    tier_meta: dict[str, tuple[str, str]] = {}
    for tier_key in ("tier1", "tier2"):
        for entry in (tiers.get(tier_key) or []):
            if isinstance(entry, dict) and "field" in entry:
                tier_meta[entry["field"]] = (tier_key, entry.get("reason", ""))

    # Build SFC slug lookup: canonical_field → (slug, entry_dict)
    sfc_by_canonical: dict[str, tuple[str, dict]] = {}
    for slug, entry in (sfc.get("fields") or {}).items():
        cf = entry.get("canonical_field")
        if cf and cf not in sfc_by_canonical:
            sfc_by_canonical[cf] = (slug, entry)

    # Populate table for every tier1 + tier2 field
    for field_name, (tier_key, reason) in tier_meta.items():
        sop_section: str | None = None
        severity:    str = _SEVERITY_TIER.get(tier_key, "Medium")
        label:       str = field_name.replace("_", " ").title()

        if field_name in sfc_by_canonical:
            slug, sfc_entry = sfc_by_canonical[field_name]

            # Apply client_type_override if present
            eff_class = sfc_entry.get("class", "deficiency_eligible")
            overrides = sfc_entry.get("client_type_overrides") or {}
            if client_type and client_type in overrides:
                eff_class = overrides[client_type].get("class", eff_class)

            # Only generate findings for deficiency_eligible fields
            if eff_class != "deficiency_eligible":
                continue

            # sop_section_override takes precedence; fallback to slug → §Section
            sop_section = sfc_entry.get("sop_section_override") or _slug_to_section(slug)
            raw_sev     = sfc_entry.get("severity_default")
            severity    = _coerce_severity(raw_sev, tier_key)
            label       = sfc_entry.get("label") or label

        else:
            # Fallback: extract §N.N from tier reason text
            sop_section = _extract_sop_ref_from_reason(reason)

        table[field_name] = SopMapping(
            canonical_field = field_name,
            sop_section     = sop_section,
            severity        = severity,
            label           = label,
        )

    logger.debug(
        "completion_renderer: FIELD_TO_SOP built — %d fields mapped "
        "(client_type=%s sop_version=%s)",
        len(table), client_type or "generic", sop_mapping_version(),
    )
    return table


# ---------------------------------------------------------------------------
# Version key
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def sop_mapping_version() -> str:
    """
    Stable version key derived from the content hashes of field_tiers.yaml
    and sop_field_classes.yaml. Changes when either config changes.
    Included in every training pair's metadata for reproducibility.
    """
    h = hashlib.sha256()
    for path in (_TIERS_PATH, _SFC_PATH):
        if path.exists():
            h.update(path.read_bytes())
    short_hash = h.hexdigest()[:8]
    # Use today's date as a human-readable prefix
    from datetime import date
    return f"{date.today().isoformat()}_{short_hash}"


# ---------------------------------------------------------------------------
# Client type context (matches completion_drafter for format compatibility)
# ---------------------------------------------------------------------------

_CLIENT_CONTEXT: dict[str, dict] = {
    "NPO": {
        "label":       "Nonprofit Organization (501(c)(3))",
        "standards":   "GAAS",
        "rec_context": (
            "As a nonprofit entity, corrective actions must address donor restrictions, "
            "board oversight requirements, and Form 990 filing obligations."
        ),
        "compliance":  "AICPA AU-C standards applicable to nonprofit entities",
    },
    "Government": {
        "label":       "State / Local Government Entity",
        "standards":   "GAGAS",
        "rec_context": (
            "As a governmental entity, corrective actions must address Yellow Book "
            "independence requirements, CPE obligations, and public accountability standards."
        ),
        "compliance":  "Government Auditing Standards (Yellow Book) and applicable state law",
    },
    "Tribal": {
        "label":       "Federally Recognized Tribal Government",
        "standards":   "GAGAS + Single Audit",
        "rec_context": (
            "As a tribal government, corrective actions must address both Yellow Book "
            "requirements and Uniform Guidance compliance for federal program expenditures."
        ),
        "compliance":  "Government Auditing Standards and 2 CFR Part 200 (Uniform Guidance)",
    },
    "For-Profit": {
        "label":       "For-Profit Commercial Entity",
        "standards":   "GAAS",
        "rec_context": (
            "As a for-profit entity, corrective actions must address PCAOB or AICPA "
            "standards as applicable and focus on internal control over financial reporting."
        ),
        "compliance":  "AICPA AU-C standards applicable to for-profit entities",
    },
}

_DEFAULT_CLIENT_CONTEXT = {
    "label":       "Audit Client",
    "standards":   "GAAS",
    "rec_context": "Corrective actions should address applicable professional standards.",
    "compliance":  "Applicable AICPA auditing standards",
}


def _client_ctx(client_type: str) -> dict:
    return _CLIENT_CONTEXT.get(client_type, _DEFAULT_CLIENT_CONTEXT)


# ---------------------------------------------------------------------------
# Deterministic renderer
# ---------------------------------------------------------------------------

def render_completion(
    validated,               # ValidatedClassification from claim_mapper
    sop_table: SopTable,
    client_type:        str,
    is_gagas:           bool,
    has_single_audit:   bool,
    mapping_version:    str | None = None,
) -> RenderResult:
    """
    Render a deterministic audit completion from a ValidatedClassification.

    No LLM inference. Every finding comes from the precompiled sop_table.
    Absent fields generate numbered findings. Present/uncertain fields do not.

    Parameters
    ----------
    validated : ValidatedClassification
        Corrected classification from claim_mapper.validate_classification().
    sop_table : SopTable
        Precompiled FIELD_TO_SOP table from build_sop_mapping_table().
    client_type : str
    is_gagas : bool
    has_single_audit : bool
    mapping_version : str | None
        Pass sop_mapping_version() result from startup. If None, recomputes.

    Returns
    -------
    RenderResult
    """
    version = mapping_version or sop_mapping_version()
    ctx     = _client_ctx(client_type)

    # Engagement type line
    standards = []
    if is_gagas:
        standards.append("GAGAS (Yellow Book)")
    if has_single_audit:
        standards.append("Single Audit (2 CFR 200)")
    if not standards:
        standards.append(ctx["standards"])
    engagement_type = f"{ctx['label']} — {' + '.join(standards)}"

    # Generate findings for absent fields only
    finding_lines: list[str] = []
    findings_generated: list[str] = []
    sop_unverified_fields: list[str] = []

    for field_name in validated.absent_fields:
        mapping = sop_table.get(field_name)
        if mapping is None:
            # Field not in FIELD_TO_SOP (not SOP-mappable for this client type)
            logger.debug(
                "completion_renderer: %s not in sop_table for client_type=%s — "
                "skipping finding",
                field_name, client_type,
            )
            continue

        citation = f" ({mapping.sop_section})" if mapping.sop_section else ""
        if not mapping.sop_section:
            sop_unverified_fields.append(field_name)

        consequence   = _FIELD_CONSEQUENCE_TEXT.get(field_name, _DEFAULT_CONSEQUENCE_TEXT)
        standards_str = _resolve_standards(field_name, is_gagas, has_single_audit)
        risk_text     = (
            f"{consequence} Applicable standards: {standards_str}."
            if standards_str else consequence
        )
        finding_label = _FIELD_FINDING_LABEL.get(field_name, mapping.label)

        idx = len(finding_lines) + 1
        finding_lines.append(
            f"{idx}. {finding_label}{citation}\n"
            f"   Severity: {mapping.severity}\n"
            f"   Risk: {risk_text}"
        )
        findings_generated.append(field_name)

    # Clean completion when no absent fields map to findings
    if not finding_lines:
        findings_block = "No deficiencies identified — engagement documentation appears complete."
    else:
        findings_block = "\n\n".join(finding_lines)

    # Compliance reference adapts to engagement-level flags, overriding the
    # client-type default when GAGAS or Single Audit flags are set.
    if is_gagas and has_single_audit:
        compliance = (
            "Government Auditing Standards (Yellow Book) and "
            "2 CFR Part 200 (Uniform Guidance)"
        )
    elif is_gagas:
        compliance = "Government Auditing Standards (Yellow Book)"
    elif has_single_audit:
        compliance = (
            "2 CFR Part 200 (Uniform Guidance) and applicable AICPA AU-C standards"
        )
    else:
        compliance = ctx["compliance"]

    # Recommendation (template, client-type-aware)
    n = len(finding_lines)
    if n > 0:
        rec = (
            f"Management of this {ctx['label']} should implement corrective procedures "
            f"to address the {n} identified documentation deficienc{'y' if n == 1 else 'ies'}. "
            f"{ctx['rec_context']} "
            f"All corrections should be completed and documented prior to report issuance "
            f"in accordance with {compliance}."
        )
    else:
        rec = (
            f"No corrective action is required. This {ctx['label']} engagement form "
            f"appears complete and in compliance with {compliance}."
        )

    completion = (
        f"ENGAGEMENT TYPE: {engagement_type}\n\n"
        f"FINDINGS:\n{findings_block}\n\n"
        f"RECOMMENDATION:\n{rec}"
    )

    sop_unverified = len(sop_unverified_fields) > 0

    metadata: dict[str, Any] = {
        "sop_mapping_version":  version,
        "findings_count":       n,
        "absent_fields":        validated.absent_fields,
        "present_fields":       validated.present_fields,
        "uncertain_fields":     validated.uncertain_fields,
        "provisional_fields":   validated.provisional_fields,
        "sop_unverified":       sop_unverified,
        "sop_unverified_fields": sop_unverified_fields,
    }

    logger.info(
        "completion_renderer: rendered %d finding(s) for %s "
        "(absent=%d uncertain=%d provisional=%d sop_unverified=%s)",
        n, client_type, len(validated.absent_fields),
        len(validated.uncertain_fields), len(validated.provisional_fields),
        sop_unverified,
    )

    return RenderResult(
        completion          = completion,
        absent_fields       = findings_generated,
        uncertain_fields    = validated.uncertain_fields,
        provisional_fields  = validated.provisional_fields,
        sop_unverified      = sop_unverified,
        sop_mapping_version = version,
        metadata            = metadata,
    )
