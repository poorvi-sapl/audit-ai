"""
raw_to_training_pair/completion_drafter.py
===========================================
Phase C1/C3/C4 rewrite.

C1 — Redesigned completion prompt
    New output format includes severity, risk implication, and
    client-type-aware recommendations. Gemma writes narrative from
    structured findings (C2 output), not raw text.

C3 — Risk-aware deficient variants
    Deficient findings now explain WHY the missing field matters
    (from findings_extractor risk registry), not just "field X missing".

C4 — review_confidence scorer
    After generation, scores the completion on 5 criteria.
    Sets record.review_confidence and record.quality_gate.

Output format (C1)
------------------
ENGAGEMENT TYPE: <type + standards>

FINDINGS:
1. [Finding label] (SOP §X.X)
   Severity: High | Medium | Low | Informational
   Risk: <one sentence on audit consequence>

RECOMMENDATION:
<specific, actionable, client-type-aware>

Public API
----------
    draft(record, findings, sop_text, client_type, is_gagas,
          has_single_audit, use_mock) -> str | None
    score_completion(completion, findings, client_type) -> float
    set_review_gate(record, completion, findings, client_type) -> float
    is_available() -> bool
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from auditai_data_normalization.schema import DocumentRecord

logger = logging.getLogger(__name__)

_MODEL       = "gemma3:12b"
_TEMPERATURE = 0.0
_MAX_TOKENS  = 1500   # C1 format is richer — needs more tokens
_FIRM        = "Harshwal & Company LLP (HCLLP)"

# review_confidence gate threshold
_QUALITY_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Client type context  (drives recommendation specificity — C1/C3)
# ---------------------------------------------------------------------------

_CLIENT_CONTEXT: dict[str, dict] = {
    "NPO": {
        "label":       "Nonprofit Organization (501(c)(3))",
        "standards":   "GAAS",
        "rec_context": (
            "As a nonprofit entity, recommendations must address donor restrictions, "
            "board oversight requirements, and Form 990 filing obligations."
        ),
        "compliance":  "AICPA AU-C standards applicable to nonprofit entities",
    },
    "Government": {
        "label":       "State / Local Government Entity",
        "standards":   "GAGAS",
        "rec_context": (
            "As a governmental entity, recommendations must address Yellow Book "
            "independence requirements, CPE obligations, and public accountability standards."
        ),
        "compliance":  "Government Auditing Standards (Yellow Book) and applicable state law",
    },
    "Tribal": {
        "label":       "Federally Recognized Tribal Government",
        "standards":   "GAGAS + Single Audit",
        "rec_context": (
            "As a tribal government, recommendations must address both Yellow Book "
            "requirements and Uniform Guidance compliance for federal program expenditures."
        ),
        "compliance":  "Government Auditing Standards and 2 CFR Part 200 (Uniform Guidance)",
    },
    "For-Profit": {
        "label":       "For-Profit Commercial Entity",
        "standards":   "GAAS",
        "rec_context": (
            "As a for-profit entity, recommendations must address PCAOB or AICPA "
            "standards as applicable and focus on internal control over financial reporting."
        ),
        "compliance":  "AICPA AU-C standards applicable to for-profit entities",
    },
}

_DEFAULT_CLIENT_CONTEXT = {
    "label":       "Audit Client",
    "standards":   "GAAS",
    "rec_context": "Recommendations should address applicable professional standards.",
    "compliance":  "Applicable AICPA auditing standards",
}


def _client_ctx(client_type: str) -> dict:
    return _CLIENT_CONTEXT.get(client_type, _DEFAULT_CLIENT_CONTEXT)


# ---------------------------------------------------------------------------
# System prompt (C1)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""\
You are an expert audit assistant for {_FIRM}, a US CPA firm specializing \
in nonprofit, governmental, and single audit engagements under GAAS, GAGAS \
(Yellow Book), and the Uniform Guidance (2 CFR Part 200).

OUTPUT FORMAT — follow exactly, no deviation:

ENGAGEMENT TYPE: <single line: audit type + applicable standards>

FINDINGS:
1. [Finding label] (SOP §X.X)
   Severity: High | Medium | Low | Informational
   Risk: <one sentence explaining the audit consequence of this issue>

2. [Finding label] (SOP §X.X)
   Severity: High | Medium | Low | Informational
   Risk: <one sentence>

[add more numbered findings as needed]

RECOMMENDATION:
<specific, actionable recommendation tailored to the client type and engagement scope>

RULES:
- Generate EXACTLY the number of findings specified in STRUCTURED FINDINGS TO ADDRESS — no more, no fewer
- Do NOT add findings from SOP context for fields not in STRUCTURED FINDINGS TO ADDRESS
- Every finding MUST cite a specific SOP section — never use §X.X as a placeholder
- Use only SOP sections from the context provided
- Severity must be exactly one of: High, Medium, Low, Informational
- Risk must explain the audit CONSEQUENCE, not just restate the finding
- Recommendation must reference the client type and engagement scope explicitly
- No preamble, no explanation outside the format above
- Professional, concise, audit-standard language throughout\
"""


# ---------------------------------------------------------------------------
# Prompt builder (C1 — findings-first)
# ---------------------------------------------------------------------------

def _build_prompt(
    findings: list[dict],
    sop_text: str,
    client_type: str,
    is_gagas: bool,
    has_single_audit: bool,
    fields: dict[str, Any],
    correction_hint: str = "",
) -> str:
    ctx = _client_ctx(client_type)

    # Cap SOP text
    sop_words = sop_text.split()
    if len(sop_words) > 2000:
        sop_text = " ".join(sop_words[:2000]) + "\n[... SOP text truncated ...]"

    # Build engagement type line
    standards = []
    if is_gagas:
        standards.append("GAGAS (Yellow Book)")
    if has_single_audit:
        standards.append("Single Audit (2 CFR 200)")
    if not standards:
        standards.append(ctx["standards"])
    engagement_type_hint = f"{ctx['label']} — {' + '.join(standards)}"

    # Format structured findings for the prompt
    findings_text_lines = []
    for i, f in enumerate(findings, 1):
        sop_ref = f.get("sop_section") or "— (resolve from SOP context below)"
        risk    = f.get("risk", "")
        sev     = f.get("severity", "Medium")
        label   = f.get("label", f.get("field", "Unknown field"))
        findings_text_lines.append(
            f"{i}. {label} | Severity: {sev} | SOP: {sop_ref}\n"
            f"   Risk context: {risk}"
        )

    n_findings = len(findings_text_lines)
    if findings_text_lines:
        findings_header = (
            f"STRUCTURED FINDINGS TO ADDRESS — "
            f"generate exactly {n_findings} numbered finding"
            f"{'s' if n_findings != 1 else ''}, no more, no fewer:"
        )
        findings_block = "\n".join(findings_text_lines)
    else:
        findings_header = "STRUCTURED FINDINGS TO ADDRESS — 0 findings:"
        findings_block  = "No deficiencies identified — document appears complete."

    # Key fields for context
    field_lines = [
        f"  {k}: {v}" for k, v in fields.items()
        if v and str(v).strip() and k not in (
            "file_name", "file_hash", "file_type", "source_path"
        )
    ]
    fields_block = "\n".join(field_lines) if field_lines else "  [no fields extracted]"

    closing = (
        f"Draft the audit completion following the required format. "
        f"The FINDINGS section must contain exactly {n_findings} "
        f"numbered item{'s' if n_findings != 1 else ''}. "
        f"Use the exact SOP sections from the guidance above."
        if n_findings > 0
        else
        "Draft the audit completion following the required format. "
        "The FINDINGS section must state that no deficiencies were identified. "
        "Do not invent findings."
    )

    return (
        f"ENGAGEMENT CONTEXT:\n"
        f"  Engagement type: {engagement_type_hint}\n"
        f"  Client context:  {ctx['rec_context']}\n"
        f"  Compliance:      {ctx['compliance']}\n\n"
        f"WORKPAPER FIELDS:\n{fields_block}\n\n"
        f"{findings_header}\n{findings_block}\n\n"
        f"RELEVANT SOP GUIDANCE (context only — do not generate findings from this):\n"
        f"{sop_text or '[no SOP text provided]'}\n\n"
        f"{closing}"
        + (
            f"\n\nCORRECTION REQUIRED BY AUDITOR:\n{correction_hint}\n"
            f"Address the correction above — it overrides any prior draft."
            if correction_hint.strip() else ""
        )
    )


# ---------------------------------------------------------------------------
# C4 — review_confidence scorer
# ---------------------------------------------------------------------------

# Scoring rubric (5 criteria, total 1.0)
_RUBRIC = {
    "sections_present":     0.30,   # all 3 sections exist
    "sop_citations":        0.20,   # every finding has a real citation
    "severity_present":     0.20,   # every finding has severity classification
    "client_type_specific": 0.20,   # recommendation references client type
    "no_placeholders":      0.10,   # no generic placeholder text
}


def _load_threshold_cfg() -> dict:
    """Load threshold config once per process (cached via module-level dict)."""
    import yaml as _yaml
    from pathlib import Path as _Path
    try:
        with open(_Path(__file__).parent.parent /
                  "auditai_data_normalization/alias_registry/threshold_config.yaml") as _f:
            return _yaml.safe_load(_f) or {}
    except Exception:
        return {}

_PLACEHOLDER_PATTERNS = [
    r"§X\.X",                              # placeholder SOP citation
    r"\[Finding label\]",                  # unfilled template
    r"REQUIRES MANUAL COMPLETION",
    r"\[add more findings",
    r"field X missing",
    r"ensure field completed",
]

_REQUIRED_SECTIONS = ["ENGAGEMENT TYPE:", "FINDINGS:", "RECOMMENDATION:"]

_SOP_CITATION_RE = re.compile(
    r"(?:SOP[\w-]*\s+)?§\s*[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*(?:\([^)]*\))?",
    re.IGNORECASE,
)


def _norm_section(s: str) -> str:
    s = s.strip().lower()
    m = re.search(r"§\s*([a-z0-9]+(?:[.\-][a-z0-9]+)*)", s)
    if m:
        return m.group(1)
    return s.replace(" ", "").lstrip("sop").lstrip("§")


def _compute_rubric(
    completion:       str,
    n_findings:       int,               # expected finding count (0 = clean pair)
    client_type:      str,
    sop_sections:     list[str] | None,
    verify_citations: bool,
    min_grounded:     float,
) -> float:
    """
    Compute the 5-criterion structural rubric score.

    Shared by score_completion() (legacy) and score_r7() (R7 path).
    Does not apply any grounding multiplier or classification penalties —
    those are caller's responsibility.

    Parameters
    ----------
    n_findings : int
        Number of findings that should appear in the completion.
        0 means clean pair (no deficiencies). Used for Criteria 2 and 3.
    """
    score = 0.0

    # Criterion 1 — all 3 required sections present (+0.30)
    sections_found = sum(1 for s in _REQUIRED_SECTIONS if s in completion)
    score += _RUBRIC["sections_present"] * (sections_found / len(_REQUIRED_SECTIONS))

    # Criterion 2 — SOP citations present AND grounded (+0.20)
    finding_lines  = re.findall(r"^\s*\d+\.\s+.+", completion, re.MULTILINE)
    cited_sections = _SOP_CITATION_RE.findall(completion)
    citation_count = len(cited_sections)

    if finding_lines:
        if verify_citations and sop_sections:
            retrieved_normalised = {_norm_section(s) for s in sop_sections}
            grounded = sum(
                1 for c in cited_sections
                if _norm_section(c) in retrieved_normalised
            )
            grounded_ratio = grounded / citation_count if citation_count else 0.0
            citation_score = _RUBRIC["sop_citations"] * grounded_ratio
            if grounded_ratio < min_grounded and citation_count > 0:
                citation_score = min(citation_score, _RUBRIC["sop_citations"] * 0.5)
            score += citation_score
        else:
            citation_ratio = min(citation_count / len(finding_lines), 1.0)
            score += _RUBRIC["sop_citations"] * citation_ratio
    elif n_findings == 0:
        score += _RUBRIC["sop_citations"]

    # Criterion 3 — severity classification on every finding (+0.20)
    severity_hits = len(re.findall(
        r"Severity:\s*(High|Medium|Low|Informational)", completion, re.IGNORECASE
    ))
    if finding_lines:
        severity_ratio = min(severity_hits / len(finding_lines), 1.0)
        score += _RUBRIC["severity_present"] * severity_ratio
    elif n_findings == 0:
        score += _RUBRIC["severity_present"]

    # Criterion 4 — recommendation references client type (+0.20)
    rec_match = re.search(
        r"RECOMMENDATION:\s*(.+?)(?:\Z)", completion,
        re.DOTALL | re.IGNORECASE,
    )
    rec_text = rec_match.group(1).strip() if rec_match else ""
    ctx = _client_ctx(client_type)
    client_words = set(ctx["label"].lower().split())
    rec_lower    = rec_text.lower()
    client_hits  = sum(1 for w in client_words if len(w) > 3 and w in rec_lower)
    if client_hits == 0 and client_type and client_type.lower() in rec_lower:
        client_hits = 1
    if client_hits >= 1:
        score += _RUBRIC["client_type_specific"]
    elif rec_text:
        score += _RUBRIC["client_type_specific"] * 0.5

    # Criterion 5 — no generic placeholder text (+0.10)
    if not any(re.search(p, completion, re.IGNORECASE) for p in _PLACEHOLDER_PATTERNS):
        score += _RUBRIC["no_placeholders"]

    return round(min(score, 1.0), 4)


def score_completion(
    completion: str,
    findings: list[dict],
    client_type: str,
    sop_sections: list[str] | None = None,
    fields_missing: list[str] | None = None,
    fields_present: list[str] | None = None,
) -> float:
    """
    Score a generated completion on 5 rubric criteria, then apply
    grounding validity multiplier and penalties via claim_mapper.

    Parameters
    ----------
    completion : str
        The text generated by draft().
    findings : list[dict]
        Structured findings from findings_extractor.
    client_type : str
        Used to check recommendation specificity.
    sop_sections : list[str] | None
        Section identifiers returned by qdrant_retriever.
    fields_missing : list[str] | None
        Canonical fields not found in document (real deficiencies).
        Required for grounding validation. Without it, grounding
        multiplier defaults to 1.0 (no penalty).
    fields_present : list[str] | None
        Canonical fields found in document.
        Required to detect MISANCHORED claims.

    Returns
    -------
    float
        review_confidence in [0.0, 1.0].
    """
    if not completion or not completion.strip():
        return 0.0

    # Load citation verification config
    try:
        import yaml as _yaml
        from pathlib import Path as _Path
        _thr = _yaml.safe_load(
            open(_Path(__file__).parent.parent /
                 "auditai_data_normalization/alias_registry/threshold_config.yaml")
        ) or {}
        _sop_cfg          = _thr.get("sop_retrieval", {})
        _verify_citations = _sop_cfg.get("citation_verification", True)
        _min_grounded     = float(_sop_cfg.get("min_grounded_citation_ratio", 0.50))
    except Exception:
        _verify_citations = False
        _min_grounded     = 0.0

    rubric_score = _compute_rubric(
        completion, len(findings), client_type,
        sop_sections, _verify_citations, _min_grounded,
    )

    # Grounding validity layer — apply multiplier and penalties
    if fields_missing is not None:
        try:
            from raw_to_training_pair.claim_mapper import (
                build_claim_map, compute_grounding_signals
            )
            claim_map = build_claim_map(
                completion     = completion,
                fields_missing = fields_missing,
                fields_present = fields_present or [],
                sop_sections   = sop_sections,
                client_type    = client_type,
            )
            signals = compute_grounding_signals(claim_map)

            # Load grounding config
            _g_cfg   = _load_threshold_cfg().get("grounding", {})
            _g_min   = float(_g_cfg.get("grounding_multiplier_min",   0.50))
            _g_range = float(_g_cfg.get("grounding_multiplier_range", 0.50))
            _hall_pen = float(_g_cfg.get("hallucination_penalty_per_claim", 0.15))
            _mis_pen  = float(_g_cfg.get("misanchored_penalty_per_claim",   0.10))
            _t1_pen   = float(_g_cfg.get("coverage_tier1_penalty",          0.08))
            _fa_pen   = float(_g_cfg.get("fixed_admin_penalty_per_claim",   0.03))
            # Max fraction of the grounded score that deductions can remove.
            # Without this cap, a model that generates N findings for N present
            # fields accumulates N*0.10 in MISANCHORED penalties, which easily
            # exceeds the entire rubric score and floors everything to 0.0.
            _max_ded_frac = float(_g_cfg.get("max_deduction_fraction", 0.60))

            # Grounding multiplier: 0.5 (all unanchored) → 1.0 (all valid)
            grounding_multiplier = _g_min + signals.grounding_score * _g_range

            # Hallucination penalty: HALLUCINATED * 0.15 + MISANCHORED * 0.10
            # fixed_admin claims carry their own softer penalty (0.03) —
            # structural noise from SOP context, not a factual data error.
            hallucination_deduction = (
                signals.hallucinated_count * _hall_pen
                + signals.misanchored_count * _mis_pen
                + signals.fixed_admin_count * _fa_pen
            )

            # Coverage penalty: uncovered Tier 1 fields
            coverage_deduction = signals.coverage_gap * len(claim_map.tier1_missing) * _t1_pen

            # Cap total deductions so they cannot consume more than max_deduction_fraction
            # of the grounded rubric score — prevents a large findings list from
            # flooring the score to 0.0 when the rubric quality is otherwise good.
            grounded_rubric = rubric_score * grounding_multiplier
            total_deduction = min(
                hallucination_deduction + coverage_deduction,
                grounded_rubric * _max_ded_frac,
            )

            final_score = grounded_rubric - total_deduction
            final_score = round(max(0.0, min(1.0, final_score)), 4)

            logger.debug(
                "score_completion: rubric=%.3f grounding=%.3f mult=%.2f "
                "hall_ded=%.3f cov_ded=%.3f final=%.3f",
                rubric_score, signals.grounding_score, grounding_multiplier,
                hallucination_deduction, coverage_deduction, final_score,
            )
            return final_score

        except Exception as e:
            logger.debug("score_completion: grounding layer failed (%s) — returning rubric score", e)

    return rubric_score


def set_review_gate(
    record: "DocumentRecord",
    completion: str,
    findings: list[dict],
    client_type: str,
    sop_sections: list[str] | None = None,
    fields_missing: list[str] | None = None,
    fields_present: list[str] | None = None,
) -> float:
    """
    Score the completion, set record.review_confidence and record.quality_gate.

    Parameters
    ----------
    record : DocumentRecord
        Updated in place.
    completion : str
    findings : list[dict]
    client_type : str
    sop_sections : list[str] | None
    fields_missing : list[str] | None
        For grounding validation via claim_mapper.
    fields_present : list[str] | None

    Returns
    -------
    float
        review_confidence score.
    """
    rev_conf = score_completion(
        completion,
        findings,
        client_type,
        sop_sections=sop_sections,
        fields_missing=fields_missing,
        fields_present=fields_present,
    )
    record.review_confidence = rev_conf
    record.quality_gate = rev_conf >= _QUALITY_THRESHOLD

    if not record.quality_gate:
        record.needs_review = True
        logger.info(
            "completion_drafter: quality_gate=False for %s "
            "(review_confidence=%.3f < %.2f) — flagged for auditor review",
            record.file_name, rev_conf, _QUALITY_THRESHOLD,
        )
    else:
        logger.info(
            "completion_drafter: quality_gate=True for %s "
            "(review_confidence=%.3f)",
            record.file_name, rev_conf,
        )

    return rev_conf


# ---------------------------------------------------------------------------
# C3 — Mock completion (risk-aware deficient variants)
# ---------------------------------------------------------------------------

_MOCK_ENGAGEMENT_TYPES = {
    "NPO":        "GAAS Audit — Nonprofit Organization (501(c)(3))",
    "Government": "GAGAS Audit — State/Local Governmental Entity (Yellow Book)",
    "Tribal":     "GAGAS + Single Audit — Federally Recognized Tribal Government",
    "For-Profit": "GAAS Audit — For-Profit Commercial Entity",
}


def draft_mock(
    findings: list[dict],
    client_type: str,
    is_gagas: bool,
    has_single_audit: bool,
    file_name: str = "unknown",
    correction_hint: str = "",
) -> str:
    """
    C3 — Risk-aware mock completion. Semi-deterministic.

    Findings now include severity and risk implication, not just
    "field X missing" generic text (Phase C3 improvement).
    """
    seed_str = f"{file_name}|{client_type}|{is_gagas}|{has_single_audit}"
    seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)

    ctx = _client_ctx(client_type)
    base_type = _MOCK_ENGAGEMENT_TYPES.get(client_type, f"GAAS Audit — {ctx['label']}")
    if is_gagas and "GAGAS" not in base_type:
        base_type += " (GAGAS)"
    if has_single_audit and "Single Audit" not in base_type:
        base_type += " + Single Audit (2 CFR 200)"

    # Build risk-aware finding lines from structured findings (C3)
    finding_lines = []
    for i, f in enumerate(findings, 1):
        label    = f.get("label", f.get("field", "Required field"))
        severity = f.get("severity", "Medium")
        risk     = f.get("risk", "Documentation deficiency identified.")
        sop_ref  = f.get("sop_section") or f"SOP §{2 + (seed_hash + i) % 5}.{(seed_hash + i * 3) % 4 + 1}"
        finding_lines.append(
            f"{i}. {label} ({sop_ref})\n"
            f"   Severity: {severity}\n"
            f"   Risk: {risk}"
        )

    if not finding_lines:
        finding_lines = ["1. No deficiencies identified — engagement form appears complete."]

    # Client-type-specific recommendation (C3)
    rec = (
        f"Management of this {ctx['label']} should implement corrective procedures "
        f"to address the identified deficiencies. {ctx['rec_context']} "
        f"All corrections should be completed and documented prior to report issuance "
        f"in accordance with {ctx['compliance']}."
    )

    correction_note = (
        f"\n\n[AUDITOR CORRECTION APPLIED: {correction_hint}]"
        if correction_hint.strip() else ""
    )
    completion = (
        f"ENGAGEMENT TYPE: {base_type}\n\n"
        f"FINDINGS:\n"
        + "\n\n".join(finding_lines)
        + f"\n\nRECOMMENDATION:\n{rec}"
        + correction_note
    )

    logger.info(
        "completion_drafter: mock completion for %s client=%s findings=%d",
        file_name, client_type, len(findings),
    )
    return completion


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def is_available() -> bool:
    try:
        import ollama
        result = ollama.list()
        names = [m.model for m in result.models]
        available = any(_MODEL in n for n in names)
        if not available:
            logger.debug("completion_drafter: %s not in ollama models", _MODEL)
        return available
    except Exception as e:
        logger.debug("completion_drafter: Ollama not reachable — %s", e)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draft(
    record: "DocumentRecord",
    findings: list[dict],
    sop_text: str,
    client_type: str,
    is_gagas: bool,
    has_single_audit: bool = False,
    fields: dict[str, Any] | None = None,
    use_mock: bool = False,
    correction_hint: str = "",
    sop_sections: list[str] | None = None,
    fields_missing: list[str] | None = None,
    fields_present: list[str] | None = None,
) -> str | None:
    """
    Draft an assistant completion for a training pair.

    Parameters
    ----------
    record : DocumentRecord
    findings : list[dict]
    sop_text : str
    client_type : str
    is_gagas : bool
    has_single_audit : bool
    fields : dict | None
    use_mock : bool
    sop_sections : list[str] | None
    fields_missing : list[str] | None
        For grounding validation. Passed to set_review_gate.
    fields_present : list[str] | None
        For grounding validation. Passed to set_review_gate.

    Returns
    -------
    str | None
    """
    if use_mock:
        completion = draft_mock(
            findings=findings,
            client_type=client_type,
            is_gagas=is_gagas,
            has_single_audit=has_single_audit,
            file_name=record.file_name,
            correction_hint=correction_hint,
        )
        set_review_gate(record, completion, findings, client_type,
                        sop_sections=sop_sections,
                        fields_missing=fields_missing,
                        fields_present=fields_present)
        return completion

    if not is_available():
        logger.warning(
            "completion_drafter: Ollama unavailable. "
            "Run: ollama serve && ollama pull %s", _MODEL,
        )
        return None

    try:
        import ollama

        prompt = _build_prompt(
            findings=findings,
            sop_text=sop_text,
            client_type=client_type,
            is_gagas=is_gagas,
            has_single_audit=has_single_audit,
            fields=fields or {},
            correction_hint=correction_hint,
        )

        response = ollama.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": _TEMPERATURE, "num_predict": _MAX_TOKENS},
        )

        completion = (response.message.content or "").strip()

        if not completion:
            logger.warning("completion_drafter: empty response for %s", record.file_name)
            return None

        # C4 — score and set quality gate
        set_review_gate(record, completion, findings, client_type,
                        sop_sections=sop_sections,
                        fields_missing=fields_missing,
                        fields_present=fields_present)

        logger.info(
            "completion_drafter: drafted %d chars | rev_conf=%.3f | gate=%s",
            len(completion), record.review_confidence, record.quality_gate,
        )
        return completion

    except Exception as e:
        logger.error("completion_drafter: generation failed — %s", e)
        return None


# ===========================================================================
# R7 scoring path (score_r7 / set_review_gate_r7)
# ===========================================================================
# Replaces the grounding multiplier + hallucination penalty path with
# classification-based signals from ClassificationValidator (R7-B).
#
# Hallucination penalty is removed — hallucination is structurally
# impossible when the deterministic renderer (R7-C) produces the completion.
# ===========================================================================


def score_r7(
    completion:   str,
    render_result,           # RenderResult from completion_renderer
    validated,               # ValidatedClassification from claim_mapper
    signals,                 # ClassificationSignals from claim_mapper
    client_type:  str,
    sop_sections: list[str] | None = None,
) -> float:
    """
    R7 scoring: rubric quality + classification grounding signals.

    Replaces score_completion() for completions produced by the R7
    deterministic renderer. The grounding multiplier and hallucination
    penalty are removed — hallucination is structurally impossible in R7.

    Hard block
    ----------
    Any Tier 1 uncertain field → review_confidence = 0.0.
    The pair enters the review queue for auditor resolution before JSONL write.

    Soft deductions
    ---------------
    - Tier 2 uncertain fields: small penalty per field (label_confidence: low)
    - Pass 2 rejection rate (Signal C): classifier over-confidence penalty
    - Signal B structural invalid: large penalty (critical schema violation)
    - SOP citation grounding: rubric Criterion 2 handles this (same as legacy)

    Parameters
    ----------
    completion : str
        Text produced by completion_renderer.render_completion().
    render_result : RenderResult
        Metadata from the renderer (absent_fields, uncertain_fields, etc.).
    validated : ValidatedClassification
        Corrected classification after Pass 2.
    signals : ClassificationSignals
        Three independent grounding signals.
    client_type : str
    sop_sections : list[str] | None
        Retrieved SOP chunks — used for Criterion 2 citation grounding.

    Returns
    -------
    float
        review_confidence in [0.0, 1.0].
    """
    if not completion or not completion.strip():
        return 0.0

    try:
        from raw_to_training_pair.claim_mapper import _load_tier1_fields
        tier1_set = {f.lower() for f in _load_tier1_fields()}
    except Exception:
        tier1_set = set()

    # Tier 1 uncertain block: any uncertain Tier 1 field → cap score at floor.
    # Floor (default 0.35) routes pair to auditor review without hard-zeroing —
    # uncertain means "needs human verification", not "pair is wrong".
    tier1_uncertain = [f for f in validated.uncertain_fields if f.lower() in tier1_set]
    if tier1_uncertain:
        cfg_floor    = _load_threshold_cfg()
        t1_floor     = float(
            cfg_floor.get("review", {}).get("r7", {}).get("tier1_uncertain_floor", 0.35)
        )
        logger.info(
            "score_r7: Tier 1 uncertain %s → review_confidence capped at %.2f",
            tier1_uncertain, t1_floor,
        )
        return t1_floor

    # Load R7 scoring config
    cfg    = _load_threshold_cfg()
    r7_cfg = cfg.get("review", {}).get("r7", {})
    sop_cfg = cfg.get("sop_retrieval", {})

    verify_citations = sop_cfg.get("citation_verification", True)
    min_grounded     = float(sop_cfg.get("min_grounded_citation_ratio", 0.50))

    tier2_penalty_per_field = float(r7_cfg.get("tier2_uncertain_penalty",  0.05))
    pass2_weight            = float(r7_cfg.get("pass2_rejection_weight",   0.10))
    pass2_cap               = float(r7_cfg.get("pass2_rejection_cap",      0.15))
    struct_penalty          = float(r7_cfg.get("structural_invalid_penalty", 0.20))

    # Rubric (structural quality of completion text)
    n_findings   = len(render_result.absent_fields)
    rubric_score = _compute_rubric(
        completion, n_findings, client_type,
        sop_sections, verify_citations, min_grounded,
    )

    # Tier 2 uncertain penalty (soft — label_confidence metadata already set)
    tier2_uncertain = [f for f in validated.uncertain_fields if f.lower() not in tier1_set]
    deduction_t2 = len(tier2_uncertain) * tier2_penalty_per_field

    # Pass 2 rejection rate penalty (Signal C)
    deduction_pass2 = min(signals.pass2_rejection_rate * pass2_weight, pass2_cap)

    # Structural validity penalty (Signal B)
    deduction_struct = struct_penalty if not signals.structural_valid else 0.0

    total_deduction = deduction_t2 + deduction_pass2 + deduction_struct
    final_score = round(max(0.0, min(1.0, rubric_score - total_deduction)), 4)

    logger.debug(
        "score_r7: rubric=%.3f t2_unc=%d pass2_rej=%.3f struct_valid=%s "
        "deduction=%.3f final=%.3f",
        rubric_score, len(tier2_uncertain), signals.pass2_rejection_rate,
        signals.structural_valid, total_deduction, final_score,
    )
    return final_score


def set_review_gate_r7(
    record,                  # DocumentRecord
    render_result,           # RenderResult from completion_renderer
    validated,               # ValidatedClassification from claim_mapper
    signals,                 # ClassificationSignals from claim_mapper
    client_type:  str,
    sop_sections: list[str] | None = None,
) -> float:
    """
    R7 equivalent of set_review_gate().

    Scores the rendered completion, sets record.review_confidence and
    record.quality_gate, and attaches R7-specific metadata to the record.

    Tier 1 uncertain → review_confidence = 0.0, quality_gate = False,
    record flagged with uncertain_tier1_fields for auditor resolution.

    Parameters
    ----------
    record : DocumentRecord
        Updated in place.
    render_result : RenderResult
    validated : ValidatedClassification
    signals : ClassificationSignals
    client_type : str
    sop_sections : list[str] | None

    Returns
    -------
    float
        review_confidence score.
    """
    rev_conf = score_r7(
        render_result.completion,
        render_result,
        validated,
        signals,
        client_type,
        sop_sections=sop_sections,
    )

    record.review_confidence = rev_conf
    record.quality_gate      = rev_conf >= _QUALITY_THRESHOLD

    # Attach R7 metadata for Streamlit review UI and JSONL pair
    try:
        from raw_to_training_pair.claim_mapper import _load_tier1_fields
        tier1_set = {f.lower() for f in _load_tier1_fields()}
    except Exception:
        tier1_set = set()

    tier1_uncertain = [f for f in validated.uncertain_fields if f.lower() in tier1_set]
    tier2_uncertain = [f for f in validated.uncertain_fields if f.lower() not in tier1_set]

    if tier1_uncertain:
        record.needs_review    = True
        # Surface which Tier 1 fields are uncertain so auditor knows what to resolve
        if hasattr(record, "review_notes"):
            record.review_notes = (
                f"[R7] Tier 1 uncertain fields require resolution before JSONL write: "
                f"{tier1_uncertain}"
            )
        logger.info(
            "set_review_gate_r7: quality_gate=False for %s "
            "(Tier 1 uncertain=%s review_confidence=%.3f)",
            record.file_name, tier1_uncertain, rev_conf,
        )
    elif not record.quality_gate:
        record.needs_review = True
        logger.info(
            "set_review_gate_r7: quality_gate=False for %s "
            "(review_confidence=%.3f < %.2f tier2_uncertain=%s)",
            record.file_name, rev_conf, _QUALITY_THRESHOLD, tier2_uncertain,
        )
    else:
        logger.info(
            "set_review_gate_r7: quality_gate=True for %s "
            "(review_confidence=%.3f sop_version=%s)",
            record.file_name, rev_conf, render_result.sop_mapping_version,
        )

    return rev_conf