"""
raw_to_training_pair/claim_mapper.py
======================================
Truth alignment validator between extracted fields and generated completions.

Purpose
-------
The rubric in completion_drafter.score_completion() measures structural quality
(does the completion look right). This module measures truth alignment (does
the completion correspond to what was actually extracted).

A completion can score 0.80+ on the rubric with entirely hallucinated content.
This module catches that by grounding every finding claim against the real
extracted field data.

ClaimMap
--------
Each numbered finding in the completion is parsed into a Claim and assigned
one of four statuses:

    VALID         — anchored to a real field in fields_missing,
                    severity label is present and attached to this finding,
                    SOP citation (if present) is in retrieved sop_sections.

    UNANCHORED    — no canonical field found via alias matching (Layer A)
                    and no semantic match (Layer B) above threshold.
                    Could be a paraphrase the system missed or a real gap.

    MISANCHORED   — a canonical field was found, but that field is in
                    fields_present (not fields_missing). The model is
                    flagging a correctly-documented field as a deficiency.
                    This is a factual error.

    HALLUCINATED  — a canonical field was found and it IS in fields_missing,
                    but the specific claims about it (severity, citation)
                    are fabricated — severity is unusually high for this field
                    type OR citation does not appear in retrieved sections
                    despite citation_verification being enabled.

Grounding signals (three only)
-------------------------------
    grounding_score     = count(fields_covered) / count(canonical fields_missing)
                          where canonical = tier1 + tier2 only (non-canonical OCR
                          extraction artifacts like preparer_id are excluded from
                          the denominator — they can never be covered by a VALID claim)
    hallucination_count = count(HALLUCINATED) + count(MISANCHORED)
    coverage_gap        = uncovered Tier 1 fields / total Tier 1 missing fields

Final score formula (applied in completion_drafter.score_completion())
-----------------------------------------------------------------------
    grounding_multiplier = grounding_min + grounding_score * grounding_range
    final_score = (rubric_score * grounding_multiplier)
                - (hallucination_count * penalty_per_claim)
                - (coverage_gap * tier1_fields_missing * tier1_penalty)
    final_score = clamp(final_score, 0.0, 1.0)

Two-layer anchoring
-------------------
Layer A — alias matching (fast, deterministic):
    Scan finding text for alias variants of canonical fields.
    Uses field_aliases.yaml reverse lookup.

Layer B — semantic fallback (only for UNANCHORED after Layer A fails):
    Compare finding sentence embedding against field_descriptions.yaml
    descriptions using cosine similarity. Only fires when
    grounding.semantic_fallback_enabled: true in threshold_config.
    Model: all-MiniLM-L6-v2 (80MB, local, no network).

Public API
----------
    build_claim_map(completion, fields_missing, fields_present,
                    sop_sections, client_type) -> ClaimMap

    compute_grounding_signals(claim_map) -> GroundingSignals
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PKG_DIR        = Path(__file__).parent
_PROJECT_DIR    = _PKG_DIR.parent
_ALIASES_PATH   = _PROJECT_DIR / "auditai_data_normalization" / "field_aliases.yaml"
_TIERS_PATH     = _PROJECT_DIR / "config" / "field_tiers.yaml"
_THRESHOLD_PATH = _PROJECT_DIR / "auditai_data_normalization" / "alias_registry" / "threshold_config.yaml"
_DESC_PATH      = _PKG_DIR / "field_descriptions.yaml"

_LLAMA_MODEL   = "llama3.1:8b"   # Pass 2 LLM verifier (evidence spot-check)

_SEVERITY_RE   = re.compile(r"Severity:\s*(High|Medium|Low|Informational)", re.IGNORECASE)
_FINDING_RE    = re.compile(r"^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.|^\s*RECOMMENDATION:|$)",
                             re.MULTILINE | re.DOTALL)
_SOP_CITE_RE   = re.compile(
    r"(?:SOP[\w-]*\s+)?§\s*[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*(?:\([^)]*\))?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class ClaimStatus(str, Enum):
    VALID        = "VALID"
    UNANCHORED   = "UNANCHORED"
    MISANCHORED  = "MISANCHORED"
    HALLUCINATED = "HALLUCINATED"


@dataclass
class Claim:
    index:           int
    text:            str
    linked_field:    str | None       # canonical field name or None
    status:          ClaimStatus
    severity:        str | None       # "High" | "Medium" | "Low" | "Informational" | None
    sop_citation:    str | None
    anchor_method:   str              # "alias" | "semantic" | "none"
    anchor_score:    float            # similarity score (1.0 for alias match)
    semantic_suggestion: str | None  # field suggested by Layer B when UNANCHORED


@dataclass
class ClaimMap:
    claims:            list[Claim]
    fields_missing:    list[str]
    fields_present:    list[str]
    tier1_missing:     list[str]      # Tier 1 fields that are missing
    fields_covered:    list[str]      # missing fields that appear in at least one VALID claim
    tier1_uncovered:   list[str]      # Tier 1 missing fields with no VALID claim


@dataclass
class GroundingSignals:
    grounding_score:      float    # fields_covered / fields_missing (coverage-based)
    hallucination_count:  int      # HALLUCINATED + MISANCHORED (excludes fixed_admin)
    coverage_gap:         float    # tier1_uncovered / tier1_missing (0 if no tier1 missing)
    claim_count:          int
    valid_count:          int
    unanchored_count:     int
    misanchored_count:    int
    hallucinated_count:   int
    fixed_admin_count:    int      # claims reclassified from MISANCHORED due to fixed_admin field


# ---------------------------------------------------------------------------
# Config loaders (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_thresholds() -> dict:
    try:
        with open(_THRESHOLD_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_sop_field_classes():
    try:
        from config.settings import load_sop_field_classes
        return load_sop_field_classes()
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_reverse_aliases() -> dict[str, str]:
    """Build {alias_variant_lower: canonical_field} reverse lookup."""
    if not _ALIASES_PATH.exists():
        return {}
    with open(_ALIASES_PATH) as f:
        aliases = yaml.safe_load(f) or {}
    reverse: dict[str, str] = {}
    for canonical, variants in aliases.items():
        reverse[canonical.lower().replace("_", " ")] = canonical
        for v in (variants or []):
            reverse[str(v).lower().strip()] = canonical
    return reverse


@lru_cache(maxsize=1)
def _load_tier1_fields() -> frozenset[str]:
    if not _TIERS_PATH.exists():
        return frozenset()
    with open(_TIERS_PATH) as f:
        tiers = yaml.safe_load(f) or {}
    return frozenset(
        e["field"] for e in (tiers.get("tier1") or [])
        if isinstance(e, dict) and "field" in e
    )


@lru_cache(maxsize=1)
def _load_canonical_fields() -> frozenset[str]:
    """
    Load all canonical field names (tier1 + tier2), lowercased.

    Used to filter fields_missing before computing grounding_score so that
    non-canonical fields (e.g. preparer_id, extracted by OCR but not in
    field_tiers.yaml) do not inflate the denominator and floor grounding to 0.
    """
    if not _TIERS_PATH.exists():
        return frozenset()
    with open(_TIERS_PATH) as f:
        tiers = yaml.safe_load(f) or {}
    fields: set[str] = set()
    for tier_key in ("tier1", "tier2"):
        for e in (tiers.get(tier_key) or []):
            if isinstance(e, dict) and "field" in e:
                fields.add(e["field"].lower())
    return frozenset(fields)


@lru_cache(maxsize=1)
def _load_field_descriptions() -> dict[str, str]:
    if not _DESC_PATH.exists():
        return {}
    with open(_DESC_PATH) as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def _load_field_aliases() -> dict[str, list[str]]:
    """
    Build {canonical_field: [alias_variants_lower]} forward lookup.

    Used in Pass 2 keyword check: for each field marked "present" by Pass 1,
    check whether any of its aliases appear in the workpaper text.
    A keyword hit → strong_present (skip embedding and Llama checks).
    """
    if not _ALIASES_PATH.exists():
        return {}
    with open(_ALIASES_PATH) as f:
        aliases = yaml.safe_load(f) or {}
    forward: dict[str, list[str]] = {}
    for canonical, variants in aliases.items():
        terms = [canonical.lower().replace("_", " ")]
        for v in (variants or []):
            terms.append(str(v).lower().strip())
        forward[canonical] = terms
    return forward


def _reset_caches() -> None:
    _load_thresholds.cache_clear()
    _load_sop_field_classes.cache_clear()
    _load_reverse_aliases.cache_clear()
    _load_field_aliases.cache_clear()
    _load_tier1_fields.cache_clear()
    _load_canonical_fields.cache_clear()
    _load_field_descriptions.cache_clear()


# ---------------------------------------------------------------------------
# Semantic fallback (Layer B)
# ---------------------------------------------------------------------------

_SEMANTIC_MODEL = None
_DESC_EMBEDDINGS: dict[str, Any] = {}


def _get_semantic_model():
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.debug("claim_mapper: loaded all-MiniLM-L6-v2")
        except Exception as e:
            logger.debug("claim_mapper: sentence-transformers unavailable — %s", e)
            _SEMANTIC_MODEL = False  # sentinel: tried and failed
    return _SEMANTIC_MODEL if _SEMANTIC_MODEL is not False else None


def _get_desc_embeddings(model) -> dict[str, Any]:
    global _DESC_EMBEDDINGS
    if not _DESC_EMBEDDINGS:
        descriptions = _load_field_descriptions()
        if descriptions and model:
            try:
                import numpy as np
                texts  = list(descriptions.values())
                fields = list(descriptions.keys())
                embeds = model.encode(texts, convert_to_numpy=True)
                _DESC_EMBEDDINGS = {
                    fields[i]: embeds[i] for i in range(len(fields))
                }
            except Exception as e:
                logger.debug("claim_mapper: description embedding failed — %s", e)
    return _DESC_EMBEDDINGS


def _semantic_anchor(text: str, threshold: float) -> tuple[str | None, float]:
    """
    Layer B — semantic fallback anchoring.
    Returns (canonical_field, similarity_score) or (None, 0.0).
    Only called when Layer A finds no alias match.
    """
    model = _get_semantic_model()
    if model is None:
        return None, 0.0

    desc_embeddings = _get_desc_embeddings(model)
    if not desc_embeddings:
        return None, 0.0

    try:
        import numpy as np
        query_embed = model.encode([text], convert_to_numpy=True)[0]

        best_field, best_score = None, 0.0
        for field_name, desc_embed in desc_embeddings.items():
            # Cosine similarity
            norm_q = np.linalg.norm(query_embed)
            norm_d = np.linalg.norm(desc_embed)
            if norm_q == 0 or norm_d == 0:
                continue
            sim = float(np.dot(query_embed, desc_embed) / (norm_q * norm_d))
            if sim > best_score:
                best_score = sim
                best_field = field_name

        if best_score >= threshold:
            return best_field, best_score
        return None, best_score

    except Exception as e:
        logger.debug("claim_mapper: semantic anchor failed — %s", e)
        return None, 0.0


# ---------------------------------------------------------------------------
# Layer A — alias matching
# ---------------------------------------------------------------------------

def _alias_anchor(text: str) -> str | None:
    """
    Layer A — scan text for alias variants of canonical fields.
    Returns canonical field name or None.
    Matches longest alias first to avoid partial matches.
    """
    reverse = _load_reverse_aliases()
    text_lower = text.lower()

    # Sort by length descending — match longest alias first
    for alias in sorted(reverse.keys(), key=len, reverse=True):
        if len(alias) < 3:
            continue
        if alias in text_lower:
            return reverse[alias]
    return None


# ---------------------------------------------------------------------------
# SOP citation normalisation
# ---------------------------------------------------------------------------

def _norm_citation(s: str) -> str:
    s = s.strip().lower()
    # Extract the section identifier after § (stops before parentheticals)
    # so §Q1(a) and §Q1 both normalize to "q1" and match each other
    m = re.search(r"§\s*([a-z0-9]+(?:[.\-][a-z0-9]+)*)", s)
    if m:
        return m.group(1)
    return s.replace(" ", "").lstrip("sop").lstrip("§")


# ---------------------------------------------------------------------------
# Core claim parser
# ---------------------------------------------------------------------------

def _parse_findings(completion: str) -> list[tuple[int, str]]:
    """
    Extract numbered findings from completion text.
    Returns list of (index, full_finding_text) tuples.
    """
    findings_section = completion
    # Trim to just the FINDINGS section if possible
    findings_match = re.search(
        r"FINDINGS:\s*\n(.*?)(?=\nRECOMMENDATION:|\Z)",
        completion, re.DOTALL | re.IGNORECASE,
    )
    if findings_match:
        findings_section = findings_match.group(1)

    results = []
    for m in _FINDING_RE.finditer(findings_section):
        idx  = int(m.group(1))
        text = m.group(2).strip()
        if text:
            results.append((idx, text))

    # Fallback: simple numbered line detection
    if not results:
        for line in completion.splitlines():
            m = re.match(r"^\s*(\d+)\.\s+(.+)", line)
            if m:
                results.append((int(m.group(1)), m.group(2).strip()))

    return results


# ---------------------------------------------------------------------------
# Main ClaimMap builder
# ---------------------------------------------------------------------------

def build_claim_map(
    completion: str,
    fields_missing: list[str],
    fields_present: list[str],
    sop_sections: list[str] | None = None,
    client_type: str = "",
) -> ClaimMap:
    """
    Build a ClaimMap by grounding each finding in the completion against
    the extracted field data.

    Parameters
    ----------
    completion : str
        The generated assistant completion text.
    fields_missing : list[str]
        Canonical field names not found in the document (real deficiencies).
    fields_present : list[str]
        Canonical field names found in the document.
    sop_sections : list[str] | None
        Retrieved SOP section identifiers (for citation verification).
    client_type : str
        Client type label (not used in anchoring, kept for future use).

    Returns
    -------
    ClaimMap
    """
    cfg          = _load_thresholds()
    grounding_cfg = cfg.get("grounding", {})
    sem_enabled  = grounding_cfg.get("semantic_fallback_enabled", True)
    sem_threshold = float(grounding_cfg.get("semantic_fallback_threshold", 0.45))

    missing_set  = {f.lower() for f in fields_missing}
    present_set  = {f.lower() for f in fields_present}

    # R4 — fixed-admin reclassification
    # Load non-deficient canonical fields for this client type so that claims
    # anchoring to structurally fixed fields are marked UNANCHORED (structural
    # noise) rather than MISANCHORED (factual error), carrying the softer
    # fixed_admin_penalty instead of the misanchored_penalty.
    _sfc = _load_sop_field_classes()
    _fixed_admin_canonicals: frozenset[str] = (
        _sfc.non_deficient_canonical_fields(client_type or None)
        if _sfc else frozenset()
    )
    tier1_fields = _load_tier1_fields()
    tier1_missing = [f for f in fields_missing if f.lower() in
                     {t.lower() for t in tier1_fields}]

    retrieved_normalised = (
        {_norm_citation(s) for s in sop_sections}
        if sop_sections else set()
    )

    finding_pairs = _parse_findings(completion)
    claims: list[Claim] = []

    for idx, text in finding_pairs:
        # Extract severity from this finding block
        sev_match = _SEVERITY_RE.search(text)
        severity  = sev_match.group(1) if sev_match else None

        # Extract SOP citation from this finding block
        cite_match  = _SOP_CITE_RE.search(text)
        sop_citation = cite_match.group(0) if cite_match else None
        citation_grounded = (
            _norm_citation(sop_citation) in retrieved_normalised
            if sop_citation and retrieved_normalised else None
        )

        # Layer A — alias matching
        linked_field   = _alias_anchor(text)
        anchor_method  = "alias" if linked_field else "none"
        anchor_score   = 1.0 if linked_field else 0.0
        semantic_sug   = None

        # Layer B — semantic fallback for unanchored claims
        if linked_field is None and sem_enabled:
            sem_field, sem_score = _semantic_anchor(text, sem_threshold)
            if sem_field:
                linked_field  = sem_field
                anchor_method = "semantic"
                anchor_score  = sem_score
                logger.debug(
                    "claim_mapper: semantic anchor %r → %s (%.3f)",
                    text[:60], sem_field, sem_score,
                )
            else:
                semantic_sug  = sem_field  # None — no suggestion above threshold
                anchor_score  = sem_score  # best score even if below threshold

        # Determine status
        if linked_field is None:
            status = ClaimStatus.UNANCHORED

        elif linked_field.lower() in present_set:
            if linked_field.lower() in {f.lower() for f in _fixed_admin_canonicals}:
                # Field is present AND classified as fixed_administrative for
                # this client type — structural noise, not a factual error.
                # Reclassify to UNANCHORED so the softer fixed_admin_penalty
                # applies instead of misanchored_penalty. anchor_method flags
                # the reason for observability in the review queue.
                status = ClaimStatus.UNANCHORED
                anchor_method = "fixed_admin"
            else:
                # Model flagged a field that IS documented — factual error
                status = ClaimStatus.MISANCHORED

        elif linked_field.lower() in missing_set:
            # Field is genuinely missing — check for hallucinated specifics
            if (
                sop_citation
                and retrieved_normalised
                and citation_grounded is False
            ):
                # Has a specific citation but it's not in retrieved chunks
                status = ClaimStatus.HALLUCINATED
            else:
                status = ClaimStatus.VALID

        else:
            # Field found but not in either present or missing set —
            # field exists in aliases but wasn't extracted at all
            status = ClaimStatus.UNANCHORED

        claims.append(Claim(
            index            = idx,
            text             = text[:500],  # truncate for storage
            linked_field     = linked_field,
            status           = status,
            severity         = severity,
            sop_citation     = sop_citation,
            anchor_method    = anchor_method,
            anchor_score     = round(anchor_score, 4),
            semantic_suggestion = semantic_sug,
        ))

    # Coverage: which missing fields have at least one VALID claim
    fields_covered = list({
        c.linked_field for c in claims
        if c.status == ClaimStatus.VALID and c.linked_field
    })
    covered_set = {f.lower() for f in fields_covered}
    tier1_uncovered = [
        f for f in tier1_missing
        if f.lower() not in covered_set
    ]

    return ClaimMap(
        claims          = claims,
        fields_missing  = list(fields_missing),
        fields_present  = list(fields_present),
        tier1_missing   = tier1_missing,
        fields_covered  = fields_covered,
        tier1_uncovered = tier1_uncovered,
    )


# ---------------------------------------------------------------------------
# Grounding signals
# ---------------------------------------------------------------------------

def compute_grounding_signals(claim_map: ClaimMap) -> GroundingSignals:
    """
    Compute the three grounding signals from a ClaimMap.

    Returns GroundingSignals with:
        grounding_score     — VALID / total (0.0 if no findings)
        hallucination_count — HALLUCINATED + MISANCHORED
        coverage_gap        — tier1_uncovered / tier1_missing
    """
    claims = claim_map.claims
    total  = len(claims)

    valid_count        = sum(1 for c in claims if c.status == ClaimStatus.VALID)
    unanchored_count   = sum(1 for c in claims if c.status == ClaimStatus.UNANCHORED)
    misanchored_count  = sum(1 for c in claims if c.status == ClaimStatus.MISANCHORED)
    hallucinated_count = sum(1 for c in claims if c.status == ClaimStatus.HALLUCINATED)
    fixed_admin_count  = sum(1 for c in claims if c.anchor_method == "fixed_admin")

    # Denominator: only canonical fields (tier1+tier2) count toward grounding.
    # Non-canonical fields (e.g. preparer_id) come from raw OCR extraction but
    # have no alias mappings, so they can never be covered by a VALID claim.
    # Including them in the denominator floors grounding_score to 0 on every
    # clean pair, making the scoring unresponsive to actual completion quality.
    canonical_fields = _load_canonical_fields()
    canonical_missing_count = sum(
        1 for f in claim_map.fields_missing if f.lower() in canonical_fields
    )
    grounding_score = (
        len(claim_map.fields_covered) / canonical_missing_count
        if canonical_missing_count > 0
        else 1.0   # nothing canonical missing → no findings expected → fully grounded
    )

    # fixed_admin claims are excluded from hallucination_count — they carry
    # their own softer penalty applied separately in score_completion().
    hallucination_count = hallucinated_count + misanchored_count

    tier1_total   = len(claim_map.tier1_missing)
    tier1_uncov   = len(claim_map.tier1_uncovered)
    coverage_gap  = tier1_uncov / tier1_total if tier1_total > 0 else 0.0

    return GroundingSignals(
        grounding_score      = round(grounding_score, 4),
        hallucination_count  = hallucination_count,
        coverage_gap         = round(coverage_gap, 4),
        claim_count          = total,
        valid_count          = valid_count,
        unanchored_count     = unanchored_count,
        misanchored_count    = misanchored_count,
        hallucinated_count   = hallucinated_count,
        fixed_admin_count    = fixed_admin_count,
    )


# ===========================================================================
# R7-B — ClassificationValidator
# ===========================================================================
# Validates the constrained classifier output (Pass 1), runs Pass 2 evidence
# verification on "present" labels, and computes three independent grounding
# signals. See docs/architecture/09_r7_classifier_invariants.md.
# ===========================================================================


# ---------------------------------------------------------------------------
# R7-B dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ValidatedClassification:
    """
    Corrected FieldClassification after Pass 2 verification.

    field_states reflects Pass 2 corrections:
      "present"   fields confirmed by keyword evidence (strong) or
                  embedding/Llama evidence (provisional)
      "uncertain" fields downgraded from "present" by Pass 2 (no evidence found)
                  plus original "uncertain" from Pass 1
      "absent"    unchanged from Pass 1

    provisional_fields is the subset of present_fields confirmed only by weak
    evidence (embedding or Llama). These are present in the completion output
    but flagged in pair metadata so auditors can distinguish confidence levels.
    """
    field_states:        dict[str, str]
    canonical_fields:    list[str]
    model:               str
    absent_fields:       list[str]
    present_fields:      list[str]        # strong + provisional combined
    uncertain_fields:    list[str]        # original uncertain + Pass-2 downgrades
    provisional_fields:  list[str]        # subset of present_fields (weak evidence only)


@dataclass
class ClassificationSignals:
    """
    Three independent classification grounding signals.

    These MUST NOT be summed or collapsed before being stored. Each signal
    measures a different failure mode. A pair can score well on two signals
    and poorly on the third — that distinction is diagnostically valuable.

    Signal A — classification entropy (LLM uncertainty from Pass 1)
    Signal B — structural validity (schema + canonical registry check)
    Signal C — cross-pass agreement (Pass 2 rejection rate)
    """
    # Signal A
    uncertain_rate:           float   # uncertain_count / total_canonical_fields
    uncertain_count:          int
    total_canonical:          int

    # Signal B
    structural_valid:         bool    # True if full keyspace + no illegal keys
    drift_count:              int     # fields in workpaper not in canonical registry
    schema_violation_count:   int     # missing canonical keys in classifier output
                                      # (should be 0 with outlines — if nonzero: critical bug)
    # Signal C
    pass2_rejection_rate:     float   # pass2_downgrades / pass1_present_count
    pass2_downgrades:         int     # present → uncertain transitions in Pass 2
    pass1_present_count:      int     # how many fields Pass 1 marked "present"


# ---------------------------------------------------------------------------
# Pass 2 — evidence verification internals
# ---------------------------------------------------------------------------

def _pass2_alias_check(field: str, workpaper_text_lower: str) -> bool:
    """
    Check whether any alias of `field` appears in the workpaper text.
    Keyword hit is strong evidence — confirms "present" without further checks.
    Uses field_aliases.yaml forward lookup (canonical → [alias variants]).
    """
    aliases = _load_field_aliases().get(field, [])
    return any(len(alias) >= 3 and alias in workpaper_text_lower for alias in aliases)


def _pass2_embedding_check(field: str, workpaper_text: str, threshold: float) -> float:
    """
    Cosine similarity between workpaper text embedding and field description.
    Returns similarity score (0.0 if model or descriptions unavailable).
    """
    model = _get_semantic_model()
    if model is None:
        return 0.0
    desc_embeds = _get_desc_embeddings(model)
    if field not in desc_embeds:
        return 0.0
    try:
        import numpy as np
        text_snippet = " ".join(workpaper_text.split()[:400])
        query_embed  = model.encode([text_snippet], convert_to_numpy=True)[0]
        field_embed  = desc_embeds[field]
        norm_q = np.linalg.norm(query_embed)
        norm_f = np.linalg.norm(field_embed)
        if norm_q == 0 or norm_f == 0:
            return 0.0
        return float(np.dot(query_embed, field_embed) / (norm_q * norm_f))
    except Exception as e:
        logger.debug("claim_mapper: Pass 2 embedding check failed for %s — %s", field, e)
        return 0.0


def _pass2_llama_check(field: str, workpaper_text: str) -> str:
    """
    Ask Llama 3.1 8B whether the workpaper contains evidence of `field`.
    Returns "yes", "no", or "unclear".
    Called only when both keyword and embedding checks find no evidence.
    """
    descriptions = _load_field_descriptions()
    desc = descriptions.get(field, field.replace("_", " "))

    text_snippet = " ".join(workpaper_text.split()[:600])
    prompt = (
        f'Does the following audit workpaper text contain clear evidence that '
        f'"{desc}" is documented?\n\n'
        f"Text:\n{text_snippet}\n\n"
        f"Answer with exactly one word: yes, no, or unclear."
    )
    try:
        import ollama as _ollama
        response = _ollama.chat(
            model   = _LLAMA_MODEL,
            messages= [{"role": "user", "content": prompt}],
            options = {"temperature": 0.0, "num_predict": 10},
        )
        answer = (response.message.content or "").strip().lower().split()
        return answer[0] if answer and answer[0] in ("yes", "no", "unclear") else "unclear"
    except Exception as e:
        logger.debug("claim_mapper: Llama evidence check failed for %s — %s", field, e)
        return "unclear"


def _llama_available() -> bool:
    try:
        import ollama as _ollama
        result = _ollama.list()
        return any(_LLAMA_MODEL in (m.model or "") for m in result.models)
    except Exception:
        return False


def _pass2_verify_field(
    field:                str,
    workpaper_text:       str,
    workpaper_text_lower: str,
    embed_threshold:      float,
    use_llama:            bool,
) -> str:
    """
    Three-tier evidence cascade for a single "present" field.

    Returns one of:
        "strong_present"      — keyword alias found in text (high precision)
        "provisional_present" — embedding or Llama confirms presence (weaker signal)
        "uncertain"           — no evidence found; field downgraded from "present"

    Cascade order (exit on first hit):
        1. Keyword  → strong_present
        2. Embedding ≥ threshold → provisional_present
        3. Llama "yes" → provisional_present
        4. Default → uncertain
    """
    if _pass2_alias_check(field, workpaper_text_lower):
        return "strong_present"

    embed_score = _pass2_embedding_check(field, workpaper_text, embed_threshold)
    if embed_score >= embed_threshold:
        return "provisional_present"

    if use_llama:
        if _pass2_llama_check(field, workpaper_text) == "yes":
            return "provisional_present"

    return "uncertain"


# ---------------------------------------------------------------------------
# R7-B — Public API
# ---------------------------------------------------------------------------

def validate_classification(
    classification,           # FieldClassification from field_classifier.py
    workpaper_text: str,
    client_type:    str = "",
) -> tuple["ValidatedClassification", "ClassificationSignals"]:
    """
    Validate Pass 1 classifier output and run Pass 2 evidence verification.

    Steps
    -----
    1. Schema check (Signal B):
       - Full keyspace: every canonical field has a state
       - No illegal keys (alias leakage / unknown fields)
    2. Pass 2 for each "present" field:
       - Keyword → strong_present
       - Embedding → provisional_present
       - Llama → provisional_present or uncertain
    3. Compute three independent signals (A, B, C).

    Parameters
    ----------
    classification : FieldClassification
        Raw output from field_classifier.classify().
    workpaper_text : str
        Full extracted workpaper text (for keyword and embedding checks).
    client_type : str

    Returns
    -------
    (ValidatedClassification, ClassificationSignals)
    """
    cfg          = _load_thresholds()
    grounding_cfg = cfg.get("grounding", {})
    embed_threshold = float(
        grounding_cfg.get("pass2_embedding_threshold",
        grounding_cfg.get("semantic_fallback_threshold", 0.45))
    )

    canonical_set = set(classification.canonical_fields)
    states        = dict(classification.field_states)  # mutable copy

    # --- Signal B: schema check -----------------------------------------

    output_keys   = set(states.keys())
    missing_keys  = [f for f in canonical_set if f not in output_keys]
    # extra_keys: read from FieldClassification.unknown_keys (captured before
    # field_states was filtered to canonical-only in field_classifier.classify()).
    # Recomputing from field_states would always give [] — field_classifier
    # already filters field_states to canonical keys.
    extra_keys    = list(getattr(classification, "unknown_keys", []))

    if missing_keys:
        logger.error(
            "claim_mapper: SCHEMA VIOLATION — missing_field_detected: %s "
            "(outlines should prevent this — check model/schema compatibility)",
            missing_keys,
        )
    if extra_keys:
        logger.warning(
            "claim_mapper: unknown_field_detected: %s "
            "(alias leakage or schema built from stale field list)",
            extra_keys,
        )

    structural_valid      = len(missing_keys) == 0 and len(extra_keys) == 0
    schema_violation_count = len(missing_keys)
    drift_count           = len(extra_keys)

    # --- Pass 2: verify "present" labels ---------------------------------

    workpaper_text_lower = workpaper_text.lower()
    use_llama            = _llama_available()
    if not use_llama:
        logger.debug("claim_mapper: Llama (%s) not available — Pass 2 uses keyword+embedding only",
                     _LLAMA_MODEL)

    pass1_present = [f for f in classification.canonical_fields if states.get(f) == "present"]
    pass2_downgrades  = 0
    provisional_fields: list[str] = []

    for field in pass1_present:
        evidence = _pass2_verify_field(
            field, workpaper_text, workpaper_text_lower, embed_threshold, use_llama,
        )
        if evidence == "strong_present":
            pass  # keep as "present"
        elif evidence == "provisional_present":
            provisional_fields.append(field)
            # state stays "present" — provisional flag is in metadata only
        else:  # "uncertain"
            states[field] = "uncertain"
            pass2_downgrades += 1
            logger.debug(
                "claim_mapper: Pass 2 downgraded %s present→uncertain "
                "(no keyword/embedding/Llama evidence)",
                field,
            )

    # --- Recompute convenience lists after Pass 2 ------------------------

    absent_fields   = [f for f in classification.canonical_fields if states.get(f) == "absent"]
    present_fields  = [f for f in classification.canonical_fields if states.get(f) == "present"]
    uncertain_fields = [f for f in classification.canonical_fields if states.get(f) == "uncertain"]

    # --- Signal A: classification entropy --------------------------------

    total_canonical = len(classification.canonical_fields)
    uncertain_count = len(uncertain_fields)
    uncertain_rate  = round(uncertain_count / total_canonical, 4) if total_canonical > 0 else 0.0

    # --- Signal C: cross-pass agreement ----------------------------------

    pass1_present_count  = len(pass1_present)
    pass2_rejection_rate = (
        round(pass2_downgrades / pass1_present_count, 4)
        if pass1_present_count > 0 else 0.0
    )

    logger.info(
        "claim_mapper: validate_classification %s — "
        "absent=%d present=%d uncertain=%d provisional=%d "
        "pass2_downgrades=%d rejection_rate=%.3f structural_valid=%s",
        client_type or "unknown",
        len(absent_fields), len(present_fields), len(uncertain_fields),
        len(provisional_fields), pass2_downgrades, pass2_rejection_rate,
        structural_valid,
    )

    validated = ValidatedClassification(
        field_states      = {f: states[f] for f in classification.canonical_fields
                             if f in states},
        canonical_fields  = list(classification.canonical_fields),
        model             = classification.model,
        absent_fields     = absent_fields,
        present_fields    = present_fields,
        uncertain_fields  = uncertain_fields,
        provisional_fields = provisional_fields,
    )
    signals = ClassificationSignals(
        uncertain_rate          = uncertain_rate,
        uncertain_count         = uncertain_count,
        total_canonical         = total_canonical,
        structural_valid        = structural_valid,
        drift_count             = drift_count,
        schema_violation_count  = schema_violation_count,
        pass2_rejection_rate    = pass2_rejection_rate,
        pass2_downgrades        = pass2_downgrades,
        pass1_present_count     = pass1_present_count,
    )
    return validated, signals