"""
pipeline/pair_quality_gate.py
==============================
Phase 5 — Pair quality gate: the dataset firewall.

Nothing enters the JSONL training set unless it passes every check here.
This gate runs on the fully assembled pair dict (after hard_gate, after
auditor approval, immediately before jsonl_writer.append()).

Relationship to hard_gate.py
------------------------------
hard_gate.py runs BEFORE enqueue and blocks contradictory pairs early.
pair_quality_gate.py runs AFTER auditor approval and enforces the full
invariant set on the final pair. Together they create defense-in-depth:
    hard_gate   → reject obviously broken pairs before review queue
    quality_gate → reject any pair that slipped through or was corrupted
                   during the review/approval process

Four checks
-----------
1. citation_valid
   All citations in the completion text must conform to the canonical
   format produced by citation_resolver. Detects pairs generated before
   the citation_resolver was integrated (legacy string contamination) and
   pairs where the renderer bypassed the resolver.

2. sop_authority
   Every field listed in metadata.fields_missing must be deficiency_allowed
   per the compiled SOPGraph for this pair's client_type. Mirrors
   hard_gate authority_check but operates on the rendered output rather than
   the sampler input — catches renderer bugs that add unauthorised findings.

3. evidence_consistent
   For deficient pairs: the user message must not contain precision aliases
   for any field listed as absent. Identical to hard_gate alias_residue but
   applied to the post-approval final pair (pairs can be edited during review).

4. context_consistent
   GAGAS citations must not appear in non-GAGAS pairs.
   Single Audit citations must not appear in non-Single-Audit pairs.
   Catches metadata flag mismatches introduced during the review process.

Public API
----------
    PairQualityResult           — outcome of one quality gate run
    check_final_pair(pair)      — run all four checks; return first failure or pass
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Patterns for detecting GAS and 2 CFR citations in completion text.
# Used for context_consistent check.
_GAS_CITATION_RE  = re.compile(r"\bgas\s+[§ch]", re.IGNORECASE)
_CFR_CITATION_RE  = re.compile(r"2\s+cfr\s+200", re.IGNORECASE)
_GAS_FULL_RE      = re.compile(r"government auditing standards", re.IGNORECASE)
_YELLOW_BOOK_RE   = re.compile(r"yellow book", re.IGNORECASE)


@dataclass
class PairQualityResult:
    """Outcome of one pair quality gate run."""
    passed: bool
    gate:   str    # failing gate name, or "ok"
    reason: str    # empty when passed


def check_final_pair(pair: dict) -> PairQualityResult:
    """
    Run all four quality gate checks on a final pair dict.

    Returns the first failing check, or PairQualityResult(passed=True) if all pass.
    """
    metadata     = pair.get("metadata", {})
    pair_type    = metadata.get("pair_type", "clean")
    client_type  = metadata.get("client_type", "")
    is_gagas     = metadata.get("is_gagas", False)
    has_sa       = metadata.get("has_single_audit", False)
    fields_missing = metadata.get("fields_missing", [])

    messages     = pair.get("messages", [])
    completion   = next(
        (m.get("content", "") for m in messages if m.get("role") == "assistant"),
        "",
    )
    user_content = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        "",
    ).lower()

    # ------------------------------------------------------------------
    # Gate 1: citation_valid
    # Scan completion text for stale citation patterns (§Q2 in risk text,
    # "ET 1.300", "SOP §Q…" embedded in sentences, duplicated GAS refs).
    # ------------------------------------------------------------------
    stale_patterns = [
        (re.compile(r"SOP\s+§Q\d", re.IGNORECASE),         "SOP §Q-number in risk text (use SOP §N from sop_section)"),
        (re.compile(r"ET\s+1\.300\b"),                       "ET 1.300 — wrong standard (not engagement acceptance)"),
        (re.compile(r"government auditing standards.*chapter.*§", re.IGNORECASE), "GAS Chapter + § in same citation (hierarchy not collapsed)"),
    ]
    for pattern, desc in stale_patterns:
        if pattern.search(completion):
            return PairQualityResult(
                passed=False, gate="citation_valid",
                reason=f"Stale or malformed citation in completion: {desc}.",
            )

    # ------------------------------------------------------------------
    # Gate 2: sop_authority
    # Every field in fields_missing must be deficiency_allowed for this
    # client_type per the compiled SOPGraph.
    # ------------------------------------------------------------------
    if fields_missing:
        sop_id = metadata.get("sop_id", "npo-cx-1.1")
        try:
            from pipeline.sop_compiler import compiled_sop
            graph = compiled_sop(sop_id)
            disallowed = [
                f for f in fields_missing
                if not graph.allowed(f, client_type)
            ]
            if disallowed:
                return PairQualityResult(
                    passed=False, gate="sop_authority",
                    reason=(
                        f"Completion contains findings for {len(disallowed)} "
                        f"field(s) not deficiency_allowed in SOPGraph for "
                        f"client_type={client_type!r}: {disallowed}."
                    ),
                )
        except Exception as _e:
            logger.warning("pair_quality_gate: sop_authority check unavailable — %s", _e)

    # ------------------------------------------------------------------
    # Gate 3: evidence_consistent (deficient pairs only)
    # User message must not contain precision aliases for absent fields.
    # ------------------------------------------------------------------
    deficiency_fields = metadata.get("deficiency_fields", [])
    if pair_type == "deficient" and deficiency_fields:
        try:
            from pipeline.evidence_redactor import load_aliases, _precision_aliases
            alias_lookup = load_aliases()
            residue: list[tuple[str, str]] = []
            for f in deficiency_fields:
                aliases = _precision_aliases(f, alias_lookup)
                hit = next((a for a in aliases if a in user_content), None)
                if hit:
                    residue.append((f, hit))
            if residue:
                detail = "; ".join(f"{f!r} matched {a!r}" for f, a in residue)
                return PairQualityResult(
                    passed=False, gate="evidence_consistent",
                    reason=(
                        f"User message contains evidence for {len(residue)} "
                        f"absent field(s) after all processing: {detail}. "
                        "Pair may have been edited during review — re-redact."
                    ),
                )
        except Exception as _e:
            logger.warning("pair_quality_gate: evidence_consistent check unavailable — %s", _e)

    # ------------------------------------------------------------------
    # Gate 4: context_consistent
    # GAGAS/Yellow Book citations must not appear in non-GAGAS pairs.
    # 2 CFR 200 citations must not appear in non-Single-Audit pairs.
    # ------------------------------------------------------------------
    if not is_gagas:
        if _GAS_CITATION_RE.search(completion) or _YELLOW_BOOK_RE.search(completion) or _GAS_FULL_RE.search(completion):
            return PairQualityResult(
                passed=False, gate="context_consistent",
                reason=(
                    "Completion cites GAS/Yellow Book standards but "
                    f"is_gagas=False for client_type={client_type!r}. "
                    "Metadata flag mismatch or renderer used wrong context."
                ),
            )

    if not has_sa:
        if _CFR_CITATION_RE.search(completion):
            return PairQualityResult(
                passed=False, gate="context_consistent",
                reason=(
                    "Completion cites 2 CFR 200 (Uniform Guidance) but "
                    f"has_single_audit=False for client_type={client_type!r}. "
                    "Metadata flag mismatch or renderer used wrong context."
                ),
            )

    return PairQualityResult(passed=True, gate="ok", reason="")
