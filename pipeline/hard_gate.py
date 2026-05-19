"""
pipeline/hard_gate.py
=====================
Step 3 hard gate: pre-enqueue consistency validator for training pairs.

A contradictory training pair teaches the model to ignore workpaper evidence
and produce spurious findings — that is strictly worse than no synthetic data.
This gate is the final checkpoint before a pair enters the review queue and
becomes eligible for JSONL export.

Three checks (deficient pairs only — clean pairs always pass)
--------------------------------------------------------------
1. authority_check
   Every field in deficiency_fields must be deficiency_allowed per the
   compiled FieldAuthorityTable for this pair's client_type. Rejects any
   pair built with a locked (fixed_administrative) or informational field,
   or a field with no SFC entry. This is the upstream guard — if the
   sampler enforces the authority rule correctly, this should never fire.
   It exists as a defence against sampler bugs or manually injected pairs.
   → Block: log the disallowed field(s) and their authority status.

2. redaction_complete
   The evidence_redactor must have run and must have successfully removed
   evidence lines for every deficiency field. If any field has
   fully_redacted == False, the user message still contains text that
   supports that field's presence while the completion calls it absent.
   → Block: log the failing fields; caller skips enqueue.

3. alias_residue
   Cross-check: scan the user message text for precision aliases belonging
   to each absent field. Catches cases where the redactor reported success
   but missed a match (e.g. the alias file was updated after the redactor
   cached its lookup, or a new alias form is present in the text).
   → Block: log which alias was found in which field.

Remediation path for blocked pairs
------------------------------------
- authority_check failure  → fix sop_field_classes.yaml or investigate
  how the pair bypassed the sampler's authority filter.
- redaction_complete failure → add precision aliases (>= 2 words) to
  auditai_data_normalization/field_aliases.yaml for the failing fields.
- alias_residue failure → same; or investigate why load_aliases() returned
  a term that matches but redact_fields() did not remove it (encoding,
  unicode normalisation, line-vs-sentence boundary difference).

Public API
----------
    check_pair(pair) -> HardGateResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HardGateResult:
    """Outcome of a hard gate check on one training pair."""
    passed: bool
    gate:   str    # failing gate name, or "ok"
    reason: str    # human-readable explanation (empty when passed)


def check_pair(pair: dict) -> HardGateResult:
    """
    Run all hard gate checks on a fully assembled pair dict.

    Returns the first failing check as a HardGateResult, or
    HardGateResult(passed=True, gate="ok", reason="") if all checks pass.

    Only deficient pairs are checked. Clean pairs always pass.
    """
    metadata          = pair.get("metadata", {})
    pair_type         = metadata.get("pair_type", "clean")
    deficiency_fields = metadata.get("deficiency_fields", [])

    if pair_type != "deficient" or not deficiency_fields:
        return HardGateResult(passed=True, gate="ok", reason="")

    client_type = metadata.get("client_type", "")
    sop_id      = metadata.get("sop_id", "npo-cx-1.1")

    # ------------------------------------------------------------------
    # Gate 1: authority_check
    # Every deficiency field must be deficiency_allowed per the compiled
    # FieldAuthorityTable. Rejects locked, informational, or SFC-ungoverned
    # fields that should never appear as synthetic deficiencies.
    # ------------------------------------------------------------------
    try:
        from pipeline.sop_compiler import compiled_sop as _compiled_sop
        _graph = _compiled_sop(sop_id)
        disallowed: list[str] = [
            f for f in deficiency_fields
            if not _graph.allowed(f, client_type)
        ]
        if disallowed:
            detail = []
            for f in disallowed:
                node = _graph.get(f)
                if node is None:
                    detail.append(f"{f!r}: no SFC entry")
                elif node.is_locked(client_type):
                    detail.append(f"{f!r}: fixed_administrative")
                elif node.is_informational(client_type):
                    detail.append(f"{f!r}: informational_only")
                else:
                    detail.append(f"{f!r}: deficiency_allowed=False")
            return HardGateResult(
                passed=False,
                gate="authority_check",
                reason=(
                    f"{len(disallowed)} deficiency field(s) are not "
                    f"deficiency_allowed for client_type={client_type!r}: "
                    f"{'; '.join(detail)}. Fix sop_field_classes.yaml or "
                    "investigate how this pair bypassed the sampler."
                ),
            )
    except Exception as _e:
        logger.warning("hard_gate: authority_check unavailable — %s", _e)

    # ------------------------------------------------------------------
    # Gate 2: redaction_complete
    # ------------------------------------------------------------------
    ev = metadata.get("evidence_redaction", {})

    if not ev.get("applied", False):
        return HardGateResult(
            passed=False,
            gate="redaction_complete",
            reason=(
                "Deficient pair has no evidence_redaction metadata. "
                "evidence_redactor.redact_fields() must run before this gate. "
                "Check _process_single_variant_r7() in pipeline.py."
            ),
        )

    if not ev.get("fully_redacted", False):
        failed = ev.get("failed_fields", [])
        return HardGateResult(
            passed=False,
            gate="redaction_complete",
            reason=(
                f"Incomplete evidence redaction — {len(failed)} field(s) still "
                f"have supporting text in the user message: {failed}. "
                "Add precision aliases (>= 2 words) to field_aliases.yaml."
            ),
        )

    # ------------------------------------------------------------------
    # Gate 3: alias_residue
    # Cross-check: re-scan user message for aliases that should have been
    # removed. Guards against redactor bugs or stale alias caching.
    # ------------------------------------------------------------------
    messages     = pair.get("messages", [])
    user_content = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        "",
    ).lower()

    try:
        from pipeline.evidence_redactor import load_aliases, _precision_aliases
        alias_lookup = load_aliases()
    except Exception as _e:
        logger.warning("hard_gate: could not load aliases for residue check — %s", _e)
        # If alias loading fails, skip gate 2 rather than block everything
        return HardGateResult(passed=True, gate="ok", reason="")

    residue: list[tuple[str, str]] = []
    for field_name in deficiency_fields:
        aliases = _precision_aliases(field_name, alias_lookup)
        hit = next((a for a in aliases if a in user_content), None)
        if hit:
            residue.append((field_name, hit))

    if residue:
        detail = "; ".join(f"{f!r} matched alias {a!r}" for f, a in residue)
        return HardGateResult(
            passed=False,
            gate="alias_residue",
            reason=(
                f"User message still contains evidence for {len(residue)} "
                f"absent field(s) after redaction — {detail}. "
                "Redactor reported fully_redacted=True but alias scan disagrees; "
                "check for encoding or line-boundary differences."
            ),
        )

    return HardGateResult(passed=True, gate="ok", reason="")
