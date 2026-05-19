"""
pipeline/derived_fields.py
==========================
Deterministic derivation rules for fields that are logical implications
of other fields rather than explicit form checkboxes.

Field taxonomy
--------------
  explicit  — direct checkbox or input on the form
              (includes_single_audit, audit_type, engagement_partner, ...)

  text      — extracted from keyword presence in document text
              (reporting_framework mentions, standalone GAGAS text, ...)

  derived   — deterministic legal/standards implication of an explicit field
              (includes_gagas ← includes_single_audit, ...)

Derived fields bypass LLM classification entirely. When a source field is
confirmed present by Phase 1 extraction, the derived field is set to
"present" with source tracking — it cannot become "uncertain" because there
is no ambiguity: the implication is definitional, not observational.

Priority order (highest wins)
------------------------------
1. Explicit checkbox from Phase 1 extraction
2. Structural derivation (this module)
3. Text keyword fallback (for observation / debug only, not truth)
4. LLM uncertainty (last resort — only for fields with no other signal)

Public API
----------
    apply_derivations(field_states) -> (updated_states, derived_map)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DerivationRule:
    source_field:  str
    derived_field: str
    authority:     str   # regulatory / standards justification for the implication


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------
#
# Each rule encodes: if source_field == "present" → derived_field = "present".
# Only "present" triggers are supported — absence of a source field does NOT
# imply absence of the derived field (there may be other sources we haven't
# modelled yet).

DERIVATION_RULES: tuple[DerivationRule, ...] = (
    DerivationRule(
        source_field  = "includes_single_audit",
        derived_field = "includes_gagas",
        authority     = (
            "2 CFR §200.514 — Single Audit engagements must be conducted under "
            "Government Auditing Standards (the Yellow Book). GAGAS applicability "
            "is a legal consequence of the Single Audit election, not a separate "
            "checkbox on this form."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_derivations(
    field_states: dict[str, str],
) -> tuple[dict[str, str], dict[str, dict]]:
    """
    Apply deterministic derivation rules to a field_states dict.

    Only fires when the source field is "present" and the derived field is
    not already "present" — this is a one-way upgrade, never a downgrade.

    Parameters
    ----------
    field_states : dict[str, str]
        Current field states (e.g. from Phase 1 reconciliation).

    Returns
    -------
    updated_states : dict[str, str]
        field_states with derived fields set to "present" (the effective value)
        where rules fire.  field_states is mutated at the effective layer only.
    derived_map : dict[str, dict]
        Maps derived_field → {"observed", "effective", "source", "authority"}.
        "observed" preserves the pre-derivation state for audit traceability;
        "effective" is the value the R7 pipeline uses.
        Empty dict if no rules fired.
    """
    derived_map: dict[str, dict] = {}

    for rule in DERIVATION_RULES:
        if rule.derived_field not in field_states:
            continue
        if field_states.get(rule.source_field) != "present":
            continue
        if field_states[rule.derived_field] == "present":
            continue

        observed = field_states[rule.derived_field]      # what the extractor/LLM saw
        derived_map[rule.derived_field] = {
            "observed":  observed,                       # "absent" / "uncertain" / etc.
            "effective": "present",                      # logically resolved value
            "source":    f"derived_from_{rule.source_field}",
            "authority": rule.authority,
        }
        logger.info(
            "derived_fields: %s  observed=%s → effective=present  "
            "rule=%s→%s  (%s)",
            rule.derived_field, observed,
            rule.source_field, rule.derived_field,
            rule.authority[:60],
        )

    # field_states is NEVER mutated — derivation is annotation only.
    # Call resolve_effective_states() to get the merged dict for R7 Pass 2.
    return field_states, derived_map


def resolve_effective_states(
    field_states: dict[str, str],
    derived_map: dict[str, dict],
) -> dict[str, str]:
    """
    Merge observed field_states with effective values from derived_map.

    This is the ONLY place where derived logic overrides observed extraction.
    It is called once, immediately before Pass 2, and nowhere else.

    field_states is not modified.
    """
    if not derived_map:
        return field_states
    effective = dict(field_states)
    for f, info in derived_map.items():
        effective[f] = info["effective"]
    return effective
