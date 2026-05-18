"""
auditai_data_normalization/pii.py
==================================
PII scrubber. Must be called before any extractor output enters
sections, cleaned_text, or any downstream consumer.

Two-tier design
---------------
Tier 1 (preferred) — Presidio + spaCy en_core_web_lg
    Full NER: catches PERSON, ORGANIZATION, LOCATION, DATE_TIME,
    EMAIL_ADDRESS, PHONE_NUMBER, URL in addition to regex patterns.
    Requires: python -m spacy download en_core_web_lg

Tier 2 (automatic fallback) — pure regex
    Runs when spaCy model is not installed (e.g. CI, fresh machine).
    Catches EIN, SSN, ITIN, ACCOUNT_NUM, ROUTING_NUM, CLIENT_ENTITY
    via the patterns in config/pii_patterns.yaml.
    Less coverage than Tier 1 but zero external dependencies.

On your machine, always use Tier 1.  Run once:
    python -m spacy download en_core_web_lg

Public API
----------
    scrub(text: str) -> ScrubResult
        The single entry point. Detects tier automatically.
        Returns cleaned text + redaction log.

    scrub_record(record: DocumentRecord) -> DocumentRecord
        Convenience wrapper — scrubs every section and cleaned_text
        on a DocumentRecord in-place and sets pii_scrubbed=True.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from auditai_data_normalization.schema import DocumentRecord, PIIRedaction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load PII patterns from config/pii_patterns.yaml
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "pii_patterns.yaml"


def _load_patterns() -> dict:
    """Load pii_patterns.yaml. Returns empty dict if file not found."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    logger.warning(
        "pii_patterns.yaml not found at %s — using built-in defaults", _CONFIG_PATH
    )
    return {}


_RAW_CONFIG = _load_patterns()

# Built-in US patterns — always active regardless of yaml
_BUILTIN_PATTERNS: list[dict] = [
    {
        "name": "EIN",
        "pattern": r"\b\d{2}-\d{7}\b",
        "replacement": "[EIN]",
        "require_context": False,
        "context_keywords": [],
    },
    {
        "name": "SSN",
        "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        "replacement": "[SSN]",
        "require_context": False,
        "context_keywords": [],
    },
    {
        "name": "ITIN",
        "pattern": r"\b9\d{2}-\d{2}-\d{4}\b",
        "replacement": "[ITIN]",
        "require_context": False,
        "context_keywords": [],
    },
    {
        # Preparer initials — "Completed by: MS1", "Completed by: AS", etc.
        # Short initials (1-3 uppercase letters + optional digit) are too short
        # for spaCy NER PERSON detection. Pattern preserves the label and
        # replaces only the initials value using a backreference (\g<1>).
        # Handles Unicode spaces (\u2002 EN SPACE, \u200b ZWSP) from DOCX cells.
        "name": "PREPARER_INITIALS",
        "pattern": r"(Completed\s+by[\s:\u2002\u200b]+)([A-Z]{1,3}\d?)(?=\b|\s|$|\|)",
        "replacement": r"\g<1>[PREPARER]",
        "require_context": False,
        "context_keywords": [],
    },
    {
        "name": "CLIENT_ENTITY",
        "pattern": (
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
            r"(?:Inc\.|LLC|Corp\.|L\.P\.|LLP|County|City of|Township of"
            r"|School District|Housing Authority|Nonprofit|Foundation"
            r"|Institute|Project|Center|Program|STEP|Association|Society"
            r"|Trust|Authority|Agency|Bureau|Department|Coalition)\b"
        ),
        "replacement": "[CLIENT_ENTITY]",
        "require_context": False,
        "context_keywords": [],
    },
    {
        "name": "ACCOUNT_NUM",
        "pattern": r"\b\d{8,17}\b",
        "replacement": "[ACCOUNT_NUM]",
        "require_context": True,
        "context_keywords": [
            "account", "acct", "checking", "savings", "routing", "deposit",
        ],
    },
    {
        "name": "ROUTING_NUM",
        "pattern": r"\b\d{9}\b",
        "replacement": "[ROUTING_NUM]",
        "require_context": True,
        "context_keywords": ["routing", "aba", "transit", "wire"],
    },
]

# Presidio entity types to enable when spaCy NER is available
_PRESIDIO_ENTITIES = [
    "PERSON",
    "LOCATION",
    "DATE_TIME",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "URL",
]

# Map Presidio entity type → replacement token
_PRESIDIO_REPLACEMENTS = {
    "PERSON":           "[PERSON]",
    "LOCATION":         "[LOCATION]",
    "DATE_TIME":        "[DATE]",
    "EMAIL_ADDRESS":    "[EMAIL]",
    "PHONE_NUMBER":     "[PHONE]",
    "URL":              "[URL]",
    "EIN":              "[EIN]",
    "SSN":              "[SSN]",
    "ITIN":             "[ITIN]",
    "ACCOUNT_NUM":      "[ACCOUNT_NUM]",
    "ROUTING_NUM":      "[ROUTING_NUM]",
    "CLIENT_ENTITY":    "[CLIENT_ENTITY]",
    "PREPARER_INITIALS":"[PREPARER]",
}


# ---------------------------------------------------------------------------
# ScrubResult
# ---------------------------------------------------------------------------

@dataclass
class ScrubResult:
    """Output of scrub(). Contains cleaned text and a redaction log."""

    cleaned_text: str
    """Text with all detected PII replaced by tokens."""

    redactions: list[PIIRedaction] = field(default_factory=list)
    """One entry per PII type found, with count of replacements."""

    tier_used: str = "unknown"
    """'presidio_ner' or 'regex_only' — which tier ran."""

    def total_replacements(self) -> int:
        return sum(r.count for r in self.redactions)

    def types_found(self) -> list[str]:
        return [r.pii_type for r in self.redactions]


# ---------------------------------------------------------------------------
# Presidio engine (Tier 1) — lazy-loaded once
# ---------------------------------------------------------------------------

_presidio_analyzer = None
_presidio_anonymizer = None
_presidio_available = False


def _init_presidio() -> bool:
    """
    Try to initialise Presidio with spaCy en_core_web_lg.
    Returns True on success, False if spaCy model is not installed.
    Sets module-level globals so init runs only once.
    """
    global _presidio_analyzer, _presidio_anonymizer, _presidio_available

    if _presidio_analyzer is not None:
        return _presidio_available

    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

        # Register custom pattern recognizers on top of spaCy NER
        for p in _BUILTIN_PATTERNS:
            pat = Pattern(name=p["name"], regex=p["pattern"], score=0.85)
            recognizer = PatternRecognizer(
                supported_entity=p["name"], patterns=[pat]
            )
            analyzer.registry.add_recognizer(recognizer)

        _presidio_analyzer = analyzer
        _presidio_anonymizer = AnonymizerEngine()
        _presidio_available = True
        logger.info("Presidio initialised with spaCy en_core_web_lg (Tier 1)")
        return True

    except Exception as e:
        logger.warning(
            "spaCy model not available (%s). "
            "Falling back to regex-only PII scrubbing (Tier 2). "
            "Run: python -m spacy download en_core_web_lg",
            e,
        )
        _presidio_analyzer = False   # sentinel: attempted, failed
        _presidio_available = False
        return False


# ---------------------------------------------------------------------------
# Tier 1 — Presidio + spaCy NER
# ---------------------------------------------------------------------------

def _scrub_presidio(text: str) -> ScrubResult:
    """Run Presidio + spaCy NER over text. Returns ScrubResult."""
    from presidio_anonymizer.entities import OperatorConfig

    results = _presidio_analyzer.analyze(text=text, language="en",
                                         entities=_PRESIDIO_ENTITIES + [
                                             "EIN", "SSN", "ITIN",
                                             "ACCOUNT_NUM", "ROUTING_NUM",
                                             "CLIENT_ENTITY",
                                         ])

    if not results:
        # Even if Presidio finds nothing, still run regex for PREPARER_INITIALS
        # since short initials are below Presidio's NER confidence threshold
        regex_result = _scrub_regex(text)
        return ScrubResult(
            cleaned_text=regex_result.cleaned_text,
            redactions=regex_result.redactions,
            tier_used="presidio_ner",
        )

    # Build operator config: replace each entity type with its token
    operators = {
        entity: OperatorConfig(
            "replace",
            {"new_value": _PRESIDIO_REPLACEMENTS.get(entity, f"[{entity}]")},
        )
        for entity in set(r.entity_type for r in results)
    }

    anonymized = _presidio_anonymizer.anonymize(
        text=text, analyzer_results=results, operators=operators
    )

    # Count replacements per type
    counts: dict[str, int] = {}
    for r in results:
        counts[r.entity_type] = counts.get(r.entity_type, 0) + 1

    redactions = [
        PIIRedaction(
            pii_type=ptype,
            replacement=_PRESIDIO_REPLACEMENTS.get(ptype, f"[{ptype}]"),
            count=cnt,
            detection_method="presidio",
        )
        for ptype, cnt in counts.items()
    ]

    # Always run regex pass after Presidio to catch PREPARER_INITIALS
    # which Presidio misses due to short length
    regex_result = _scrub_regex(anonymized.text)
    final_text = regex_result.cleaned_text

    # Merge regex redactions into the result
    for r in regex_result.redactions:
        redactions.append(r)

    return ScrubResult(
        cleaned_text=final_text,
        redactions=redactions,
        tier_used="presidio_ner",
    )


# ---------------------------------------------------------------------------
# Tier 2 — pure regex (no external model required)
# ---------------------------------------------------------------------------

def _context_window(text: str, match: re.Match, window: int = 60) -> str:
    """Return text around a match for context-keyword checking."""
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    return text[start:end].lower()


def _scrub_regex(text: str) -> ScrubResult:
    """
    Apply all built-in regex patterns to text.

    Supports backreference replacements (e.g. \\g<1>) via m.expand().
    Patterns with require_context=True only replace when a context
    keyword appears within 60 characters of the match.
    """
    result = text
    counts: dict[str, int] = {}

    for p in _BUILTIN_PATTERNS:
        name = p["name"]
        replacement = p["replacement"]
        require_ctx = p["require_context"]
        ctx_keywords = p["context_keywords"]
        compiled = re.compile(p["pattern"])

        if not require_ctx:
            # Use m.expand() to support backreference replacements (\g<1>)
            # Plain subn() treats replacement as a literal string for \g<1>
            _count_holder = [0]

            def _replacer(m, _repl=replacement, _counter=_count_holder):
                _counter[0] += 1
                return m.expand(_repl)

            new = compiled.sub(_replacer, result)
            n = _count_holder[0]
            if n:
                result = new
                counts[name] = counts.get(name, 0) + n
        else:
            # Replace only when a context keyword is nearby
            def _replace_if_context(m: re.Match) -> str:
                window = _context_window(result, m)
                if any(kw in window for kw in ctx_keywords):
                    counts[name] = counts.get(name, 0) + 1
                    return replacement
                return m.group(0)

            result = compiled.sub(_replace_if_context, result)

    redactions = [
        PIIRedaction(
            pii_type=ptype,
            replacement=_PRESIDIO_REPLACEMENTS.get(ptype, f"[{ptype}]"),
            count=cnt,
            detection_method="regex",
        )
        for ptype, cnt in counts.items()
    ]

    return ScrubResult(
        cleaned_text=result,
        redactions=redactions,
        tier_used="regex_only",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrub(text: str) -> ScrubResult:
    """
    Scrub PII from text. Auto-selects Tier 1 (Presidio + spaCy) if
    available, falls back to Tier 2 (regex only) if not.

    Parameters
    ----------
    text : str
        Raw extracted text from any document type.

    Returns
    -------
    ScrubResult
        .cleaned_text  — text with PII tokens substituted
        .redactions    — list of PIIRedaction log entries
        .tier_used     — 'presidio_ner' or 'regex_only'

    Notes
    -----
    Always call this before storing or processing extracted text.
    The result's cleaned_text is what goes into DocumentRecord.sections
    and DocumentRecord.cleaned_text.
    """
    if not text or not text.strip():
        return ScrubResult(cleaned_text=text, tier_used="none")

    if _init_presidio():
        return _scrub_presidio(text)
    return _scrub_regex(text)


def scrub_record(record: DocumentRecord) -> DocumentRecord:
    """
    Scrub all text fields on a DocumentRecord in-place.

    Runs scrub() on:
      - record.cleaned_text
      - every section.content
      - every table's raw_text and cell values

    Sets record.pii_scrubbed = True on completion.
    Aggregates all redaction log entries into record.pii_redactions.

    Parameters
    ----------
    record : DocumentRecord
        A record that has been populated by an extractor.
        raw_text is intentionally NOT scrubbed — it is preserved
        as-is for SHA-256 dedup and MinHash similarity.

    Returns
    -------
    DocumentRecord
        The same record, modified in-place, with pii_scrubbed=True.
    """
    all_redactions: dict[str, PIIRedaction] = {}

    def _merge(redactions: list[PIIRedaction]) -> None:
        """Aggregate counts across multiple scrub calls."""
        for r in redactions:
            if r.pii_type in all_redactions:
                all_redactions[r.pii_type].count += r.count
            else:
                all_redactions[r.pii_type] = PIIRedaction(
                    pii_type=r.pii_type,
                    replacement=r.replacement,
                    count=r.count,
                    detection_method=r.detection_method,
                )

    # Scrub cleaned_text
    if record.cleaned_text:
        result = scrub(record.cleaned_text)
        record.cleaned_text = result.cleaned_text
        _merge(result.redactions)

    # Scrub each section's content
    for section in record.sections:
        if section.content:
            result = scrub(section.content)
            section.content = result.cleaned_text
            _merge(result.redactions)

    # Scrub table raw_text and cell values
    for table in record.tables:
        if table.raw_text:
            result = scrub(table.raw_text)
            table.raw_text = result.cleaned_text
            _merge(result.redactions)

        scrubbed_rows = []
        for row in table.rows:
            scrubbed_row = []
            for cell in row:
                if cell:
                    r = scrub(cell)
                    scrubbed_row.append(r.cleaned_text)
                    _merge(r.redactions)
                else:
                    scrubbed_row.append(cell)
            scrubbed_rows.append(scrubbed_row)
        table.rows = scrubbed_rows

    record.pii_redactions = list(all_redactions.values())
    record.pii_scrubbed = True

    if all_redactions:
        logger.info(
            "PII scrubbed from %s: %s",
            record.file_name,
            {k: v.count for k, v in all_redactions.items()},
        )

    return record