"""
config/settings.py
===================
Pydantic-validated configuration loader for the AuditAI pipeline.

Loads config/stack.yaml once at startup and validates all fields.
All phases import from here — never use yaml.safe_load() directly.

Usage
-----
    from config.settings import get_settings

    settings = get_settings()
    print(settings.embedding.model)
    print(settings.qdrant.collection_sop)
    print(settings.sop_chunker.chunk_size_tokens)

Design rules
------------
- Single source of truth: stack.yaml holds values, settings.py validates them
- Fail fast: missing or wrong-typed config raises ValidationError at import time
- Cached: get_settings() loads once, returns same instance on every call
- Read-only: all models use frozen=True — no runtime mutation of config
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

# Path to stack.yaml — relative to this file
_STACK_YAML = Path(__file__).parent / "stack.yaml"
_SOP_FIELD_CLASSES_YAML = Path(__file__).parent / "sop_field_classes.yaml"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class EmbeddingSettings(BaseModel, frozen=True):
    """Embedding model configuration."""

    model: str = Field(..., description="HuggingFace model ID")
    dimensions: int = Field(..., gt=0, description="Vector dimensions — immutable after first Qdrant collection is created")
    distance: Literal["Cosine", "Dot", "Euclid"] = Field(..., description="Qdrant distance metric")
    device: Literal["cuda", "cpu"] = Field(..., description="Compute device")
    embed_batch_size: int = Field(..., gt=0, description="Chunks per embed call")
    qdrant_upsert_batch: int = Field(..., gt=0, description="Points per Qdrant upsert")

    @field_validator("dimensions")
    @classmethod
    def dimensions_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"dimensions must be > 0, got {v}")
        return v


class QdrantPayloadIndex(BaseModel, frozen=True):
    """One payload index definition."""

    field: str
    type: Literal["keyword", "integer", "bool", "float", "geo"]


class QdrantSettings(BaseModel, frozen=True):
    """Qdrant collection configuration."""

    collection_workpapers: str = Field(..., description="Workpaper chunks collection name")
    collection_sop: str = Field(..., description="SOP chunks collection name")
    payload_indexes: list[QdrantPayloadIndex] = Field(
        default_factory=list,
        description="Payload indexes to create before first upsert",
    )


class OCRSettings(BaseModel, frozen=True):
    """OCR pipeline configuration."""

    primary: str = Field(..., description="Primary OCR engine e.g. 'docling+surya'")
    fallback: str = Field(..., description="Fallback OCR engine e.g. 'tesseract'")
    tesseract_lang: str = "eng"
    surya_confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    tesseract_confidence_threshold: int = Field(..., ge=0, le=100)
    dpi: int = Field(..., gt=0)
    page_window_size: int = Field(..., gt=0)
    ocr_timeout_seconds: int = Field(..., gt=0)
    tesseract_timeout_seconds: int = Field(..., gt=0)


class ExtractionLLMSettings(BaseModel, frozen=True):
    """LLM extractor configuration."""

    provider: str
    model: str
    temperature: float = Field(..., ge=0.0, le=1.0)
    format: Literal["json", "text"] = "json"


class TrainingTargetSettings(BaseModel, frozen=True):
    """Fine-tuning target model configuration."""

    base_model: str
    quantization: str
    instruction_format: str
    lora_rank: int = Field(..., gt=0)
    lora_alpha: int = Field(..., gt=0)
    max_seq_length: int = Field(..., gt=0)


class ChunkerSettings(BaseModel, frozen=True):
    """General document chunker configuration."""

    max_tokens: int = Field(..., gt=0)
    overlap_tokens: int = Field(..., ge=0)
    tokenizer: str
    pdf_page_split_threshold: int = Field(..., gt=0)
    ocr_page_split_threshold: int = Field(..., gt=0)


class SOPChunkerSettings(BaseModel, frozen=True):
    """SOP-specific chunker configuration (used by Phase 3)."""

    chunk_size_tokens: int = Field(..., gt=0)
    chunk_overlap_tokens: int = Field(..., ge=0)
    tokenizer: str
    prepend_section_prefix: bool = True

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size_tokens", 0)
        if chunk_size and v >= chunk_size:
            raise ValueError(
                f"chunk_overlap_tokens ({v}) must be less than "
                f"chunk_size_tokens ({chunk_size})"
            )
        return v


class DedupSettings(BaseModel, frozen=True):
    """Deduplication configuration."""

    minhash_threshold: float = Field(..., ge=0.0, le=1.0)
    exact_hash_algorithm: str = "sha256"
    redis_lock_ttl_seconds: int = Field(..., gt=0)


class TaskBalanceSettings(BaseModel, frozen=True):
    """Training task balance constraints."""

    max_fraction_per_task: float = Field(..., gt=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Root settings model
# ---------------------------------------------------------------------------

class Settings(BaseModel, frozen=True):
    """
    Full validated configuration for the AuditAI pipeline.

    All phases import from here via get_settings().
    """

    embedding: EmbeddingSettings
    qdrant: QdrantSettings
    ocr: OCRSettings
    extraction_llm: ExtractionLLMSettings
    training_target: TrainingTargetSettings
    chunker: ChunkerSettings
    sop_chunker: SOPChunkerSettings
    dedup: DedupSettings
    task_balance: TaskBalanceSettings


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load and validate config/stack.yaml. Cached after first call.

    Returns
    -------
    Settings
        Fully validated, frozen settings instance.

    Raises
    ------
    FileNotFoundError
        If stack.yaml does not exist.
    pydantic.ValidationError
        If any required field is missing or has wrong type.
        Fails fast at startup — never silently uses wrong config.
    """
    if not _STACK_YAML.exists():
        raise FileNotFoundError(
            f"stack.yaml not found at {_STACK_YAML}. "
            "This file is required for all pipeline phases."
        )

    with open(_STACK_YAML) as f:
        raw = yaml.safe_load(f) or {}

    return Settings(**raw)


def reload_settings() -> Settings:
    """
    Force reload of stack.yaml on next get_settings() call.
    Useful in tests when stack.yaml is modified at runtime.
    """
    get_settings.cache_clear()
    return get_settings()


# ---------------------------------------------------------------------------
# SOP field class loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SopFieldEntry:
    """
    Classification of one SOP question/section.

    slug              : Q-number key e.g. "q1a", "part_ii_q2j"
    field_class       : "deficiency_eligible" | "fixed_administrative" | "informational_only"
    deficiency_allowed: whether a finding may be generated for this field
    citation_allowed  : whether a SOP §ref to this section is valid in a completion
    canonical_field   : linked canonical field name from field_tiers.yaml, or None
    label             : human-readable label for UI / logging
    severity_default  : optional severity override for findings on this field
    client_type_overrides: {client_type: class} — per-client-type class overrides
    retrieval_weight  : optional float to boost/suppress during SOP retrieval
    sampling_weight   : optional float to override deficiency_sampler tier weight
    """
    slug:                  str
    field_class:           str
    deficiency_allowed:    bool
    citation_allowed:      bool
    canonical_field:       str | None
    label:                 str
    severity_default:      str | None
    client_type_overrides: dict[str, str]
    retrieval_weight:      float | None
    sampling_weight:       float | None

    def effective_class(self, client_type: str | None = None) -> str:
        """Return class for this client type, falling back to field_class."""
        if client_type and self.client_type_overrides:
            return self.client_type_overrides.get(client_type, self.field_class)
        return self.field_class

    def is_deficiency_eligible(self, client_type: str | None = None) -> bool:
        return self.effective_class(client_type) == "deficiency_eligible"

    def is_fixed_administrative(self, client_type: str | None = None) -> bool:
        return self.effective_class(client_type) == "fixed_administrative"


@dataclass
class SopFieldClasses:
    """
    Parsed sop_field_classes.yaml — all SOP field classifications.

    Primary lookup is by Q-number slug (e.g. "q11", "part_ii_q2j").
    For finding-guard purposes use by_canonical_field() to look up
    a canonical field name (e.g. "includes_gagas") across all entries.
    """
    entries:        dict[str, SopFieldEntry] = field(default_factory=dict)
    version:        str = "unversioned"
    effective_date: str = ""
    sop_document:   str = ""

    def get(self, slug: str) -> SopFieldEntry | None:
        return self.entries.get(slug)

    def by_canonical_field(self, canonical: str) -> list[SopFieldEntry]:
        """Return all entries that map to this canonical field name."""
        return [e for e in self.entries.values() if e.canonical_field == canonical]

    def non_deficient_canonical_fields(
        self, client_type: str | None = None
    ) -> frozenset[str]:
        """
        Return canonical field names that are NOT deficiency-eligible for
        this client type. Used by findings_extractor and deficiency_sampler
        as an exclusion set.
        """
        result = set()
        for entry in self.entries.values():
            if not entry.is_deficiency_eligible(client_type) and entry.canonical_field:
                result.add(entry.canonical_field)
        return frozenset(result)

    def fixed_admin_slugs(self, client_type: str | None = None) -> frozenset[str]:
        """All Q-number slugs classified as fixed_administrative."""
        return frozenset(
            slug for slug, entry in self.entries.items()
            if entry.is_fixed_administrative(client_type)
        )


@lru_cache(maxsize=1)
def load_sop_field_classes(
    yaml_path: str | None = None,
) -> SopFieldClasses:
    """
    Load and parse config/sop_field_classes.yaml. Cached after first call.

    Parameters
    ----------
    yaml_path : str | None
        Override path for testing. Default: config/sop_field_classes.yaml.

    Returns
    -------
    SopFieldClasses
        Parsed classifications. Returns empty SopFieldClasses if file not found,
        so callers never crash when the config is absent.
    """
    path = Path(yaml_path) if yaml_path else _SOP_FIELD_CLASSES_YAML

    if not path.exists():
        import warnings
        warnings.warn(
            f"sop_field_classes.yaml not found at {path}. "
            "Field class guards will not be enforced.",
            stacklevel=2,
        )
        return SopFieldClasses()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    entries: dict[str, SopFieldEntry] = {}
    for slug, data in (raw.get("fields") or {}).items():
        overrides = data.get("client_type_overrides") or {}
        entries[slug] = SopFieldEntry(
            slug=slug,
            field_class=data.get("class", "deficiency_eligible"),
            deficiency_allowed=bool(data.get("deficiency_allowed", True)),
            citation_allowed=bool(data.get("citation_allowed", True)),
            canonical_field=data.get("canonical_field") or None,
            label=data.get("label", slug),
            severity_default=data.get("severity_default") or None,
            client_type_overrides=dict(overrides),
            retrieval_weight=data.get("retrieval_weight"),
            sampling_weight=data.get("sampling_weight"),
        )

    sfc = SopFieldClasses(
        entries=entries,
        version=str(raw.get("version", "unversioned")),
        effective_date=str(raw.get("effective_date", "")),
        sop_document=str(raw.get("sop_document", "")),
    )

    # Write versioned snapshot so every onboarding is reproducible + diffable.
    _write_snapshot(sfc, path)

    return sfc


def _write_snapshot(sfc: "SopFieldClasses", source_path: Path) -> None:
    """
    Write a versioned snapshot of sop_field_classes.yaml to
    config/sop_field_classes_snapshots/v<version>.yaml if one does not
    already exist for this version. Snapshots are append-only — never
    overwritten — so diffs between versions are always reproducible.
    """
    import shutil
    snapshot_dir = source_path.parent / "sop_field_classes_snapshots"
    snapshot_path = snapshot_dir / f"v{sfc.version}.yaml"
    if snapshot_path.exists():
        return
    try:
        snapshot_dir.mkdir(exist_ok=True)
        shutil.copy2(source_path, snapshot_path)
    except Exception:
        pass  # snapshot failure must never block the pipeline


def reload_sop_field_classes() -> SopFieldClasses:
    """Force reload on next call. Useful in tests."""
    load_sop_field_classes.cache_clear()
    return load_sop_field_classes()