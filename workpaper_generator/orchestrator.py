"""
workpaper_generator/orchestrator.py
=====================================
Glue layer between the PDF section detector, the rule engine, and the
.docx renderer (Phase 5). One engagement in, one structured run-result out.

A run consists of:
  1. Auto-discover the PY audit PDF and the workpaper template .docx
     inside the engagement directory.
  2. Detect named sections in the PDF (drives Q1c, Q1d, Q2 reference).
  3. Auto-fill auditor inputs from detector hints where unambiguous
     (e.g. organization_name).
  4. Run the rule engine over the manifest to resolve every field.
  5. Emit a JSON trace under data/workpaper_runs/<engagement>/ for audit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from workpaper_generator.pdf_section_detector import PDFDetectionReport, detect
from workpaper_generator.rule_engine import (
    ENGAGEMENT_INITIAL,
    ENGAGEMENT_RECURRING,
    ResolvedField,
    resolve_workpaper,
    summarize,
)

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "data" / "workpaper_runs"
_ENGAGEMENTS_ROOT = _PROJECT_ROOT / "Engagement Accept and Cont Form"
CANONICAL_BLANK_TEMPLATE = _PROJECT_ROOT / "config" / "templates" / "npo_cx_1_1_blank.docx"


@dataclass
class EngagementResult:
    engagement_name: str
    engagement_dir: str
    pdf_path: str
    workpaper_template_path: str
    engagement_type: str
    detection: PDFDetectionReport
    resolved: dict[str, ResolvedField]
    output_dir: str
    auditor_inputs_used: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, int]:
        return summarize(self.resolved)


def _discover_pdf(engagement_dir: Path) -> Path:
    """
    Locate the client's PY audit report PDF in an engagement folder.
    Any .docx files present are intentionally ignored — the renderer
    always uses CANONICAL_BLANK_TEMPLATE so output is deterministic
    regardless of historical workpapers in the folder.
    """
    pdfs = sorted(engagement_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {engagement_dir}")
    if len(pdfs) > 1:
        raise ValueError(f"Multiple PDFs found in {engagement_dir}: {[p.name for p in pdfs]}")
    return pdfs[0]


def _load_inputs_file(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported inputs file format: {path.suffix}")


def _merge_hints(
    detection: PDFDetectionReport,
    explicit_inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Layer detector hints UNDER explicit auditor inputs. Explicit inputs
    always win — hints only fill gaps the auditor didn't address.

    Only `organization_name` is auto-filled from hints. `prior_year_fye_date`
    is not, because the workpaper field `financial_position_date` refers to
    the CURRENT engagement, not the PY audit period.
    """
    merged = {}
    org_hint = detection.header_hints.get("organization_name")
    if org_hint:
        merged["organization_name"] = org_hint
    merged.update(explicit_inputs)
    return merged


def _serialize_for_trace(result: EngagementResult) -> dict:
    return {
        "engagement_name": result.engagement_name,
        "engagement_type": result.engagement_type,
        "pdf_path": result.pdf_path,
        "workpaper_template_path": result.workpaper_template_path,
        "detection": {
            "page_count": result.detection.page_count,
            "extraction_quality": result.detection.extraction_quality,
            "sections": {k: asdict(v) for k, v in result.detection.sections.items()},
            "header_hints": result.detection.header_hints,
        },
        "summary": result.summary,
        "fields": {fid: asdict(rf) for fid, rf in result.resolved.items()},
        "auditor_inputs_used": result.auditor_inputs_used,
    }


def run_engagement(
    engagement_dir: str | Path,
    engagement_type: str = ENGAGEMENT_RECURRING,
    auditor_inputs: Optional[dict[str, Any]] = None,
    auditor_inputs_path: Optional[str | Path] = None,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    write_trace: bool = True,
) -> EngagementResult:
    engagement_dir = Path(engagement_dir)
    if not engagement_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {engagement_dir}")

    engagement_name = engagement_dir.name
    pdf_path = _discover_pdf(engagement_dir)
    if not CANONICAL_BLANK_TEMPLATE.exists():
        raise FileNotFoundError(
            f"Canonical blank template missing: {CANONICAL_BLANK_TEMPLATE}. "
            "Regenerate it before running the pipeline."
        )
    template_path = CANONICAL_BLANK_TEMPLATE

    file_inputs = _load_inputs_file(Path(auditor_inputs_path) if auditor_inputs_path else None)
    explicit = {**file_inputs, **(auditor_inputs or {})}

    detection = detect(pdf_path)
    merged_inputs = _merge_hints(detection, explicit)

    resolved = resolve_workpaper(
        pdf_path=pdf_path,
        engagement_type=engagement_type,
        auditor_inputs=merged_inputs,
    )

    output_dir = Path(output_root) / engagement_name
    output_dir.mkdir(parents=True, exist_ok=True)

    result = EngagementResult(
        engagement_name=engagement_name,
        engagement_dir=str(engagement_dir),
        pdf_path=str(pdf_path),
        workpaper_template_path=str(template_path),
        engagement_type=engagement_type,
        detection=detection,
        resolved=resolved,
        output_dir=str(output_dir),
        auditor_inputs_used=merged_inputs,
    )

    if write_trace:
        trace_path = output_dir / "trace.json"
        trace_path.write_text(json.dumps(_serialize_for_trace(result), indent=2))

    return result


def register_new_client(
    client_name: str,
    pdf_bytes: bytes,
    pdf_filename: str,
    engagements_root: str | Path = _ENGAGEMENTS_ROOT,
) -> Path:
    """
    Create a new engagement folder under `engagements_root/<client_name>/`
    seeded with the uploaded PDF. Returns the new folder path.

    The renderer always uses CANONICAL_BLANK_TEMPLATE, so no template
    file is copied into the folder. The folder only needs to contain
    the client's PY audit report PDF.

    Idempotent: if the folder already exists, the PDF is overwritten.
    """
    client_name = client_name.strip()
    if not client_name:
        raise ValueError("client_name must be non-empty")
    if any(ch in client_name for ch in ('/', '\\', '..')):
        raise ValueError(f"client_name contains illegal path characters: {client_name!r}")

    engagements_root = Path(engagements_root)
    engagements_root.mkdir(parents=True, exist_ok=True)
    folder = engagements_root / client_name
    folder.mkdir(parents=True, exist_ok=True)

    pdf_target = folder / pdf_filename
    pdf_target.write_bytes(pdf_bytes)

    return folder


def run_all(
    parent_dir: str | Path,
    engagement_type: str = ENGAGEMENT_RECURRING,
    auditor_inputs_by_engagement: Optional[dict[str, dict[str, Any]]] = None,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
) -> list[EngagementResult]:
    """Run every subdirectory of `parent_dir` that contains a PDF + DOCX."""
    parent_dir = Path(parent_dir)
    inputs_map = auditor_inputs_by_engagement or {}
    results = []
    for sub in sorted(parent_dir.iterdir()):
        if not sub.is_dir():
            continue
        try:
            _discover_pdf(sub)
        except (FileNotFoundError, ValueError):
            continue
        results.append(
            run_engagement(
                engagement_dir=sub,
                engagement_type=engagement_type,
                auditor_inputs=inputs_map.get(sub.name),
                output_root=output_root,
            )
        )
    return results
