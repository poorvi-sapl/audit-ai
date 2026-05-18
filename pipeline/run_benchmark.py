#!/usr/bin/env python3
"""
pipeline/run_benchmark.py
==========================
Phase F1 — benchmark runner.

Runs all client workpapers through the full pipeline and records
extraction_confidence and review_confidence for each.

Targets (Phase F exit criteria):
    extraction_confidence >= 0.65 for all workpapers
    review_confidence     >= 0.75 for all workpapers

Usage:
    python pipeline/run_benchmark.py
    python pipeline/run_benchmark.py --data-dir data/ --mock
    python pipeline/run_benchmark.py --file data/NPO-CX-1_1.docx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.WARNING,  # suppress pipeline noise — summary only
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")

# Targets
_EXTRACTION_TARGET = 0.65
_REVIEW_TARGET     = 0.75


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WorkpaperResult:
    file_name:            str
    file_type:            str
    extraction_confidence: float
    review_confidence:    float
    extraction_gate:      bool
    quality_gate:         bool
    tier1_found:          int
    tier1_total:          int
    tier1_missing:        list[str]
    flagged_fields:       list[str]
    llm_assisted:         bool
    pairs_generated:      int
    errors:               list[str] = field(default_factory=list)

    @property
    def extraction_pass(self) -> bool:
        return self.extraction_confidence >= _EXTRACTION_TARGET

    @property
    def review_pass(self) -> bool:
        return self.review_confidence >= _REVIEW_TARGET

    @property
    def both_pass(self) -> bool:
        return self.extraction_pass and self.review_pass

    def row(self) -> str:
        ext_flag = "✓" if self.extraction_pass else "✗"
        rev_flag = "✓" if self.review_pass     else "✗"
        llm_tag  = " [LLM]" if self.llm_assisted else ""
        return (
            f"  {ext_flag} ext={self.extraction_confidence:.3f}  "
            f"{rev_flag} rev={self.review_confidence:.3f}  "
            f"t1={self.tier1_found}/{self.tier1_total}  "
            f"pairs={self.pairs_generated}{llm_tag}  "
            f"{self.file_name}"
        )


# ---------------------------------------------------------------------------
# Per-file runner
# ---------------------------------------------------------------------------

def _run_one(
    file_path: Path,
    queue_path: Path,
    data_dir: Path,
    client_type: str = "NPO",
    use_mock: bool = True,
) -> WorkpaperResult:
    errors: list[str] = []

    try:
        from pipeline.pipeline import process_workpaper
        result = process_workpaper(
            file_path=str(file_path),
            queue_path=queue_path,
            data_dir=data_dir,
            run_parallel=False,
            client_types=[client_type],
            use_mock=use_mock,
        )
    except Exception as e:
        return WorkpaperResult(
            file_name=file_path.name,
            file_type="unknown",
            extraction_confidence=0.0,
            review_confidence=0.0,
            extraction_gate=False,
            quality_gate=False,
            tier1_found=0,
            tier1_total=8,
            tier1_missing=[],
            flagged_fields=[],
            llm_assisted=False,
            pairs_generated=0,
            errors=[f"process_workpaper failed: {e}"],
        )

    if result.skipped:
        return WorkpaperResult(
            file_name=file_path.name,
            file_type="unknown",
            extraction_confidence=0.0,
            review_confidence=0.0,
            extraction_gate=False,
            quality_gate=False,
            tier1_found=0,
            tier1_total=8,
            tier1_missing=[],
            flagged_fields=[],
            llm_assisted=False,
            pairs_generated=0,
            errors=[f"skipped: {result.skip_reason}"],
        )

    record = result.record
    conf   = record.metadata.get("confidence_summary", {})

    return WorkpaperResult(
        file_name=record.file_name,
        file_type=record.file_type,
        extraction_confidence=record.extraction_confidence,
        review_confidence=record.review_confidence,
        extraction_gate=record.extraction_gate,
        quality_gate=record.quality_gate,
        tier1_found=conf.get("tier1_found", 0),
        tier1_total=conf.get("tier1_total", 8),
        tier1_missing=conf.get("tier1_missing", []),
        flagged_fields=record.flagged_fields or [],
        llm_assisted=record.llm_assisted,
        pairs_generated=result.pairs_queued,
        errors=([result.skip_reason] if result.skipped and result.skip_reason else []),
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    data_dir: Path,
    use_mock: bool = True,
    target_files: list[Path] | None = None,
    client_type: str = "NPO",
    report_path: Path | None = None,
) -> list[WorkpaperResult]:
    """
    Run all workpapers in data_dir through the pipeline and collect metrics.

    Parameters
    ----------
    data_dir : Path
    use_mock : bool      Use mock completions (fast, no Ollama needed)
    target_files : list  Specific files to run. Defaults to all supported in data_dir.
    client_type : str    Client type for pair generation.
    report_path : Path   If set, writes JSON report here.
    """
    _SUPPORTED = {".docx", ".pdf", ".xlsx", ".xls", ".csv", ".json"}
    _SKIP_PATTERNS = {"SOP", "review_queue", "suggested_aliases"}

    if target_files:
        files = target_files
    else:
        files = [
            f for f in sorted(data_dir.iterdir())
            if f.suffix.lower() in _SUPPORTED
            and not any(p in f.name for p in _SKIP_PATTERNS)
        ]

    if not files:
        print(f"No workpaper files found in {data_dir}")
        return []

    queue_path = data_dir / "benchmark_queue.jsonl"
    results: list[WorkpaperResult] = []

    print(f"\nBenchmark — {len(files)} workpaper(s) | mock={use_mock} | target: "
          f"ext>={_EXTRACTION_TARGET} rev>={_REVIEW_TARGET}\n")
    print("-" * 72)

    for f in files:
        print(f"  Running: {f.name} ...", end="", flush=True)
        r = _run_one(f, queue_path, data_dir, client_type, use_mock)
        results.append(r)
        print(f"\r{r.row()}")
        if r.errors:
            for e in r.errors:
                print(f"    ERROR: {e}")
        if r.tier1_missing:
            print(f"    tier1 missing: {r.tier1_missing}")
        if r.flagged_fields:
            print(f"    flagged:       {r.flagged_fields}")

    # ── Summary ───────────────────────────────────────────────────────
    passed     = [r for r in results if r.both_pass]
    ext_failed = [r for r in results if not r.extraction_pass]
    rev_failed = [r for r in results if not r.review_pass]

    print("-" * 72)
    print(f"\nSUMMARY  {len(passed)}/{len(results)} workpapers meet both targets\n")

    if ext_failed:
        print(f"  ✗ Extraction target failed ({len(ext_failed)}):")
        for r in ext_failed:
            print(f"    {r.file_name} → {r.extraction_confidence:.3f} "
                  f"(tier1={r.tier1_found}/8 missing={r.tier1_missing})")

    if rev_failed:
        print(f"  ✗ Review target failed ({len(rev_failed)}):")
        for r in rev_failed:
            print(f"    {r.file_name} → {r.review_confidence:.3f}")

    if not ext_failed and not rev_failed:
        print("  ✓ All workpapers meet both targets — Phase F complete")

    # ── JSON report ───────────────────────────────────────────────────
    if report_path:
        report = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "targets": {
                "extraction": _EXTRACTION_TARGET,
                "review":     _REVIEW_TARGET,
            },
            "summary": {
                "total":          len(results),
                "both_pass":      len(passed),
                "extraction_fail": len(ext_failed),
                "review_fail":    len(rev_failed),
            },
            "results": [asdict(r) for r in results],
        }
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\n  Report written to {report_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuditAI Phase F benchmark runner")
    parser.add_argument("--data-dir",   default="data",  help="Directory with workpaper files")
    parser.add_argument("--mock",       action="store_true", default=True,
                        help="Use mock completions (default True)")
    parser.add_argument("--real",       action="store_true", help="Use real Ollama completions")
    parser.add_argument("--client",     default="NPO", help="Client type (default NPO)")
    parser.add_argument("--file",       help="Run a single file instead of all")
    parser.add_argument("--report",     help="Write JSON report to this path")
    args = parser.parse_args()

    use_mock   = not args.real
    data_dir   = Path(args.data_dir)
    target     = [Path(args.file)] if args.file else None
    report     = Path(args.report) if args.report else None

    results = run_benchmark(
        data_dir=data_dir,
        use_mock=use_mock,
        target_files=target,
        client_type=args.client,
        report_path=report,
    )

    # Exit 1 if any workpaper failed targets
    failed = [r for r in results if not r.both_pass and not r.errors]
    sys.exit(1 if failed else 0)