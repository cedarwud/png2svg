#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

REPO_ROOT = Path(__file__).resolve().parents[1]

app = typer.Typer(add_completion=False, help="Triage real regression failures.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _candidate_dir(case_dir: Path, template_id: str | None) -> Path | None:
    candidates_root = case_dir / "candidates"
    if template_id:
        candidate = candidates_root / template_id
        if candidate.exists():
            return candidate
    if not candidates_root.exists():
        return None
    candidates = sorted([path for path in candidates_root.iterdir() if path.is_dir()])
    return candidates[0] if candidates else None


@app.command()
def main(
    root: Path = typer.Argument(
        REPO_ROOT / "output" / "regress_real",
        exists=False,
        help="Root directory of real regression outputs.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        dir_okay=False,
        help="Optional path to write JSON triage output.",
    ),
) -> None:
    if not root.exists():
        typer.echo(f"ERROR E1500_REAL_OUTPUT_MISSING: {root} not found.", err=True)
        typer.echo("HINT: Run real regression first to populate output/regress_real.", err=True)
        raise typer.Exit(code=1)

    failures: list[dict[str, Any]] = []
    failures_by_reason: dict[str, int] = {}
    for report_path in sorted(root.glob("*/report.json")):
        report = _read_json(report_path)
        if report.get("status") == "pass":
            continue
        case_dir = report_path.parent
        case_id = report.get("id") or case_dir.name
        reasons = report.get("failure_reasons") or ["UNKNOWN"]
        for reason in reasons:
            failures_by_reason[str(reason)] = failures_by_reason.get(str(reason), 0) + 1

        template_id = report.get("selected_template")
        candidate_dir = _candidate_dir(case_dir, template_id)

        def _path_if_exists(path: Path | None) -> str | None:
            if path is None or not path.exists():
                return None
            return str(path)

        debug_paths = {
            "report_json": str(report_path),
            "convert_report": _path_if_exists(case_dir / "convert_report.json"),
            "diff_png": _path_if_exists(case_dir / "diff.png"),
            "classification_overlay": _path_if_exists(case_dir / "classify" / "overlay.png"),
            "extract_overlay": _path_if_exists(candidate_dir / "extract" / "02_overlay.png")
            if candidate_dir
            else None,
            "ocr_json": _path_if_exists(candidate_dir / "extract" / "03_ocr_raw.json")
            if candidate_dir
            else None,
            "gate_report": _path_if_exists(candidate_dir / "gate_report.json")
            if candidate_dir
            else None,
        }

        failures.append(
            {
                "id": case_id,
                "reasons": reasons,
                "paths": debug_paths,
            }
        )

    payload = {
        "total_failures": len(failures),
        "failures_by_reason": failures_by_reason,
        "failures": failures,
    }
    output = json.dumps(payload, indent=2, sort_keys=True)
    if out is not None:
        out.write_text(output)
    typer.echo(output)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in {"-h", "--help"}:
        app(prog_name="triage_real_failures")
    else:
        app(prog_name="triage_real_failures")
