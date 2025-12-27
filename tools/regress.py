#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import typer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from png2svg.regress_manifest import _load_real_manifest  # noqa: E402
from png2svg.regress_runner import _print_summary, _run_dataset, _run_real_case  # noqa: E402
from png2svg.regress_utils import _has_rasterizer, _write_real_summary  # noqa: E402

app = typer.Typer(add_completion=False, help="Run regression cases.")


@app.command()
def main(
    dataset: Path = typer.Argument(
        REPO_ROOT / "datasets" / "regression_v0",
        exists=True,
        readable=True,
        help="Dataset directory or manifest.yaml path.",
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        dir_okay=False,
        help="Optional path to write the JSON report.",
    ),
    contract: Path = typer.Option(
        REPO_ROOT / "config" / "figure_contract.v1.yaml",
        "--contract",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the figure contract YAML.",
    ),
    thresholds: Path = typer.Option(
        REPO_ROOT / "config" / "validator_thresholds.v1.yaml",
        "--thresholds",
        "-t",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the validator thresholds YAML.",
    ),
    pipeline: str = typer.Option(
        "render",
        "--pipeline",
        help="Pipeline to run: render or convert.",
    ),
    tier: str = typer.Option(
        "fast",
        "--tier",
        help="Regression tier: fast, hard, or all.",
    ),
    input_variant: str = typer.Option(
        "fast",
        "--input-variant",
        help="Input variant: fast (input.png) or hard (input_hard.png).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Optional limit on number of cases to run.",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Only run a specific case id.",
    ),
    real_manifest: Path | None = typer.Option(
        REPO_ROOT / "datasets" / "real_regression_v1" / "manifest.yaml",
        "--real-manifest",
        dir_okay=False,
        help="Optional manifest for real PNG regression (uses REAL_PNG_DIR).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output (alias; no behavioral change).",
    ),
) -> None:
    """Run regression cases listed in manifest.yaml."""
    dataset = dataset.resolve()

    if pipeline not in {"render", "convert"}:
        typer.echo("ERROR E1302_PIPELINE_INVALID: pipeline must be 'render' or 'convert'.", err=True)
        typer.echo("HINT: Use --pipeline render (default) or --pipeline convert.", err=True)
        raise typer.Exit(code=1)
    if tier not in {"fast", "hard", "all"}:
        typer.echo("ERROR E1304_TIER_INVALID: tier must be fast, hard, or all.", err=True)
        typer.echo("HINT: Use --tier fast, --tier hard, or --tier all.", err=True)
        raise typer.Exit(code=1)
    if input_variant not in {"fast", "hard"}:
        typer.echo("ERROR E1303_INPUT_VARIANT_INVALID: input variant must be fast or hard.", err=True)
        typer.echo("HINT: Use --input-variant fast or --input-variant hard.", err=True)
        raise typer.Exit(code=1)

    output_root = REPO_ROOT / "output"
    fast_dataset = dataset
    hard_dataset = REPO_ROOT / "datasets" / "regression_hard_v1"
    if tier == "hard":
        hard_dataset = dataset

    tier_payload: dict[str, Any] = {}
    failed = 0

    if tier in {"fast", "all"}:
        fast_results = _run_dataset(
            fast_dataset,
            contract,
            thresholds,
            output_root / "regress",
            pipeline,
            input_variant,
            limit,
            only,
            "fast",
        )
        tier_payload["fast"] = {
            "dataset": str(fast_dataset),
            "summary": fast_results["summary"],
            "cases": fast_results["cases"],
            "input_variant": input_variant,
        }
        failed += int(fast_results["failed"])
        _print_summary(
            "fast",
            fast_results["summary"],
            fast_results["template_counts"],
            fast_results["tag_counts"],
            input_variant,
            pipeline,
        )

    if tier in {"hard", "all"}:
        if tier == "all" and input_variant != "fast":
            typer.echo("Hard regression: using input.png (fast) for hard tier.")
        hard_input_variant = "fast" if tier == "all" else input_variant
        hard_manifest = hard_dataset if hard_dataset.is_file() else hard_dataset / "manifest.yaml"
        if not hard_manifest.exists():
            typer.echo("Hard regression: skipped (manifest missing).")
            tier_payload["hard"] = {
                "dataset": str(hard_dataset),
                "summary": {"skipped": True, "reason": "manifest missing"},
                "cases": [],
                "input_variant": hard_input_variant,
            }
        elif not _has_rasterizer():
            typer.echo("Hard regression: skipped (no rasterizer available).")
            tier_payload["hard"] = {
                "dataset": str(hard_dataset),
                "summary": {"skipped": True, "reason": "no rasterizer"},
                "cases": [],
                "input_variant": hard_input_variant,
            }
        else:
            hard_results = _run_dataset(
                hard_dataset,
                contract,
                thresholds,
                output_root / "regress_hard",
                pipeline,
                hard_input_variant,
                limit,
                only,
                "hard",
            )
            tier_payload["hard"] = {
                "dataset": str(hard_dataset),
                "summary": hard_results["summary"],
                "cases": hard_results["cases"],
                "input_variant": hard_input_variant,
            }
            failed += int(hard_results["failed"])
            _print_summary(
                "hard",
                hard_results["summary"],
                hard_results["template_counts"],
                hard_results["tag_counts"],
                hard_input_variant,
                pipeline,
            )

    payload_dict: dict[str, Any] = {
        "tier": tier,
        "pipeline": pipeline,
        "input_variant": input_variant,
    }
    if tier == "all":
        payload_dict.update(tier_payload)
    elif tier == "fast":
        payload_dict.update(tier_payload.get("fast", {}))
    else:
        payload_dict.update(tier_payload.get("hard", {}))

    real_results: list[dict[str, Any]] = []
    real_summary: dict[str, Any] | None = None
    real_failed = 0
    if real_manifest is not None and real_manifest.exists():
        real_root_value = os.environ.get("REAL_PNG_DIR")
        if not real_root_value:
            typer.echo("Real regression: skipped (no REAL_PNG_DIR).")
            real_summary = {"skipped": True, "reason": "no REAL_PNG_DIR"}
        else:
            real_root = Path(real_root_value).expanduser()
            typer.echo(f"Real regression: using REAL_PNG_DIR={real_root}")
            try:
                real_manifest_data = _load_real_manifest(real_manifest)
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"ERROR E1399_REAL_MANIFEST_INVALID: {exc}", err=True)
                typer.echo("HINT: Ensure real manifest has a cases list.", err=True)
                raise typer.Exit(code=1)
            for entry in real_manifest_data["cases"]:
                if not isinstance(entry, dict):
                    continue
                case_id = str(entry.get("id") or "unknown")
                case_result = _run_real_case(
                    entry,
                    real_root,
                    contract,
                    thresholds,
                    REPO_ROOT / "output" / "regress_real",
                )
                case_result["id"] = case_id
                real_results.append(case_result)
                if case_result["status"] != "pass":
                    real_failed += 1
                status_line = f"{case_id}: {case_result['status']} (report: {case_result['report_path']})"
                if case_result.get("diff_path"):
                    status_line += f" diff: {case_result['diff_path']}"
                typer.echo(status_line)
            real_summary = {
                "total": len(real_results),
                "passed": len(real_results) - real_failed,
                "failed": real_failed,
            }
        payload_dict["real_summary"] = real_summary
        payload_dict["real_cases"] = real_results
    elif real_manifest is not None:
        typer.echo("Real regression: skipped (manifest missing).")

    if real_summary is not None:
        _write_real_summary(REPO_ROOT / "output" / "regress_real", real_results, real_summary)

    payload = json.dumps(payload_dict, indent=2, sort_keys=True)
    if report is not None:
        report.write_text(payload)
    typer.echo(payload)
    if tier == "all":
        combined_total = 0
        combined_passed = 0
        combined_failed = 0
        for label in ("fast", "hard"):
            entry = tier_payload.get(label, {})
            summary = entry.get("summary", {})
            if summary.get("skipped"):
                continue
            combined_total += int(summary.get("total", 0))
            combined_passed += int(summary.get("passed", 0))
            combined_failed += int(summary.get("failed", 0))
        typer.echo(
            f"Combined summary: total={combined_total} passed={combined_passed} failed={combined_failed} "
            f"pipeline={pipeline}"
        )
    if real_summary is not None and real_summary.get("failed"):
        failed += int(real_summary["failed"])
    raise typer.Exit(code=0 if failed == 0 else 1)


if __name__ == "__main__":
    app(prog_name="regress")
