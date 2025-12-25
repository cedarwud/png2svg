#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from png2svg import Png2SvgError, render_svg  # noqa: E402
from validators import validate_svg  # noqa: E402

app = typer.Typer(add_completion=False, help="Run regression cases.")


def _load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("manifest.yaml must contain a mapping.")
    if "cases" not in data or not isinstance(data["cases"], list):
        raise ValueError("manifest.yaml must contain a list under 'cases'.")
    return data


def _resolve_manifest(dataset: Path) -> tuple[Path, Path | None]:
    if dataset.is_file():
        return dataset, dataset.parent
    manifest = dataset / "manifest.yaml"
    if manifest.exists():
        return manifest, dataset
    return dataset, None


def _case_entry_dir(base_dir: Path, entry: Any) -> Path:
    if isinstance(entry, str):
        return base_dir / entry
    if isinstance(entry, dict) and "dir" in entry:
        return base_dir / str(entry["dir"])
    raise ValueError("Each case entry must be a string or have a 'dir' field.")


def _case_entry_id(entry: Any, case_dir: Path) -> str:
    if isinstance(entry, dict) and "id" in entry:
        return str(entry["id"])
    return case_dir.name


def _run_case(
    case_dir: Path,
    case_id: str,
    contract: Path,
    thresholds: Path,
    output_root: Path,
) -> dict[str, Any]:
    input_png = case_dir / "input.png"
    params = case_dir / "params.json"
    expected_png = case_dir / "expected.png"
    output_dir = output_root / case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_svg = output_dir / "generated.svg"
    output_png = output_dir / "generated.png"
    report_path = output_dir / "report.json"
    diff_path = output_dir / "diff.png"

    def _write_report(payload: dict[str, Any]) -> None:
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if not input_png.exists():
        payload = {
            "status": "fail",
            "error": {
                "code": "E1200_INPUT_MISSING",
                "message": "input.png not found.",
                "hint": "Place input.png in the case directory.",
            },
        }
        _write_report(payload)
        return {
            "status": "fail",
            "report_path": str(report_path),
            "generated_svg": str(output_svg),
            "generated_png": str(output_png),
            "diff_path": None,
            "report": payload,
        }
    if not params.exists():
        payload = {
            "status": "fail",
            "error": {
                "code": "E1201_PARAMS_MISSING",
                "message": "params.json not found.",
                "hint": "Place params.json in the case directory.",
            },
        }
        _write_report(payload)
        return {
            "status": "fail",
            "report_path": str(report_path),
            "generated_svg": str(output_svg),
            "generated_png": str(output_png),
            "diff_path": None,
            "report": payload,
        }
    if not expected_png.exists():
        payload = {
            "status": "fail",
            "error": {
                "code": "E1202_EXPECTED_MISSING",
                "message": "expected.png not found.",
                "hint": "Place expected.png in the case directory.",
            },
        }
        _write_report(payload)
        return {
            "status": "fail",
            "report_path": str(report_path),
            "generated_svg": str(output_svg),
            "generated_png": str(output_png),
            "diff_path": None,
            "report": payload,
        }

    try:
        render_svg(input_png, params, output_svg)
    except Png2SvgError as exc:
        payload = {
            "status": "fail",
            "error": {"code": exc.code, "message": exc.message, "hint": exc.hint},
        }
        _write_report(payload)
        return {
            "status": "fail",
            "report_path": str(report_path),
            "generated_svg": str(output_svg),
            "generated_png": str(output_png),
            "diff_path": None,
            "report": payload,
        }
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "fail",
            "error": {
                "code": "E1299_RENDER_FAILED",
                "message": f"{exc}",
                "hint": "Check input.png and params.json.",
            },
        }
        _write_report(payload)
        return {
            "status": "fail",
            "report_path": str(report_path),
            "generated_svg": str(output_svg),
            "generated_png": str(output_png),
            "diff_path": None,
            "report": payload,
        }

    result = validate_svg(
        output_svg,
        contract,
        thresholds,
        expected_png,
        actual_png_path=output_png,
        diff_png_path=diff_path,
    )
    report_payload = result.to_dict()
    _write_report(report_payload)
    diff_path_value = str(diff_path) if diff_path.exists() else None
    return {
        "status": result.status,
        "report_path": str(report_path),
        "generated_svg": str(output_svg),
        "generated_png": str(output_png),
        "diff_path": diff_path_value,
        "report": report_payload,
    }


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
) -> None:
    """Run regression cases listed in manifest.yaml."""
    dataset = dataset.resolve()
    manifest_path, base_dir = _resolve_manifest(dataset)
    if base_dir is None:
        case_dir = manifest_path
        if not case_dir.is_dir():
            typer.echo("ERROR E1300_MANIFEST_MISSING: manifest.yaml not found.", err=True)
            typer.echo("HINT: Provide a dataset directory with manifest.yaml.", err=True)
            raise typer.Exit(code=1)
        entries = [{"dir": case_dir.name, "id": case_dir.name}]
        base_dir = case_dir.parent
    else:
        try:
            manifest = _load_manifest(manifest_path)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"ERROR E1301_MANIFEST_INVALID: {exc}", err=True)
            typer.echo("HINT: Ensure manifest.yaml has a cases list.", err=True)
            raise typer.Exit(code=1)
        entries = manifest["cases"]

    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    for entry in entries:
        case_dir = _case_entry_dir(base_dir, entry)
        case_id = _case_entry_id(entry, case_dir)
        case_result = _run_case(case_dir, case_id, contract, thresholds, REPO_ROOT / "output" / "regress")
        case_result["id"] = case_id
        case_result["dir"] = str(case_dir)
        if case_result["status"] == "pass":
            passed += 1
        else:
            failed += 1
        results.append(case_result)
        status_line = f"{case_id}: {case_result['status']} (report: {case_result['report_path']})"
        if case_result["diff_path"]:
            status_line += f" diff: {case_result['diff_path']}"
        typer.echo(status_line)

    summary = {"total": len(results), "passed": passed, "failed": failed}
    payload = json.dumps({"summary": summary, "cases": results}, indent=2, sort_keys=True)
    if report is not None:
        report.write_text(payload)
    typer.echo(payload)
    raise typer.Exit(code=0 if failed == 0 else 1)


if __name__ == "__main__":
    app(prog_name="regress")
