#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import typer
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

app = typer.Typer(add_completion=False, help="Summarize regression case coverage.")


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


def _hash_params(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@app.command()
def main(
    dataset: Path = typer.Argument(
        REPO_ROOT / "datasets" / "regression_v0",
        exists=True,
        readable=True,
        help="Dataset directory or manifest.yaml path.",
    )
) -> None:
    """Generate a coverage report for regression cases."""
    dataset = dataset.resolve()
    manifest_path, base_dir = _resolve_manifest(dataset)
    if base_dir is None:
        raise typer.BadParameter("manifest.yaml not found in dataset directory.")
    manifest = _load_manifest(manifest_path)
    entries = manifest["cases"]

    template_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    hash_to_cases: dict[str, list[str]] = {}
    warnings: list[str] = []

    for entry in entries:
        case_dir = _case_entry_dir(base_dir, entry)
        case_id = _case_entry_id(entry, case_dir)
        params_path = case_dir / "params.json"
        if not params_path.exists():
            warnings.append(f"{case_id}: params.json missing")
            continue
        try:
            params = json.loads(params_path.read_text())
        except json.JSONDecodeError:
            warnings.append(f"{case_id}: params.json invalid JSON")
            continue
        template = str(params.get("template") or "unknown")
        template_counts[template] = template_counts.get(template, 0) + 1

        if isinstance(entry, dict):
            tags = entry.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    tag_counts[str(tag)] = tag_counts.get(str(tag), 0) + 1
            elif tags:
                warnings.append(f"{case_id}: tags must be a list")
        params_hash = _hash_params(params)
        hash_to_cases.setdefault(params_hash, []).append(case_id)

    duplicate_groups = {h: ids for h, ids in hash_to_cases.items() if len(ids) > 1}

    typer.echo("Regression Case Coverage Report")
    typer.echo(f"Total cases: {len(entries)}")
    typer.echo("Templates:")
    for template, count in sorted(template_counts.items()):
        typer.echo(f"  {template}: {count}")
    typer.echo("Tags:")
    for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
        typer.echo(f"  {tag}: {count}")
    if duplicate_groups:
        typer.echo("Duplicate-risk params hashes:")
        for params_hash, case_ids in sorted(duplicate_groups.items()):
            short_hash = params_hash[:10]
            typer.echo(f"  {short_hash}: {', '.join(sorted(case_ids))}")
    else:
        typer.echo("Duplicate-risk params hashes: none")
    if warnings:
        typer.echo("Warnings:")
        for warning in warnings:
            typer.echo(f"  {warning}")


if __name__ == "__main__":
    app(prog_name="make_case_report")
