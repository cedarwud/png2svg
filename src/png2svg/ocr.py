from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


def has_tesseract() -> bool:
    return shutil.which("tesseract") is not None


def _run_tesseract(
    image: Image.Image,
    lang: str = "eng",
    psm: int = 6,
    config: str | None = None,
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "input.png"
        image.save(input_path)
        cmd = [
            "tesseract",
            str(input_path),
            "stdout",
            "--psm",
            str(psm),
            "-l",
            lang,
            "tsv",
        ]
        if config:
            cmd.extend(config.split())
        result = subprocess.run(  # noqa: S603
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().splitlines()
        if not lines:
            return []
        header = lines[0].split("\t")
        rows: list[dict[str, Any]] = []
        for raw in lines[1:]:
            parts = raw.split("\t")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            text = row.get("text", "").strip()
            if not text:
                continue
            try:
                conf = float(row.get("conf", "-1"))
            except ValueError:
                conf = -1.0
            if conf < 0:
                continue
            try:
                left = int(row.get("left", "0"))
                top = int(row.get("top", "0"))
                width = int(row.get("width", "0"))
                height = int(row.get("height", "0"))
            except ValueError:
                continue
            rows.append(
                {
                    "text": text,
                    "conf": conf / 100.0,
                    "bbox": {
                        "x": left,
                        "y": top,
                        "width": width,
                        "height": height,
                    },
                }
            )
        rows.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"], item["text"]))
        return rows


def ocr_image(
    image: Image.Image,
    backend: str = "auto",
    rois: list[dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    backend_value = backend.lower()
    if backend_value == "auto":
        backend_value = "tesseract" if has_tesseract() else "none"
    if backend_value == "none":
        return []
    if backend_value != "tesseract":
        raise ValueError(f"Unsupported OCR backend: {backend}")

    results: list[dict[str, Any]] = []
    if not rois:
        return _run_tesseract(image)
    for roi in rois:
        try:
            x = int(roi["x"])
            y = int(roi["y"])
            width = int(roi["width"])
            height = int(roi["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        crop = image.crop((x, y, x + width, y + height))
        roi_results = _run_tesseract(crop)
        for item in roi_results:
            bbox = item.get("bbox", {})
            try:
                bx = int(bbox.get("x", 0))
                by = int(bbox.get("y", 0))
                bw = int(bbox.get("width", 0))
                bh = int(bbox.get("height", 0))
            except (TypeError, ValueError):
                continue
            item["bbox"] = {
                "x": bx + x,
                "y": by + y,
                "width": bw,
                "height": bh,
            }
            results.append(item)
    results.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"], item["text"]))
    return results


def write_ocr_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
