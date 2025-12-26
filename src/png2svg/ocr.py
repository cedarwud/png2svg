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


def has_pytesseract() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return has_tesseract()


def _pytesseract_module():
    try:
        import pytesseract
    except ImportError:
        return None
    return pytesseract


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


def _run_pytesseract(
    image: Image.Image,
    lang: str = "eng",
    psm: int = 6,
    config: str | None = None,
) -> list[dict[str, Any]]:
    pytesseract = _pytesseract_module()
    if pytesseract is None:
        return []
    if not has_tesseract():
        return []
    config_args = f"--psm {psm}"
    if config:
        config_args = f"{config_args} {config}"
    try:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=config_args,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    texts = data.get("text", [])
    for idx, text in enumerate(texts):
        text = str(text).strip()
        if not text:
            continue
        try:
            conf = float(data.get("conf", [])[idx])
        except (IndexError, ValueError, TypeError):
            conf = -1.0
        if conf < 0:
            continue
        try:
            left = int(data.get("left", [])[idx])
            top = int(data.get("top", [])[idx])
            width = int(data.get("width", [])[idx])
            height = int(data.get("height", [])[idx])
        except (IndexError, ValueError, TypeError):
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
        backend_value = "pytesseract" if has_pytesseract() else "tesseract"
        if backend_value == "tesseract" and not has_tesseract():
            backend_value = "none"
    if backend_value == "none":
        return []
    if backend_value == "pytesseract" and not has_pytesseract():
        raise ValueError("pytesseract requested but not available")
    if backend_value not in {"tesseract", "pytesseract"}:
        raise ValueError(f"Unsupported OCR backend: {backend}")

    results: list[dict[str, Any]] = []
    if not rois:
        if backend_value == "pytesseract":
            return _run_pytesseract(image)
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
        if backend_value == "pytesseract":
            roi_results = _run_pytesseract(crop)
        else:
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
