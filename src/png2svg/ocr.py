from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_OCR_MAX_DIM = 1200
DEFAULT_OCR_TIMEOUT_SEC = 6.0


class OcrTimeoutError(RuntimeError):
    def __init__(
        self,
        message: str,
        timeouts: list[str],
        partial_results: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.timeouts = timeouts
        self.partial_results = partial_results


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
    timeout_sec: float | None = None,
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
        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise OcrTimeoutError(
                f"Tesseract timed out after {timeout_sec}s.",
                timeouts=["tesseract"],
                partial_results=[],
            ) from exc
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
    timeout_sec: float | None = None,
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
            timeout=timeout_sec or 0,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "timeout" in message or "timed out" in message:
            raise OcrTimeoutError(
                f"Tesseract timed out after {timeout_sec}s.",
                timeouts=["pytesseract"],
                partial_results=[],
            ) from exc
        return []
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
    timeout_sec: float | None = None,
    max_dim: int | None = None,
    cache_path: Path | None = None,
    cache_key_prefix: str | None = None,
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

    if rois is None:
        raise ValueError("OCR requires explicit ROI list; full-image OCR is disabled.")
    if not rois:
        return []

    if timeout_sec is None:
        timeout_sec = DEFAULT_OCR_TIMEOUT_SEC
    if max_dim is None:
        max_dim = DEFAULT_OCR_MAX_DIM

    cache_entries: dict[str, Any] = {}
    if cache_path is not None and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if isinstance(cached, dict):
                cache_entries = cached.get("entries", {}) if isinstance(cached.get("entries"), dict) else {}
        except Exception:
            cache_entries = {}

    results: list[dict[str, Any]] = []
    timeouts: list[str] = []
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
        roi_id = roi.get("id") if isinstance(roi, dict) else None
        crop = image.crop((x, y, x + width, y + height))
        key_payload = {
            "roi": {"x": x, "y": y, "width": width, "height": height, "id": roi_id},
            "lang": "eng",
            "psm": 6,
            "config": None,
            "max_dim": max_dim,
            "prefix": cache_key_prefix or "",
        }
        cache_key = json.dumps(key_payload, sort_keys=True)
        if cache_key in cache_entries and isinstance(cache_entries[cache_key], list):
            cached_results = cache_entries[cache_key]
            results.extend(cached_results)
            continue

        scale = 1.0
        if max_dim and max(crop.size) > max_dim:
            scale = max_dim / float(max(crop.size))
            new_w = max(1, int(round(crop.size[0] * scale)))
            new_h = max(1, int(round(crop.size[1] * scale)))
            crop = crop.resize((new_w, new_h), resample=Image.LANCZOS)

        try:
            if backend_value == "pytesseract":
                roi_results = _run_pytesseract(crop, timeout_sec=timeout_sec)
            else:
                roi_results = _run_tesseract(crop, timeout_sec=timeout_sec)
        except OcrTimeoutError:
            timeouts.append(str(roi_id or "roi"))
            continue
        adjusted_results: list[dict[str, Any]] = []
        for item in roi_results:
            bbox = item.get("bbox", {})
            try:
                bx = int(bbox.get("x", 0))
                by = int(bbox.get("y", 0))
                bw = int(bbox.get("width", 0))
                bh = int(bbox.get("height", 0))
            except (TypeError, ValueError):
                continue
            if scale != 1.0:
                inv = 1.0 / scale
                bx = int(round(bx * inv))
                by = int(round(by * inv))
                bw = int(round(bw * inv))
                bh = int(round(bh * inv))
            item["bbox"] = {
                "x": bx + x,
                "y": by + y,
                "width": bw,
                "height": bh,
            }
            if isinstance(roi_id, str) and roi_id:
                item["roi_id"] = roi_id
            item["roi"] = {"x": x, "y": y, "width": width, "height": height}
            results.append(item)
            adjusted_results.append(item)
        cache_entries[cache_key] = adjusted_results

    if cache_path is not None:
        payload = {"version": 1, "entries": cache_entries}
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if timeouts:
        raise OcrTimeoutError(
            "OCR timed out on one or more ROIs.",
            timeouts=timeouts,
            partial_results=results,
        )
    results.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"], item["text"]))
    return results


def write_ocr_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
