from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from png2svg.ocr import write_ocr_json


def _write_debug_artifacts(
    debug_dir: Path,
    rgba: np.ndarray,
    preprocessed: np.ndarray,
    overlay: dict[str, Any],
    ocr: list[dict[str, Any]],
    report: dict[str, Any],
    params: dict[str, Any],
    input_png: Path,
    effective_config: dict[str, Any],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "effective_config.json").write_text(
        json.dumps(effective_config, indent=2, sort_keys=True)
    )
    pre = Image.fromarray(preprocessed, mode="L")
    pre.save(debug_dir / "01_preprocessed.png")
    pre.save(debug_dir / "preprocessed.png")

    overlay_img = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(overlay_img, "RGBA")

    for panel in overlay.get("panels", []):
        draw.rectangle(
            [
                panel["x"],
                panel["y"],
                panel["x"] + panel["width"],
                panel["y"] + panel["height"],
            ],
            outline=(0, 128, 255, 200),
            width=2,
        )
    plot = overlay.get("axes_plot")
    if plot:
        draw.rectangle(
            [plot["x"], plot["y"], plot["x"] + plot["width"], plot["y"] + plot["height"]],
            outline=(0, 200, 0, 200),
            width=2,
        )
    for node in overlay.get("nodes", []):
        draw.rectangle(
            [
                node["x"],
                node["y"],
                node["x"] + node["width"],
                node["y"] + node["height"],
            ],
            outline=(255, 165, 0, 200),
            width=2,
        )
    for line in overlay.get("lines", []):
        try:
            draw.line(
                [(line["x1"], line["y1"]), (line["x2"], line["y2"])],
                fill=(120, 120, 255, 180),
                width=2,
            )
        except Exception:
            continue
    for marker in overlay.get("markers", []):
        try:
            x = marker["x"]
            y = marker["y"]
            r = marker.get("radius", 3)
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0, 180), width=2)
        except Exception:
            continue
    for text in overlay.get("text_boxes", []):
        bbox = text["bbox"]
        draw.rectangle(
            [
                bbox["x"],
                bbox["y"],
                bbox["x"] + bbox["width"],
                bbox["y"] + bbox["height"],
            ],
            outline=(255, 0, 0, 180),
            width=1,
        )

    overlay_img.save(debug_dir / "02_overlay.png")
    overlay_img.save(debug_dir / "overlays.png")
    write_ocr_json(debug_dir / "03_ocr_raw.json", ocr)
    (debug_dir / "04_params.json").write_text(json.dumps(params, indent=2, sort_keys=True))
    (debug_dir / "extracted.json").write_text(json.dumps(params, indent=2, sort_keys=True))
    (debug_dir / "extract_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    extracted = params.get("extracted", {})
    curve_points = extracted.get("curve_points") if isinstance(extracted, dict) else None
    if isinstance(curve_points, list) and curve_points:
        (debug_dir / "curves_points.json").write_text(
            json.dumps(curve_points, indent=2, sort_keys=True)
        )
        curves_img = Image.fromarray(rgba, mode="RGBA")
        curves_draw = ImageDraw.Draw(curves_img, "RGBA")
        color_map = {
            "serving": (43, 108, 176, 220),
            "neighbor": (221, 107, 32, 220),
        }
        for curve in curve_points:
            curve_id = str(curve.get("curve_id", ""))
            color = color_map.get(curve_id, (0, 0, 0, 200))
            points = curve.get("points", [])
            if not isinstance(points, list):
                continue
            coords: list[tuple[float, float]] = []
            for point in points:
                if not isinstance(point, dict):
                    continue
                try:
                    x = float(point.get("x", 0.0))
                    y = float(point.get("y", 0.0))
                except (TypeError, ValueError):
                    continue
                coords.append((x, y))
                curves_draw.ellipse([x - 2, y - 2, x + 2, y + 2], outline=color, width=2)
            if len(coords) >= 2:
                curves_draw.line(coords, fill=color, width=2)
        curves_img.save(debug_dir / "overlays_curves.png")

    try:
        from png2svg.renderer import render_svg
        from validators.visual_diff import RasterizeError, rasterize_svg_to_png

        snap_svg = debug_dir / "05_snap_preview.svg"
        snap_png = debug_dir / "05_snap_preview.png"
        snap_params = debug_dir / "04_params.json"
        render_svg(input_png, snap_params, snap_svg)
        try:
            rasterize_svg_to_png(snap_svg, snap_png)
        except RasterizeError:
            pass
    except Exception:
        pass
