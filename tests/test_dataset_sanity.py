from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "regression_v0" / "manifest.yaml"
SANITY_CONFIG = ROOT / "config" / "dataset_sanity.v1.yaml"


def _load_manifest() -> list[dict[str, str]]:
    data = yaml.safe_load(MANIFEST.read_text())
    return data["cases"]


def _metrics(image: Image.Image, edge_threshold: float) -> dict[str, float]:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3]
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    ink = (alpha > 10) & (luminance < 245)
    ink_ratio = float(np.mean(ink))
    variance = float(np.var(luminance))
    dx = np.abs(np.diff(luminance, axis=1))
    dy = np.abs(np.diff(luminance, axis=0))
    edges = np.zeros(luminance.shape, dtype=bool)
    edges[:, 1:] |= dx > edge_threshold
    edges[1:, :] |= dy > edge_threshold
    edge_ratio = float(np.mean(edges))
    return {
        "ink_ratio": ink_ratio,
        "variance": variance,
        "edge_ratio": edge_ratio,
    }


def test_dataset_sanity_sample() -> None:
    cases = _load_manifest()[:5]
    cfg = yaml.safe_load(SANITY_CONFIG.read_text())
    variance_min = float(cfg.get("variance_min", 2.0))
    ink_ratio_min = float(cfg.get("ink_ratio_min", 0.002))
    ink_ratio_max = float(cfg.get("ink_ratio_max", 0.98))
    edge_ratio_min = float(cfg.get("edge_ratio_min", 0.001))
    edge_threshold = float(cfg.get("edge_threshold", 8.0))

    for entry in cases:
        case_dir = ROOT / "datasets" / "regression_v0" / entry["dir"]
        input_png = case_dir / "input.png"
        assert input_png.exists()
        with Image.open(input_png) as image:
            stats = _metrics(image, edge_threshold)
        assert stats["variance"] >= variance_min
        assert ink_ratio_min <= stats["ink_ratio"] <= ink_ratio_max
        assert stats["edge_ratio"] >= edge_ratio_min
