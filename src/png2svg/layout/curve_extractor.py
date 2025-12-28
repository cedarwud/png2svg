from __future__ import annotations

from typing import Any

import numpy as np

from png2svg.layout.types import LayoutNode
from png2svg.extractor_curves import (
    _rgb_to_hsv,
    _curve_color_mask,
    _curve_centerline_points,
    _extract_dominant_color,
    _hue_distance
)


class GenericCurveExtractor:
    """Extracts curves from layout nodes by automatically detecting colors."""

    def __init__(self, adaptive_cfg: dict[str, Any] | None = None, debug: bool = False):
        self.cfg = adaptive_cfg or {}
        self.debug = debug

    def extract(self, node: LayoutNode, rgba: np.ndarray) -> None:
        """Process the node and extract curves if it appears to be a chart panel.
        
        Mutates node.content['curves'] with a list of extracted curve dicts.
        """
        # Only process LEAF nodes or specific PANEL nodes
        if node.type not in ("LEAF", "PANEL"):
            return

        # 1. Crop to node area
        x, y, w, h = node.bbox
        if w < 20 or h < 20:
            return
            
        # Ensure bounds
        if x < 0 or y < 0 or x + w > rgba.shape[1] or y + h > rgba.shape[0]:
            return

        crop = rgba[y : y + h, x : x + w]
        
        # 2. Detect dominant hues (colors used for curves)
        target_hues = self._detect_dominant_hues(crop)
        
        extracted_curves = []
        
        # 3. Extract curve for each hue
        for i, hue in enumerate(target_hues):
            # Create mask
            mask = _curve_color_mask(crop, hue, self.cfg)
            
            if self.debug:
                print(f"[DEBUG] Processing Hue {hue:.1f}: Mask pixels {mask.sum()}")
            
            # Extract centerline points
            points = _curve_centerline_points(mask, self.cfg)
            
            if not points:
                if self.debug:
                    print(f"[DEBUG] No points found for hue {hue}")
                continue
                
            # Convert to absolute coordinates
            abs_points = [(px + x, py + y) for px, py in points]
            
            # Get hex color
            hex_color = _extract_dominant_color(crop, mask)
            
            extracted_curves.append({
                "id": f"curve_{node.id}_{i}",
                "hue_center": hue,
                "stroke": hex_color,
                "points": abs_points,
                "relative_points": self._normalize_points(abs_points, node.bbox)
            })
            
        if extracted_curves:
            node.content["curves"] = extracted_curves

    def _detect_dominant_hues(self, rgba: np.ndarray) -> list[float]:
        """Analyze image to find significant colored line hues."""
        rgb = rgba[:, :, :3]
        hue, sat, val = _rgb_to_hsv(rgb)
        
        # Filter for "colored curve" pixels:
        mask = (sat > 0.25) & (val > 0.2) & (val < 0.95)
        
        valid_hues = hue[mask]
        
        if valid_hues.size < 50: 
            return []
            
        # Histogram: 0-360 degrees, bin size 10
        bins = 36
        hist, bin_edges = np.histogram(valid_hues, bins=bins, range=(0, 360))
        
        # Find peaks
        min_count = max(50, valid_hues.size * 0.05)
        
        peaks = []
        for i in range(bins):
            count = hist[i]
            if count < min_count:
                continue
                
            # Circular neighbors
            prev_i = (i - 1) % bins
            next_i = (i + 1) % bins
            
            if count >= hist[prev_i] and count >= hist[next_i]:
                # Found a peak
                center_hue = (bin_edges[i] + bin_edges[i+1]) / 2
                
                # Check if close to existing peak (merge)
                is_distinct = True
                for existing in peaks:
                    if _hue_distance(center_hue, existing) < 20:
                        is_distinct = False
                        break
                
                if is_distinct:
                    peaks.append(center_hue)
        
        if self.debug:
            print(f"[DEBUG] Total pixels: {valid_hues.size}, Min count: {min_count}")
            print(f"[DEBUG] Peaks found: {peaks}")
            # Optional: print nonzero bins
            for i in range(bins):
                if hist[i] > min_count / 2:
                    print(f"  Bin {i*10}-{(i+1)*10}: {hist[i]}")
                    
        return peaks

    def _normalize_points(self, points: list[tuple[float, float]], bbox: tuple[int, int, int, int]) -> list[dict[str, float]]:
        """Convert absolute points to 0-1 relative ratio."""
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return []
            
        return [
            {
                "x": round((px - x) / w, 4),
                "y": round((py - y) / h, 4) # SVG usually Y increases down, keep logic consistent
            }
            for px, py in points
        ]
