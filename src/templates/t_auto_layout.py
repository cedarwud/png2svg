from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from common.svg_builder import SvgBuilder
from png2svg.extractor_preprocess import _ink_mask, _load_image
from png2svg.layout.analyzer import LayoutAnalyzer
from png2svg.layout.curve_extractor import GenericCurveExtractor
from png2svg.layout.role_assigner import RoleAssigner
from png2svg.layout.types import LayoutNode


def extract(
    width: int,
    height: int,
    mask: np.ndarray,
    rgba: Any,
    text_items: list[dict[str, Any]],
    warnings: list[Any],
    debug_dir: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Auto-layout extraction pipeline (V1)."""
    
    # 2. Layout Analysis
    analyzer = LayoutAnalyzer(debug=bool(debug_dir))
    root = analyzer.analyze(mask)
    
    if debug_dir:
        _render_debug_overlay(root, rgba, Path(debug_dir) / "layout_tree.png")
    
    # 4. Role Assignment
    assigner = RoleAssigner()
    assigner.assign(root, text_items)
    
    # 5. Component Extraction (Curves)
    curve_extractor = GenericCurveExtractor(debug=bool(debug_dir))
    _recurse_extract_curves(root, rgba, curve_extractor)
    
    # 6. Convert to params format
    params = _layout_to_params(root, width, height)
    
    # 7. Overlay for debug
    overlay = {
        "panels": params.get("panels", []),
    }
    
    return params, overlay


def _recurse_extract_curves(node: LayoutNode, rgba: Any, extractor: GenericCurveExtractor) -> None:
    if node.type in ("PANEL", "LEAF"):
        extractor.extract(node, rgba)
    
    for child in node.children:
        _recurse_extract_curves(child, rgba, extractor)


def _snap_color(hex_color: str) -> str:
    """Map color to nearest standard palette color."""
    if not hex_color or not hex_color.startswith("#"):
        return "#000000"
        
    # Standard matplotlib/d3 palette + black
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#000000"
    ]
    
    def hex_to_rgb(h):
        return tuple(int(h.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        
    try:
        r, g, b = hex_to_rgb(hex_color)
    except ValueError:
        return "#000000"
        
    best_dist = float("inf")
    best_color = "#000000"
    
    for p in palette:
        pr, pg, pb = hex_to_rgb(p)
        dist = ((r-pr)**2 + (g-pg)**2 + (b-pb)**2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_color = p
            
    return best_color


def _layout_to_params(root: LayoutNode, width: int, height: int) -> dict[str, Any]:
    """Convert LayoutTree to standard params dictionary."""
    panels = []
    
    def _visit(node: LayoutNode):
        if node.type in ("PANEL", "LEAF"):
            # Try to find a title
            title = f"Panel {node.id}"
            texts = node.content.get("texts", [])
            for t in texts:
                if t.get("role") == "title":
                    title = str(t.get("text", "")).strip()
                    break
            
            panel_def = {
                "id": node.id,
                "x": float(node.x),
                "y": float(node.y),
                "width": float(node.width),
                "height": float(node.height),
                "title": title
            }
            
            if "curves" in node.content:
                curves = node.content["curves"]
                for c in curves:
                    if "stroke" in c:
                        c["stroke"] = _snap_color(c["stroke"])
                panel_def["extracted_curves"] = curves
                
            panels.append(panel_def)
            
        for child in node.children:
            _visit(child)
            
    _visit(root)
    
    return {
        "template": "t_auto_layout",
        "canvas": {"width": width, "height": height},
        "panels": panels,
        "geometry": {"lines": []},
        "style": {}
    }


def render(builder: SvgBuilder, params: dict[str, Any], canvas: tuple[int, int]) -> None:
    """Render the auto-extracted layout."""
    panels = params.get("panels", [])
    used_colors = set()
    
    for panel in panels:
        # Draw Panel Box with lighter stroke
        builder.groups["g_axes"].add(
            builder.drawing.rect(
                insert=(panel["x"], panel["y"]),
                size=(panel["width"], panel["height"]),
                fill="none",
                stroke="#e0e0e0",
                stroke_width=1
            )
        )
        
        # Draw Panel Title
        title = panel.get("title")
        if title:
            # Simple heuristic: title at top center of panel
            tx = panel["x"] + panel["width"] / 2
            ty = panel["y"] + 20
            builder.groups["g_text"].add(
                builder.drawing.text(
                    title,
                    insert=(tx, ty),
                    text_anchor="middle",
                    font_family="sans-serif",
                    font_size=12,
                    fill="#333333",
                    font_weight="bold"
                )
            )
        
        # Draw Extracted Curves
        curves = panel.get("extracted_curves", [])
        for c in curves:
            pts = c.get("points", [])
            if len(pts) < 2:
                continue
            
            stroke = c.get("stroke", "#000000")
            used_colors.add(stroke)
            
            builder.groups["g_curves"].add(
                builder.drawing.polyline(
                    points=pts,
                    fill="none",
                    stroke=stroke,
                    stroke_width=2,
                    stroke_linecap="round",
                    stroke_linejoin="round"
                )
            )

    # Draw Simple Legend at bottom if curves exist
    if used_colors:
        legend_y = canvas[1] - 30
        legend_x = 20
        builder.groups["g_text"].add(
            builder.drawing.text(
                "Legend:", 
                insert=(legend_x, legend_y), 
                font_family="sans-serif", 
                font_size=10, 
                fill="#666666"
            )
        )
        legend_x += 50
        
        for i, color in enumerate(sorted(used_colors)):
            builder.groups["g_markers"].add(
                builder.drawing.circle(
                    center=(legend_x, legend_y - 4),
                    r=4,
                    fill=color
                )
            )
            builder.groups["g_text"].add(
                builder.drawing.text(
                    f"Series {i+1}",
                    insert=(legend_x + 10, legend_y),
                    font_family="sans-serif",
                    font_size=10,
                    fill="#333333"
                )
            )
            legend_x += 80


def _render_debug_overlay(root: LayoutNode, rgba: np.ndarray, output_path: Path) -> None:
    """Render layout tree structure on the image for debugging."""
    try:
        img = Image.fromarray(rgba)
        draw = ImageDraw.Draw(img)
        
        # Colors for different node types
        colors = {
            "ROOT": "red",
            "ROW": "green",
            "COL": "blue",
            "PANEL": "orange",
            "LEAF": "yellow",
            "HEADER": "purple",
            "FOOTER": "cyan"
        }
        
        def _draw_node(node: LayoutNode, level: int):
            color = colors.get(node.type, "gray")
            # Draw bbox
            x, y, w, h = node.bbox
            if w > 0 and h > 0:
                draw.rectangle([x, y, x + w, y + h], outline=color, width=max(1, 6 - level))
                
                # Draw label
                text = f"{node.type} {node.id}"
                # Draw text background to make it readable
                text_x = x + 2
                text_y = y + 2
                
                # Simple text drawing (default font)
                draw.text((text_x, text_y), text, fill=color)
            
            for child in node.children:
                _draw_node(child, level + 1)
                
        _draw_node(root, 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        
    except Exception as e:
        print(f"Failed to render debug overlay: {e}")
