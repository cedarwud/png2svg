# System Design Document: Phase 1 - Generalized Layout-Based Extraction

**Version:** 1.0
**Status:** Draft
**Date:** 2025-12-28

## 1. Executive Summary

The `png2svg` project currently operates on a "Template-Based" extraction model (V0). This model relies on a classifier to select a rigid template (e.g., `t_3gpp_events`), which dictates exact Regions of Interest (ROIs) and extraction logic. While effective for known chart types, it is brittle and scales poorly to variations in layout (e.g., 4-panel charts, mixed grids).

**Phase 1** introduces a **Generalized Layout-Based Extraction** pipeline. Instead of matching a predefined template, the system will analyze the image's geometric structure to dynamically build a hierarchy of "Layout Nodes" (Canvas -> Panels -> Axes -> Components). This enables the system to handle unknown layouts by recursively applying generic component extractors.

---

## 2. Architecture Overview

### 2.1 The New Pipeline

The transformation from `Input -> Output` evolves from a linear template script to a recursive analysis process:

1.  **Input**: Raw PNG image.
2.  **Preprocess**: Binarization, Edge Detection, Line Detection (H/V).
3.  **Layout Analysis (New)**:
    *   Decompose Canvas into `LayoutNodes` using separators (whitespace, lines).
    *   Result: A Tree structure (e.g., Root -> [Header, Grid -> [Panel 1, Panel 2]]).
4.  **Role Assignment (New)**:
    *   Context-aware classification of Text/Geometry within each LayoutNode.
    *   Example: "Text at bottom-center of PanelNode is likely an X-Axis Label."
5.  **Component Extraction**:
    *   Apply specialized extractors (Curves, Arrows, Bars) within the context of confirmed Panels.
6.  **SVG Composition**:
    *   Traverse the Layout Tree to render the SVG, maintaining the hierarchical group structure.

### 2.2 Data Structure: The Layout Tree

We introduce a `LayoutNode` class to replace flat lists of parameters.

```python
from dataclasses import dataclass, field
from typing import List, Optional, Literal

NodeType = Literal["ROOT", "ROW", "COL", "PANEL", "HEADER", "FOOTER"]

@dataclass
class LayoutNode:
    id: str
    type: NodeType
    bbox: tuple[float, float, float, float]  # x, y, w, h
    children: List['LayoutNode'] = field(default_factory=list)
    
    # Content assigned to this node
    content: dict = field(default_factory=dict) 
    # e.g., {'title': TextItem, 'axes': AxisObject, 'curves': [Curve...]}
```

---

## 3. Core Modules

### 3.1 LayoutAnalyzer

**Responsibility**: Detect the structural skeleton of the image without understanding the content semantics yet.

**Algorithms**:
1.  **XY-Cut (Recursive Projection Profile)**:
    *   Calculate pixel density histograms on X and Y axes.
    *   Find "valleys" (whitespace) or "peaks" (separator lines) that span the entire width/height.
    *   Recursively split the image into sub-regions until leaf nodes (atomic panels) are found.
2.  **Grid Detection**:
    *   Identify aligned intersections of horizontal and vertical separators to detect grids (e.g., Performance Grid).

**Interface**:
```python
class LayoutAnalyzer:
    def analyze(self, image: np.ndarray, edges: List[Line]) -> LayoutNode:
        """
        Returns the root LayoutNode containing the structural hierarchy.
        """
        # 1. Detect Header/Footer via horizontal separators
        # 2. Detect main body split (Cols vs Rows)
        # 3. Recursively decompose
        pass
```

### 3.2 RoleAssigner

**Responsibility**: Assign semantic roles to unclassified geometric and text elements based on their spatial relationship with `LayoutNodes`.

**Logic**:
*   **Titles**: Text centered at the top of a `ROOT` or `PANEL` node.
*   **Axis Labels**: Text centered below (X) or rotated left of (Y) a `PANEL` node.
*   **Legends**: Clusters of (Marker + Text) inside a `PANEL` but outside the data area, or in a dedicated `LEGEND` node.

**Interface**:
```python
class RoleAssigner:
    def assign_roles(self, node: LayoutNode, text_items: List[TextItem], geometry: List[Shape]):
        """
        Mutates node.content and text_items, assigning roles like 
        'axis_title', 'panel_label', etc.
        """
        pass
```

### 3.3 GenericCurveExtractor

**Responsibility**: A unified entry point that takes a defined region (Panel) and extracts data curves, agnostic of the original template.

**Consolidation**:
*   Refactor `_curve_color_mask` and `_curve_centerline_points` from `extractor_curves.py` into a class that accepts configuration (colors, line types) but determines coordinates relative to the passed `LayoutNode`.

---

## 4. Migration Strategy

### 4.1 Hybrid V0/V1 Adapter
To ensure stability, we will not delete V0 templates immediately.

1.  **Step 1**: Implement `LayoutAnalyzer` and `GenericCurveExtractor`.
2.  **Step 2**: Create a new "Template" called `t_auto_layout`.
3.  **Step 3**: The Classifier (V0) continues to direct known types (3GPP, Flow) to V0 templates.
4.  **Step 4**: "Unknown" or low-confidence images are directed to `t_auto_layout`.
5.  **Step 5**: Gradually refactor V0 templates (e.g., `t_performance_grid`) to use `LayoutAnalyzer` internally, proving the V1 logic works.

### 4.2 Code Reuse
*   **Reuse**: `extractor_text.py` (filtering, OCR), `extractor_curves.py` (core algo), `arrow_detector.py`.
*   **Refactor**: `extractor_geometry.py` needs to expose its line detection primitives more cleanly for the `LayoutAnalyzer`.

---

## 5. Data Contract Changes

The `params.json` output schema will evolve to support nesting.

**Current (Flat)**:
```json
{
  "panels": [{"id": "p1", ...}, {"id": "p2", ...}],
  "texts": [...]
}
```

**Proposed (Hierarchical)**:
```json
{
  "layout": {
    "type": "ROOT",
    "children": [
      {
        "type": "HEADER",
        "content": {"title": "..."}
      },
      {
        "type": "ROW",
        "children": [
          {"type": "PANEL", "id": "p1", "content": {...}},
          {"type": "PANEL", "id": "p2", "content": {...}}
        ]
      }
    ]
  }
}
```
*Note: We may flatten this for the final SVG renderer if the renderer expects a flat list, or update the renderer to traverse the tree.*

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| **Over-segmentation** | High | Medium | Text paragraphs might be split into separate panels. **Fix**: Text grouping heuristics before layout analysis. |
| **Implicit Separators** | Medium | High | Some charts use whitespace only (no lines) to separate panels. **Fix**: Tune XY-Cut thresholds for whitespace detection. |
| **Legacy Regression** | Low | High | New logic breaks `fig1`. **Fix**: Keep V0 pipeline active for `fig1` until V1 passes Visual Diff tests. |
| **Performance** | Medium | Low | Recursive analysis is slower. **Fix**: Downscale images for layout analysis (structural pass), use full res for extraction. |

## 7. Comparison: V0 vs V1

| Feature | V0 (Template-Based) | V1 (Layout-Based) |
| :--- | :--- | :--- |
| **Entry Point** | Classifier -> Specific Script | Layout Analyzer -> Recursive Process |
| **ROI Definition** | Hardcoded ratios (e.g., `y=0.2*height`) | Dynamic detection (XY-Cut, Lines) |
| **Scalability** | Linear (1 new template = 1 new script) | Exponential (Handles combinatorics of layouts) |
| **Maintenance** | High (Copy-paste code) | Low (Shared logic) |
| **Failure Mode** | "Unknown Template" error | Partial extraction (might miss some panels) |
