from __future__ import annotations

import numpy as np
from png2svg.layout.types import LayoutNode, NodeType


class LayoutAnalyzer:
    """Analyzes image structure to build a hierarchical layout tree."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._node_counter = 0

    def _next_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def analyze(self, binary_image: np.ndarray) -> LayoutNode:
        """Analyze the binary image (0=background, 1=foreground/ink).

        Args:
            binary_image: Boolean or uint8 numpy array where True/1 is content.

        Returns:
            Root LayoutNode of the detected structure.
        """
        height, width = binary_image.shape
        root = LayoutNode(
            id="root",
            type="ROOT",
            bbox=(0, 0, width, height),
        )
        
        # Start recursive decomposition
        self._recursive_xy_cut(root, binary_image, axis=1) # Start cutting horizontally (Rows)
        
        return root

    def _compute_projection(self, image: np.ndarray, axis: int) -> np.ndarray:
        """Compute projection profile (sum of ink pixels).
        
        axis=0: Column profile (project onto X axis) -> sum over rows
        axis=1: Row profile (project onto Y axis) -> sum over cols
        """
        return np.sum(image, axis=axis)

    def _mask_long_lines(self, image: np.ndarray, axis: int) -> np.ndarray:
        """Create a mask where long lines perpendicular to the projection axis are removed.
        
        If detecting vertical gaps (axis=0), we sum cols. Horizontal lines crossing these cols 
        should be ignored to find gaps.
        So we remove lines that run along axis=1 (rows).
        
        Args:
            axis: The axis of the projection (0 for X-profile, 1 for Y-profile).
                  We want to remove lines ORTHOGONAL to this axis.
                  If axis=0 (Col profile), remove Horizontal lines (axis=1).
        """
        # We want to remove lines running along 'line_axis'
        line_axis = 1 - axis
        
        # Simple run-length based removal using morph-like logic or just detecting rows/cols with high fill?
        # A horizontal line is a row (axis=0 index) with high sum?
        # No, a horizontal line is a row where pixels are connected.
        
        # Fast approximation: High density rows/cols
        # If we want to ignore horizontal lines (axis=1), we check row sums?
        # A solid horizontal line has row_sum ~ width.
        # But a dashed line or text also has some sum.
        
        # Better: morphological opening? 
        # To remove horizontal lines: Erase anything that is horizontally long.
        # But we don't have cv2.
        
        # Heuristic: Calculate profile along the line_axis.
        # If a specific coordinate in line_axis has very high density (> 50%?), ignore it?
        # But a text line also has high density.
        
        # Let's try a simpler approach:
        # Just use median filtering on the projection? 
        # Or: if we are looking for vertical gaps, horizontal lines add a constant offset to the projection profile.
        # We can subtract the "baseline" noise?
        
        # If a horizontal line crosses the whole image, every column in projection increases by 1.
        # Gaps will still be "valleys", but not zero.
        # So our threshold logic should handle this.
        
        # My current threshold logic is absolute (max(3, ...)).
        # It should be relative to the local minimum?
        # "Find valleys"
        
        return image

    def _find_separators(
        self, 
        projection: np.ndarray, 
        min_gap: int = 10, 
        threshold: int | None = None
    ) -> list[tuple[int, int]]:
        """Find gaps (zero/low ink) in the projection profile."""
        
        if threshold is None:
            # Dynamic threshold: use 10th percentile as base noise floor?
            # If there are grid lines crossing, the "empty" areas will have height = num_lines.
            # We want to find areas where projection is close to the minimum.
            
            # Use lower quartile as the "empty" reference
            if len(projection) > 0:
                base_noise = np.percentile(projection, 10)
                threshold = int(base_noise) + 2 # slightly above noise floor
            else:
                threshold = 3
            
        has_content = projection > threshold
        
        # Find runs of False (gaps)
        padded = np.concatenate(([True], has_content, [True]))
        diff = padded[:-1] != padded[1:]
        indices = np.where(diff)[0]
        
        separators = []
        
        for i in range(0, len(indices), 2):
            gap_start = indices[i]
            if i + 1 < len(indices):
                gap_end = indices[i+1]
            else:
                gap_end = len(has_content)
                
            if gap_end - gap_start >= min_gap:
                separators.append((gap_start, gap_end))
                
        return separators

    def _recursive_xy_cut(
        self, 
        node: LayoutNode, 
        image: np.ndarray, 
        axis: int, 
        depth: int = 0
    ) -> None:
        """Recursively split node content."""
        if depth > 4: 
            node.type = "LEAF"
            return

        h, w = image.shape
        if h < 20 or w < 20:
            node.type = "LEAF"
            return

        # No masking implemented yet, relying on dynamic threshold in _find_separators
        projection = self._compute_projection(image, 1 - axis)
        
        if self.debug:
            max_val = np.max(projection)
            mean_val = np.mean(projection)
            print(f"[DEBUG] Depth {depth} Axis {axis} ({w}x{h}): Max proj {max_val}, Mean {mean_val}")

        dim_size = h if axis == 1 else w
        min_gap = max(10, int(dim_size * 0.02))
        
        # Pass None to trigger dynamic percentile-based thresholding
        separators = self._find_separators(projection, min_gap=min_gap, threshold=None)
        
        valid_separators = []
        margin = max(5, int(dim_size * 0.01))
        
        for start, end in separators:
            if start > margin and end < dim_size - margin:
                valid_separators.append((start, end))
        
        separators = valid_separators
        
        # Aggressive Pass
        if not separators and dim_size > 500:
            if self.debug:
                print(f"[DEBUG] Aggressive cut attempt on Axis {axis} (size {dim_size})")
            
            aggressive_gap = max(5, int(dim_size * 0.005))
            # For aggressive, we might want a slightly stricter threshold relative to base?
            # Or just rely on the gap being smaller.
            # Passing None uses percentile logic again.
            
            separators = self._find_separators(projection, min_gap=aggressive_gap, threshold=None)
            
            valid_separators = []
            for start, end in separators:
                if start > margin and end < dim_size - margin:
                    valid_separators.append((start, end))
            separators = valid_separators

        if self.debug and separators:
            print(f"[DEBUG] Found separators: {separators}")

        if not separators:
            next_axis = 1 - axis
            
            other_proj = self._compute_projection(image, 1 - next_axis)
            dim_other = h if next_axis == 1 else w
            min_gap_other = max(10, int(dim_other * 0.02))
            
            other_seps = self._find_separators(other_proj, min_gap=min_gap_other, threshold=None)
            valid_other = [s for s in other_seps if s[0] > margin and s[1] < dim_other - margin]
            
            if not valid_other:
                node.type = "LEAF"
                return
            
            if self.debug:
                print(f"[DEBUG] Swapping axis to {next_axis}")
            self._recursive_xy_cut(node, image, next_axis, depth + 1)
            return

        # We have splits!
        node.type = "ROW" if axis == 1 else "COL"