from __future__ import annotations

from typing import Any

from png2svg.layout.types import LayoutNode


class RoleAssigner:
    """Assigns semantic roles to elements within the layout hierarchy."""

    def assign(self, root: LayoutNode, text_items: list[dict[str, Any]]) -> None:
        """Main entry point: distributes text and guesses roles.

        Args:
            root: The root LayoutNode of the analyzed tree.
            text_items: List of text dictionaries (must have 'bbox').
        """
        # 1. Spatial Join: Attach text items to the specific nodes that contain them
        self._distribute_text_items(root, text_items)

        # 2. Semantic Analysis: Guess roles based on position within the node
        self._analyze_node_semantics(root)

    def _distribute_text_items(self, node: LayoutNode, items: list[dict[str, Any]]) -> None:
        """Recursively assign text items to the deepest containing node."""
        node_items = []
        
        for item in items:
            # Check if item is roughly inside this node
            if self._is_contained(item, node):
                # Try to push down to children
                placed_in_child = False
                for child in node.children:
                    if self._is_contained(item, child):
                        # Collect for batch processing on child
                        # (We don't recurse immediately to avoid iterating full list multiple times)
                        # Actually, cleaner to filter list for child.
                        pass
                
                # If we rely on the caller passing only relevant items, this is cleaner.
                # But here we receive the FULL list at root.
                # Let's change strategy: Filter items for *this* node, then pass subsets to children.
                pass

        # Optimized approach:
        # Items assigned to *this* node are those contained in this node
        # AND NOT contained in any child (or we push them down).
        
        node.content.setdefault("texts", [])
        
        # Identify items strictly inside this node
        my_items = [item for item in items if self._is_contained(item, node)]
        
        remaining_items = []
        
        for item in my_items:
            # Check if it fits into any child
            fitted_child = None
            for child in node.children:
                if self._is_contained(item, child):
                    fitted_child = child
                    break
            
            if fitted_child:
                # Delegate to child later (grouping by child would be more efficient but loop is fine for now)
                # We can't verify which child bucket it goes to easily here without a map.
                pass
            else:
                # Belongs to this node (e.g. Header text in a ROOT node that has ROW children)
                item["_layout_node_id"] = node.id
                node.content["texts"].append(item)

        # Recurse for children
        for child in node.children:
            # Filter items relevant for this child
            child_items = [item for item in my_items if self._is_contained(item, child)]
            self._distribute_text_items(child, child_items)

    def _is_contained(self, item: dict[str, Any], node: LayoutNode) -> bool:
        """Check if text item bbox is mostly inside the node bbox."""
        bbox = item.get("bbox")
        if not bbox:
            return False
        
        # Item coords
        ix, iy = float(bbox.get("x", 0)), float(bbox.get("y", 0))
        iw, ih = float(bbox.get("width", 0)), float(bbox.get("height", 0))
        ix2, iy2 = ix + iw, iy + ih
        
        # Node coords
        nx, ny = node.x, node.y
        nx2, ny2 = node.x2, node.y2
        
        # Tolerance for slightly overlapping borders
        tol = 5.0
        
        return (ix >= nx - tol and iy >= ny - tol and 
                ix2 <= nx2 + tol and iy2 <= ny2 + tol)

    def _analyze_node_semantics(self, node: LayoutNode) -> None:
        """Analyze text positions to assign roles (Title, Axis, etc.)."""
        
        texts = node.content.get("texts", [])
        
        # Heuristics applied to every node layer
        self._guess_global_titles(node, texts)
        self._guess_axis_labels(node, texts)
        
        if node.type == "LEAF":
            self._guess_leaf_content(node, texts)
            
        # Recurse
        for child in node.children:
            self._analyze_node_semantics(child)

    def _guess_global_titles(self, node: LayoutNode, texts: list[dict[str, Any]]) -> None:
        """Identify titles at the top of a node."""
        if not texts:
            return

        # Top 15% of the node is "Header" area
        header_threshold = node.height * 0.15
        
        for item in texts:
            if item.get("role"): # Skip if already assigned
                continue
                
            # Check overlap with top region
            bbox = item.get("bbox", {})
            y = float(bbox.get("y", 0))
            h = float(bbox.get("height", 0))
            
            # Text must be fully within the top margin relative to the node
            # (Use relative coordinate)
            rel_y = y - node.y
            
            if rel_y < header_threshold:
                # Also checks if it is somewhat centered? 
                # For now, just top position is strong indicator.
                item["role"] = "title"
                node.content["has_title"] = True

    def _guess_axis_labels(self, node: LayoutNode, texts: list[dict[str, Any]]) -> None:
        """Identify axis labels (bottom X, left Y)."""
        if not texts:
            return
            
        bottom_threshold = node.height * 0.85
        left_threshold = node.width * 0.15
        
        for item in texts:
            if item.get("role"):
                continue
                
            bbox = item.get("bbox", {})
            y = float(bbox.get("y", 0))
            x = float(bbox.get("x", 0))
            rel_y = y - node.y
            rel_x = x - node.x
            
            # Bottom region -> X Axis
            if rel_y > bottom_threshold:
                item["role"] = "axis_x"
                continue
                
            # Left region -> Y Axis (and usually vertical, but we assume raw OCR here)
            if rel_x < left_threshold:
                item["role"] = "axis_y"
                continue

    def _guess_leaf_content(self, node: LayoutNode, texts: list[dict[str, Any]]) -> None:
        """Identify roles in LEAF nodes (Labels, Legends)."""
        # If node has many texts, might be a text block or axis labels
        # This is a placeholder for complex logic (e.g. detecting numerical sequences)
        for item in texts:
            if item.get("role"):
                continue
            item["role"] = "annotation" # Default fallback
