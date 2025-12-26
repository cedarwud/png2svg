from __future__ import annotations

import re
from typing import Any

import numpy as np

from common.svg_builder import DEFAULT_FONT_FAMILY
from png2svg.extractor_curves import _hue_distance, _rgb_to_hsv
from png2svg.extractor_preprocess import _connected_components, _ink_mask


def _text_items_from_boxes(text_boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not text_boxes:
        return []
    baselines: list[float] = []
    heights: list[float] = []
    boxes_sorted: list[tuple[float, dict[str, Any]]] = []
    for entry in text_boxes:
        bbox = entry.get("bbox")
        if not isinstance(bbox, dict):
            continue
        try:
            x = float(bbox["x"])
            y = float(bbox["y"])
            width = float(bbox["width"])
            height = float(bbox["height"])
        except (KeyError, TypeError, ValueError):
            continue
        baseline = y + height
        baselines.append(baseline)
        heights.append(height)
        boxes_sorted.append((baseline, {"x": x, "y": y, "width": width, "height": height}))
    if not boxes_sorted:
        return []
    heights.sort()
    median_height = heights[len(heights) // 2]
    tolerance = max(4.0, median_height * 0.6)
    boxes_sorted.sort(key=lambda item: (item[0], item[1]["x"]))

    clusters: list[list[dict[str, Any]]] = []
    cluster_baselines: list[float] = []
    for baseline, bbox in boxes_sorted:
        if not clusters or abs(baseline - cluster_baselines[-1]) > tolerance:
            clusters.append([bbox])
            cluster_baselines.append(baseline)
        else:
            clusters[-1].append(bbox)
            cluster_baselines[-1] = (cluster_baselines[-1] + baseline) / 2.0

    line_boxes: list[dict[str, Any]] = []
    for cluster, baseline in zip(clusters, cluster_baselines):
        min_x = min(box["x"] for box in cluster)
        max_x = max(box["x"] + box["width"] for box in cluster)
        min_y = min(box["y"] for box in cluster)
        max_y = max(box["y"] + box["height"] for box in cluster)
        line_boxes.append(
            {
                "min_x": float(min_x),
                "max_x": float(max_x),
                "min_y": float(min_y),
                "max_y": float(max_y),
                "baseline": float(baseline),
            }
        )

    line_boxes.sort(key=lambda item: (item["min_y"], item["min_x"]))
    line_gap = max(4.0, median_height * 1.8)
    blocks: list[dict[str, Any]] = []

    for line in line_boxes:
        best_idx: int | None = None
        best_overlap = 0.0
        for idx, block in enumerate(blocks):
            overlap = min(line["max_x"], block["max_x"]) - max(line["min_x"], block["min_x"])
            min_width = min(line["max_x"] - line["min_x"], block["max_x"] - block["min_x"])
            if min_width <= 0:
                continue
            overlap_ratio = overlap / min_width
            vertical_gap = line["min_y"] - block["max_y"]
            if 0 <= vertical_gap <= line_gap and overlap_ratio >= 0.4:
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_idx = idx
        if best_idx is None:
            blocks.append(
                {
                    "min_x": line["min_x"],
                    "max_x": line["max_x"],
                    "min_y": line["min_y"],
                    "max_y": line["max_y"],
                    "baseline": line["baseline"],
                }
            )
        else:
            block = blocks[best_idx]
            block["min_x"] = min(block["min_x"], line["min_x"])
            block["max_x"] = max(block["max_x"], line["max_x"])
            block["min_y"] = min(block["min_y"], line["min_y"])
            block["max_y"] = max(block["max_y"], line["max_y"])
            block["baseline"] = (block["baseline"] + line["baseline"]) / 2.0

    items: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        items.append(
            {
                "text": None,
                "x": float(block["min_x"]),
                "y": float(block["baseline"]),
                "role": "annotation",
                "anchor": "start",
                "baseline_group": f"block_{idx}",
                "conf": 0.0,
                "bbox": {
                    "x": float(block["min_x"]),
                    "y": float(block["min_y"]),
                    "width": float(block["max_x"] - block["min_x"]),
                    "height": float(block["max_y"] - block["min_y"]),
                },
            }
        )
    return items


def _group_ocr_lines(
    ocr_results: list[dict[str, Any]], tolerance: float
) -> list[list[dict[str, Any]]]:
    if not ocr_results:
        return []
    sorted_results = sorted(
        ocr_results,
        key=lambda item: (
            float(item["bbox"]["y"]) + float(item["bbox"]["height"]),
            float(item["bbox"]["x"]),
        ),
    )
    clusters: list[list[dict[str, Any]]] = []
    cluster_baselines: list[float] = []
    for item in sorted_results:
        bbox = item["bbox"]
        baseline = float(bbox["y"]) + float(bbox["height"])
        if not clusters or abs(baseline - cluster_baselines[-1]) > tolerance:
            clusters.append([item])
            cluster_baselines.append(baseline)
        else:
            clusters[-1].append(item)
            cluster_baselines[-1] = (cluster_baselines[-1] + baseline) / 2.0
    split_clusters: list[list[dict[str, Any]]] = []
    for cluster in clusters:
        cluster.sort(key=lambda item: float(item["bbox"]["x"]))
        if len(cluster) <= 1:
            split_clusters.append(cluster)
            continue
        heights = sorted(float(item["bbox"]["height"]) for item in cluster)
        median_height = heights[len(heights) // 2] if heights else 10.0
        gap_threshold = max(median_height * 2.5, 24.0)
        current: list[dict[str, Any]] = [cluster[0]]
        for item in cluster[1:]:
            prev = current[-1]
            gap = float(item["bbox"]["x"]) - (
                float(prev["bbox"]["x"]) + float(prev["bbox"]["width"])
            )
            if gap > gap_threshold:
                split_clusters.append(current)
                current = [item]
            else:
                current.append(item)
        if current:
            split_clusters.append(current)
    return split_clusters


_TOKEN_CLEAN_RE = re.compile(r"[^0-9A-Za-z_./+-]+")


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _sanitize_text(value: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    tokens: list[str] = []
    for raw in cleaned.split():
        token = _TOKEN_CLEAN_RE.sub("", raw).strip("._/+-")
        if token:
            tokens.append(token)
    if not tokens:
        return ""
    if len(tokens) > 1:
        pruned = [token for token in tokens if not (len(token) == 1 and token.isalpha())]
        tokens = pruned or tokens
    return " ".join(tokens)


def _dedupe_consecutive_tokens(value: str) -> str:
    tokens = value.split()
    if not tokens:
        return value
    deduped = [tokens[0]]
    for token in tokens[1:]:
        if token.lower() == deduped[-1].lower():
            continue
        deduped.append(token)
    return " ".join(deduped)


def _text_items_from_ocr(
    ocr_results: list[dict[str, Any]],
    width: int,
    height: int,
    adaptive: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not ocr_results:
        return []
    heights = sorted(float(item["bbox"]["height"]) for item in ocr_results)
    median_height = heights[len(heights) // 2]
    ocr_cfg = adaptive.get("ocr") if adaptive else None
    layout_cfg = adaptive.get("text_layout") if adaptive else None
    max_height_ratio = float(ocr_cfg.get("max_bbox_height_ratio", 2.2)) if isinstance(ocr_cfg, dict) else 2.2
    font_ratio = float(layout_cfg.get("ocr_font_size_ratio", 0.6)) if isinstance(layout_cfg, dict) else 0.6
    line_height_ratio = float(ocr_cfg.get("max_line_height_ratio", 1.3)) if isinstance(ocr_cfg, dict) else 1.3
    max_line_aspect = float(ocr_cfg.get("max_line_aspect_ratio", 3.0)) if isinstance(ocr_cfg, dict) else 3.0
    max_height = max(median_height * max_height_ratio, median_height + 1.0)
    filtered_results = [
        item
        for item in ocr_results
        if float(item.get("bbox", {}).get("height", 0.0)) <= max_height
    ]
    tolerance = max(4.0, median_height * 0.6)
    by_roi: dict[str | None, list[dict[str, Any]]] = {}
    for item in filtered_results:
        by_roi.setdefault(item.get("roi_id"), []).append(item)
    items: list[dict[str, Any]] = []
    for roi_id in sorted(by_roi.keys(), key=lambda value: "" if value is None else str(value)):
        roi_items = by_roi[roi_id]
        lines = _group_ocr_lines(roi_items, tolerance)
        for idx, line in enumerate(lines):
            texts: list[str] = []
            word_heights: list[float] = []
            for item in line:
                raw = item.get("text")
                if not raw:
                    continue
                text = _sanitize_text(str(raw))
                if not text:
                    continue
                if len(text) == 1 and not text.isalnum():
                    continue
                if texts and text == texts[-1]:
                    continue
                texts.append(text)
                try:
                    word_heights.append(float(item["bbox"]["height"]))
                except (TypeError, ValueError):
                    continue
            if not texts:
                continue
            if len(texts) > 1:
                pruned = [token for token in texts if not (len(token) == 1 and token.isalpha())]
                texts = pruned or texts
            content = _dedupe_consecutive_tokens(" ".join(texts))
            min_x = min(float(item["bbox"]["x"]) for item in line)
            min_y = min(float(item["bbox"]["y"]) for item in line)
            max_x = max(float(item["bbox"]["x"]) + float(item["bbox"]["width"]) for item in line)
            max_y = max(float(item["bbox"]["y"]) + float(item["bbox"]["height"]) for item in line)
            line_height = max_y - min_y
            line_width = max_x - min_x
            if (
                line_height > max_height * line_height_ratio
                and line_width > 0
                and (line_height / line_width) > max_line_aspect
            ):
                continue
            baseline = max_y
            conf_values = [float(item.get("conf", 0.0)) for item in line]
            conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
            if word_heights:
                word_heights.sort()
                median_word_height = word_heights[len(word_heights) // 2]
            else:
                median_word_height = median_height
            font_size = max(6.0, median_word_height * font_ratio)
            bbox = {
                "x": float(min_x),
                "y": float(min_y),
                "width": float(max_x - min_x),
                "height": float(max_y - min_y),
            }
            bbox["x0"] = bbox["x"]
            bbox["y0"] = bbox["y"]
            bbox["x1"] = bbox["x"] + bbox["width"]
            bbox["y1"] = bbox["y"] + bbox["height"]
            item_payload = {
                "content": content,
                "text": content,
                "x": float(min_x),
                "y": float(baseline),
                "role": "annotation",
                "anchor": "start",
                "baseline_group": f"line_{roi_id or 'global'}_{idx}",
                "conf": conf,
                "confidence": conf,
                "font_size": font_size,
                "bbox": bbox,
            }
            if roi_id:
                item_payload["roi_id"] = roi_id
            items.append(item_payload)
    return items


def _text_char_stats(value: str) -> dict[str, float]:
    total = len(value)
    if total == 0:
        return {"total": 0, "alnum": 0, "alpha": 0, "digit": 0, "ascii": 0}
    alnum = sum(1 for ch in value if ch.isalnum())
    alpha = sum(1 for ch in value if ch.isalpha())
    digit = sum(1 for ch in value if ch.isdigit())
    ascii_count = sum(1 for ch in value if ord(ch) < 128)
    return {
        "total": total,
        "alnum": alnum,
        "alpha": alpha,
        "digit": digit,
        "ascii": ascii_count,
    }


def _keep_text_item(item: dict[str, Any], cfg: dict[str, Any] | None) -> bool:
    text = _clean_text(str(item.get("content") or item.get("text") or ""))
    if not text:
        return False
    try:
        conf = float(item.get("conf", 1.0))
    except (TypeError, ValueError):
        conf = 1.0
    min_conf = float(cfg.get("min_conf", 0.5)) if isinstance(cfg, dict) else 0.5
    roi_id = str(item.get("roi_id") or "")
    panel_mid_min_conf = float(cfg.get("panel_mid_min_conf", min_conf)) if isinstance(cfg, dict) else min_conf
    panel_bottom_min_conf = float(cfg.get("panel_bottom_min_conf", min_conf)) if isinstance(cfg, dict) else min_conf
    if roi_id.startswith("panel_mid"):
        keyword_hits = any(
            token in text.lower()
            for token in (
                "threshold",
                "hys",
                "trigger",
                "serving",
                "neighbor",
                "target",
                "both",
                "condition",
                "ttt",
                "a3",
                "a4",
                "a5",
            )
        )
        if conf < panel_mid_min_conf and not keyword_hits:
            return False
    if roi_id.startswith("panel_bottom"):
        keyword_hits = any(
            token in text.lower()
            for token in ("condition", "report", "event", "trigger", "ttt")
        )
        if conf < panel_bottom_min_conf and not keyword_hits:
            return False
    if conf < min_conf:
        return False
    min_chars = int(cfg.get("min_chars", 2)) if isinstance(cfg, dict) else 2
    if len(text) < min_chars:
        return False
    bbox = item.get("bbox")
    if isinstance(bbox, dict):
        min_h = int(cfg.get("min_bbox_height", 6)) if isinstance(cfg, dict) else 6
        min_w = int(cfg.get("min_bbox_width", 6)) if isinstance(cfg, dict) else 6
        try:
            if float(bbox.get("height", 0.0)) < min_h or float(bbox.get("width", 0.0)) < min_w:
                return False
        except (TypeError, ValueError):
            pass
    stats = _text_char_stats(text)
    total = stats["total"]
    if total <= 0:
        return False
    alnum_ratio = stats["alnum"] / total
    alpha_ratio = stats["alpha"] / total
    digit_ratio = stats["digit"] / total
    ascii_ratio = stats["ascii"] / total
    min_alnum = float(cfg.get("min_alnum_ratio", 0.35)) if isinstance(cfg, dict) else 0.35
    min_alpha = float(cfg.get("min_alpha_ratio", 0.2)) if isinstance(cfg, dict) else 0.2
    min_digit = float(cfg.get("min_digit_ratio", 0.15)) if isinstance(cfg, dict) else 0.15
    min_ascii = float(cfg.get("min_ascii_ratio", 0.7)) if isinstance(cfg, dict) else 0.7
    if ascii_ratio < min_ascii:
        return False
    if alnum_ratio < min_alnum:
        return False
    if alpha_ratio < min_alpha and digit_ratio < min_digit:
        return False
    return True


def _filter_text_items(text_items: list[dict[str, Any]], cfg: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not text_items:
        return []
    filtered: list[dict[str, Any]] = []
    seen: dict[tuple[str, int, int], dict[str, Any]] = {}
    for item in text_items:
        if not _keep_text_item(item, cfg):
            continue
        content = _clean_text(str(item.get("content") or item.get("text") or ""))
        if not content:
            continue
        item["content"] = content
        item["text"] = content
        bbox = item.get("bbox")
        if isinstance(bbox, dict):
            try:
                key = (content, int(round(float(bbox.get("x", 0.0)))), int(round(float(bbox.get("y", 0.0)))))
            except (TypeError, ValueError):
                key = (content, 0, 0)
        else:
            key = (content, 0, 0)
        existing = seen.get(key)
        if existing is None:
            seen[key] = item
            filtered.append(item)
        else:
            try:
                if float(item.get("conf", 0.0)) > float(existing.get("conf", 0.0)):
                    seen[key] = item
                    idx = filtered.index(existing)
                    filtered[idx] = item
            except (TypeError, ValueError):
                continue
    filtered.sort(
        key=lambda item: (
            float(item.get("bbox", {}).get("y", 0.0)),
            float(item.get("bbox", {}).get("x", 0.0)),
            str(item.get("content") or item.get("text") or ""),
        )
    )
    return filtered


def _count_renderable_texts(text_items: list[dict[str, Any]]) -> int:
    count = 0
    for item in text_items:
        if item.get("render") is False:
            continue
        text_value = _clean_text(str(item.get("content") or item.get("text") or ""))
        if text_value:
            count += 1
    return count


def _detect_text_boxes(rgba: np.ndarray, adaptive: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    mask = _ink_mask(rgba, adaptive)
    height, width = mask.shape
    text_cfg = adaptive.get("text_boxes") if adaptive else None
    if text_cfg:
        min_area = int(text_cfg.get("min_area", 8))
        max_area = int(text_cfg.get("max_area", min_area + 1))
        min_size = int(text_cfg.get("min_size", 3))
        max_width_ratio = float(text_cfg.get("max_width_ratio", 0.6))
        max_height_ratio = float(text_cfg.get("max_height_ratio", 0.4))
    else:
        min_area = max(int(height * width * 0.00005), 8)
        max_area = max(int(height * width * 0.02), min_area + 1)
        min_size = 3
        max_width_ratio = 0.6
        max_height_ratio = 0.4
    components = _connected_components(mask, min_area=min_area)
    boxes: list[dict[str, Any]] = []
    for comp in components:
        area = comp["area"]
        if area > max_area:
            continue
        w = comp["width"]
        h = comp["height"]
        if w < min_size or h < min_size:
            continue
        if w > width * max_width_ratio or h > height * max_height_ratio:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.2 or aspect > 12:
            continue
        boxes.append(
            {
                "text": None,
                "bbox": {
                    "x": comp["x"],
                    "y": comp["y"],
                    "width": w,
                    "height": h,
                },
                "confidence": 0.0,
            }
        )
    boxes.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))
    return boxes


def _text_bbox_center(item: dict[str, Any]) -> tuple[float, float]:
    bbox = item.get("bbox", {})
    try:
        x = float(bbox.get("x", 0.0))
        y = float(bbox.get("y", 0.0))
        w = float(bbox.get("width", 0.0))
        h = float(bbox.get("height", 0.0))
    except (TypeError, ValueError):
        return (0.0, 0.0)
    return (x + w / 2.0, y + h / 2.0)


def _text_bbox_width(item: dict[str, Any]) -> float:
    bbox = item.get("bbox", {})
    try:
        return float(bbox.get("width", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _text_bbox_height(item: dict[str, Any]) -> float:
    bbox = item.get("bbox", {})
    try:
        return float(bbox.get("height", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _panel_label_from_text(text: str) -> str | None:
    lowered = text.strip().lower()
    if lowered.startswith("a3"):
        return "A3"
    if lowered.startswith("a4"):
        return "A4"
    if lowered.startswith("a5"):
        return "A5"
    return None


def _assign_roles_3gpp(
    text_items: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[str | None, dict[str, Any] | None, list[dict[str, Any]]]:
    title: str | None = None
    title_style: dict[str, Any] | None = None
    top_candidates = [
        item
        for item in text_items
        if float(item.get("bbox", {}).get("y", 0.0)) < height * 0.1
        and _text_bbox_width(item) > width * 0.5
    ]
    if not top_candidates:
        top_candidates = [
            item
            for item in text_items
            if float(item.get("bbox", {}).get("y", 0.0)) < height * 0.14
            and _text_bbox_width(item) > width * 0.45
        ]
    if top_candidates:
        top_sorted = sorted(
            top_candidates,
            key=lambda item: (
                float(item.get("bbox", {}).get("y", 0.0)),
                float(item.get("bbox", {}).get("x", 0.0)),
            ),
        )
        lines: list[str] = []
        font_sizes: list[float] = []
        min_x = None
        max_x = None
        min_y = None
        y_positions: list[float] = []
        for item in top_sorted:
            text_value = _clean_text(str(item.get("content") or item.get("text") or ""))
            if not text_value:
                continue
            if not lines or text_value != lines[-1]:
                lines.append(text_value)
            try:
                font_sizes.append(float(item.get("font_size", 0.0)))
            except (TypeError, ValueError):
                pass
            bbox = item.get("bbox", {})
            try:
                x0 = float(bbox.get("x", 0.0))
                x1 = float(bbox.get("x", 0.0)) + float(bbox.get("width", 0.0))
                y0 = float(bbox.get("y", 0.0))
            except (TypeError, ValueError):
                x0 = x1 = y0 = 0.0
            min_x = x0 if min_x is None else min(min_x, x0)
            max_x = x1 if max_x is None else max(max_x, x1)
            min_y = y0 if min_y is None else min(min_y, y0)
            try:
                y_positions.append(float(item.get("y", 0.0)))
            except (TypeError, ValueError):
                pass
        if lines:
            title = "\n".join(lines)
            if font_sizes:
                font_sizes.sort()
                font_size = font_sizes[len(font_sizes) // 2]
            else:
                font_size = max(12.0, height * 0.02)
            anchor = "start"
            title_x = min_x if min_x is not None else 10.0
            if min_x is not None and max_x is not None:
                center_x = (min_x + max_x) / 2.0
                if abs(center_x - width / 2.0) <= width * 0.1:
                    anchor = "middle"
                    title_x = width / 2.0
            title_y = min(y_positions) if y_positions else (min_y + font_size if min_y is not None else 20.0)
            title_style = {
                "x": float(title_x),
                "y": float(title_y),
                "font_size": float(font_size),
                "font_weight": "bold",
                "anchor": anchor,
            }
        for item in top_candidates:
            item["role"] = "title"
            item["anchor"] = "middle"
            item["x"] = float(width) / 2.0
            item["render"] = False
        for item in text_items:
            if item.get("roi_id") == "title" and item.get("role") != "title":
                item["render"] = False
    for panel in panels:
        px = panel["x"]
        py = panel["y"]
        pw = panel["width"]
        ph = panel["height"]
        for item in text_items:
            cx, cy = _text_bbox_center(item)
            if (
                px <= cx <= px + pw
                and py <= cy <= py + ph * 0.3
                and _text_bbox_width(item) <= pw * 0.4
            ):
                item["role"] = "panel_label"
                item["anchor"] = "start"
                item["render"] = False
    for item in text_items:
        if item.get("role") == "panel_label" or item.get("role") == "title":
            continue
        cx, cy = _text_bbox_center(item)
        matched = False
        for panel in panels:
            if (
                panel["x"] <= cx <= panel["x"] + panel["width"]
                and panel["y"] + panel["height"] * 0.8 <= cy <= panel["y"] + panel["height"]
            ):
                item["role"] = "axis_label"
                item["anchor"] = "middle"
                matched = True
                break
        if not matched:
            item.setdefault("role", "annotation")
    return title, title_style, text_items


def _merge_stacked_text_items(
    text_items: list[dict[str, Any]],
    roi_prefix: str,
    max_gap_ratio: float = 1.8,
    min_overlap_ratio: float = 0.6,
) -> list[dict[str, Any]]:
    if not text_items:
        return text_items
    by_roi: dict[str, list[dict[str, Any]]] = {}
    for item in text_items:
        roi_id = str(item.get("roi_id") or "")
        if not roi_id.startswith(roi_prefix):
            continue
        if item.get("render") is False:
            continue
        by_roi.setdefault(roi_id, []).append(item)

    for roi_id, items in by_roi.items():
        items.sort(key=lambda item: float(item.get("bbox", {}).get("y", 0.0)))
        idx = 0
        while idx < len(items):
            group = [items[idx]]
            idx += 1
            while idx < len(items):
                prev = group[-1]
                curr = items[idx]
                prev_bbox = prev.get("bbox", {})
                curr_bbox = curr.get("bbox", {})
                try:
                    prev_x0 = float(prev_bbox.get("x", 0.0))
                    prev_x1 = prev_x0 + float(prev_bbox.get("width", 0.0))
                    curr_x0 = float(curr_bbox.get("x", 0.0))
                    curr_x1 = curr_x0 + float(curr_bbox.get("width", 0.0))
                    prev_h = float(prev_bbox.get("height", 0.0))
                    curr_h = float(curr_bbox.get("height", 0.0))
                    prev_y = float(prev.get("y", 0.0))
                    curr_y = float(curr.get("y", 0.0))
                except (TypeError, ValueError):
                    break
                overlap = max(0.0, min(prev_x1, curr_x1) - max(prev_x0, curr_x0))
                min_width = max(min(prev_x1 - prev_x0, curr_x1 - curr_x0), 1.0)
                overlap_ratio = overlap / min_width
                max_gap = max(prev_h, curr_h) * max_gap_ratio
                if overlap_ratio < min_overlap_ratio or (curr_y - prev_y) > max_gap:
                    break
                if prev.get("role") != curr.get("role"):
                    break
                group.append(curr)
                idx += 1
            if len(group) <= 1:
                continue
            content_lines = [_clean_text(str(item.get("content") or item.get("text") or "")) for item in group]
            content_lines = [line for line in content_lines if line]
            if not content_lines:
                continue
            first = group[0]
            first["content"] = "\n".join(content_lines)
            first["text"] = first["content"]
            first["baseline_group"] = f"stacked_{roi_id}_{idx}"
            for item in group[1:]:
                item["render"] = False
        for item in items:
            if item.get("render") is False:
                continue
            item["render"] = True
    return text_items


def _normalize_panel_mid_text(text_items: list[dict[str, Any]]) -> None:
    for item in text_items:
        roi_id = str(item.get("roi_id") or "")
        if not roi_id.startswith("panel_mid"):
            continue
        content = str(item.get("content") or item.get("text") or "")
        if not content:
            continue
        lines = []
        changed = False
        for line in content.splitlines():
            tokens = line.split()
            if len(tokens) > 1 and tokens[0].isdigit():
                if any(token.isalpha() for token in tokens[1:]):
                    tokens = tokens[1:]
                    changed = True
            lines.append(" ".join(tokens))
        if changed:
            normalized = "\n".join([line for line in lines if line])
            if normalized:
                item["content"] = normalized
                item["text"] = normalized


def _normalize_curve_labels_3gpp(text_items: list[dict[str, Any]]) -> None:
    for item in text_items:
        role = str(item.get("role") or "")
        if role == "curve_label_serving":
            item["content"] = "Serving beam"
            item["text"] = item["content"]
        elif role == "curve_label_neighbor":
            item["content"] = "Neighbor/Target beam"
            item["text"] = item["content"]


def _snap_grid(value: float, grid: float) -> float:
    if grid <= 0:
        return value
    return round(value / grid) * grid


def _text_layout_config(adaptive: dict[str, Any] | None, height: int) -> dict[str, float]:
    cfg = adaptive.get("text_layout") if adaptive else None
    if not isinstance(cfg, dict):
        cfg = {}
    grid = float(cfg.get("grid_px", 5.0))
    baseline_tol = float(cfg.get("baseline_tolerance_px", 2.0))
    ref_height = float(cfg.get("font_scale_ref_height", 360.0))
    scale_min = float(cfg.get("font_scale_min", 0.8))
    scale_max = float(cfg.get("font_scale_max", 3.0))
    scale = height / ref_height if ref_height > 0 else 1.0
    scale = max(scale_min, min(scale_max, scale))
    return {
        "grid_px": grid,
        "baseline_tolerance_px": baseline_tol,
        "font_scale": scale,
    }


def _baseline_groups(text_items: list[dict[str, Any]], tolerance: float) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in text_items:
        group = item.get("baseline_group")
        if isinstance(group, str) and group:
            groups.setdefault(group, []).append(item)
    if groups:
        return groups
    candidates = []
    for idx, item in enumerate(text_items):
        try:
            y = float(item.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        candidates.append((idx, y))
    candidates.sort(key=lambda pair: pair[1])
    clusters: list[list[int]] = []
    current: list[int] = []
    last_y = None
    for idx, y in candidates:
        if last_y is None or abs(y - last_y) <= tolerance:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
        last_y = y
    if current:
        clusters.append(current)
    for group_idx, indices in enumerate(clusters):
        group_key = f"baseline_{group_idx}"
        groups[group_key] = [text_items[idx] for idx in indices]
        for item in groups[group_key]:
            item["baseline_group"] = group_key
    return groups


def _apply_text_layout(
    text_items: list[dict[str, Any]],
    template_id: str,
    width: int,
    height: int,
    adaptive: dict[str, Any] | None,
) -> None:
    if not text_items:
        return
    layout_cfg = _text_layout_config(adaptive, height)
    grid = layout_cfg["grid_px"]
    baseline_tol = layout_cfg["baseline_tolerance_px"]
    font_scale = layout_cfg["font_scale"]
    role_styles = {
        "title": {"font_size": 16, "font_weight": "bold", "anchor": "middle"},
        "panel_label": {"font_size": 12, "font_weight": "bold", "anchor": "start"},
        "axis_label": {"font_size": 10, "font_weight": "normal", "anchor": "middle"},
        "axis_label_x": {"font_size": 10, "font_weight": "normal", "anchor": "middle"},
        "axis_label_y": {"font_size": 10, "font_weight": "normal", "anchor": "middle"},
        "legend": {"font_size": 9, "font_weight": "normal", "anchor": "start"},
        "node_label": {"font_size": 11, "font_weight": "normal", "anchor": "start"},
        "curve_label_serving": {"font_size": 9, "font_weight": "normal", "anchor": "start"},
        "curve_label_neighbor": {"font_size": 9, "font_weight": "normal", "anchor": "start"},
        "annotation": {"font_size": 7, "font_weight": "normal", "anchor": "start"},
    }
    for item in text_items:
        role = str(item.get("role") or "annotation")
        style = role_styles.get(role, role_styles["annotation"])
        item.setdefault("font_size", style["font_size"] * font_scale)
        item.setdefault("font_weight", style["font_weight"])
        item.setdefault("anchor", style["anchor"])
        item.setdefault("font_family", DEFAULT_FONT_FAMILY)
        item.setdefault("dominant_baseline", "alphabetic")
        if role == "title" and "x" not in item:
            item["x"] = float(width) / 2.0

    groups = _baseline_groups(text_items, baseline_tol)
    for group_items in groups.values():
        if not group_items:
            continue
        baselines = []
        for item in group_items:
            try:
                baselines.append(float(item.get("y", 0.0)))
            except (TypeError, ValueError):
                continue
        if not baselines:
            continue
        baseline = sum(baselines) / len(baselines)
        baseline = _snap_grid(baseline, grid)
        for item in group_items:
            try:
                item["y"] = baseline
                item["x"] = _snap_grid(float(item.get("x", 0.0)), grid)
            except (TypeError, ValueError):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, dict):
                continue
            width_val = float(bbox.get("width", 0.0))
            height_val = float(bbox.get("height", 0.0))
            anchor = str(item.get("anchor") or "start")
            if anchor == "middle":
                x0 = float(item.get("x", 0.0)) - width_val / 2.0
            elif anchor == "end":
                x0 = float(item.get("x", 0.0)) - width_val
            else:
                x0 = float(item.get("x", 0.0))
            y0 = baseline - height_val
            x0 = _snap_grid(x0, grid)
            y0 = _snap_grid(y0, grid)
            bbox.update(
                {
                    "x": x0,
                    "y": y0,
                    "width": width_val,
                    "height": height_val,
                    "x0": x0,
                    "y0": y0,
                    "x1": x0 + width_val,
                    "y1": y0 + height_val,
                }
            )


def _estimate_text_color_3gpp(rgba: np.ndarray, bbox: dict[str, Any]) -> str | None:
    try:
        x0 = int(max(float(bbox.get("x", 0.0)), 0.0))
        y0 = int(max(float(bbox.get("y", 0.0)), 0.0))
        x1 = int(float(bbox.get("x", 0.0)) + float(bbox.get("width", 0.0)))
        y1 = int(float(bbox.get("y", 0.0)) + float(bbox.get("height", 0.0)))
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    x1 = min(x1, rgba.shape[1])
    y1 = min(y1, rgba.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    region = rgba[y0:y1, x0:x1, :3]
    if region.size == 0:
        return None
    hue, sat, val = _rgb_to_hsv(region)
    mask = (sat > 0.3) & (val > 0.2) & (val < 0.9)
    if mask.sum() < 6:
        return None
    hue_vals = hue[mask]
    median_hue = float(np.median(hue_vals)) if hue_vals.size else None
    if median_hue is None:
        return None
    if _hue_distance(median_hue, 220.0) <= 25.0:
        return "#2b6cb0"
    if _hue_distance(median_hue, 30.0) <= 20.0:
        return "#dd6b20"
    return None


def _assign_text_colors_3gpp(text_items: list[dict[str, Any]], rgba: np.ndarray) -> None:
    for item in text_items:
        if item.get("render") is False:
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, dict):
            continue
        color = _estimate_text_color_3gpp(rgba, bbox)
        if color:
            item["fill"] = color


def _enforce_curve_label_colors_3gpp(text_items: list[dict[str, Any]]) -> None:
    for item in text_items:
        role = str(item.get("role") or "")
        if role == "curve_label_serving":
            item["fill"] = "#2b6cb0"
        elif role == "curve_label_neighbor":
            item["fill"] = "#dd6b20"


def _assign_roles_lineplot(
    text_items: list[dict[str, Any]], plot: dict[str, Any], width: int, height: int
) -> tuple[str | None, str | None, str | None, list[str]]:
    title: str | None = None
    axis_x: str | None = None
    axis_y: str | None = None
    legend_labels: list[str] = []

    top_candidates = [
        item
        for item in text_items
        if float(item.get("bbox", {}).get("y", 0.0)) < height * 0.1
        and _text_bbox_width(item) > width * 0.5
    ]
    if top_candidates:
        title_item = sorted(
            top_candidates,
            key=lambda item: (
                float(item.get("bbox", {}).get("y", 0.0)),
                float(item.get("bbox", {}).get("x", 0.0)),
            ),
        )[0]
        title = _clean_text(str(title_item.get("content") or title_item.get("text") or ""))
        title_item["role"] = "title"
        title_item["render"] = False

    for item in text_items:
        if item.get("role") == "title":
            continue
        cx, cy = _text_bbox_center(item)
        if plot["x"] <= cx <= plot["x"] + plot["width"] and cy >= plot["y"] + plot["height"]:
            axis_x = _clean_text(str(item.get("content") or item.get("text") or ""))
            item["role"] = "axis_label_x"
            item["anchor"] = "middle"
            item["render"] = False
        elif cx <= plot["x"] and plot["y"] <= cy <= plot["y"] + plot["height"]:
            axis_y = _clean_text(str(item.get("content") or item.get("text") or ""))
            item["role"] = "axis_label_y"
            item["anchor"] = "middle"
            item["render"] = False
        elif cy <= plot["y"] and cx >= plot["x"] + plot["width"] * 0.6:
            label = _clean_text(str(item.get("content") or item.get("text") or ""))
            if label:
                legend_labels.append(label)
                item["role"] = "legend"
                item["render"] = False

    return title, axis_x, axis_y, legend_labels


def _assign_roles_flow(
    text_items: list[dict[str, Any]], nodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    for item in text_items:
        cx, cy = _text_bbox_center(item)
        for node in nodes:
            if (
                node["x"] <= cx <= node["x"] + node["width"]
                and node["y"] <= cy <= node["y"] + node["height"]
            ):
                item["role"] = "node_label"
                item["anchor"] = "middle"
                break
        item.setdefault("role", "annotation")
    return text_items
