from __future__ import annotations

from typing import Any


def _snap_value(value: float, grid: float) -> float:
    return round(value / grid) * grid


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _round_value(value: float, places: int = 3) -> float:
    return round(value, places)


def _snap_box(box: dict[str, Any], grid: float, min_size: float = 1.0) -> dict[str, Any]:
    x = _snap_value(float(box.get("x", 0)), grid)
    y = _snap_value(float(box.get("y", 0)), grid)
    width = _snap_value(float(box.get("width", min_size)), grid)
    height = _snap_value(float(box.get("height", min_size)), grid)
    if width < min_size:
        width = min_size
    if height < min_size:
        height = min_size
    box.update({"x": x, "y": y, "width": width, "height": height})
    return box


def _snap_point(point: dict[str, Any], grid: float) -> dict[str, Any]:
    if "x" in point:
        point["x"] = _snap_value(float(point["x"]), grid)
    if "y" in point:
        point["y"] = _snap_value(float(point["y"]), grid)
    return point


def _point_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    if start == end:
        dx = point[0] - start[0]
        dy = point[1] - start[1]
        return (dx * dx + dy * dy) ** 0.5
    x0, y0 = point
    x1, y1 = start
    x2, y2 = end
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return num / den


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    max_dist = 0.0
    index = 0
    for idx, point in enumerate(points[1:-1], start=1):
        dist = _point_distance(point, start, end)
        if dist > max_dist:
            max_dist = dist
            index = idx
    if max_dist > epsilon:
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return left[:-1] + right
    return [start, end]


def _simplify_points(
    points: list[tuple[float, float]],
    max_points: int,
    min_points: int,
    epsilon: float,
) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    simplified = _rdp(points, epsilon)
    step = 0
    while len(simplified) > max_points and step < 6:
        epsilon *= 1.6
        simplified = _rdp(points, epsilon)
        step += 1
    if len(simplified) > max_points:
        keep = [points[0]]
        if max_points > 2:
            stride = max(1, int((len(points) - 2) / (max_points - 2)))
            keep.extend(points[1:-1:stride])
        keep.append(points[-1])
        simplified = keep[:max_points]
    if len(simplified) < min_points:
        if len(points) >= min_points:
            simplified = points[:min_points]
        else:
            simplified = points
    return simplified


def _simplify_collinear(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    for idx in range(1, len(points) - 1):
        prev = simplified[-1]
        current = points[idx]
        nxt = points[idx + 1]
        dx1 = current[0] - prev[0]
        dy1 = current[1] - prev[1]
        dx2 = nxt[0] - current[0]
        dy2 = nxt[1] - current[1]
        if abs(dx1) <= tolerance and abs(dx2) <= tolerance:
            continue
        if abs(dy1) <= tolerance and abs(dy2) <= tolerance:
            continue
        simplified.append(current)
    simplified.append(points[-1])
    return simplified


def _align_baselines(text_blocks: list[dict[str, Any]], tolerance: float, grid: float) -> None:
    baselines: list[float] = []
    for block in text_blocks:
        bbox = block.get("bbox")
        if not isinstance(bbox, dict):
            continue
        y = bbox.get("y")
        height = bbox.get("height")
        if y is None or height is None:
            continue
        baselines.append(float(y) + float(height))
    if not baselines:
        return
    baselines.sort()
    clusters: list[list[float]] = [[baselines[0]]]
    for value in baselines[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    cluster_centers = [sum(cluster) / len(cluster) for cluster in clusters]
    for block in text_blocks:
        bbox = block.get("bbox")
        if not isinstance(bbox, dict):
            continue
        y = bbox.get("y")
        height = bbox.get("height")
        if y is None or height is None:
            continue
        baseline = float(y) + float(height)
        nearest = min(cluster_centers, key=lambda value: abs(value - baseline))
        snapped = _snap_value(nearest, grid)
        bbox["y"] = snapped - float(height)


def _normalize_text_items(text_items: list[dict[str, Any]], grid: float) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in text_items:
        try:
            item["x"] = _snap_value(float(item.get("x", 0.0)), grid)
            item["y"] = _snap_value(float(item.get("y", 0.0)), grid)
        except (TypeError, ValueError):
            continue
        group = item.get("baseline_group")
        if isinstance(group, str) and group:
            groups.setdefault(group, []).append(item)

    for group_items in groups.values():
        if not group_items:
            continue
        baseline = sum(float(item.get("y", 0.0)) for item in group_items) / len(group_items)
        snapped = _snap_value(baseline, grid)
        for item in group_items:
            item["y"] = snapped


def _normalize_3gpp(params: dict[str, Any], grid: float) -> dict[str, Any]:
    canvas = params.get("canvas", {})
    canvas_width = float(canvas.get("width", 0))
    canvas_height = float(canvas.get("height", 0))
    for panel in params.get("panels", []):
        _snap_box(panel, grid)
        panel["x"] = _clamp(panel["x"], 0, max(canvas_width - panel["width"], 0))
        panel["y"] = _clamp(panel["y"], 0, max(canvas_height - panel["height"], 0))

    t_start = params.get("t_start_ratio")
    if t_start is not None:
        params["t_start_ratio"] = _round_value(_clamp(float(t_start), 0.0, 1.0), 3)
    t_trigger = params.get("t_trigger_ratio")
    if t_trigger is not None:
        params["t_trigger_ratio"] = _round_value(_clamp(float(t_trigger), 0.0, 1.0), 3)

    curves = params.get("curves", {})
    for curve_key in ("serving", "neighbor"):
        raw_points = curves.get(curve_key, [])
        points: list[tuple[float, float]] = []
        for point in raw_points:
            if isinstance(point, dict) and "x" in point and "y" in point:
                points.append((float(point["x"]), float(point["y"])))
        if not points:
            continue
        max_range = max(
            max(p[0] for p in points) - min(p[0] for p in points),
            max(p[1] for p in points) - min(p[1] for p in points),
            1.0,
        )
        simplified = _simplify_points(points, max_points=4, min_points=3, epsilon=max_range * 0.02)
        curves[curve_key] = [
            {"x": _round_value(_clamp(point[0], 0.0, 1.0), 3), "y": _round_value(_clamp(point[1], 0.0, 1.0), 3)}
            for point in simplified
        ]
    params["curves"] = curves
    return params


def _normalize_lineplot(params: dict[str, Any], grid: float) -> dict[str, Any]:
    axes = params.get("axes")
    if isinstance(axes, dict) and isinstance(axes.get("plot"), dict):
        _snap_box(axes["plot"], grid)

    for series in params.get("series", []):
        if series.get("dashed") and not series.get("dasharray"):
            stroke_width = float(series.get("stroke_width", 2))
            dash = max(4.0, stroke_width * 3.0)
            gap = max(3.0, stroke_width * 2.0)
            series["dasharray"] = [dash, gap]

        raw_points = series.get("points", [])
        points: list[tuple[float, float]] = []
        for point in raw_points:
            if isinstance(point, dict) and "x" in point and "y" in point:
                points.append((float(point["x"]), float(point["y"])))
        if not points:
            continue
        max_range = max(
            max(p[0] for p in points) - min(p[0] for p in points),
            max(p[1] for p in points) - min(p[1] for p in points),
            1.0,
        )
        simplified = _simplify_points(points, max_points=10, min_points=2, epsilon=max_range * 0.01)
        series["points"] = [
            {"x": _round_value(point[0], 4), "y": _round_value(point[1], 4)}
            for point in simplified
        ]
    return params


def _normalize_flow(params: dict[str, Any], grid: float) -> dict[str, Any]:
    for lane in params.get("lanes", []):
        _snap_box(lane, grid)

    nodes = params.get("nodes", [])
    for node in nodes:
        _snap_box(node, grid)
        if "rx" in node:
            node["rx"] = _snap_value(float(node["rx"]), grid)
        if "ry" in node:
            node["ry"] = _snap_value(float(node["ry"]), grid)

    nodes_by_id = {node.get("id"): node for node in nodes if isinstance(node, dict)}
    for edge in params.get("edges", []):
        if edge.get("dashed") and not edge.get("dasharray"):
            stroke_width = float(edge.get("stroke_width", 2))
            dash = max(4.0, stroke_width * 3.0)
            gap = max(3.0, stroke_width * 2.0)
            edge["dasharray"] = [dash, gap]
        points_raw = edge.get("points")
        if not isinstance(points_raw, list):
            continue
        points: list[tuple[float, float]] = []
        for point in points_raw:
            if isinstance(point, dict) and "x" in point and "y" in point:
                points.append((float(point["x"]), float(point["y"])))
        if len(points) < 2:
            continue
        points = [(float(_snap_value(x, grid)), float(_snap_value(y, grid))) for x, y in points]
        points = _simplify_collinear(points, tolerance=grid * 0.4)
        from_node = nodes_by_id.get(edge.get("from"))
        to_node = nodes_by_id.get(edge.get("to"))
        if isinstance(from_node, dict):
            start = (
                float(from_node.get("x", 0)) + float(from_node.get("width", 0)) / 2,
                float(from_node.get("y", 0)) + float(from_node.get("height", 0)) / 2,
            )
            points[0] = (float(_snap_value(start[0], grid)), float(_snap_value(start[1], grid)))
        if isinstance(to_node, dict):
            end = (
                float(to_node.get("x", 0)) + float(to_node.get("width", 0)) / 2,
                float(to_node.get("y", 0)) + float(to_node.get("height", 0)) / 2,
            )
            points[-1] = (float(_snap_value(end[0], grid)), float(_snap_value(end[1], grid)))
        edge["points"] = [{"x": _round_value(x, 3), "y": _round_value(y, 3)} for x, y in points]
    return params


def normalize_params(template_id: str, params: dict[str, Any], grid: float = 1.0) -> dict[str, Any]:
    if template_id == "t_3gpp_events_3panel":
        params = _normalize_3gpp(params, grid)
    elif template_id == "t_performance_lineplot":
        params = _normalize_lineplot(params, grid)
    elif template_id == "t_procedure_flow":
        params = _normalize_flow(params, grid)

    extracted = params.get("extracted")
    if isinstance(extracted, dict):
        text_blocks = extracted.get("text_blocks")
        if isinstance(text_blocks, list):
            _align_baselines(text_blocks, tolerance=4.0, grid=grid)
        text_items = extracted.get("text_items")
        if isinstance(text_items, list):
            _normalize_text_items(text_items, grid=grid)
        extracted["normalization"] = {
            "grid": grid,
        }
    return params
