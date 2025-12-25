from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from common.svg_builder import DEFAULT_FONT_FAMILY, SvgBuilder
from png2svg.errors import Png2SvgError


PADDING = 12
FONT_SIZE = 12
LABEL_SIZE = 10
LINE_HEIGHT = 14


@dataclass(frozen=True)
class Lane:
    lane_id: str
    label: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Node:
    node_id: str
    x: float
    y: float
    width: float
    height: float
    rx: float
    ry: float
    lines: list[str]

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class Edge:
    from_id: str
    to_id: str
    label: str | None
    dashed: bool
    points: list[tuple[float, float]] | None


def _require_list(params: dict[str, Any], key: str) -> list[Any]:
    value = params.get(key)
    if not isinstance(value, list):
        raise Png2SvgError(
            code="E2200_LIST_MISSING",
            message=f"'{key}' must be a list.",
            hint=f"Provide a list for '{key}'.",
        )
    return value


def _parse_lanes(params: dict[str, Any]) -> list[Lane]:
    lanes_raw = params.get("lanes")
    if lanes_raw is None:
        return []
    if not isinstance(lanes_raw, list):
        raise Png2SvgError(
            code="E2201_LANES_TYPE",
            message="lanes must be a list of lane objects.",
            hint="Provide lanes as a list of objects with id/label/x/y/width/height.",
        )
    lanes: list[Lane] = []
    for raw in lanes_raw:
        if not isinstance(raw, dict):
            raise Png2SvgError(
                code="E2202_LANES_ENTRY",
                message="Each lane must be an object.",
                hint="Use {id, label, x, y, width, height} for each lane.",
            )
        lane_id = str(raw.get("id", "")).strip()
        if not lane_id:
            raise Png2SvgError(
                code="E2203_LANES_ID",
                message="Lane id is required.",
                hint="Provide a non-empty lane id.",
            )
        label = str(raw.get("label", lane_id)).strip() or lane_id
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            width = float(raw["width"])
            height = float(raw["height"])
        except KeyError as exc:
            raise Png2SvgError(
                code="E2204_LANES_GEOMETRY",
                message=f"Lane missing geometry field: {exc}",
                hint="Provide x, y, width, height for each lane.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise Png2SvgError(
                code="E2205_LANES_GEOMETRY",
                message="Lane geometry must be numeric.",
                hint="Use numeric x/y/width/height.",
            ) from exc
        lanes.append(Lane(lane_id=lane_id, label=label, x=x, y=y, width=width, height=height))
    return lanes


def _parse_nodes(params: dict[str, Any]) -> list[Node]:
    nodes_raw = _require_list(params, "nodes")
    nodes: list[Node] = []
    for raw in nodes_raw:
        if not isinstance(raw, dict):
            raise Png2SvgError(
                code="E2210_NODE_TYPE",
                message="Each node must be an object.",
                hint="Provide node objects with id/x/y/width/height/text.",
            )
        node_id = str(raw.get("id", "")).strip()
        if not node_id:
            raise Png2SvgError(
                code="E2211_NODE_ID",
                message="Node id is required.",
                hint="Provide a non-empty node id.",
            )
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            width = float(raw["width"])
            height = float(raw["height"])
        except KeyError as exc:
            raise Png2SvgError(
                code="E2212_NODE_GEOMETRY",
                message=f"Node missing geometry field: {exc}",
                hint="Provide x, y, width, height for each node.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise Png2SvgError(
                code="E2213_NODE_GEOMETRY",
                message="Node geometry must be numeric.",
                hint="Use numeric x/y/width/height values.",
            ) from exc
        rx = raw.get("rx", 8)
        ry = raw.get("ry", 8)
        try:
            rx = float(rx)
            ry = float(ry)
        except (TypeError, ValueError) as exc:
            raise Png2SvgError(
                code="E2214_NODE_RADIUS",
                message="Node corner radius must be numeric.",
                hint="Provide numeric rx/ry values.",
            ) from exc
        text = raw.get("text", node_id)
        lines = _normalize_lines(text, node_id)
        nodes.append(
            Node(
                node_id=node_id,
                x=x,
                y=y,
                width=width,
                height=height,
                rx=rx,
                ry=ry,
                lines=lines,
            )
        )
    return nodes


def _normalize_lines(text: Any, fallback: str) -> list[str]:
    if isinstance(text, list):
        lines = [str(item).strip() for item in text if str(item).strip()]
    else:
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        lines = [fallback]
    return lines


def _parse_edges(params: dict[str, Any]) -> list[Edge]:
    edges_raw = params.get("edges", [])
    if not isinstance(edges_raw, list):
        raise Png2SvgError(
            code="E2220_EDGE_TYPE",
            message="edges must be a list.",
            hint="Provide edges as a list of objects.",
        )
    edges: list[Edge] = []
    for raw in edges_raw:
        if not isinstance(raw, dict):
            raise Png2SvgError(
                code="E2221_EDGE_ENTRY",
                message="Each edge must be an object.",
                hint="Use {from, to, label, dashed, points}.",
            )
        from_id = str(raw.get("from", "")).strip()
        to_id = str(raw.get("to", "")).strip()
        if not from_id or not to_id:
            raise Png2SvgError(
                code="E2222_EDGE_IDS",
                message="Edge must include from/to ids.",
                hint="Provide from and to node ids.",
            )
        label = raw.get("label")
        if label is not None:
            label = str(label)
        dashed = bool(raw.get("dashed", False))
        points = None
        if raw.get("points") is not None:
            points_raw = raw.get("points")
            if not isinstance(points_raw, list) or len(points_raw) < 2:
                raise Png2SvgError(
                    code="E2223_EDGE_POINTS",
                    message="Edge points must be a list of at least 2 points.",
                    hint="Provide at least two points for a polyline edge.",
                )
            parsed_points: list[tuple[float, float]] = []
            for point in points_raw:
                if not isinstance(point, dict):
                    raise Png2SvgError(
                        code="E2224_EDGE_POINTS",
                        message="Edge points must be objects with x/y.",
                        hint="Use {x, y} for each point.",
                    )
                try:
                    x = float(point["x"])
                    y = float(point["y"])
                except KeyError as exc:
                    raise Png2SvgError(
                        code="E2225_EDGE_POINTS",
                        message=f"Edge point missing field: {exc}",
                        hint="Provide x and y for each edge point.",
                    ) from exc
                except (TypeError, ValueError) as exc:
                    raise Png2SvgError(
                        code="E2226_EDGE_POINTS",
                        message="Edge point values must be numeric.",
                        hint="Provide numeric x/y values.",
                    ) from exc
                parsed_points.append((x, y))
            points = parsed_points
        edges.append(
            Edge(
                from_id=from_id,
                to_id=to_id,
                label=label,
                dashed=dashed,
                points=points,
            )
        )
    return edges


def _edge_points(edge: Edge, nodes: dict[str, Node]) -> list[tuple[float, float]]:
    if edge.points:
        return edge.points
    if edge.from_id not in nodes or edge.to_id not in nodes:
        raise Png2SvgError(
            code="E2227_EDGE_NODE",
            message="Edge references unknown node id.",
            hint="Ensure edge from/to ids match node ids.",
        )
    from_node = nodes[edge.from_id]
    to_node = nodes[edge.to_id]
    if from_node.center_x <= to_node.center_x:
        start = (from_node.x + from_node.width, from_node.center_y)
        end = (to_node.x, to_node.center_y)
    else:
        start = (from_node.x, from_node.center_y)
        end = (to_node.x + to_node.width, to_node.center_y)
    return [start, end]


def _arrow_points(
    start: tuple[float, float],
    end: tuple[float, float],
    length: float = 10,
    width: float = 6,
) -> list[tuple[float, float]] | None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist <= 0.001:
        return None
    ux = dx / dist
    uy = dy / dist
    base_x = end[0] - ux * length
    base_y = end[1] - uy * length
    perp_x = -uy
    perp_y = ux
    half = width / 2
    p1 = (base_x + perp_x * half, base_y + perp_y * half)
    p2 = (base_x - perp_x * half, base_y - perp_y * half)
    return [end, p1, p2]


def _add_multiline_text(
    builder: SvgBuilder,
    lines: list[str],
    x: float,
    y: float,
    text_id: str,
    font_size: int = FONT_SIZE,
) -> None:
    text = builder.drawing.text(
        "",
        insert=(x, y),
        id=text_id,
        font_family=DEFAULT_FONT_FAMILY,
        font_size=font_size,
        fill="#000000",
    )
    for idx, line in enumerate(lines):
        if idx == 0:
            text.add(builder.drawing.tspan(line, x=[x], y=[y], id=f"{text_id}_line{idx}"))
        else:
            text.add(
                builder.drawing.tspan(line, x=[x], dy=[LINE_HEIGHT], id=f"{text_id}_line{idx}")
            )
    builder.groups["g_text"].add(text)


def _draw_lanes(builder: SvgBuilder, lanes: list[Lane]) -> None:
    annotations = builder.groups["g_annotations"]
    for lane in lanes:
        annotations.add(
            builder.drawing.rect(
                insert=(lane.x, lane.y),
                size=(lane.width, lane.height),
                rx=6,
                ry=6,
                fill="#f2f2f2",
                stroke="#000000",
                stroke_width=1,
            )
        )
        _add_multiline_text(
            builder,
            [lane.label],
            lane.x + 8,
            lane.y + 16,
            f"txt_lane_{lane.lane_id}",
            font_size=LABEL_SIZE,
        )


def _draw_nodes(builder: SvgBuilder, nodes: list[Node]) -> None:
    annotations = builder.groups["g_annotations"]
    for node in nodes:
        annotations.add(
            builder.drawing.rect(
                insert=(node.x, node.y),
                size=(node.width, node.height),
                rx=node.rx,
                ry=node.ry,
                fill="#ffffff",
                stroke="#000000",
                stroke_width=2,
                id=f"node_{node.node_id}",
            )
        )
        text_x = node.x + PADDING
        text_y = node.y + PADDING + FONT_SIZE
        _add_multiline_text(builder, node.lines, text_x, text_y, f"txt_node_{node.node_id}")


def _draw_edges(builder: SvgBuilder, edges: list[Edge], nodes: dict[str, Node]) -> None:
    curves = builder.groups["g_curves"]
    markers = builder.groups["g_markers"]
    for edge in edges:
        points = _edge_points(edge, nodes)
        stroke_kwargs = {"stroke": "#000000", "stroke_width": 2, "fill": "none"}
        if edge.dashed:
            stroke_kwargs["stroke_dasharray"] = "6,4"
            stroke_kwargs["class_"] = "dashed"
        if len(points) == 2:
            curves.add(
                builder.drawing.line(start=points[0], end=points[1], **stroke_kwargs)
            )
        else:
            curves.add(builder.drawing.polyline(points=points, **stroke_kwargs))

        arrow = _arrow_points(points[-2], points[-1])
        if arrow:
            markers.add(builder.drawing.polygon(points=arrow, fill="#000000"))

        if edge.label:
            mid_x = (points[0][0] + points[-1][0]) / 2
            mid_y = (points[0][1] + points[-1][1]) / 2 - 6
            builder.groups["g_text"].add(
                builder.drawing.text(
                    edge.label,
                    insert=(mid_x, mid_y),
                    id=f"txt_edge_{edge.from_id}_{edge.to_id}",
                    font_family=DEFAULT_FONT_FAMILY,
                    font_size=LABEL_SIZE,
                    fill="#000000",
                )
            )


def render(builder: SvgBuilder, params: dict[str, Any], canvas: tuple[int, int]) -> None:
    title = params.get("title")
    if title:
        builder.add_title(str(title))

    lanes = _parse_lanes(params)
    nodes = _parse_nodes(params)
    edges = _parse_edges(params)

    nodes_by_id = {node.node_id: node for node in nodes}
    _draw_lanes(builder, lanes)
    _draw_nodes(builder, nodes)
    _draw_edges(builder, edges, nodes_by_id)
