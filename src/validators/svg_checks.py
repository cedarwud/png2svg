from __future__ import annotations

import re
from typing import Iterable


COMMAND_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
HEX_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
RGB_RE = re.compile(r"^rgba?\(([^)]+)\)$", re.IGNORECASE)
NUMBER_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_style(style: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for chunk in style.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def iter_with_style(element: Iterable) -> Iterable[tuple]:
    for node in element:
        style = parse_style(node.get("style", ""))
        yield node, style


def extract_property_values(node, style: dict[str, str], prop: str) -> list[str]:
    values: list[str] = []
    direct = node.get(prop)
    if direct:
        values.append(direct)
    style_value = style.get(prop)
    if style_value:
        values.append(style_value)
    return values


def parse_color(value: str) -> tuple[int, int, int] | str | None:
    raw = value.strip()
    lowered = raw.lower()
    if lowered in {"none", "transparent", "currentcolor", "inherit"}:
        return None
    if lowered.startswith("url("):
        return lowered
    hex_match = HEX_RE.match(raw)
    if hex_match:
        hex_value = hex_match.group(1)
        if len(hex_value) == 3:
            hex_value = "".join(ch * 2 for ch in hex_value)
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
        return (r, g, b)
    rgb_match = RGB_RE.match(raw)
    if rgb_match:
        parts = [p.strip() for p in rgb_match.group(1).split(",")]
        if len(parts) >= 3:
            rgb: list[int] = []
            for part in parts[:3]:
                if part.endswith("%"):
                    try:
                        rgb.append(round(float(part[:-1]) * 2.55))
                    except ValueError:
                        return lowered
                else:
                    try:
                        rgb.append(int(float(part)))
                    except ValueError:
                        return lowered
            return tuple(rgb)  # type: ignore[return-value]
    return lowered


def parse_number(value: str) -> float | None:
    match = NUMBER_RE.match(value)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def count_path_commands(d: str) -> int:
    return len(COMMAND_RE.findall(d))
