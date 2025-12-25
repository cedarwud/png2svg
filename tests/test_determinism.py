from __future__ import annotations

import base64
from pathlib import Path

from png2svg import render_svg


def _write_png(path: Path) -> None:
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
    )
    path.write_bytes(data)


def test_svg_output_is_deterministic(tmp_path: Path) -> None:
    input_png = tmp_path / "input.png"
    _write_png(input_png)
    params_path = tmp_path / "params.json"
    params_path.write_text(
        """{\n"""
        """  "template": "t_procedure_flow",\n"""
        """  "canvas": {"width": 400, "height": 200},\n"""
        """  "nodes": [\n"""
        """    {"id": "a", "x": 40, "y": 60, "width": 120, "height": 60, "text": "Start"},\n"""
        """    {"id": "b", "x": 220, "y": 60, "width": 120, "height": 60, "text": "End"}\n"""
        """  ],\n"""
        """  "edges": [\n"""
        """    {"from": "a", "to": "b", "label": "next"}\n"""
        """  ]\n"""
        """}\n"""
    )

    output_a = tmp_path / "out_a.svg"
    output_b = tmp_path / "out_b.svg"

    render_svg(input_png, params_path, output_a)
    render_svg(input_png, params_path, output_b)

    assert output_a.read_bytes() == output_b.read_bytes()
