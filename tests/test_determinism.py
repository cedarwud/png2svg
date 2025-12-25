from __future__ import annotations

from pathlib import Path

from png2svg import render_svg


def test_png2svg_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_png = root / "samples" / "input.png"
    params = root / "samples" / "t_3gpp_events_3panel.json"
    output_a = tmp_path / "a.svg"
    output_b = tmp_path / "b.svg"

    render_svg(input_png, params, output_a)
    render_svg(input_png, params, output_b)

    assert output_a.read_bytes() == output_b.read_bytes()
