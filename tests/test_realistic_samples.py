from __future__ import annotations

from pathlib import Path

from png2svg import convert_png


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples" / "realistic"


def test_realistic_noisy_samples_convert(tmp_path: Path) -> None:
    cases = [
        ("lineplot_noisy.png", "t_performance_lineplot"),
        ("3gpp_noisy.png", "t_3gpp_events_3panel"),
    ]
    for filename, template_id in cases:
        input_png = SAMPLES / filename
        output_svg = tmp_path / f"{template_id}.svg"
        result = convert_png(
            input_png,
            output_svg,
            debug_dir=tmp_path / f"debug_{template_id}",
            topk=1,
            force_template=template_id,
            gate_rmse_max=0.75,
            gate_bad_pixel_max=0.65,
        )
        assert result["status"] == "pass"
