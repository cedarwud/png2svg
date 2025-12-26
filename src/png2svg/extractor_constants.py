from __future__ import annotations

from pathlib import Path


TEMPLATE_ALIASES = {
    "3gpp_3panel": "t_3gpp_events_3panel",
    "t_3gpp_events_3panel": "t_3gpp_events_3panel",
    "procedure_flow": "t_procedure_flow",
    "t_procedure_flow": "t_procedure_flow",
    "performance_lineplot": "t_performance_lineplot",
    "t_performance_lineplot": "t_performance_lineplot",
    "project_architecture_v1": "t_project_architecture_v1",
    "t_project_architecture_v1": "t_project_architecture_v1",
}

DEFAULT_SERIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
DEFAULT_DASHARRAY = [6, 4]
DEFAULT_ADAPTIVE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "extract_adaptive.v1.yaml"
