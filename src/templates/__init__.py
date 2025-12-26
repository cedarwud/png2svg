"""Template renderers."""

from .t_3gpp_events_3panel import render as render_3gpp_events_3panel
from .t_procedure_flow import render as render_procedure_flow
from .t_performance_lineplot import render as render_performance_lineplot
from .t_project_architecture_v1 import render as render_project_architecture_v1

__all__ = [
    "render_3gpp_events_3panel",
    "render_procedure_flow",
    "render_performance_lineplot",
    "render_project_architecture_v1",
]
