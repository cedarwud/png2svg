#!/usr/bin/env python3
from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="Extract PNGs from PDF.")


def _not_implemented(tool_name: str) -> None:
    typer.echo(
        f"ERROR E0001 NOT_IMPLEMENTED: {tool_name} is not implemented in Phase 0.",
        err=True,
    )
    typer.echo("HINT: Continue with Phase 1+ to add functionality.", err=True)
    raise typer.Exit(code=1)


@app.command()
def main() -> None:
    """Placeholder CLI entrypoint."""
    _not_implemented("pdf2png")


if __name__ == "__main__":
    app(prog_name="pdf2png")
