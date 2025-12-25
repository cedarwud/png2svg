# Agent Rules (for Codex CLI)

## Golden Rule
Follow spec/00-constitution.md and spec/04-acceptance.md as hard constraints.

## Implementation Rules
- Python 3.10+
- Deterministic output: no randomness unless explicitly controlled
- Keep dependencies minimal; prefer pure python for XML parsing and SVG generation
- Provide CLI scripts under tools/ using Typer
- Prefer svgwrite or lxml (choose one and be consistent)

## Testing Rules
- All validators must have unit tests
- regress must be runnable in one command and fail non-zero on any failure

## UX Rules
- Error reports must include:
  - stable error code
  - human-readable message
  - hint for fixing
