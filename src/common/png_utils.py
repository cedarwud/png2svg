from __future__ import annotations

from pathlib import Path

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def has_png_magic(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(PNG_MAGIC)) == PNG_MAGIC
    except OSError:
        return False
