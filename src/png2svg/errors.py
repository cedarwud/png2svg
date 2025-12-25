from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Png2SvgError(Exception):
    code: str
    message: str
    hint: str

    def __str__(self) -> str:
        return self.message
