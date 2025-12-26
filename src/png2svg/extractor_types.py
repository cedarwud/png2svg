from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExtractIssue:
    code: str
    message: str
    hint: str
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"code": self.code, "message": self.message, "hint": self.hint}
        if self.context:
            payload["context"] = self.context
        return payload
