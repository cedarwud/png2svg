from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationIssue:
    code: str
    message: str
    hint: str
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"code": self.code, "message": self.message, "hint": self.hint}
        if self.context:
            data["context"] = self.context
        return data


@dataclass
class ValidationReport:
    status: str
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "stats": self.stats,
        }
