"""Session notebook — records analytics actions for review and export."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class NotebookEntry:
    timestamp: str
    action: str          # "load", "transform", "visualize", "analyze", etc.
    description: str     # Human-readable description
    code: str           # The generated pandas/sklearn code
    output_type: str    # "table", "figure", "text", "stats"
    output_summary: str # Brief summary of the output


class SessionNotebook:
    """Accumulates entries during a workbench session."""

    def __init__(self):
        self.entries: list[NotebookEntry] = []

    def record(self, action: str, description: str, code: str,
               output_type: str = "text", output_summary: str = "") -> None:
        """Record an analytics action."""
        self.entries.append(NotebookEntry(
            timestamp=datetime.now().isoformat(),
            action=action,
            description=description,
            code=code,
            output_type=output_type,
            output_summary=output_summary,
        ))

    def get_entries(self) -> list[NotebookEntry]:
        return list(self.entries)

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)
