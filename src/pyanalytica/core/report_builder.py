"""Report builder — notebook-style report with code + markdown cells."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyanalytica.core.procedure import ProcedureRecorder


class CellType(Enum):
    """Type of cell in a report."""
    CODE = "code"
    MARKDOWN = "markdown"


@dataclass
class ReportCell:
    """A single cell in a report."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    order: int = 0
    cell_type: CellType = CellType.CODE
    enabled: bool = True
    # Code cell fields
    action: str = ""
    description: str = ""
    code: str = ""
    imports: list[str] = field(default_factory=list)
    # Markdown cell fields
    markdown: str = ""


class ReportBuilder:
    """Builds a notebook-style report from procedure steps and markdown cells."""

    def __init__(self) -> None:
        self._cells: list[ReportCell] = []
        self.title: str = "PyAnalytica Report"
        self.author: str = ""

    def get_cells(self) -> list[ReportCell]:
        return list(self._cells)

    def cell_count(self) -> int:
        return len(self._cells)

    def _renumber(self) -> None:
        for i, c in enumerate(self._cells):
            c.order = i + 1

    def import_from_recorder(self, recorder: ProcedureRecorder) -> int:
        """Import procedure steps as code cells. Returns number imported."""
        steps = recorder.get_steps()
        count = 0
        for s in steps:
            cell = ReportCell(
                order=len(self._cells) + 1,
                cell_type=CellType.CODE,
                enabled=s.enabled,
                action=s.action,
                description=s.description,
                code=s.code,
                imports=list(s.imports),
            )
            self._cells.append(cell)
            count += 1
        self._renumber()
        return count

    def add_markdown_cell(self, after_cell_id: str | None = None, markdown: str = "") -> ReportCell:
        """Insert a markdown cell. If after_cell_id is given, insert after that cell."""
        cell = ReportCell(
            cell_type=CellType.MARKDOWN,
            markdown=markdown,
        )
        if after_cell_id is None:
            self._cells.append(cell)
        else:
            idx = self._find_index(after_cell_id)
            if idx is not None:
                self._cells.insert(idx + 1, cell)
            else:
                self._cells.append(cell)
        self._renumber()
        return cell

    def add_title_cell(self) -> ReportCell:
        """Insert a title markdown cell at position 0."""
        cell = ReportCell(
            cell_type=CellType.MARKDOWN,
            markdown=f"# {self.title}\n\n*Author: {self.author or 'N/A'}*",
        )
        self._cells.insert(0, cell)
        self._renumber()
        return cell

    def remove_cell(self, cell_id: str) -> None:
        self._cells = [c for c in self._cells if c.id != cell_id]
        self._renumber()

    def move_cell(self, cell_id: str, direction: str) -> None:
        """Move a cell 'up' or 'down'."""
        idx = self._find_index(cell_id)
        if idx is None:
            return
        if direction == "up" and idx > 0:
            self._cells[idx], self._cells[idx - 1] = self._cells[idx - 1], self._cells[idx]
        elif direction == "down" and idx < len(self._cells) - 1:
            self._cells[idx], self._cells[idx + 1] = self._cells[idx + 1], self._cells[idx]
        self._renumber()

    def toggle_cell(self, cell_id: str) -> None:
        for c in self._cells:
            if c.id == cell_id:
                c.enabled = not c.enabled
                break

    def update_markdown(self, cell_id: str, text: str) -> None:
        for c in self._cells:
            if c.id == cell_id and c.cell_type == CellType.MARKDOWN:
                c.markdown = text
                break

    def clear(self) -> None:
        self._cells = []

    def _find_index(self, cell_id: str) -> int | None:
        for i, c in enumerate(self._cells):
            if c.id == cell_id:
                return i
        return None

    # --- Serialization ---

    def export_json(self) -> str:
        """Export the report as JSON."""
        data = {
            "title": self.title,
            "author": self.author,
            "cells": [
                {
                    "id": c.id,
                    "order": c.order,
                    "cell_type": c.cell_type.value,
                    "enabled": c.enabled,
                    "action": c.action,
                    "description": c.description,
                    "code": c.code,
                    "imports": c.imports,
                    "markdown": c.markdown,
                }
                for c in self._cells
            ],
        }
        return json.dumps(data, indent=2)

    def import_json(self, json_str: str) -> None:
        """Load a report from JSON, replacing current cells."""
        data = json.loads(json_str)
        self.title = data.get("title", "PyAnalytica Report")
        self.author = data.get("author", "")
        self._cells = []
        for c in data.get("cells", []):
            self._cells.append(ReportCell(
                id=c.get("id", str(uuid.uuid4())[:8]),
                order=c.get("order", 0),
                cell_type=CellType(c.get("cell_type", "code")),
                enabled=c.get("enabled", True),
                action=c.get("action", ""),
                description=c.get("description", ""),
                code=c.get("code", ""),
                imports=c.get("imports", []),
                markdown=c.get("markdown", ""),
            ))
        self._renumber()
