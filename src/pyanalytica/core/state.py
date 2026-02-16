"""Dataset store and operation history for the workbench."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from pyanalytica.core.codegen import CodeGenerator, CodeSnippet
from pyanalytica.core.model_store import ModelStore
from pyanalytica.core.procedure import ProcedureRecorder


@dataclass
class Operation:
    """A recorded analytics operation."""
    timestamp: datetime
    action: str          # "load", "merge", "transform", "filter", etc.
    description: str     # Human-readable
    dataset: str         # Which dataset was affected
    details: dict = field(default_factory=dict)


MAX_UNDO = 20  # Maximum number of undo snapshots to keep in memory


class WorkbenchState:
    """Stores loaded datasets and operation history. No UI logic.

    The Shiny reactive graph owns which dataset is currently selected.
    This is a dumb store — it holds datasets and history, nothing else.
    """

    def __init__(self) -> None:
        self.datasets: dict[str, pd.DataFrame] = {}
        self.originals: dict[str, pd.DataFrame] = {}
        self._history_stack: list[tuple[str, pd.DataFrame]] = []  # For undo
        self.history: list[Operation] = []
        self.codegen: CodeGenerator = CodeGenerator()
        self.model_store: ModelStore = ModelStore()
        self.procedure_recorder: ProcedureRecorder = ProcedureRecorder()
        self._decimals = lambda: 4  # Default; overridden per-module by UI
        self._change_signal = None  # Set externally by UI layer (shiny reactive.value)
        self._change_counter = 0

        # Auto-forward codegen records to the procedure recorder
        def _on_codegen_record(snippet: CodeSnippet, *, action: str | None = None,
                               description: str | None = None) -> None:
            # Prefer explicit params; fall back to last history entry
            dataset = ""
            if action is None or description is None:
                if self.history:
                    last = self.history[-1]
                    action = action or last.action
                    description = description or last.description
                    dataset = last.dataset
                else:
                    action = action or "unknown"
                    description = description or "Recorded operation"
            else:
                # Try to get dataset from latest history entry
                if self.history:
                    dataset = self.history[-1].dataset
            self.procedure_recorder.record_step(action, description, snippet,
                                                dataset=dataset)

        self.codegen.set_on_record(_on_codegen_record)

    def _notify(self) -> None:
        """Bump the reactive change signal if one is attached."""
        if self._change_signal is not None:
            self._change_counter += 1
            self._change_signal.set(self._change_counter)

    def load(self, name: str, df: pd.DataFrame) -> None:
        """Load a new dataset into the store."""
        self.datasets[name] = df
        self.originals[name] = df.copy()
        self.history.append(Operation(
            timestamp=datetime.now(),
            action="load",
            description=f"Loaded dataset '{name}' ({df.shape[0]} rows, {df.shape[1]} columns)",
            dataset=name,
            details={"rows": df.shape[0], "cols": df.shape[1]},
        ))
        self._notify()

    def get(self, name: str) -> pd.DataFrame:
        """Get a dataset by name. Raises KeyError if not found."""
        if name not in self.datasets:
            raise KeyError(f"Dataset '{name}' not found. Available: {list(self.datasets.keys())}")
        return self.datasets[name]

    def update(self, name: str, df: pd.DataFrame, operation: Operation) -> None:
        """Update a dataset and record the operation."""
        if name in self.datasets:
            self._history_stack.append((name, self.datasets[name].copy()))
            # Cap undo stack to prevent unbounded memory growth
            if len(self._history_stack) > MAX_UNDO:
                self._history_stack = self._history_stack[-MAX_UNDO:]
        self.datasets[name] = df
        self.history.append(operation)
        self._notify()

    def undo(self) -> str | None:
        """Undo the last operation. Returns the dataset name that was restored, or None."""
        if not self._history_stack:
            return None
        name, prev_df = self._history_stack.pop()
        self.datasets[name] = prev_df
        self.history.append(Operation(
            timestamp=datetime.now(),
            action="undo",
            description=f"Undid last operation on '{name}'",
            dataset=name,
        ))
        self._notify()
        return name

    def reset(self, name: str) -> None:
        """Reset a dataset to its original loaded state."""
        if name not in self.originals:
            raise KeyError(f"No original data for '{name}'")
        self.datasets[name] = self.originals[name].copy()
        self._history_stack = [
            (n, df) for n, df in self._history_stack if n != name
        ]
        self.history.append(Operation(
            timestamp=datetime.now(),
            action="reset",
            description=f"Reset '{name}' to original state",
            dataset=name,
        ))
        self._notify()

    def dataset_names(self) -> list[str]:
        """Return sorted list of loaded dataset names."""
        return sorted(self.datasets.keys())

    def remove(self, name: str) -> None:
        """Remove a dataset from the store."""
        self.datasets.pop(name, None)
        self.originals.pop(name, None)
        self._history_stack = [
            (n, df) for n, df in self._history_stack if n != name
        ]
        self._notify()

    def has(self, name: str) -> bool:
        """Check if a dataset exists."""
        return name in self.datasets

    def __len__(self) -> int:
        return len(self.datasets)

    def __contains__(self, name: str) -> bool:
        return name in self.datasets
