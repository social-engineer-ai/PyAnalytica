"""Code generation engine — emits real pandas/matplotlib/sklearn code."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodeSnippet:
    """A single code operation with its required imports."""
    code: str
    imports: list[str] = field(default_factory=list)


class CodeGenerator:
    """Accumulates pandas/matplotlib/sklearn code for the entire session.

    Every UI action records a CodeSnippet. The generator tracks imports
    and code lines, and can export a full runnable Python script.
    """

    def __init__(self) -> None:
        self.imports: set[str] = {"import pandas as pd"}
        self.snippets: list[CodeSnippet] = []

    def record(self, snippet: CodeSnippet) -> None:
        """Record a code snippet and its imports."""
        for imp in snippet.imports:
            self.imports.add(imp)
        self.snippets.append(snippet)

    def export_script(self) -> str:
        """Export the full accumulated script as runnable Python."""
        header = sorted(self.imports)
        code_lines = [s.code for s in self.snippets]
        return "\n".join(header) + "\n\n" + "\n\n".join(code_lines) + "\n"

    def export_last(self) -> str:
        """Export just the most recent operation's code."""
        if not self.snippets:
            return ""
        last = self.snippets[-1]
        header = sorted(set(last.imports))
        if header:
            return "\n".join(header) + "\n\n" + last.code + "\n"
        return last.code + "\n"

    def clear(self) -> None:
        """Clear all recorded code."""
        self.imports = {"import pandas as pd"}
        self.snippets = []

    def __len__(self) -> int:
        return len(self.snippets)

    def __bool__(self) -> bool:
        return len(self.snippets) > 0
