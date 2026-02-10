"""Export session as HTML report, Python script, or Jupyter notebook."""
from __future__ import annotations

import html
import json
from datetime import datetime
from typing import TYPE_CHECKING

from pyanalytica.core.codegen import CodeGenerator
from pyanalytica.report.notebook import SessionNotebook

# ---------------------------------------------------------------------------
# Action badge colours — maps action keywords to colour pairs (bg, text)
# ---------------------------------------------------------------------------
_ACTION_COLORS: dict[str, tuple[str, str]] = {
    "load":      ("#e3f2fd", "#1565c0"),
    "transform": ("#fff3e0", "#e65100"),
    "visualize": ("#e8f5e9", "#2e7d32"),
    "analyze":   ("#f3e5f5", "#6a1b9a"),
    "export":    ("#fce4ec", "#b71c1c"),
}
_DEFAULT_ACTION_COLOR = ("#f5f5f5", "#424242")

# ---------------------------------------------------------------------------
# Output-type icons (plain-text, no external deps)
# ---------------------------------------------------------------------------
_OUTPUT_ICONS: dict[str, str] = {
    "table":  "&#x1f4ca;",  # bar chart
    "figure": "&#x1f5bc;",  # framed picture
    "text":   "&#x1f4dd;",  # memo
    "stats":  "&#x1f4c8;",  # chart increasing
}


def _escape(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(text, quote=True)


# ---- HTML Export ----------------------------------------------------------

def export_html(notebook: SessionNotebook) -> str:
    """Export session notebook as a standalone HTML report.

    Produces a self-contained HTML page with inline CSS.  Each recorded
    entry is rendered as a card showing the description, the generated
    code in a ``<pre>`` block, and an output summary.
    """
    entries = notebook.get_entries()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- inline CSS --------------------------------------------------------
    css = """\
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     Helvetica, Arial, sans-serif;
        background: #fafafa;
        color: #212121;
        padding: 2rem;
        line-height: 1.6;
    }
    header {
        text-align: center;
        margin-bottom: 2rem;
    }
    header h1 {
        font-size: 1.8rem;
        color: #1a237e;
    }
    header p {
        color: #757575;
        font-size: 0.9rem;
    }
    .card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .timestamp {
        font-size: 0.78rem;
        color: #9e9e9e;
        margin-left: auto;
    }
    .description {
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }
    pre {
        background: #263238;
        color: #eeffff;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        overflow-x: auto;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 0.75rem;
    }
    .output-summary {
        background: #f5f5f5;
        border-left: 3px solid #90caf9;
        padding: 0.5rem 0.75rem;
        font-size: 0.85rem;
        color: #616161;
        border-radius: 0 4px 4px 0;
    }
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #bdbdbd;
        font-size: 1.1rem;
    }
    footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #bdbdbd;
    }
    """

    # --- build cards -------------------------------------------------------
    cards_html_parts: list[str] = []

    for idx, entry in enumerate(entries, start=1):
        bg, fg = _ACTION_COLORS.get(entry.action, _DEFAULT_ACTION_COLOR)
        icon = _OUTPUT_ICONS.get(entry.output_type, "")

        card = (
            f'<div class="card">'
            f'  <div class="card-header">'
            f'    <span class="badge" style="background:{bg};color:{fg};">'
            f'      {_escape(entry.action)}'
            f'    </span>'
            f'    <strong>Step {idx}</strong>'
            f'    <span class="timestamp">{_escape(entry.timestamp)}</span>'
            f'  </div>'
            f'  <p class="description">{_escape(entry.description)}</p>'
        )

        if entry.code.strip():
            card += f'  <pre><code>{_escape(entry.code)}</code></pre>'

        if entry.output_summary.strip():
            card += (
                f'  <div class="output-summary">'
                f'    {icon} {_escape(entry.output_summary)}'
                f'  </div>'
            )

        card += '</div>'
        cards_html_parts.append(card)

    if not cards_html_parts:
        body = '<div class="empty-state">No entries recorded in this session.</div>'
    else:
        body = "\n".join(cards_html_parts)

    # --- assemble full page ------------------------------------------------
    page = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "  <title>PyAnalytica Session Report</title>\n"
        f"  <style>\n{css}  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <header>\n"
        "    <h1>PyAnalytica Session Report</h1>\n"
        f"    <p>Generated on {_escape(now)}</p>\n"
        "  </header>\n"
        f"  {body}\n"
        "  <footer>\n"
        "    <p>Produced by PyAnalytica</p>\n"
        "  </footer>\n"
        "</body>\n"
        "</html>\n"
    )
    return page


# ---- Python script export -------------------------------------------------

def export_python_script(codegen: CodeGenerator) -> str:
    """Export accumulated code as a runnable Python script.

    Includes sorted imports at the top followed by all recorded code lines.
    """
    return codegen.export_script()


# ---- Jupyter notebook export ----------------------------------------------

def _build_notebook_json(codegen: CodeGenerator) -> str:
    """Manually construct a valid Jupyter notebook (.ipynb) JSON string.

    The output conforms to nbformat v4.5 and can be opened by JupyterLab,
    VS Code, or any notebook viewer.
    """
    # Gather imports and code snippets from the code generator.
    import_lines = sorted(codegen.imports)
    code_snippets = [s.code for s in codegen.snippets]

    cells: list[dict] = []

    # First cell: imports
    if import_lines:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in import_lines],
        })

    # Subsequent cells: one per code snippet
    for snippet in code_snippets:
        source_lines = snippet.splitlines(keepends=True)
        # Ensure the last line ends with a newline for consistency
        if source_lines and not source_lines[-1].endswith("\n"):
            source_lines[-1] += "\n"
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines,
        })

    # If there are no cells at all, add a single empty cell so the
    # notebook is still valid.
    if not cells:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [],
        })

    notebook_dict = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }

    return json.dumps(notebook_dict, indent=1, ensure_ascii=False)


def export_jupyter_notebook(codegen: CodeGenerator) -> str:
    """Export as a Jupyter notebook (.ipynb) JSON string.

    Uses ``nbformat`` to create a proper notebook structure when available.
    Falls back to manual JSON construction otherwise, so there is no hard
    dependency on ``nbformat``.
    """
    try:
        import nbformat  # type: ignore[import-untyped]

        nb = nbformat.v4.new_notebook()

        # First cell: imports
        import_lines = sorted(codegen.imports)
        if import_lines:
            nb.cells.append(nbformat.v4.new_code_cell("\n".join(import_lines)))

        # One cell per code snippet
        for snippet in codegen.snippets:
            nb.cells.append(nbformat.v4.new_code_cell(snippet.code))

        return nbformat.writes(nb)

    except ImportError:
        return _build_notebook_json(codegen)


def export_jupyter_bytes(codegen: CodeGenerator) -> bytes:
    """Export as bytes suitable for a file download."""
    return export_jupyter_notebook(codegen).encode("utf-8")
