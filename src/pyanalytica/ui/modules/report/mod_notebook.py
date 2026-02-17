"""Report module — view session history, preview code, and export reports."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.report.notebook import SessionNotebook
from pyanalytica.report.export import (
    export_html,
    export_jupyter_bytes,
    export_python_script,
)


# ---------------------------------------------------------------------------
# Action badge colour mapping (mirrors report.export._ACTION_COLORS)
# ---------------------------------------------------------------------------
_ACTION_COLORS: dict[str, tuple[str, str]] = {
    "load":      ("#e3f2fd", "#1565c0"),
    "transform": ("#fff3e0", "#e65100"),
    "visualize": ("#e8f5e9", "#2e7d32"),
    "analyze":   ("#f3e5f5", "#6a1b9a"),
    "export":    ("#fce4ec", "#b71c1c"),
    "merge":     ("#e0f7fa", "#00695c"),
    "filter":    ("#fff9c4", "#f57f17"),
    "model":     ("#ede7f6", "#4527a0"),
    "undo":      ("#efebe9", "#4e342e"),
    "reset":     ("#fbe9e7", "#bf360c"),
}
_DEFAULT_COLOR = ("#f5f5f5", "#424242")


def _badge_html(action: str) -> str:
    """Return an inline-styled HTML badge for an action type."""
    bg, fg = _ACTION_COLORS.get(action, _DEFAULT_COLOR)
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'font-size:0.75rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.04em;background:{bg};color:{fg};">'
        f'{action}</span>'
    )


# ---------------------------------------------------------------------------
# Module UI
# ---------------------------------------------------------------------------

@module.ui
def notebook_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Export Session"),
            ui.input_select(
                "export_format", "Export Format",
                choices={
                    "html": "HTML Report",
                    "python": "Python Script (.py)",
                    "jupyter": "Jupyter Notebook (.ipynb)",
                },
            ),
            ui.download_button(
                "download_report", "Download",
                class_="btn-primary w-100 mt-2",
            ),
            ui.tags.hr(),
            ui.h6("Session Info"),
            ui.output_ui("session_stats"),
            ui.input_action_button("refresh_stats", "Refresh", class_="btn-sm btn-outline-secondary w-100 mt-1"),
            width=280,
        ),
        # --- Main panel ---
        ui.navset_tab(
            ui.nav_panel(
                "Session History",
                ui.tags.div(
                    ui.output_ui("history_list"),
                    class_="mt-3",
                ),
            ),
            ui.nav_panel(
                "Code Preview",
                ui.tags.div(
                    ui.output_ui("code_preview"),
                    class_="mt-3",
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Module Server
# ---------------------------------------------------------------------------

@module.server
def notebook_server(input, output, session, state: WorkbenchState, get_current_df):

    # Local refresh counter — bumped by button click
    _refresh = reactive.value(0)

    @reactive.effect
    @reactive.event(input.refresh_stats)
    def _bump_refresh():
        _refresh.set(_refresh() + 1)

    # ------------------------------------------------------------------
    # Session stats
    # ------------------------------------------------------------------
    @render.ui
    def session_stats():
        _refresh()
        n_ops = len(state.history)
        n_datasets = len(state.datasets)
        n_snippets = len(state.codegen)
        return ui.tags.small(
            f"Operations: {n_ops} | Datasets: {n_datasets} | Snippets: {n_snippets}",
            class_="text-muted",
        )

    # ------------------------------------------------------------------
    # History list — styled entries with action badges
    # ------------------------------------------------------------------
    @render.ui
    def history_list():
        _refresh()
        ops = state.history
        if not ops:
            return ui.tags.div(
                ui.tags.p(
                    "No operations recorded yet. "
                    "Load a dataset and perform some analysis to build your session history.",
                    class_="text-muted",
                ),
                class_="text-center py-4",
            )

        rows = []
        for idx, op in enumerate(ops, start=1):
            badge = _badge_html(op.action)
            ts = op.timestamp.strftime("%H:%M:%S") if hasattr(op.timestamp, "strftime") else str(op.timestamp)
            row = ui.tags.div(
                ui.tags.div(
                    ui.HTML(badge),
                    ui.tags.span(f"  Step {idx}", class_="fw-bold ms-2"),
                    ui.tags.span(f"  {ts}", class_="text-muted ms-auto", style="font-size:0.8rem;"),
                    class_="d-flex align-items-center mb-1",
                ),
                ui.tags.p(op.description, class_="mb-0", style="font-size:0.9rem;"),
                ui.tags.small(
                    f"Dataset: {op.dataset}" if op.dataset else "",
                    class_="text-muted",
                ),
                class_="border rounded p-2 mb-2",
                style="background:#fafafa;",
            )
            rows.append(row)

        return ui.tags.div(
            ui.h5(f"Session History ({len(ops)} operations)"),
            *rows,
        )

    # ------------------------------------------------------------------
    # Code preview — shows the full generated Python script
    # ------------------------------------------------------------------
    @render.ui
    def code_preview():
        _refresh()
        codegen = state.codegen
        if not codegen:
            return ui.tags.div(
                ui.tags.p(
                    "No code generated yet. Perform some analysis actions to generate code.",
                    class_="text-muted",
                ),
                class_="text-center py-4",
            )

        script = export_python_script(codegen)
        return ui.tags.div(
            ui.h5("Generated Python Script"),
            ui.tags.p(
                f"{len(codegen.snippets)} code snippet{'s' if len(codegen.snippets) != 1 else ''} "
                f"| {len(codegen.imports)} import{'s' if len(codegen.imports) != 1 else ''}",
                class_="text-muted",
            ),
            ui.tags.pre(
                ui.tags.code(script),
                style=(
                    "background:#263238;color:#eeffff;padding:1rem 1.2rem;"
                    "border-radius:6px;overflow-x:auto;font-size:0.85rem;"
                    "line-height:1.5;max-height:600px;overflow-y:auto;"
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Build a SessionNotebook from state for HTML export
    # ------------------------------------------------------------------
    def _build_session_notebook() -> SessionNotebook:
        """Construct a SessionNotebook from state history and codegen."""
        nb = SessionNotebook()
        snippets = state.codegen.snippets
        for idx, op in enumerate(state.history):
            code = snippets[idx].code if idx < len(snippets) else ""
            nb.record(
                action=op.action,
                description=op.description,
                code=code,
                output_type="text",
                output_summary=op.details.get("summary", "") if op.details else "",
            )
        return nb

    # ------------------------------------------------------------------
    # Download handler
    # ------------------------------------------------------------------
    @render.download(filename=lambda: _export_filename())
    def download_report():
        fmt = input.export_format()
        if fmt == "html":
            nb = _build_session_notebook()
            html_content = export_html(nb)
            yield html_content.encode("utf-8")
        elif fmt == "python":
            script = export_python_script(state.codegen)
            yield script.encode("utf-8")
        elif fmt == "jupyter":
            yield export_jupyter_bytes(state.codegen)

    def _export_filename() -> str:
        fmt = input.export_format()
        if fmt == "html":
            return "pyanalytica_session_report.html"
        elif fmt == "python":
            return "pyanalytica_session.py"
        elif fmt == "jupyter":
            return "pyanalytica_session.ipynb"
        return "pyanalytica_session.txt"
