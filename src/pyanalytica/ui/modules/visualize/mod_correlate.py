"""Visualize > Correlate module — correlation matrix, pair plot."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_numeric_columns
from pyanalytica.visualize.correlate import correlation_matrix, pair_plot
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def correlate_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("cols", "Variables", choices=[], multiple=True),
            ui.input_select("chart_type", "Chart Type",
                choices=["correlation_matrix", "pair_plot"]),
            ui.input_select("method", "Method", choices=["pearson", "spearman"]),
            ui.input_slider("threshold", "|r| Threshold", 0.0, 1.0, 0.0, step=0.05),
            ui.input_action_button("run_btn", "Plot", class_="btn-primary w-100 mt-2"),
            width=280,
        ),
        ui.output_plot("chart", height="600px"),
        code_panel_ui("code"),
    )


@module.server
def correlate_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            ui.update_select("cols", choices=get_numeric_columns(df))

    @render.plot
    @reactive.event(input.run_btn)
    def chart():
        df = get_current_df()
        req(df is not None)
        cols = list(input.cols())
        req(len(cols) >= 2)
        ct = input.chart_type()

        if ct == "correlation_matrix":
            fig, snippet = correlation_matrix(df, cols, method=input.method(), threshold=input.threshold())
        else:
            fig, snippet = pair_plot(df, cols)

        state.codegen.record(snippet)
        last_code.set(snippet.code)
        return fig

    code_panel_server("code", get_code=last_code)
