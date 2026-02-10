"""Visualize > Relate module — scatter, hexbin."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_categorical_columns, get_numeric_columns
from pyanalytica.visualize.relate import hexbin, scatter
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def relate_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("x", "X Variable", choices=[]),
            ui.input_select("y", "Y Variable", choices=[]),
            ui.input_select("chart_type", "Chart Type",
                choices=["scatter", "hexbin"]),
            ui.input_select("color_by", "Color By (optional)", choices=[""]),
            ui.input_checkbox("trend", "Show Trend Line", value=True),
            ui.input_action_button("run_btn", "Plot", class_="btn-primary w-100 mt-2"),
            width=280,
        ),
        ui.output_plot("chart", height="500px"),
        code_panel_ui("code"),
    )


@module.server
def relate_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            num_cols = get_numeric_columns(df)
            cat_cols = get_categorical_columns(df)
            ui.update_select("x", choices=num_cols)
            ui.update_select("y", choices=num_cols)
            ui.update_select("color_by", choices=[""] + cat_cols)

    @render.plot
    @reactive.event(input.run_btn)
    def chart():
        df = get_current_df()
        req(df is not None)
        x, y = input.x(), input.y()
        req(x, y)

        color = input.color_by() or None
        ct = input.chart_type()

        if ct == "scatter":
            fig, snippet = scatter(df, x, y, color_by=color, trend_line=input.trend())
        else:
            fig, snippet = hexbin(df, x, y)

        state.codegen.record(snippet)
        last_code.set(snippet.code)
        return fig

    code_panel_server("code", get_code=last_code)
