"""Visualize > Distribute module — histogram, box, violin, bar."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import ColumnType, classify_column
from pyanalytica.visualize.distribute import bar_chart, boxplot, histogram, violin
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def distribute_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("col", "Variable", choices=[]),
            ui.input_select("chart_type", "Chart Type",
                choices=["histogram", "boxplot", "violin", "bar"]),
            ui.output_ui("chart_options"),
            ui.input_action_button("run_btn", "Plot", class_="btn-primary w-100 mt-2"),
            width=280,
        ),
        ui.output_plot("chart", height="500px"),
        code_panel_ui("code"),
    )


@module.server
def distribute_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            ui.update_select("col", choices=list(df.columns))

    @render.ui
    def chart_options():
        ct = input.chart_type()
        if ct == "histogram":
            return ui.TagList(
                ui.input_slider("bins", "Bins", 5, 100, 30),
                ui.input_checkbox("kde", "Show KDE", value=False),
            )
        elif ct == "bar":
            return ui.TagList(
                ui.input_checkbox("pct", "Show Percentages", value=False),
                ui.input_select("orientation", "Orientation",
                    choices=["vertical", "horizontal"]),
            )
        return ui.div()

    @render.plot
    @reactive.event(input.run_btn)
    def chart():
        df = get_current_df()
        req(df is not None)
        col = input.col()
        req(col)
        ct = input.chart_type()

        if ct == "histogram":
            fig, snippet = histogram(df, col, bins=input.bins(), kde=input.kde())
        elif ct == "boxplot":
            fig, snippet = boxplot(df, col)
        elif ct == "violin":
            fig, snippet = violin(df, col)
        elif ct == "bar":
            fig, snippet = bar_chart(df, col, orientation=input.orientation(), pct=input.pct())
        else:
            return

        state.codegen.record(snippet)
        last_code.set(snippet.code)
        return fig

    code_panel_server("code", get_code=last_code)
