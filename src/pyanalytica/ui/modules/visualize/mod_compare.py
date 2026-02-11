"""Visualize > Compare module — grouped box, violin, bar of means, strip."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_categorical_columns, get_numeric_columns
from pyanalytica.visualize.compare import bar_of_means, grouped_boxplot, grouped_violin, strip_plot
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def compare_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("x_cat", "Category (X)", choices=[]),
            ui.input_select("y_num", "Numeric (Y)", choices=[]),
            ui.input_select("chart_type", "Chart Type",
                choices=["boxplot", "violin", "bar_of_means", "strip"]),
            ui.input_select("hue", "Hue (optional)", choices=[""]),
            ui.input_select("facet_col", "Facet Column (optional)", choices=[""]),
            ui.input_select("facet_row", "Facet Row (optional)", choices=[""]),
            ui.input_action_button("run_btn", "Plot", class_="btn-primary w-100 mt-2"),
            width=280,
        ),
        ui.card(
            ui.card_header(
                ui.div(
                    {"class": "d-flex justify-content-between align-items-center"},
                    ui.span("Chart"),
                    ui.input_action_button("expand_btn", "Expand",
                        class_="btn btn-outline-secondary btn-sm"),
                ),
            ),
            ui.output_plot("chart", height="500px"),
            full_screen=True,
        ),
        code_panel_ui("code"),
    )


@module.server
def compare_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    _last_fig = reactive.value(None)

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            cat_cols = get_categorical_columns(df)
            ui.update_select("x_cat", choices=cat_cols)
            ui.update_select("y_num", choices=get_numeric_columns(df))
            ui.update_select("hue", choices=[""] + cat_cols)
            ui.update_select("facet_col", choices=[""] + cat_cols)
            ui.update_select("facet_row", choices=[""] + cat_cols)

    @render.plot
    @reactive.event(input.run_btn)
    def chart():
        df = get_current_df()
        req(df is not None)
        x, y = input.x_cat(), input.y_num()
        req(x, y)
        ct = input.chart_type()
        hue = input.hue() or None
        facet_c = input.facet_col() or None
        facet_r = input.facet_row() or None

        if ct == "boxplot":
            fig, snippet = grouped_boxplot(df, x, y, hue=hue, facet_col=facet_c, facet_row=facet_r)
        elif ct == "violin":
            fig, snippet = grouped_violin(df, x, y, hue=hue, facet_col=facet_c, facet_row=facet_r)
        elif ct == "bar_of_means":
            fig, snippet = bar_of_means(df, x, y, hue=hue, facet_col=facet_c, facet_row=facet_r)
        elif ct == "strip":
            fig, snippet = strip_plot(df, x, y, hue=hue, facet_col=facet_c, facet_row=facet_r)
        else:
            return

        state.codegen.record(snippet)
        last_code.set(snippet.code)
        _last_fig.set(fig)
        return fig

    @reactive.effect
    @reactive.event(input.expand_btn)
    def _show_modal():
        m = ui.modal(
            ui.output_plot("chart_full", height="80vh"),
            size="xl",
            easy_close=True,
            title="Chart (Full Screen)",
        )
        ui.modal_show(m)

    @render.plot
    def chart_full():
        fig = _last_fig()
        req(fig is not None)
        return fig

    code_panel_server("code", get_code=last_code)
