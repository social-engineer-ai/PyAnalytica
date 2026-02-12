"""Visualize > Timeline module — time series charts."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_datetime_columns, get_numeric_columns
from pyanalytica.visualize.timeline import time_series
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def timeline_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("date_col", "Date Column", choices=[]),
            ui.input_select("value_col", "Value Column", choices=[]),
            ui.input_select("group_by", "Group By (optional)", choices=[""]),
            ui.input_select("agg_level", "Aggregation",
                choices=["raw", "daily", "weekly", "monthly"]),
            ui.input_select("chart_type", "Chart Type",
                choices=["line", "area", "bar"]),
            ui.input_slider("rolling", "Rolling Window", 0, 30, 0),
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
def timeline_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    _last_fig = reactive.value(None)

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            dt_cols = get_datetime_columns(df)
            # Also include object cols that might be dates
            all_cols = list(df.columns)
            date_choices = dt_cols if dt_cols else all_cols
            ui.update_select("date_col", choices=date_choices)
            ui.update_select("value_col", choices=get_numeric_columns(df))
            ui.update_select("group_by", choices=[""] + all_cols)

    @render.plot
    @reactive.event(input.run_btn)
    def chart():
        df = get_current_df()
        req(df is not None)
        date_col = input.date_col()
        value_col = input.value_col()
        req(date_col, value_col)

        group = input.group_by() or None
        rolling = input.rolling() if input.rolling() > 0 else None

        fig, snippet = time_series(
            df, date_col, value_col,
            group_by=group,
            agg_level=input.agg_level(),
            chart_type=input.chart_type(),
            rolling_window=rolling,
        )
        state.codegen.record(snippet, action="visualize", description="Time series plot")
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
