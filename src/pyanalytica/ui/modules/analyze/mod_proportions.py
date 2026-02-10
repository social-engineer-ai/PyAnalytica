"""Analyze > Proportions module — chi-square test."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_categorical_columns
from pyanalytica.analyze.proportions import chi_square_test
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def proportions_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("row_var", "Row Variable", choices=[]),
            ui.input_select("col_var", "Column Variable", choices=[]),
            ui.input_action_button("run_btn", "Run Test", class_="btn-primary w-100 mt-2"),
            width=280,
        ),
        ui.output_ui("test_result"),
        ui.h5("Observed"),
        ui.output_data_frame("observed"),
        ui.h5("Expected"),
        ui.output_data_frame("expected"),
        code_panel_ui("code"),
    )


@module.server
def proportions_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    result = reactive.value(None)

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            cats = get_categorical_columns(df)
            choices = cats if cats else list(df.columns)
            ui.update_select("row_var", choices=choices)
            ui.update_select("col_var", choices=choices)

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        df = get_current_df()
        req(df is not None)
        row, col = input.row_var(), input.col_var()
        req(row, col)
        try:
            r = chi_square_test(df, row, col)
            result.set(r)
            state.codegen.record(r.code)
            last_code.set(r.code.code)
        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error")

    @render.ui
    def test_result():
        r = result()
        req(r is not None)
        sig_class = "alert-success" if r.p_value < 0.05 else "alert-warning"
        return ui.div(ui.p(r.interpretation), class_=f"alert {sig_class}")

    @render.data_frame
    def observed():
        r = result()
        req(r is not None)
        return render.DataGrid(r.observed.reset_index())

    @render.data_frame
    def expected():
        r = result()
        req(r is not None)
        return render.DataGrid(r.expected.reset_index())

    code_panel_server("code", get_code=last_code)
