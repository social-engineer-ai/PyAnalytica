"""Analyze > Proportions module — chi-square tests."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core import round_df
from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_categorical_columns
from pyanalytica.analyze.proportions import chi_square_test, goodness_of_fit_test
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui
from pyanalytica.ui.components.decimals_control import decimals_server, decimals_ui
from pyanalytica.ui.components.download_result import download_result_server, download_result_ui


@module.ui
def proportions_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("test_type", "Test", choices={
                "independence": "Test of Independence",
                "goodness_of_fit": "Goodness of Fit",
            }),
            ui.output_ui("test_controls"),
            ui.input_action_button("run_btn", "Run Test", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        ui.output_ui("test_result"),
        decimals_ui("dec"),
        ui.output_ui("result_tables"),
        download_result_ui("dl"),
        code_panel_ui("code"),
    )


@module.server
def proportions_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    indep_result = reactive.value(None)
    gof_result = reactive.value(None)
    last_test_type = reactive.value(None)
    get_dec = decimals_server("dec")

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            cats = get_categorical_columns(df)
            choices = cats if cats else list(df.columns)
            ui.update_select("row_var", choices=choices)
            ui.update_select("col_var", choices=choices)
            ui.update_select("gof_var", choices=choices)

    @render.ui
    def test_controls():
        tt = input.test_type()
        df = get_current_df()
        cats = []
        if df is not None:
            cats = get_categorical_columns(df)
            if not cats:
                cats = list(df.columns)
        if tt == "independence":
            return ui.div(
                ui.input_select("row_var", "Row Variable", choices=cats),
                ui.input_select("col_var", "Column Variable", choices=cats),
            )
        else:
            return ui.div(
                ui.input_select("gof_var", "Variable", choices=cats),
                ui.input_text("expected_probs", "Expected Proportions (optional)",
                              placeholder="A:0.25, B:0.50, C:0.25"),
            )

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        df = get_current_df()
        req(df is not None)
        tt = input.test_type()

        try:
            if tt == "independence":
                row, col = input.row_var(), input.col_var()
                req(row, col)
                r = chi_square_test(df, row, col)
                indep_result.set(r)
                last_test_type.set("independence")
                state.codegen.record(r.code, action="analyze", description="Chi-square test of independence")
                last_code.set(r.code.code)
            else:
                var = input.gof_var()
                req(var)
                probs_str = input.expected_probs().strip()
                expected = None
                if probs_str:
                    expected = {}
                    for pair in probs_str.split(","):
                        k, v = pair.strip().split(":")
                        expected[k.strip()] = float(v.strip())
                r = goodness_of_fit_test(df, var, expected)
                gof_result.set(r)
                last_test_type.set("goodness_of_fit")
                state.codegen.record(r.code, action="analyze", description="Chi-square goodness-of-fit test")
                last_code.set(r.code.code)
        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error")

    @render.ui
    def test_result():
        tt = last_test_type()
        req(tt is not None)
        if tt == "independence":
            r = indep_result()
        else:
            r = gof_result()
        req(r is not None)
        sig_class = "alert-success" if r.p_value < 0.05 else "alert-warning"
        return ui.div(ui.p(r.interpretation), class_=f"alert {sig_class}")

    @render.ui
    def result_tables():
        tt = last_test_type()
        req(tt is not None)
        if tt == "independence":
            r = indep_result()
            req(r is not None)
            dec = get_dec()
            obs_grid = ui.output_data_frame("observed")
            exp_grid = ui.output_data_frame("expected")
            return ui.div(
                ui.h5("Observed"),
                obs_grid,
                ui.h5("Expected"),
                exp_grid,
            )
        else:
            r = gof_result()
            req(r is not None)
            return ui.div(
                ui.h5("Goodness-of-Fit Results"),
                ui.output_data_frame("gof_table"),
            )

    @render.data_frame
    def observed():
        r = indep_result()
        req(r is not None)
        return render.DataGrid(round_df(r.observed.reset_index(), get_dec()))

    @render.data_frame
    def expected():
        r = indep_result()
        req(r is not None)
        return render.DataGrid(round_df(r.expected.reset_index(), get_dec()))

    @render.data_frame
    def gof_table():
        r = gof_result()
        req(r is not None)
        return render.DataGrid(round_df(r.table, get_dec()))

    def _get_dl_df():
        tt = last_test_type()
        if tt == "independence":
            r = indep_result()
            return r.observed.reset_index() if r else None
        else:
            r = gof_result()
            return r.table if r else None

    download_result_server("dl", get_df=_get_dl_df, filename="proportions")
    code_panel_server("code", get_code=last_code)
