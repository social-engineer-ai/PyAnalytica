"""Model > Regression module."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_numeric_columns
from pyanalytica.model.regression import linear_regression
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def regression_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("target", "Target (Y)", choices=[]),
            ui.input_select("features", "Features (X)", choices=[], multiple=True),
            ui.input_slider("test_size", "Test Split", 0.0, 0.5, 0.0, step=0.05),
            ui.input_action_button("run_btn", "Fit Model", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        ui.output_ui("model_summary"),
        ui.output_data_frame("coef_table"),
        ui.h5("VIF (Multicollinearity)"),
        ui.output_data_frame("vif_table"),
        ui.output_plot("resid_plot", height="350px"),
        ui.output_plot("qq_plot", height="350px"),
        code_panel_ui("code"),
    )


@module.server
def regression_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    result = reactive.value(None)

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            cols = get_numeric_columns(df)
            ui.update_select("target", choices=cols)
            ui.update_select("features", choices=cols)

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        df = get_current_df()
        req(df is not None)
        target = input.target()
        features = list(input.features())
        req(target, features)
        try:
            test_size = input.test_size() if input.test_size() > 0 else None
            r = linear_regression(df, target, features, test_size=test_size)
            result.set(r)
            state.codegen.record(r.code)
            last_code.set(r.code.code)
        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error")

    @render.ui
    def model_summary():
        r = result()
        req(r is not None)
        return ui.div(
            ui.h5("Linear Regression"),
            ui.p(r.interpretation),
            ui.tags.small(
                f"R\u00b2 = {r.r_squared:.4f} | Adj. R\u00b2 = {r.adj_r_squared:.4f} | "
                f"F = {r.f_stat:.2f} (p = {r.f_pvalue:.4f})",
                class_="text-muted",
            ),
            class_="alert alert-info",
        )

    @render.data_frame
    def coef_table():
        r = result()
        req(r is not None)
        return render.DataGrid(r.coefficients)

    @render.data_frame
    def vif_table():
        r = result()
        req(r is not None)
        return render.DataGrid(r.vif)

    @render.plot
    def resid_plot():
        r = result()
        req(r is not None and r.residual_plot is not None)
        return r.residual_plot

    @render.plot
    def qq_plot():
        r = result()
        req(r is not None and r.qq_plot is not None)
        return r.qq_plot

    code_panel_server("code", get_code=last_code)
