"""Analyze > Correlation module — Pearson/Spearman tests."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_numeric_columns
from pyanalytica.analyze.correlation import correlation_test
from pyanalytica.ui.components.add_to_report import add_to_report_server, add_to_report_ui
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def correlation_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("x", "Variable X", choices=[]),
            ui.input_select("y", "Variable Y", choices=[]),
            ui.input_select("method", "Method", choices=["pearson", "spearman"]),
            ui.input_select("alternative", "Alternative Hypothesis",
                choices={"two-sided": "Two-sided (!=)", "less": "Less (<)", "greater": "Greater (>)"}),
            ui.input_action_button("run_btn", "Run Test", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        ui.output_ui("test_result"),
        add_to_report_ui("rpt"),
        code_panel_ui("code"),
    )


@module.server
def correlation_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    last_report_info = reactive.value(None)
    result = reactive.value(None)

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            cols = get_numeric_columns(df)
            ui.update_select("x", choices=cols)
            ui.update_select("y", choices=cols)

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        df = get_current_df()
        req(df is not None)
        x, y = input.x(), input.y()
        req(x, y)
        try:
            r = correlation_test(df, x, y, method=input.method(), alternative=input.alternative())
            result.set(r)
            state.codegen.record(r.code, action="analyze", description="Correlation test")
            last_code.set(r.code.code)
            last_report_info.set(("analyze", "Correlation test", r.code.code, r.code.imports))
        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error")

    @render.ui
    def test_result():
        r = result()
        req(r is not None)
        sig_class = "alert-success" if r.p_value < 0.05 else "alert-warning"
        return ui.div(
            ui.h5(f"{r.method.title()} Correlation"),
            ui.p(r.interpretation),
            ui.tags.small(
                f"r = {r.r:.4f} | p = {r.p_value:.4f} | n = {r.n} | "
                f"95% CI [{r.ci_lower:.4f}, {r.ci_upper:.4f}]",
                class_="text-muted",
            ),
            class_=f"alert {sig_class}",
        )

    add_to_report_server("rpt", state=state, get_code_info=last_report_info)
    code_panel_server("code", get_code=last_code)
