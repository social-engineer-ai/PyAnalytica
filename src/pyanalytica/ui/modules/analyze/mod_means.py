"""Analyze > Means module — t-tests and ANOVA."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core import round_df
from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_categorical_columns, get_numeric_columns
from pyanalytica.analyze.means import one_sample_ttest, one_way_anova, two_sample_ttest
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def means_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("test_type", "Test",
                choices={"one_sample": "One-sample t-test", "two_sample": "Two-sample t-test",
                         "anova": "One-way ANOVA"}),
            ui.input_select("value_col", "Numeric Variable", choices=[]),
            ui.output_ui("test_controls"),
            ui.input_action_button("run_btn", "Run Test", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        ui.output_ui("test_result"),
        ui.output_data_frame("group_stats"),
        ui.output_ui("assumptions"),
        code_panel_ui("code"),
    )


@module.server
def means_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    test_result_val = reactive.value(None)

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            ui.update_select("value_col", choices=get_numeric_columns(df))

    @render.ui
    def test_controls():
        df = get_current_df()
        tt = input.test_type()
        if tt == "one_sample":
            return ui.input_numeric("mu", "Hypothesized Mean", value=0)
        elif tt in ("two_sample", "anova"):
            cats = get_categorical_columns(df) if df is not None else []
            return ui.input_select("group_col", "Group Variable", choices=cats)
        return ui.div()

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        df = get_current_df()
        req(df is not None)
        col = input.value_col()
        req(col)
        tt = input.test_type()

        try:
            if tt == "one_sample":
                result = one_sample_ttest(df, col, input.mu())
            elif tt == "two_sample":
                result = two_sample_ttest(df, col, input.group_col())
            elif tt == "anova":
                result = one_way_anova(df, col, input.group_col())
            else:
                return
            test_result_val.set(result)
            state.codegen.record(result.code)
            last_code.set(result.code.code)
        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error")

    @render.ui
    def test_result():
        r = test_result_val()
        req(r is not None)
        sig_class = "alert-success" if r.p_value < 0.05 else "alert-warning"
        return ui.div(
            ui.h5(r.test_name),
            ui.p(r.interpretation),
            ui.tags.small(f"Test statistic: {r.statistic} | p-value: {r.p_value} | "
                         f"Effect size ({r.effect_size_name}): {r.effect_size}", class_="text-muted"),
            class_=f"alert {sig_class}",
        )

    @render.data_frame
    def group_stats():
        r = test_result_val()
        req(r is not None)
        return render.DataGrid(round_df(r.group_stats, state._decimals()))

    @render.ui
    def assumptions():
        r = test_result_val()
        req(r is not None)
        checks = r.assumption_checks
        if not checks:
            return ui.div()
        items = [ui.h6("Assumption Checks")]
        for k, v in checks.items():
            items.append(ui.p(f"{k}: {v}", class_="mb-1 small"))
        return ui.div(*items, class_="mt-2 p-2 bg-light rounded")

    code_panel_server("code", get_code=last_code)
