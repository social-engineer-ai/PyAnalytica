"""Explore > Cross-tab module — cross-tabulation with chi-square."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_categorical_columns
from pyanalytica.explore.crosstab import create_crosstab
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def crosstab_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("row_var", "Row Variable", choices=[]),
            ui.input_select("col_var", "Column Variable", choices=[]),
            ui.input_select("normalize", "Display",
                choices={"": "Counts", "index": "Row %", "columns": "Column %", "all": "Total %"}),
            ui.input_checkbox("margins", "Show Margins", value=True),
            ui.input_action_button("run_btn", "Create Cross-tab", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        ui.output_ui("chi2_result"),
        ui.output_data_frame("crosstab_table"),
        code_panel_ui("code"),
    )


@module.server
def crosstab_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            cat_cols = get_categorical_columns(df)
            all_cols = list(df.columns)
            choices = cat_cols if cat_cols else all_cols
            ui.update_select("row_var", choices=choices)
            ui.update_select("col_var", choices=choices)

    @reactive.calc
    @reactive.event(input.run_btn)
    def result():
        df = get_current_df()
        req(df is not None)
        row = input.row_var()
        col = input.col_var()
        req(row, col)

        normalize = input.normalize() or None
        ct_result = create_crosstab(df, row, col, normalize=normalize, margins=input.margins())
        state.codegen.record(ct_result.code)
        last_code.set(ct_result.code.code)
        return ct_result

    @render.ui
    def chi2_result():
        r = result()
        req(r is not None)
        return ui.div(
            ui.h5("Chi-Square Test"),
            ui.p(r.interpretation),
            ui.tags.small(
                f"\u03c7\u00b2 = {r.chi2}, df = {r.dof}, p = {r.p_value}",
                class_="text-muted",
            ),
            class_="alert alert-info",
        )

    @render.data_frame
    def crosstab_table():
        r = result()
        req(r is not None)
        return render.DataGrid(r.table.reset_index(), height="400px")

    code_panel_server("code", get_code=last_code)
