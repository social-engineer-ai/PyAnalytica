"""Explore > Pivot module — pivot tables."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core import round_df
from pyanalytica.core.state import WorkbenchState
from pyanalytica.explore.pivot import create_pivot_table
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui
from pyanalytica.ui.components.decimals_control import decimals_server, decimals_ui
from pyanalytica.ui.components.download_result import download_result_server, download_result_ui


@module.ui
def pivot_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("index", "Row Variable", choices=[]),
            ui.input_select("columns", "Column Variable", choices=[]),
            ui.input_select("values", "Value Variable", choices=[]),
            ui.input_select("aggfunc", "Aggregation",
                choices=["count", "sum", "mean", "median", "min", "max"]),
            ui.input_checkbox("margins", "Show Margins", value=True),
            ui.input_select("normalize", "Normalize",
                choices={"": "None", "index": "Row %", "columns": "Column %", "all": "Total %"}),
            ui.input_action_button("run_btn", "Create Pivot", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        decimals_ui("dec"),
        ui.output_data_frame("pivot_table"),
        download_result_ui("dl"),
        code_panel_ui("code"),
    )


@module.server
def pivot_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    get_dec = decimals_server("dec")

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            cols = list(df.columns)
            ui.update_select("index", choices=cols)
            ui.update_select("columns", choices=cols)
            ui.update_select("values", choices=cols)

    @reactive.calc
    @reactive.event(input.run_btn)
    def result():
        df = get_current_df()
        req(df is not None)
        idx = input.index()
        cols = input.columns()
        vals = input.values()
        req(idx, cols, vals)

        normalize = input.normalize() or None
        result_df, snippet = create_pivot_table(
            df, idx, cols, vals,
            aggfunc=input.aggfunc(),
            margins=input.margins(),
            normalize=normalize,
        )
        state.codegen.record(snippet)
        last_code.set(snippet.code)
        return result_df.reset_index()

    @render.data_frame
    def pivot_table():
        df = result()
        req(df is not None)
        return render.DataGrid(round_df(df, get_dec()), height="500px")

    download_result_server("dl", get_df=result, filename="pivot_table")
    code_panel_server("code", get_code=last_code)
