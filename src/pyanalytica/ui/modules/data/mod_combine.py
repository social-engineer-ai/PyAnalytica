"""Data > Combine module — merge, append, reshape."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import Operation, WorkbenchState
from pyanalytica.data.combine import merge_dataframes
from pyanalytica.ui.components.add_to_report import add_to_report_server, add_to_report_ui
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui

from datetime import datetime


@module.ui
def combine_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Merge Two Datasets"),
            ui.input_select("left", "Left Dataset", choices=[]),
            ui.input_select("right", "Right Dataset", choices=[]),
            ui.input_select("join_key", "Join Key", choices=[]),
            ui.input_select("how", "Join Type",
                choices=["inner", "left", "right", "outer"]),
            ui.input_text("result_name", "Result Name", value="merged"),
            ui.input_action_button("merge_btn", "Merge", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        ui.output_text("merge_info"),
        ui.output_data_frame("merge_preview"),
        add_to_report_ui("rpt"),
        code_panel_ui("code"),
    )


@module.server
def combine_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    last_report_info = reactive.value(None)

    @reactive.effect
    def _update_datasets():
        # Read change signal to re-fire when datasets are added/removed
        if state._change_signal is not None:
            state._change_signal()
        names = state.dataset_names()
        ui.update_select("left", choices=names)
        ui.update_select("right", choices=names)

    @reactive.effect
    def _update_keys():
        left_name = input.left()
        right_name = input.right()
        if left_name and right_name and left_name in state.datasets and right_name in state.datasets:
            left_cols = set(state.get(left_name).columns)
            right_cols = set(state.get(right_name).columns)
            common = sorted(left_cols & right_cols)
            ui.update_select("join_key", choices=common)

    @reactive.effect
    @reactive.event(input.merge_btn)
    def _merge():
        left_name = input.left()
        right_name = input.right()
        key = input.join_key()
        how = input.how()
        result_name = input.result_name() or "merged"
        req(left_name, right_name, key)

        try:
            result = merge_dataframes(
                state.get(left_name), state.get(right_name),
                on=key, how=how,
                left_name=left_name, right_name=right_name,
            )
            state.load(result_name, result.merged)
            state.codegen.record(result.code)
            last_code.set(result.code.code)
            last_report_info.set(("merge", f"Merge {left_name} + {right_name}", result.code.code, result.code.imports))

            msg = (
                f"Merged: {result.result_rows} rows | "
                f"Left unmatched: {result.left_unmatched} | "
                f"Right unmatched: {result.right_unmatched}"
            )
            ui.notification_show(msg, type="message")
        except Exception as e:
            ui.notification_show(f"Merge error: {e}", type="error")

    @render.text
    def merge_info():
        return "Select two datasets and a common key column to merge."

    @render.data_frame
    def merge_preview():
        df = get_current_df()
        req(df is not None)
        return render.DataGrid(df.head(100), height="400px")

    add_to_report_server("rpt", state=state, get_code_info=last_report_info)
    code_panel_server("code", get_code=last_code)
