"""Data > Transform module — missing values, dtypes, duplicates, new columns, string ops."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import Operation, WorkbenchState
from pyanalytica.data import transform
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui

from datetime import datetime


@module.ui
def transform_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("action", "Transform Action", choices={
                "fill_missing": "Fill Missing Values",
                "drop_missing": "Drop Missing Rows",
                "convert_dtype": "Convert Data Type",
                "drop_duplicates": "Drop Duplicates",
                "add_log": "Add Log Column",
                "add_zscore": "Add Z-score Column",
                "add_rank": "Add Rank Column",
                "str_lower": "String: Lowercase",
                "str_upper": "String: Uppercase",
                "str_strip": "String: Strip Whitespace",
            }),
            ui.output_ui("action_controls"),
            ui.input_action_button("apply_btn", "Apply", class_="btn-primary w-100 mt-2"),
            width=320,
        ),
        ui.output_text("transform_info"),
        ui.output_data_frame("preview"),
        code_panel_ui("code"),
    )


@module.server
def transform_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")

    @render.ui
    def action_controls():
        df = get_current_df()
        cols = list(df.columns) if df is not None else []
        action = input.action()

        controls = [ui.input_select("col", "Column", choices=cols)]

        if action == "fill_missing":
            controls.append(ui.input_select("fill_method", "Method",
                choices=["mean", "median", "mode", "ffill", "bfill", "value"]))
            controls.append(ui.input_text("fill_value", "Value (if method=value)"))
        elif action == "convert_dtype":
            controls.append(ui.input_select("target_dtype", "Target Type",
                choices=["int", "float", "str", "category", "datetime", "bool"]))
        elif action in ("add_log", "add_zscore", "add_rank"):
            controls.append(ui.input_text("new_col_name", "New Column Name"))

        return ui.TagList(*controls)

    @reactive.effect
    @reactive.event(input.apply_btn)
    def _apply():
        df = get_current_df()
        req(df is not None)
        action = input.action()
        col = input.col()
        req(col)

        try:
            if action == "fill_missing":
                method = input.fill_method()
                val = input.fill_value() if method == "value" else None
                result, snippet = transform.fill_missing(df, col, method, val)
            elif action == "drop_missing":
                result, snippet = transform.drop_missing(df, [col])
            elif action == "convert_dtype":
                result, snippet = transform.convert_dtype(df, col, input.target_dtype())
            elif action == "drop_duplicates":
                result, snippet = transform.drop_duplicates(df, [col])
            elif action == "add_log":
                new_name = input.new_col_name() or f"{col}_log"
                result, snippet = transform.add_column_log(df, new_name, col)
            elif action == "add_zscore":
                new_name = input.new_col_name() or f"{col}_zscore"
                result, snippet = transform.add_column_zscore(df, new_name, col)
            elif action == "add_rank":
                new_name = input.new_col_name() or f"{col}_rank"
                result, snippet = transform.add_column_rank(df, new_name, col)
            elif action == "str_lower":
                result, snippet = transform.str_lower(df, col)
            elif action == "str_upper":
                result, snippet = transform.str_upper(df, col)
            elif action == "str_strip":
                result, snippet = transform.str_strip(df, col)
            else:
                return

            # Find and update the dataset
            for name in state.dataset_names():
                if state.get(name) is get_current_df():
                    state.update(name, result, Operation(
                        timestamp=datetime.now(), action="transform",
                        description=f"{action} on '{col}'", dataset=name,
                    ))
                    state.codegen.record(snippet)
                    last_code.set(snippet.code)
                    ui.notification_show(f"Transform applied: {action}", type="message")
                    break

        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error")

    @render.text
    def transform_info():
        df = get_current_df()
        if df is None:
            return "No dataset selected."
        return f"{df.shape[0]} rows × {df.shape[1]} columns"

    @render.data_frame
    def preview():
        df = get_current_df()
        req(df is not None)
        return render.DataGrid(df.head(100), height="400px")

    code_panel_server("code", get_code=last_code)
