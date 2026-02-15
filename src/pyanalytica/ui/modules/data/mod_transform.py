"""Data > Transform module — missing values, dtypes, duplicates, new columns, string ops."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

import pandas as pd

from pyanalytica.core import round_df
from pyanalytica.core.state import Operation, WorkbenchState
from pyanalytica.data import transform
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui
from pyanalytica.ui.components.decimals_control import decimals_server, decimals_ui

from datetime import datetime


@module.ui
def transform_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("action", "Transform Action", choices={
                "fill_missing": "Fill Missing Values",
                "drop_missing": "Drop Missing Rows",
                "drop_columns": "Drop Column(s)",
                "convert_dtype": "Convert Data Type",
                "drop_duplicates": "Drop Duplicates",
                "dummy_encode": "Dummy Encode (One-Hot)",
                "ordinal_encode": "Ordinal Encode",
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
        decimals_ui("dec"),
        ui.output_data_frame("preview"),
        code_panel_ui("code"),
    )


@module.server
def transform_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    get_dec = decimals_server("dec")
    _prev_ds_id = reactive.value(None)

    @reactive.effect
    def _track_dataset_change():
        """Force UI refresh when the active dataset changes."""
        df = get_current_df()
        new_id = id(df) if df is not None else None
        if new_id != _prev_ds_id():
            _prev_ds_id.set(new_id)

    @render.ui
    def action_controls():
        _prev_ds_id()  # re-render when dataset changes
        df = get_current_df()
        cols = list(df.columns) if df is not None else []
        action = input.action()

        # For encoding actions, show only object/category columns
        if action in ("dummy_encode", "ordinal_encode") and df is not None:
            cat_cols = [c for c in cols if df[c].dtype == object
                        or isinstance(df[c].dtype, pd.CategoricalDtype)]
            col_choices = cat_cols if cat_cols else cols
        else:
            col_choices = cols

        if action == "drop_columns":
            controls = [ui.input_selectize("drop_cols", "Columns to Drop",
                choices=cols, multiple=True)]
        else:
            controls = [ui.input_select("col", "Column", choices=col_choices)]

        if action == "fill_missing":
            controls.append(ui.input_select("fill_method", "Method",
                choices=["mean", "median", "mode", "ffill", "bfill", "value"]))
            controls.append(ui.input_text("fill_value", "Value (if method=value)"))
        elif action == "convert_dtype":
            controls.append(ui.input_select("target_dtype", "Target Type",
                choices=["int", "float", "str", "category", "datetime", "bool"]))
        elif action in ("add_log", "add_zscore", "add_rank"):
            controls.append(ui.input_text("new_col_name", "New Column Name"))
        elif action == "dummy_encode":
            controls.append(ui.input_checkbox("drop_first", "Drop first level", value=False))
        elif action == "ordinal_encode":
            controls.append(ui.input_text("ordinal_order",
                "Category order (comma-separated, optional)",
                placeholder="e.g. low, medium, high"))

        return ui.TagList(*controls)

    @reactive.effect
    @reactive.event(input.apply_btn)
    def _apply():
        df = get_current_df()
        req(df is not None)
        action = input.action()

        try:
            if action == "drop_columns":
                drop_cols = list(input.drop_cols())
                req(len(drop_cols) > 0)
                result, snippet = transform.drop_columns(df, drop_cols)
            else:
                col = input.col()
                req(col)
                if action == "fill_missing":
                    method = input.fill_method()
                    val = input.fill_value() if method == "value" else None
                    n_missing = df[col].isna().sum()
                    if n_missing == 0:
                        ui.notification_show(
                            f"Column '{col}' has no missing values. Nothing to fill.",
                            type="warning",
                        )
                        return
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
                elif action == "dummy_encode":
                    drop_first = input.drop_first()
                    result, snippet = transform.dummy_encode(df, col, drop_first=drop_first)
                elif action == "ordinal_encode":
                    order_str = input.ordinal_order().strip()
                    order = [s.strip() for s in order_str.split(",") if s.strip()] if order_str else None
                    result, snippet = transform.ordinal_encode(df, col, order=order)
                else:
                    return

            # Find and update the dataset
            desc = f"{action}" if action == "drop_columns" else f"{action} on '{input.col()}'"
            for name in state.dataset_names():
                if state.get(name) is get_current_df():
                    state.update(name, result, Operation(
                        timestamp=datetime.now(), action="transform",
                        description=desc, dataset=name,
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
        n_missing = df.isna().sum().sum()
        return f"{df.shape[0]} rows × {df.shape[1]} columns | {n_missing} missing values"

    @render.data_frame
    def preview():
        df = get_current_df()
        req(df is not None)
        return render.DataGrid(round_df(df.head(100), get_dec()), height="400px")

    code_panel_server("code", get_code=last_code)
