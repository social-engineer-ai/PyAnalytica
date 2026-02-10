"""Data > Load module — load datasets from bundled, upload, or URL."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core import round_df
from pyanalytica.core.state import Operation, WorkbenchState
from pyanalytica.data.load import load_bundled, load_csv, load_from_bytes, load_url
from pyanalytica.datasets import list_datasets
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui

from datetime import datetime


@module.ui
def load_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_radio_buttons(
                "source", "Data Source",
                choices={"bundled": "Bundled Dataset", "upload": "Upload File", "url": "From URL"},
            ),
            ui.output_ui("source_controls"),
            ui.input_action_button("load_btn", "Load Dataset", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        ui.h4("Preview"),
        ui.output_text("load_info"),
        ui.output_data_frame("preview_table"),
        code_panel_ui("code"),
    )


@module.server
def load_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")

    @render.ui
    def source_controls():
        src = input.source()
        if src == "bundled":
            return ui.input_select("bundled_name", "Dataset", choices=list_datasets())
        elif src == "upload":
            return ui.input_file("file_upload", "Upload CSV/Excel", accept=[".csv", ".xlsx", ".xls", ".tsv"])
        elif src == "url":
            return ui.TagList(
                ui.input_text("data_url", "URL", placeholder="https://..."),
                ui.input_text("url_name", "Dataset Name", placeholder="my_data"),
            )

    @reactive.effect
    @reactive.event(input.load_btn)
    def _load():
        src = input.source()
        try:
            if src == "bundled":
                name = input.bundled_name()
                req(name)
                df, snippet = load_bundled(name)
            elif src == "upload":
                file_info = input.file_upload()
                req(file_info)
                f = file_info[0]
                name = f["name"].rsplit(".", 1)[0]
                with open(f["datapath"], "rb") as fh:
                    content = fh.read()
                df, snippet = load_from_bytes(content, f["name"])
            elif src == "url":
                url = input.data_url()
                name = input.url_name() or "url_data"
                req(url)
                df, snippet = load_url(url)
            else:
                return

            state.load(name, df)
            state.codegen.record(snippet)
            last_code.set(snippet.code)
            ui.notification_show(f"Loaded '{name}': {df.shape[0]} rows, {df.shape[1]} columns", type="message")
        except Exception as e:
            ui.notification_show(f"Error loading data: {e}", type="error")

    @render.text
    def load_info():
        df = get_current_df()
        if df is None:
            return "No dataset loaded."
        return f"Shape: {df.shape[0]} rows × {df.shape[1]} columns | Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB"

    @render.data_frame
    def preview_table():
        df = get_current_df()
        req(df is not None)
        return render.DataGrid(round_df(df.head(100), state._decimals()), height="400px")

    code_panel_server("code", get_code=last_code)
