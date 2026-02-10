"""Data > Export module — download data as CSV or Excel."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.data.export import to_csv_bytes, to_excel_bytes


@module.ui
def export_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("format", "Export Format",
                choices={"csv": "CSV", "excel": "Excel (.xlsx)"}),
            ui.download_button("download_btn", "Download", class_="btn-primary w-100 mt-2"),
            width=250,
        ),
        ui.output_text("export_info"),
        ui.output_data_frame("export_preview"),
    )


@module.server
def export_server(input, output, session, state: WorkbenchState, get_current_df):

    @render.text
    def export_info():
        df = get_current_df()
        if df is None:
            return "No dataset selected."
        return f"Ready to export: {df.shape[0]} rows × {df.shape[1]} columns"

    @render.data_frame
    def export_preview():
        df = get_current_df()
        req(df is not None)
        return render.DataGrid(df.head(50), height="400px")

    @render.download(filename=lambda: f"data.{input.format()}" if input.format() == "csv" else "data.xlsx")
    def download_btn():
        df = get_current_df()
        req(df is not None)
        if input.format() == "csv":
            yield to_csv_bytes(df)
        else:
            yield to_excel_bytes(df)
