"""Model > Reduce module — PCA."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_numeric_columns
from pyanalytica.model.reduce import pca_analysis
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui
from pyanalytica.ui.components.download_result import download_result_server, download_result_ui


@module.ui
def reduce_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("features", "Features", choices=[], multiple=True),
            ui.input_action_button("run_btn", "Run PCA", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        ui.output_ui("pca_summary"),
        ui.output_plot("scree_plot", height="350px"),
        ui.output_plot("biplot", height="400px"),
        ui.h5("Loadings"),
        ui.output_data_frame("loadings"),
        download_result_ui("dl"),
        ui.p("PCA reveals structure in your data. It's exploratory — not predictive.", class_="text-muted small mt-2"),
        code_panel_ui("code"),
    )


@module.server
def reduce_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    result = reactive.value(None)

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            ui.update_select("features", choices=get_numeric_columns(df))

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        df = get_current_df()
        req(df is not None)
        features = list(input.features())
        req(len(features) >= 2)
        try:
            r = pca_analysis(df, features)
            result.set(r)
            state.codegen.record(r.code)
            last_code.set(r.code.code)
        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error")

    @render.ui
    def pca_summary():
        r = result()
        req(r is not None)
        return ui.div(
            ui.h5("PCA Results"),
            ui.p(f"Recommended components: {r.recommended_n} (>80% variance explained)"),
            ui.p(f"Total variance explained: {r.cumulative_variance[-1]*100:.1f}%"),
            class_="alert alert-info",
        )

    @render.plot
    def scree_plot():
        r = result()
        req(r is not None and r.scree_plot is not None)
        return r.scree_plot

    @render.plot
    def biplot():
        r = result()
        req(r is not None and r.biplot is not None)
        return r.biplot

    @render.data_frame
    def loadings():
        r = result()
        req(r is not None)
        return render.DataGrid(r.loadings.reset_index().rename(columns={"index": "Variable"}))

    download_result_server(
        "dl",
        get_df=lambda: result().loadings.reset_index(),
        filename="pca_loadings",
    )
    code_panel_server("code", get_code=last_code)
