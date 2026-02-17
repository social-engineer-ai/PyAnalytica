"""Model > Cluster module — K-means, hierarchical."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_numeric_columns
from pyanalytica.model.cluster import hierarchical_cluster, kmeans_cluster
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui
from pyanalytica.ui.components.download_result import download_result_server, download_result_ui


@module.ui
def cluster_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("method", "Method",
                choices={"kmeans": "K-Means", "hierarchical": "Hierarchical"}),
            ui.input_select("features", "Features", choices=[], multiple=True),
            ui.input_slider("n_clusters", "Number of Clusters", 2, 15, 3),
            ui.input_action_button("run_btn", "Run Clustering", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        ui.output_ui("cluster_summary"),
        ui.output_plot("elbow_plot", height="350px"),
        ui.output_plot("scatter_plot", height="350px"),
        ui.h5("Cluster Profiles"),
        ui.output_data_frame("profiles"),
        download_result_ui("dl"),
        code_panel_ui("code"),
    )


@module.server
def cluster_server(input, output, session, state: WorkbenchState, get_current_df):
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
            if input.method() == "kmeans":
                r = kmeans_cluster(df, features, chosen_k=input.n_clusters())
            else:
                r = hierarchical_cluster(df, features, n_clusters=input.n_clusters())
            result.set(r)
            state.codegen.record(r.code, action="model", description="Cluster analysis")
            last_code.set(r.code.code)
        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error")

    @render.ui
    def cluster_summary():
        r = result()
        req(r is not None)
        return ui.div(
            ui.h5(f"Clustering: {r.n_clusters} clusters"),
            ui.p("Note: Clusters are analytical conveniences, not fixed types in reality.", class_="text-muted small"),
            class_="alert alert-info",
        )

    @render.plot
    def elbow_plot():
        r = result()
        req(r is not None and r.elbow_plot is not None)
        return r.elbow_plot

    @render.plot
    def scatter_plot():
        r = result()
        req(r is not None and r.scatter_plot is not None)
        return r.scatter_plot

    @render.data_frame
    def profiles():
        r = result()
        req(r is not None)
        return render.DataGrid(r.cluster_profiles.reset_index())

    download_result_server(
        "dl",
        get_df=lambda: result().cluster_profiles.reset_index(),
        filename="cluster_profiles",
    )
    code_panel_server("code", get_code=last_code)
