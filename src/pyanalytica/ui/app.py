"""PyAnalytica — Main Shiny application entry point."""

from __future__ import annotations

from shiny import App, reactive, render, ui

from pyanalytica.core.config import CourseConfig, is_menu_visible, load_config
from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.theme import apply_theme, get_theme
from pyanalytica.ui.components.dataset_selector import dataset_selector_server, dataset_selector_ui
from pyanalytica.ui.modules.data import (
    mod_combine, mod_export, mod_load, mod_profile, mod_transform, mod_view,
)
from pyanalytica.ui.modules.explore import mod_crosstab, mod_pivot, mod_summarize
from pyanalytica.ui.modules.visualize import (
    mod_compare, mod_correlate, mod_distribute, mod_relate, mod_timeline,
)
from pyanalytica.ui.modules.analyze import mod_correlation, mod_means, mod_proportions
from pyanalytica.ui.modules.model import (
    mod_classify, mod_cluster, mod_evaluate, mod_reduce, mod_regression,
)
from pyanalytica.ui.modules.homework import mod_homework
from pyanalytica.ui.modules.report import mod_notebook
from pyanalytica.ui.modules.ai import mod_assistant


def create_app(config: CourseConfig | None = None) -> App:
    """Create the PyAnalytica Shiny application."""
    if config is None:
        config = load_config()

    apply_theme(get_theme(config.theme))

    app_ui = ui.page_navbar(
        # === DATA ===
        ui.nav_panel("Data",
            ui.navset_tab(
                ui.nav_panel("Load", mod_load.load_ui("load")),
                ui.nav_panel("Profile", mod_profile.profile_ui("profile")),
                ui.nav_panel("View", mod_view.view_ui("view")),
                ui.nav_panel("Transform", mod_transform.transform_ui("transform")),
                ui.nav_panel("Combine", mod_combine.combine_ui("combine")),
                ui.nav_panel("Export", mod_export.export_ui("export")),
            ),
        ),
        # === EXPLORE ===
        ui.nav_panel("Explore",
            ui.navset_tab(
                ui.nav_panel("Summarize", mod_summarize.summarize_ui("summarize")),
                ui.nav_panel("Pivot", mod_pivot.pivot_ui("pivot")),
                ui.nav_panel("Cross-tab", mod_crosstab.crosstab_ui("crosstab")),
            ),
        ),
        # === VISUALIZE ===
        ui.nav_panel("Visualize",
            ui.navset_tab(
                ui.nav_panel("Distribute", mod_distribute.distribute_ui("distribute")),
                ui.nav_panel("Relate", mod_relate.relate_ui("relate")),
                ui.nav_panel("Compare", mod_compare.compare_ui("compare")),
                ui.nav_panel("Correlate", mod_correlate.correlate_ui("correlate")),
                ui.nav_panel("Timeline", mod_timeline.timeline_ui("timeline")),
            ),
        ),
        # === ANALYZE ===
        ui.nav_panel("Analyze",
            ui.navset_tab(
                ui.nav_panel("Means", mod_means.means_ui("means")),
                ui.nav_panel("Proportions", mod_proportions.proportions_ui("proportions")),
                ui.nav_panel("Correlation", mod_correlation.correlation_ui("correlation")),
            ),
        ),
        # === MODEL ===
        ui.nav_panel("Model",
            ui.navset_tab(
                ui.nav_panel("Regression", mod_regression.regression_ui("regression")),
                ui.nav_panel("Classify", mod_classify.classify_ui("classify")),
                ui.nav_panel("Evaluate", mod_evaluate.evaluate_ui("evaluate")),
                ui.nav_panel("Cluster", mod_cluster.cluster_ui("cluster")),
                ui.nav_panel("Reduce", mod_reduce.reduce_ui("reduce")),
            ),
        ),
        # === HOMEWORK ===
        ui.nav_panel("Homework", mod_homework.homework_ui("homework")),
        # === REPORT ===
        ui.nav_panel("Report", mod_notebook.notebook_ui("report")),
        # === AI ASSISTANT ===
        ui.nav_panel("AI Assistant", mod_assistant.assistant_ui("assistant")),
        header=ui.div(
            dataset_selector_ui("ds"),
            class_="container-fluid py-2",
        ),
        title="PyAnalytica",
        id="main_nav",
    )

    def server(input, output, session):
        state = WorkbenchState()
        state._change_signal = reactive.value(0)

        # Dataset selector
        selected_dataset = dataset_selector_server("ds", state=state)

        @reactive.calc
        def current_df():
            # Re-read when datasets change
            state._change_signal()
            name = selected_dataset()
            if not name or name not in state.datasets:
                return None
            return state.get(name)

        # Data modules
        mod_load.load_server("load", state=state, get_current_df=current_df)
        mod_profile.profile_server("profile", state=state, get_current_df=current_df)
        mod_view.view_server("view", state=state, get_current_df=current_df)
        mod_transform.transform_server("transform", state=state, get_current_df=current_df)
        mod_combine.combine_server("combine", state=state, get_current_df=current_df)
        mod_export.export_server("export", state=state, get_current_df=current_df)

        # Explore modules
        mod_summarize.summarize_server("summarize", state=state, get_current_df=current_df)
        mod_pivot.pivot_server("pivot", state=state, get_current_df=current_df)
        mod_crosstab.crosstab_server("crosstab", state=state, get_current_df=current_df)

        # Visualize modules
        mod_distribute.distribute_server("distribute", state=state, get_current_df=current_df)
        mod_relate.relate_server("relate", state=state, get_current_df=current_df)
        mod_compare.compare_server("compare", state=state, get_current_df=current_df)
        mod_correlate.correlate_server("correlate", state=state, get_current_df=current_df)
        mod_timeline.timeline_server("timeline", state=state, get_current_df=current_df)

        # Analyze modules
        mod_means.means_server("means", state=state, get_current_df=current_df)
        mod_proportions.proportions_server("proportions", state=state, get_current_df=current_df)
        mod_correlation.correlation_server("correlation", state=state, get_current_df=current_df)

        # Model modules
        mod_regression.regression_server("regression", state=state, get_current_df=current_df)
        mod_classify.classify_server("classify", state=state, get_current_df=current_df)
        mod_evaluate.evaluate_server("evaluate", state=state, get_current_df=current_df)
        mod_cluster.cluster_server("cluster", state=state, get_current_df=current_df)
        mod_reduce.reduce_server("reduce", state=state, get_current_df=current_df)

        # Homework module
        mod_homework.homework_server("homework", state=state, get_current_df=current_df)

        # Report module
        mod_notebook.notebook_server("report", state=state, get_current_df=current_df)

        # AI Assistant module
        mod_assistant.assistant_server("assistant", state=state, get_current_df=current_df)

    return App(app_ui, server)


app = create_app()


def main():
    """CLI entry point."""
    import shiny
    shiny.run_app("pyanalytica.ui.app:app")


if __name__ == "__main__":
    main()
