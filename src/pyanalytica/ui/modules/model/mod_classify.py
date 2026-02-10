"""Model > Classify module — logistic regression, decision tree."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_categorical_columns, get_numeric_columns
from pyanalytica.model.classify import decision_tree, logistic_regression
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def classify_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("model_type", "Model",
                choices={"logistic": "Logistic Regression", "tree": "Decision Tree"}),
            ui.input_select("target", "Target", choices=[]),
            ui.input_select("features", "Features", choices=[], multiple=True),
            ui.input_slider("test_size", "Test Split", 0.1, 0.5, 0.3, step=0.05),
            ui.output_ui("model_options"),
            ui.input_action_button("run_btn", "Fit Model", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        ui.output_ui("model_summary"),
        ui.output_data_frame("detail_table"),
        code_panel_ui("code"),
    )


@module.server
def classify_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    result = reactive.value(None)

    @reactive.effect
    def _update_cols():
        df = get_current_df()
        if df is not None:
            cats = get_categorical_columns(df)
            nums = get_numeric_columns(df)
            ui.update_select("target", choices=cats + nums)
            ui.update_select("features", choices=nums)

    @render.ui
    def model_options():
        if input.model_type() == "tree":
            return ui.input_slider("max_depth", "Max Depth", 1, 20, 5)
        return ui.div()

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        df = get_current_df()
        req(df is not None)
        target = input.target()
        features = list(input.features())
        req(target, features)
        try:
            if input.model_type() == "logistic":
                r = logistic_regression(df, target, features, test_size=input.test_size())
            else:
                r = decision_tree(df, target, features, test_size=input.test_size(), max_depth=input.max_depth())
            result.set(r)
            state.codegen.record(r.code)
            last_code.set(r.code.code)
        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error")

    @render.ui
    def model_summary():
        r = result()
        req(r is not None)
        test_str = f" | Test: {r.test_accuracy:.3f}" if r.test_accuracy else ""
        return ui.div(
            ui.h5(r.model_type),
            ui.p(f"Train accuracy: {r.train_accuracy:.3f}{test_str}"),
            class_="alert alert-info",
        )

    @render.data_frame
    def detail_table():
        r = result()
        req(r is not None)
        if r.coefficients is not None:
            return render.DataGrid(r.coefficients)
        elif r.feature_importance is not None:
            return render.DataGrid(r.feature_importance)
        return render.DataGrid(None)

    code_panel_server("code", get_code=last_code)
