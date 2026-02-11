"""Model > Classify module — logistic regression, decision tree."""

from __future__ import annotations

from datetime import datetime

from shiny import module, reactive, render, req, ui

from pyanalytica.core.model_store import ModelArtifact
from pyanalytica.core.state import WorkbenchState
from pyanalytica.core.types import get_categorical_columns, get_numeric_columns
from pyanalytica.model.classify import decision_tree, logistic_regression
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui
from pyanalytica.ui.components.download_result import download_result_server, download_result_ui


@module.ui
def classify_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("model_type", "Model",
                choices={"logistic": "Logistic Regression", "tree": "Decision Tree"}),
            ui.input_select("target", "Target", choices=[]),
            ui.input_select("features", "Features", choices=[], multiple=True),
            ui.input_slider("test_size", "Test Split", 0.1, 0.5, 0.3, step=0.05),
            ui.input_numeric("random_seed", "Random Seed", value=42, min=0, max=99999),
            ui.output_ui("model_options"),
            ui.tags.hr(),
            ui.input_text("model_name", "Save Model As", placeholder="my_classifier"),
            ui.input_checkbox("save_splits", "Save train/test as datasets", value=False),
            ui.input_action_button("run_btn", "Fit Model", class_="btn-primary w-100 mt-2"),
            width=300,
        ),
        ui.output_ui("model_summary"),
        ui.output_data_frame("detail_table"),
        download_result_ui("dl"),
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
            seed = int(input.random_seed()) if input.random_seed() is not None else 42
            if input.model_type() == "logistic":
                r = logistic_regression(df, target, features,
                                        test_size=input.test_size(), random_state=seed)
                mt = "logistic_regression"
            else:
                r = decision_tree(df, target, features,
                                  test_size=input.test_size(), max_depth=input.max_depth(),
                                  random_state=seed)
                mt = "decision_tree"
            result.set(r)
            state.codegen.record(r.code)
            last_code.set(r.code.code)

            # Save model artifact
            model_name = input.model_name().strip()
            if model_name:
                artifact = ModelArtifact(
                    name=model_name,
                    model_type=mt,
                    model=r.model,
                    feature_names=r.feature_names,
                    target_name=target,
                    label_encoder=r.label_encoder,
                    created_at=datetime.now(),
                    X_train=r.X_train,
                    X_test=r.X_test,
                    y_train=r.y_train,
                    y_test=r.y_test,
                )
                state.model_store.save(model_name, artifact)
                state._notify()
                ui.notification_show(f"Model '{model_name}' saved.", type="message")

            # Save train/test splits as datasets
            if input.save_splits():
                base = model_name or "classifier"
                saved = []
                if r.X_train is not None and r.y_train is not None:
                    train_df = r.X_train.copy()
                    train_df[target] = r.y_train.values
                    state.load(f"{base}_train", train_df)
                    saved.append(f"{base}_train")
                if r.X_test is not None and r.y_test is not None:
                    test_df = r.X_test.copy()
                    test_df[target] = r.y_test.values
                    state.load(f"{base}_test", test_df)
                    saved.append(f"{base}_test")
                if saved:
                    ui.notification_show(f"Saved datasets: {', '.join(saved)}", type="message")
                else:
                    ui.notification_show("No train/test data to save (set Test Split > 0).", type="warning")

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

    def _get_classify_detail():
        r = result()
        if r is None:
            return None
        if r.coefficients is not None:
            return r.coefficients
        return r.feature_importance

    download_result_server("dl", get_df=_get_classify_detail, filename="model_details")
    code_panel_server("code", get_code=last_code)
