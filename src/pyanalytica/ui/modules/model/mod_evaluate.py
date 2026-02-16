"""Model > Evaluate module — confusion matrix, ROC, metrics for saved models."""

from __future__ import annotations

import numpy as np

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.model.evaluate import evaluate_classification
from pyanalytica.ui.components.add_to_report import add_to_report_server, add_to_report_ui
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui
from pyanalytica.ui.components.download_result import download_result_server, download_result_ui


@module.ui
def evaluate_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("model_name", "Saved Model", choices=[]),
            ui.input_select("eval_data", "Evaluate On",
                choices={"test": "Test Set", "train": "Training Set"}),
            ui.input_action_button("run_btn", "Evaluate", class_="btn-primary w-100 mt-2"),
            ui.tags.hr(),
            ui.output_ui("threshold_ui"),
            width=300,
        ),
        ui.output_ui("metrics_summary"),
        ui.h5("Confusion Matrix"),
        ui.output_data_frame("cm_table"),
        download_result_ui("dl"),
        ui.output_plot("roc_plot", height="400px"),
        add_to_report_ui("rpt"),
        code_panel_ui("code"),
    )


@module.server
def evaluate_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    last_report_info = reactive.value(None)
    eval_result = reactive.value(None)
    is_binary = reactive.value(False)

    @reactive.effect
    def _update_models():
        # Re-read when datasets/models change
        state._change_signal()
        models = state.model_store.list_models()
        ui.update_select("model_name", choices=models)

    @render.ui
    def threshold_ui():
        if is_binary():
            return ui.input_slider("threshold", "Classification Threshold", 0.0, 1.0, 0.5, step=0.01)
        return ui.div()

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        model_name = input.model_name()
        req(model_name)
        try:
            artifact = state.model_store.get(model_name)

            # Get evaluation data
            eval_on = input.eval_data()
            if eval_on == "test":
                X = artifact.X_test
                y_encoded = artifact.y_test
            else:
                X = artifact.X_train
                y_encoded = artifact.y_train

            if X is None or y_encoded is None:
                ui.notification_show("No data available for selected split.", type="warning")
                return

            y_true = np.asarray(y_encoded)

            # Get predictions
            y_pred = artifact.model.predict(X)

            # Get probabilities if available
            y_prob = None
            if hasattr(artifact.model, "predict_proba"):
                try:
                    proba = artifact.model.predict_proba(X)
                    if proba.shape[1] == 2:
                        y_prob = proba[:, 1]
                except Exception:
                    pass

            # Decode labels for display if label encoder exists
            if artifact.label_encoder is not None:
                y_true_display = artifact.label_encoder.inverse_transform(y_true.astype(int))
                y_pred_display = artifact.label_encoder.inverse_transform(y_pred.astype(int))
            else:
                y_true_display = y_true
                y_pred_display = y_pred

            is_binary.set(len(set(y_true)) == 2)

            r = evaluate_classification(y_true_display, y_pred_display, y_prob=y_prob)
            eval_result.set(r)
            state.codegen.record(r.code, action="model", description="Model evaluation")
            last_code.set(r.code.code)
            last_report_info.set(("model", "Model evaluation", r.code.code, r.code.imports))

        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error")

    @render.ui
    def metrics_summary():
        r = eval_result()
        req(r is not None)
        auc_str = f" | AUC = {r.auc:.4f}" if r.auc is not None else ""
        return ui.div(
            ui.h5("Classification Metrics"),
            ui.p(
                f"Accuracy: {r.accuracy:.4f} | Precision: {r.precision:.4f} | "
                f"Recall: {r.recall:.4f} | F1: {r.f1:.4f}{auc_str}"
            ),
            class_="alert alert-info",
        )

    @render.data_frame
    def cm_table():
        r = eval_result()
        req(r is not None)
        return render.DataGrid(r.confusion_matrix.reset_index())

    @render.plot
    def roc_plot():
        r = eval_result()
        req(r is not None and r.roc_curve_plot is not None)
        return r.roc_curve_plot

    download_result_server(
        "dl",
        get_df=lambda: eval_result().confusion_matrix.reset_index(),
        filename="confusion_matrix",
    )
    add_to_report_server("rpt", state=state, get_code_info=last_report_info)
    code_panel_server("code", get_code=last_code)
