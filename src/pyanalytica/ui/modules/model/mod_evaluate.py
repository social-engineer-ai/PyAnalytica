"""Model > Evaluate module — confusion matrix, ROC, metrics."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.ui.components.code_panel import code_panel_server, code_panel_ui


@module.ui
def evaluate_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.p("Run a classification model first (Classify tab), then evaluate results here."),
            ui.p("Evaluation integrates with the Classify module.", class_="text-muted small"),
            width=280,
        ),
        ui.div(
            ui.h5("Model Evaluation"),
            ui.p("Fit a model in the Classify tab first. Evaluation metrics will appear here."),
            ui.p("Features: Confusion matrix, Accuracy/Precision/Recall/F1, "
                 "ROC curve, AUC, Profit curve, Fairness metrics.", class_="text-muted"),
            class_="p-4",
        ),
        code_panel_ui("code"),
    )


@module.server
def evaluate_server(input, output, session, state: WorkbenchState, get_current_df):
    last_code = reactive.value("")
    code_panel_server("code", get_code=last_code)
