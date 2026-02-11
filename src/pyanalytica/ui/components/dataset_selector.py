"""Global dataset selector component."""

from __future__ import annotations

from shiny import module, reactive, render, ui

from pyanalytica.core.state import WorkbenchState


@module.ui
def dataset_selector_ui():
    """Dataset selector dropdown for the navbar."""
    return ui.div(
        ui.input_select(
            "dataset",
            "Active Dataset:",
            choices=["(none)"],
            width="250px",
        ),
        class_="d-flex align-items-center gap-3",
    )


@module.server
def dataset_selector_server(input, output, session, state: WorkbenchState):
    """Server logic for dataset selector. Returns reactive selected name."""

    @reactive.effect
    def _update_choices():
        # Read the change signal to create a reactive dependency
        if state._change_signal is not None:
            state._change_signal()
        names = state.dataset_names()
        choices = names if names else ["(none)"]
        ui.update_select("dataset", choices=choices)

    @reactive.calc
    def selected_name() -> str:
        name = input.dataset()
        if name == "(none)" or not name:
            return ""
        return name

    return selected_name
