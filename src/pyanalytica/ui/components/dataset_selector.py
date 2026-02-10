"""Global dataset selector component."""

from __future__ import annotations

from shiny import module, reactive, render, ui

from pyanalytica.core.state import WorkbenchState


@module.ui
def dataset_selector_ui():
    """Dataset selector dropdown and decimals control for the navbar."""
    return ui.div(
        ui.input_select(
            "dataset",
            "Active Dataset:",
            choices=["(none)"],
            width="250px",
        ),
        ui.input_select(
            "decimals",
            "Decimals:",
            choices={"2": "2", "3": "3", "4": "4", "5": "5", "6": "6"},
            selected="4",
            width="80px",
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

    # Attach decimals accessor to state so all modules can use it
    state._decimals = lambda: int(input.decimals())

    return selected_name
