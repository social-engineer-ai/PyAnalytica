"""Reusable 'Add to Report' button component for all modules."""

from __future__ import annotations

from typing import Callable

from shiny import module, reactive, ui

from pyanalytica.core.state import WorkbenchState


@module.ui
def add_to_report_ui():
    return ui.input_action_button(
        "add_to_report", "Add to Report",
        class_="btn-sm btn-outline-success mt-1",
    )


@module.server
def add_to_report_server(
    input, output, session,
    state: WorkbenchState,
    get_code_info: Callable[[], tuple[str, str, str, list[str]] | None],
):
    """Server logic for Add to Report button.

    Parameters
    ----------
    state : WorkbenchState
        Shared state with report_builder instance.
    get_code_info : callable
        Returns ``(action, description, code, imports)`` or ``None``.
    """

    @reactive.effect
    @reactive.event(input.add_to_report)
    def _add():
        info = get_code_info()
        if info is None:
            ui.notification_show(
                "No result to add. Run an action first.", type="warning",
            )
            return
        action, description, code, imports = info
        state.report_builder.add_code_cell(
            action=action, description=description,
            code=code, imports=imports,
        )
        state._notify_report()
        ui.notification_show("Added to Report Builder.", type="message")
