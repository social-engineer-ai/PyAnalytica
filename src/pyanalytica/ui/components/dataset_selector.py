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
        ui.input_action_link("remove_dataset", "Remove", class_="text-danger small"),
        # Confirmation guard: intercept click and require user consent
        ui.tags.script(ui.HTML("""
            $(document).on('click', '[id$="remove_dataset"]', function(e) {
                if (!confirm('Remove this dataset? This cannot be undone.')) {
                    e.preventDefault();
                    e.stopImmediatePropagation();
                }
            });
        """)),
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

    @reactive.effect
    @reactive.event(input.remove_dataset)
    def _remove_dataset():
        name = input.dataset()
        if not name or name == "(none)":
            return
        state.remove(name)
        remaining = state.dataset_names()
        if remaining:
            ui.update_select("dataset", choices=remaining, selected=remaining[0])
        else:
            ui.update_select("dataset", choices=["(none)"], selected="(none)")
        ui.notification_show(f"Removed dataset '{name}'", type="message")

    @reactive.calc
    def selected_name() -> str:
        name = input.dataset()
        if name == "(none)" or not name:
            return ""
        return name

    return selected_name
