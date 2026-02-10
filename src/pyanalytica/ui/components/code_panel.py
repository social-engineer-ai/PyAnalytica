"""Collapsible 'Show Code' panel with copy button."""

from __future__ import annotations

from typing import Callable

from shiny import module, reactive, render, ui


@module.ui
def code_panel_ui():
    """Show Code panel UI."""
    return ui.div(
        ui.input_action_button("toggle_code", "Show Code", class_="btn-sm btn-outline-secondary"),
        ui.output_ui("code_display"),
        class_="mt-3",
    )


@module.server
def code_panel_server(input, output, session, get_code: Callable[[], str]):
    """Server logic for code panel."""
    show_code = reactive.value(False)

    @reactive.effect
    @reactive.event(input.toggle_code)
    def _toggle():
        show_code.set(not show_code())
        label = "Hide Code" if show_code() else "Show Code"
        ui.update_action_button("toggle_code", label=label)

    @render.ui
    def code_display():
        if not show_code():
            return ui.div()
        code = get_code()
        if not code:
            return ui.div("No code generated yet.", class_="text-muted")
        return ui.div(
            ui.tags.pre(
                ui.tags.code(code, class_="language-python"),
                class_="bg-light p-3 rounded border",
                style="max-height: 400px; overflow-y: auto;",
            ),
            class_="mt-2",
        )
