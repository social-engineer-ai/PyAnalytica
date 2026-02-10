"""Homework task panel component — placeholder for Phase 4."""

from shiny import module, ui


@module.ui
def homework_panel_ui():
    return ui.div()


@module.server
def homework_panel_server(input, output, session, **kwargs):
    pass
