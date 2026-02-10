"""Chat panel component — placeholder for Phase 5."""

from shiny import module, ui


@module.ui
def chat_panel_ui():
    return ui.div()


@module.server
def chat_panel_server(input, output, session, **kwargs):
    pass
