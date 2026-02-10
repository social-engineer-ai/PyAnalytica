"""AI Assistant module — placeholder for Phase 5."""

from shiny import module, ui


@module.ui
def assistant_ui():
    return ui.div(
        ui.h4("AI Assistant"),
        ui.p("Socratic agent — coming in Phase 5.", class_="text-muted"),
    )


@module.server
def assistant_server(input, output, session, **kwargs):
    pass
