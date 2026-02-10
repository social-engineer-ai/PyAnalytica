"""Homework module — placeholder for Phase 4."""

from shiny import module, ui


@module.ui
def homework_ui():
    return ui.div(
        ui.h4("Homework"),
        ui.p("Load a homework YAML file to get started."),
        ui.p("Coming in Phase 4.", class_="text-muted"),
    )


@module.server
def homework_server(input, output, session, **kwargs):
    pass
