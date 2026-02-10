"""Report module — placeholder for Phase 4."""

from shiny import module, ui


@module.ui
def notebook_ui():
    return ui.div(
        ui.h4("Session Report"),
        ui.p("Export your session as a Python script, Jupyter notebook, or HTML report."),
        ui.p("Coming in Phase 4.", class_="text-muted"),
    )


@module.server
def notebook_server(input, output, session, **kwargs):
    pass
