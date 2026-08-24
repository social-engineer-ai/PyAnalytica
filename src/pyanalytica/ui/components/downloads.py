"""One download decorator that works across Shiny versions.

Shiny 1.7 renamed ``render.download`` to ``render.download_button`` and made
the old name emit a deprecation warning at import time — once per decorated
function. PyAnalytica has ten of them, several inside reusable components, so
a student starting the app saw more than twenty warnings scroll past and
reasonably assumed something had broken. One emailed the course address about
it on the first day.

``pyproject.toml`` asks for ``shiny>=1.0``, so students get whatever is current
while a developer may be pinned to something older. Picking the new name when
it exists keeps the console clean on new versions and keeps the app working on
old ones, without forcing a floor bump on everybody.

Usage is identical to the decorator it replaces::

    @render_download(filename="data.csv")
    def download_btn():
        yield csv_bytes
"""

from __future__ import annotations

from shiny import render

# download_button arrived in Shiny 1.7 with the same signature; download is
# still present there but deprecated. Prefer the new name where it exists.
render_download = getattr(render, "download_button", render.download)

__all__ = ["render_download"]
