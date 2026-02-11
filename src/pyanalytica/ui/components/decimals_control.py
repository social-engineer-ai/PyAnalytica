"""Reusable inline decimals selector for data-display modules."""

from __future__ import annotations

from shiny import module, reactive, ui

from pyanalytica.core.profile import get_profile


@module.ui
def decimals_ui():
    """Small inline decimals selector.

    The initial selection comes from the user profile
    (``~/.pyanalytica/profile.yaml`` ``defaults.decimals``).
    """
    default = str(get_profile().decimals)
    # Clamp to allowed choices
    if default not in ("2", "3", "4", "5", "6"):
        default = "4"
    return ui.div(
        ui.input_select(
            "decimals",
            None,
            choices={"2": "2", "3": "3", "4": "4", "5": "5", "6": "6"},
            selected=default,
            width="70px",
        ),
        ui.tags.label("decimals", class_="text-muted small ms-1", style="line-height:2.2;"),
        class_="d-flex align-items-center",
        style="float:right;margin-bottom:4px;",
    )


@module.server
def decimals_server(input, output, session) -> reactive.Calc:
    """Returns a reactive calc that yields the current decimals value."""

    @reactive.calc
    def get_decimals() -> int:
        return int(input.decimals())

    return get_decimals
