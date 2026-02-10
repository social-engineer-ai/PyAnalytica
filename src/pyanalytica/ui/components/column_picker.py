"""Smart column selector that filters by type."""

from __future__ import annotations

from typing import Callable

import pandas as pd
from shiny import module, reactive, render, ui

from pyanalytica.core.types import ColumnType, classify_columns


@module.ui
def column_picker_ui(label: str = "Column", multiple: bool = False):
    """Column picker input."""
    return ui.input_select(
        "column",
        label,
        choices=[],
        multiple=multiple,
    )


@module.server
def column_picker_server(
    input, output, session,
    get_df: Callable[[], pd.DataFrame | None],
    col_type: ColumnType | None = None,
):
    """Server logic — updates choices when dataset changes, filters by type."""

    @reactive.effect
    def _update_choices():
        df = get_df()
        if df is None:
            ui.update_select("column", choices=[])
            return

        if col_type is None:
            choices = list(df.columns)
        else:
            classifications = classify_columns(df)
            choices = [col for col, ct in classifications.items() if ct == col_type]

        ui.update_select("column", choices=choices)

    @reactive.calc
    def selected():
        return input.column()

    return selected
