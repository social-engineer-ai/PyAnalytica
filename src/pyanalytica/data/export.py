"""Data export functions — CSV, Excel, download bytes."""

from __future__ import annotations

import io

import pandas as pd

from pyanalytica.core.codegen import CodeSnippet


def export_csv(df: pd.DataFrame, path: str) -> CodeSnippet:
    """Export DataFrame to CSV and return equivalent code."""
    df.to_csv(path, index=False)
    return CodeSnippet(
        code=f'df.to_csv("{path}", index=False)',
        imports=["import pandas as pd"],
    )


def export_excel(df: pd.DataFrame, path: str) -> CodeSnippet:
    """Export DataFrame to Excel and return equivalent code."""
    df.to_excel(path, index=False)
    return CodeSnippet(
        code=f'df.to_excel("{path}", index=False)',
        imports=["import pandas as pd"],
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export DataFrame to CSV bytes (for Shiny download)."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Export DataFrame to Excel bytes (for Shiny download)."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()
