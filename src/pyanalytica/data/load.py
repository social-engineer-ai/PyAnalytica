"""Data loading functions — CSV, Excel, URL, bundled datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pyanalytica.core.codegen import CodeSnippet


def load_csv(path: str | Path, **kwargs: Any) -> tuple[pd.DataFrame, CodeSnippet]:
    """Load a CSV file and return the DataFrame with equivalent code."""
    path = Path(path)
    df = pd.read_csv(path, **kwargs)

    # Build kwargs string for code
    kwargs_str = _format_kwargs(kwargs)
    var_name = _sanitize_name(path.stem)
    code = f'{var_name} = pd.read_csv("{path.name}"{kwargs_str})'

    return df, CodeSnippet(code=code, imports=["import pandas as pd"])


def load_excel(
    path: str | Path, sheet: str | int | None = None, **kwargs: Any
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Load an Excel file and return the DataFrame with equivalent code."""
    path = Path(path)
    read_kwargs = {**kwargs}
    if sheet is not None:
        read_kwargs["sheet_name"] = sheet

    df = pd.read_excel(path, **read_kwargs)

    kwargs_str = _format_kwargs(read_kwargs)
    var_name = _sanitize_name(path.stem)
    code = f'{var_name} = pd.read_excel("{path.name}"{kwargs_str})'

    return df, CodeSnippet(code=code, imports=["import pandas as pd"])


def load_url(url: str, **kwargs: Any) -> tuple[pd.DataFrame, CodeSnippet]:
    """Load a CSV from a URL and return the DataFrame with equivalent code."""
    df = pd.read_csv(url, **kwargs)

    kwargs_str = _format_kwargs(kwargs)
    code = f'df = pd.read_csv("{url}"{kwargs_str})'

    return df, CodeSnippet(code=code, imports=["import pandas as pd"])


def load_bundled(name: str) -> tuple[pd.DataFrame, CodeSnippet]:
    """Load a bundled dataset by name."""
    from pyanalytica.datasets import load_dataset

    df = load_dataset(name)
    var_name = _sanitize_name(name)

    # Generate code that uses pd.read_csv with a comment about bundled data
    code = f'# Bundled dataset: {name}\n{var_name} = pd.read_csv("{name}.csv")'

    return df, CodeSnippet(code=code, imports=["import pandas as pd"])


def load_from_bytes(
    content: bytes, filename: str, **kwargs: Any
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Load from file bytes (for Shiny file upload)."""
    import io

    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content), **kwargs)
        kwargs_str = _format_kwargs(kwargs)
        var_name = _sanitize_name(Path(filename).stem)
        code = f'{var_name} = pd.read_excel("{filename}"{kwargs_str})'
        return df, CodeSnippet(code=code, imports=["import pandas as pd"])
    else:
        df = pd.read_csv(io.BytesIO(content), **kwargs)
        kwargs_str = _format_kwargs(kwargs)
        var_name = _sanitize_name(Path(filename).stem)
        code = f'{var_name} = pd.read_csv("{filename}"{kwargs_str})'
        return df, CodeSnippet(code=code, imports=["import pandas as pd"])


def _sanitize_name(name: str) -> str:
    """Convert a filename to a valid Python variable name."""
    import re
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = "df_" + name
    return name or "df"


def _format_kwargs(kwargs: dict) -> str:
    """Format kwargs for code generation."""
    if not kwargs:
        return ""
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v!r}")
    return ", " + ", ".join(parts)
