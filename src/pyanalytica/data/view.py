"""Data viewing — filter, sort, sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from pyanalytica.core.codegen import CodeSnippet


@dataclass
class FilterCondition:
    """A single filter condition."""
    column: str
    operator: str   # "==", "!=", ">", "<", ">=", "<=", "between", "in", "contains", "isnull", "notnull"
    value: Any = None
    value2: Any = None  # For "between"


def apply_filters(
    df: pd.DataFrame,
    filters: list[FilterCondition],
    logic: str = "AND",
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Apply a list of filter conditions to a DataFrame."""
    if not filters:
        return df, CodeSnippet(code="# No filters applied")

    masks = []
    code_parts = []

    for f in filters:
        mask, code = _build_filter(df, f)
        masks.append(mask)
        code_parts.append(code)

    if logic.upper() == "AND":
        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m
        logic_str = " & "
    else:
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        logic_str = " | "

    result = df[combined].copy()

    if len(code_parts) == 1:
        filter_code = code_parts[0]
    else:
        filter_code = logic_str.join(f"({c})" for c in code_parts)

    code = f"df = df[{filter_code}]"
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def _build_filter(df: pd.DataFrame, f: FilterCondition) -> tuple[pd.Series, str]:
    """Build a boolean mask and code string for a single filter."""
    col = f.column

    if f.operator == "==":
        val = _coerce(f.value, df[col].dtype)
        return df[col] == val, f'df["{col}"] == {_repr_val(val)}'

    elif f.operator == "!=":
        val = _coerce(f.value, df[col].dtype)
        return df[col] != val, f'df["{col}"] != {_repr_val(val)}'

    elif f.operator == ">":
        val = _coerce(f.value, df[col].dtype)
        return df[col] > val, f'df["{col}"] > {_repr_val(val)}'

    elif f.operator == "<":
        val = _coerce(f.value, df[col].dtype)
        return df[col] < val, f'df["{col}"] < {_repr_val(val)}'

    elif f.operator == ">=":
        val = _coerce(f.value, df[col].dtype)
        return df[col] >= val, f'df["{col}"] >= {_repr_val(val)}'

    elif f.operator == "<=":
        val = _coerce(f.value, df[col].dtype)
        return df[col] <= val, f'df["{col}"] <= {_repr_val(val)}'

    elif f.operator == "between":
        lo = _coerce(f.value, df[col].dtype)
        hi = _coerce(f.value2, df[col].dtype)
        return df[col].between(lo, hi), f'df["{col}"].between({_repr_val(lo)}, {_repr_val(hi)})'

    elif f.operator == "in":
        vals = f.value if isinstance(f.value, list) else [f.value]
        return df[col].isin(vals), f'df["{col}"].isin({vals!r})'

    elif f.operator == "contains":
        return (
            df[col].astype(str).str.contains(str(f.value), case=False, na=False),
            f'df["{col}"].str.contains("{f.value}", case=False, na=False)',
        )

    elif f.operator == "isnull":
        return df[col].isna(), f'df["{col}"].isna()'

    elif f.operator == "notnull":
        return df[col].notna(), f'df["{col}"].notna()'

    else:
        raise ValueError(f"Unknown operator: {f.operator}")


def sort_dataframe(
    df: pd.DataFrame,
    sort_cols: list[str],
    ascending: list[bool] | None = None,
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Sort a DataFrame by one or more columns."""
    if ascending is None:
        ascending = [True] * len(sort_cols)

    result = df.sort_values(sort_cols, ascending=ascending).copy()

    if len(sort_cols) == 1:
        asc_str = str(ascending[0])
        code = f'df = df.sort_values("{sort_cols[0]}", ascending={asc_str})'
    else:
        code = f"df = df.sort_values({sort_cols!r}, ascending={ascending!r})"

    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def sample_dataframe(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Return a random sample of the DataFrame."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)


def _coerce(value: Any, dtype: Any) -> Any:
    """Coerce a value to match column dtype."""
    try:
        if pd.api.types.is_numeric_dtype(dtype):
            return float(value)
    except (ValueError, TypeError):
        pass
    return value


def _repr_val(val: Any) -> str:
    """Represent a value for code generation."""
    if isinstance(val, str):
        return f'"{val}"'
    return repr(val)
