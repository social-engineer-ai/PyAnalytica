"""Data transformation — missing values, dtypes, duplicates, new columns, string ops."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pyanalytica.core.codegen import CodeSnippet


# --- Expression validation ---

_FORBIDDEN_PATTERNS = (
    "__", "import", "exec", "eval", "compile", "open", "getattr",
    "setattr", "delattr", "globals", "locals", "vars", "dir",
    "breakpoint", "exit", "quit", "input", "print",
)


def _validate_expr(expr: str) -> None:
    """Reject expressions containing dangerous patterns for df.eval()."""
    lowered = expr.lower().replace(" ", "")
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern in lowered:
            raise ValueError(
                f"Expression contains forbidden pattern '{pattern}'. "
                "Only arithmetic on column names is allowed."
            )


# --- Missing values ---

def fill_missing(
    df: pd.DataFrame, col: str, method: str, value: Any = None
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Fill missing values in a column.

    Methods: 'value', 'mean', 'median', 'mode', 'ffill', 'bfill'
    """
    result = df.copy()

    if method == "value":
        result[col] = result[col].fillna(value)
        code = f'df["{col}"] = df["{col}"].fillna({_repr_val(value)})'
    elif method == "mean":
        fill_val = result[col].mean()
        result[col] = result[col].fillna(fill_val)
        code = f'df["{col}"] = df["{col}"].fillna(df["{col}"].mean())'
    elif method == "median":
        fill_val = result[col].median()
        result[col] = result[col].fillna(fill_val)
        code = f'df["{col}"] = df["{col}"].fillna(df["{col}"].median())'
    elif method == "mode":
        fill_val = result[col].mode().iloc[0] if not result[col].mode().empty else None
        result[col] = result[col].fillna(fill_val)
        code = f'df["{col}"] = df["{col}"].fillna(df["{col}"].mode().iloc[0])'
    elif method == "ffill":
        result[col] = result[col].ffill()
        code = f'df["{col}"] = df["{col}"].ffill()'
    elif method == "bfill":
        result[col] = result[col].bfill()
        code = f'df["{col}"] = df["{col}"].bfill()'
    else:
        raise ValueError(f"Unknown fill method: {method}")

    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def drop_missing(
    df: pd.DataFrame, cols: list[str] | None = None, how: str = "any"
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Drop rows with missing values."""
    if cols:
        result = df.dropna(subset=cols, how=how).copy()
        code = f'df = df.dropna(subset={cols!r}, how="{how}")'
    else:
        result = df.dropna(how=how).copy()
        code = f'df = df.dropna(how="{how}")'

    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


# --- Data types ---

def convert_dtype(
    df: pd.DataFrame, col: str, target_dtype: str
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Convert a column to a different dtype.

    target_dtype: 'int', 'float', 'str', 'category', 'datetime', 'bool'
    """
    result = df.copy()

    if target_dtype == "int":
        result[col] = pd.to_numeric(result[col], errors="coerce").astype("Int64")
        code = f'df["{col}"] = pd.to_numeric(df["{col}"], errors="coerce").astype("Int64")'
    elif target_dtype == "float":
        result[col] = pd.to_numeric(result[col], errors="coerce").astype(float)
        code = f'df["{col}"] = pd.to_numeric(df["{col}"], errors="coerce").astype(float)'
    elif target_dtype == "str":
        result[col] = result[col].astype(str)
        code = f'df["{col}"] = df["{col}"].astype(str)'
    elif target_dtype == "category":
        result[col] = result[col].astype("category")
        code = f'df["{col}"] = df["{col}"].astype("category")'
    elif target_dtype == "datetime":
        result[col] = pd.to_datetime(result[col], errors="coerce")
        code = f'df["{col}"] = pd.to_datetime(df["{col}"], errors="coerce")'
    elif target_dtype == "bool":
        result[col] = result[col].astype(bool)
        code = f'df["{col}"] = df["{col}"].astype(bool)'
    else:
        raise ValueError(f"Unknown target dtype: {target_dtype}")

    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


# --- Rename column ---

def rename_column(
    df: pd.DataFrame, old_name: str, new_name: str
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Rename a single column."""
    result = df.rename(columns={old_name: new_name})
    code = f'df = df.rename(columns={{"{old_name}": "{new_name}"}})'
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


# --- Drop columns ---

def drop_columns(
    df: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Drop one or more columns from a DataFrame."""
    result = df.drop(columns=cols).copy()
    code = f"df = df.drop(columns={cols!r})"
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


# --- Duplicates ---

def drop_duplicates(
    df: pd.DataFrame, cols: list[str] | None = None, keep: str = "first"
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Drop duplicate rows."""
    if cols:
        result = df.drop_duplicates(subset=cols, keep=keep).copy()
        code = f'df = df.drop_duplicates(subset={cols!r}, keep="{keep}")'
    else:
        result = df.drop_duplicates(keep=keep).copy()
        code = f'df = df.drop_duplicates(keep="{keep}")'

    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


# --- New columns ---

def add_column_arithmetic(
    df: pd.DataFrame, new_col: str, expr: str
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Add a new column from an arithmetic expression using existing columns.

    expr should reference columns by name, e.g. "salary * 12" or "revenue - cost"
    The expression is evaluated using DataFrame.eval() with input validation.
    """
    _validate_expr(expr)
    result = df.copy()
    result[new_col] = result.eval(expr)
    code = f'df["{new_col}"] = df.eval("{expr}")'
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def add_column_conditional(
    df: pd.DataFrame, new_col: str, condition: str,
    true_val: Any, false_val: Any
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Add a column with values based on a condition.

    condition: pandas eval expression, e.g. "salary > 50000"
    """
    _validate_expr(condition)
    result = df.copy()
    mask = result.eval(condition)
    result[new_col] = np.where(mask, true_val, false_val)

    code = (
        f'df["{new_col}"] = np.where(\n'
        f'    df.eval("{condition}"),\n'
        f"    {_repr_val(true_val)},\n"
        f"    {_repr_val(false_val)}\n"
        f")"
    )
    return result, CodeSnippet(code=code, imports=["import numpy as np", "import pandas as pd"])


def add_column_binned(
    df: pd.DataFrame, new_col: str, source_col: str,
    bins: int | list, labels: list[str] | None = None
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Add a column with binned/discretized values."""
    result = df.copy()
    result[new_col] = pd.cut(result[source_col], bins=bins, labels=labels)

    labels_str = f", labels={labels!r}" if labels else ""
    code = f'df["{new_col}"] = pd.cut(df["{source_col}"], bins={bins!r}{labels_str})'
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def add_column_log(
    df: pd.DataFrame, new_col: str, source_col: str
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Add a log-transformed column."""
    result = df.copy()
    result[new_col] = np.log(result[source_col].clip(lower=1e-10))
    code = f'df["{new_col}"] = np.log(df["{source_col}"])'
    return result, CodeSnippet(code=code, imports=["import numpy as np", "import pandas as pd"])


def add_column_zscore(
    df: pd.DataFrame, new_col: str, source_col: str
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Add a z-score normalized column."""
    result = df.copy()
    mean = result[source_col].mean()
    std = result[source_col].std()
    result[new_col] = (result[source_col] - mean) / std

    code = (
        f'df["{new_col}"] = (\n'
        f'    (df["{source_col}"] - df["{source_col}"].mean()) / df["{source_col}"].std()\n'
        f")"
    )
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def add_column_rank(
    df: pd.DataFrame, new_col: str, source_col: str
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Add a rank column."""
    result = df.copy()
    result[new_col] = result[source_col].rank()
    code = f'df["{new_col}"] = df["{source_col}"].rank()'
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


# --- String operations ---

def str_lower(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, CodeSnippet]:
    """Convert string column to lowercase."""
    result = df.copy()
    result[col] = result[col].str.lower()
    code = f'df["{col}"] = df["{col}"].str.lower()'
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def str_upper(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, CodeSnippet]:
    """Convert string column to uppercase."""
    result = df.copy()
    result[col] = result[col].str.upper()
    code = f'df["{col}"] = df["{col}"].str.upper()'
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def str_strip(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, CodeSnippet]:
    """Strip whitespace from string column."""
    result = df.copy()
    result[col] = result[col].str.strip()
    code = f'df["{col}"] = df["{col}"].str.strip()'
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def str_extract(
    df: pd.DataFrame, new_col: str, col: str, pattern: str
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Extract substring using regex pattern."""
    result = df.copy()
    result[new_col] = result[col].str.extract(f"({pattern})", expand=False)
    code = f'df["{new_col}"] = df["{col}"].str.extract(r"({pattern})", expand=False)'
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


# --- Encoding ---

def dummy_encode(
    df: pd.DataFrame, column: str, drop_first: bool = False
) -> tuple[pd.DataFrame, CodeSnippet]:
    """One-hot / dummy encode a categorical column."""
    result = pd.get_dummies(df, columns=[column], drop_first=drop_first)
    drop_str = ", drop_first=True" if drop_first else ""
    code = f'df = pd.get_dummies(df, columns=["{column}"]{drop_str})'
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def ordinal_encode(
    df: pd.DataFrame, column: str, order: list[str] | None = None
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Map categories to integers (0, 1, 2, ...).

    If *order* is given, categories are mapped in that order.
    Otherwise, sorted unique values are used.
    """
    result = df.copy()
    if order is None:
        order = sorted(result[column].dropna().unique())
    mapping = {val: i for i, val in enumerate(order)}
    result[column] = result[column].map(mapping)

    code = (
        f'{column}_map = {mapping!r}\n'
        f'df["{column}"] = df["{column}"].map({column}_map)'
    )
    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def _repr_val(val: Any) -> str:
    """Represent a value for code generation."""
    if isinstance(val, str):
        return f'"{val}"'
    return repr(val)
