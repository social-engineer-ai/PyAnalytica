"""Natural language to pandas code translator.

Converts plain-English questions about a DataFrame into executable
pandas code snippets.  Operates in two modes:

1. **Rule-based** (always available): uses regex patterns to match
   common query patterns and maps them to pandas operations.
2. **LLM-enhanced** (optional): when an Anthropic API key is set,
   sends ambiguous queries to Claude for better code generation.

The translator is intentionally conservative -- it only generates code
for queries it can confidently parse, and returns a helpful fallback
message for anything it cannot handle.

Usage:
    from pyanalytica.ai.query import natural_language_to_pandas
    code = natural_language_to_pandas(
        "average of salary by department",
        df_info={"columns": ["salary", "department", "age"],
                 "dtypes": {"salary": "float64", "department": "object", "age": "int64"},
                 "shape": (500, 3),
                 "name": "employees"}
    )
    # Returns: 'employees.groupby("department")["salary"].mean()'
"""

from __future__ import annotations

import os
import re
from typing import Any


# ---------------------------------------------------------------------------
# LLM helper (optional)
# ---------------------------------------------------------------------------

def _try_llm(prompt: str) -> str | None:
    """Attempt to get an LLM-enhanced response from Claude.

    Returns the response text on success, or ``None`` if the anthropic
    package is not installed or the API key is not configured.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import anthropic  # type: ignore[import-untyped]

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Column matching helpers
# ---------------------------------------------------------------------------

def _find_column(name: str, columns: list[str]) -> str | None:
    """Find the best matching column name (case-insensitive, fuzzy).

    Parameters
    ----------
    name : str
        The user's reference to a column (may be partial or differently
        cased).
    columns : list[str]
        The actual column names in the DataFrame.

    Returns
    -------
    str or None
        The matched column name, or None if no match found.
    """
    name_lower = name.strip().lower()

    # Exact match (case-insensitive)
    for col in columns:
        if col.lower() == name_lower:
            return col

    # Exact match after replacing spaces/underscores
    name_normalized = name_lower.replace(" ", "_").replace("-", "_")
    for col in columns:
        col_normalized = col.lower().replace(" ", "_").replace("-", "_")
        if col_normalized == name_normalized:
            return col

    # Substring match: column name contains the query term
    for col in columns:
        if name_lower in col.lower():
            return col

    # Substring match: query term contains the column name
    for col in columns:
        if col.lower() in name_lower:
            return col

    return None


def _extract_columns_from_query(
    query: str, columns: list[str]
) -> list[str]:
    """Extract all column references from a query string.

    Returns matched column names in the order they appear in the query.
    """
    matched: list[str] = []
    query_lower = query.lower()

    # Sort columns by length (longest first) to avoid partial matches
    sorted_cols = sorted(columns, key=len, reverse=True)

    for col in sorted_cols:
        # Check if column name appears in query (case-insensitive)
        col_lower = col.lower()
        col_pattern = re.escape(col_lower)
        if re.search(r'\b' + col_pattern + r'\b', query_lower):
            if col not in matched:
                matched.append(col)

    # Also try with underscores replaced by spaces
    if not matched:
        for col in sorted_cols:
            col_spaces = col.lower().replace("_", " ")
            if col_spaces in query_lower and col not in matched:
                matched.append(col)

    return matched


def _extract_number(text: str) -> float | None:
    """Extract the first number from a text string."""
    match = re.search(r'[-+]?\d*\.?\d+', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Pattern-based query parser
# ---------------------------------------------------------------------------

def _parse_show_me(
    query: str, columns: list[str], df_name: str
) -> str | None:
    """Parse 'show me X' or 'display X' queries."""
    patterns = [
        r"show\s+(?:me\s+)?(?:the\s+)?(.+)",
        r"display\s+(?:the\s+)?(.+)",
        r"what\s+(?:is|are)\s+(?:the\s+)?(.+)",
        r"list\s+(?:the\s+)?(.+)",
        r"print\s+(?:the\s+)?(.+)",
    ]
    for pattern in patterns:
        match = re.match(pattern, query.lower().strip())
        if match:
            subject = match.group(1).strip()

            # "show me the data" / "show me everything"
            if subject in ("data", "everything", "all", "all data",
                           "the data", "dataset"):
                return f"{df_name}.head()"

            # "show me first/top N rows"
            n_match = re.search(r"(?:first|top)\s+(\d+)", subject)
            if n_match:
                n = int(n_match.group(1))
                return f"{df_name}.head({n})"

            # "show me last N rows"
            n_match = re.search(r"(?:last|bottom)\s+(\d+)", subject)
            if n_match:
                n = int(n_match.group(1))
                return f"{df_name}.tail({n})"

            # "show me column X"
            col = _find_column(subject, columns)
            if col:
                return f'{df_name}["{col}"].head()'

            # "show me columns" / "show me column names"
            if "column" in subject:
                return f"{df_name}.columns.tolist()"

            # "show me shape" / "show me size"
            if subject in ("shape", "size", "dimensions"):
                return f"{df_name}.shape"

            # "show me info"
            if subject in ("info", "information", "types", "dtypes"):
                return f"{df_name}.info()"

            return None
    return None


def _parse_aggregation(
    query: str, columns: list[str], df_name: str
) -> str | None:
    """Parse aggregation queries like 'average of X' or 'average of X by Y'."""
    agg_map = {
        "average": "mean", "mean": "mean", "avg": "mean",
        "sum": "sum", "total": "sum",
        "count": "count", "number of": "count",
        "min": "min", "minimum": "min", "smallest": "min", "lowest": "min",
        "max": "max", "maximum": "max", "largest": "max", "highest": "max",
        "median": "median", "middle": "median",
        "std": "std", "standard deviation": "std", "stdev": "std",
        "variance": "var", "var": "var",
    }

    query_lower = query.lower().strip()

    # Pattern: "aggregation of X by/for each Y"
    for agg_word, agg_func in agg_map.items():
        # "average of X by Y" / "mean of X grouped by Y" / "mean of X for each Y"
        pattern = (
            rf"{re.escape(agg_word)}\s+(?:of\s+)?(.+?)\s+"
            rf"(?:by|grouped?\s+by|for\s+each|per|across)\s+(.+)"
        )
        match = re.match(pattern, query_lower)
        if match:
            value_text = match.group(1).strip()
            group_text = match.group(2).strip()
            value_col = _find_column(value_text, columns)
            group_col = _find_column(group_text, columns)
            if value_col and group_col:
                return f'{df_name}.groupby("{group_col}")["{value_col}"].{agg_func}()'

    # Pattern: "aggregation of X" (no groupby)
    for agg_word, agg_func in agg_map.items():
        pattern = rf"{re.escape(agg_word)}\s+(?:of\s+)?(.+)"
        match = re.match(pattern, query_lower)
        if match:
            value_text = match.group(1).strip()
            value_col = _find_column(value_text, columns)
            if value_col:
                if agg_func == "count":
                    return f'{df_name}["{value_col}"].value_counts()'
                return f'{df_name}["{value_col}"].{agg_func}()'

    # Pattern: "how many X" (count)
    match = re.match(r"how\s+many\s+(.+)", query_lower)
    if match:
        subject = match.group(1).strip()
        col = _find_column(subject, columns)
        if col:
            return f'{df_name}["{col}"].value_counts()'
        # "how many rows"
        if "row" in subject:
            return f"len({df_name})"

    return None


def _parse_count_query(
    query: str, columns: list[str], df_name: str
) -> str | None:
    """Parse count/frequency queries."""
    query_lower = query.lower().strip()

    # "count of X" / "frequency of X" / "distribution of X"
    patterns = [
        r"(?:count|frequency|distribution|breakdown)\s+(?:of\s+)?(.+)",
        r"value\s+counts?\s+(?:of\s+|for\s+)?(.+)",
    ]
    for pattern in patterns:
        match = re.match(pattern, query_lower)
        if match:
            col_text = match.group(1).strip()
            col = _find_column(col_text, columns)
            if col:
                return f'{df_name}["{col}"].value_counts()'
    return None


def _parse_filter(
    query: str, columns: list[str], df_name: str
) -> str | None:
    """Parse filter/where queries."""
    query_lower = query.lower().strip()

    # "filter where X > N" / "rows where X > N" / "where X > N"
    patterns = [
        r"(?:filter|rows?|data|show)\s+(?:where|when|if)\s+(.+)",
        r"where\s+(.+)",
        r"(?:only|just)\s+(?:show\s+)?(?:rows?\s+)?(?:where\s+)?(.+)",
    ]

    operators = {
        "greater than": ">", "more than": ">", "above": ">",
        "less than": "<", "fewer than": "<", "below": "<", "under": "<",
        "equal to": "==", "equals": "==", "is": "==",
        "not equal to": "!=", "is not": "!=", "not": "!=",
        ">=": ">=", "at least": ">=",
        "<=": "<=", "at most": "<=", "no more than": "<=",
        ">": ">", "<": "<", "==": "==", "!=": "!=",
    }

    for pattern in patterns:
        match = re.match(pattern, query_lower)
        if match:
            condition = match.group(1).strip()

            # Try to parse "column operator value"
            for op_text, op_symbol in sorted(
                operators.items(), key=lambda x: len(x[0]), reverse=True
            ):
                op_pattern = re.escape(op_text)
                parts = re.split(op_pattern, condition, maxsplit=1)
                if len(parts) == 2:
                    col_text = parts[0].strip()
                    val_text = parts[1].strip()
                    col = _find_column(col_text, columns)
                    if col:
                        num = _extract_number(val_text)
                        if num is not None:
                            # Format integer if it's a whole number
                            if num == int(num):
                                val_repr = str(int(num))
                            else:
                                val_repr = str(num)
                            return f'{df_name}[{df_name}["{col}"] {op_symbol} {val_repr}]'
                        else:
                            # String value
                            val_clean = val_text.strip("'\"")
                            return f'{df_name}[{df_name}["{col}"] {op_symbol} "{val_clean}"]'

    return None


def _parse_sort(
    query: str, columns: list[str], df_name: str
) -> str | None:
    """Parse sort/order queries."""
    query_lower = query.lower().strip()

    # "sort by X" / "order by X" / "rank by X"
    patterns = [
        r"(?:sort|order|rank|arrange)\s+(?:by\s+)?(.+)",
    ]
    for pattern in patterns:
        match = re.match(pattern, query_lower)
        if match:
            rest = match.group(1).strip()

            # Check for descending
            ascending = True
            if re.search(r"\b(?:desc|descending|highest|largest|most)\b", rest):
                ascending = False
                rest = re.sub(
                    r"\b(?:desc|descending|highest|largest|most|first)\b",
                    "", rest
                ).strip()
            elif re.search(r"\b(?:asc|ascending|lowest|smallest|least)\b", rest):
                ascending = True
                rest = re.sub(
                    r"\b(?:asc|ascending|lowest|smallest|least|first)\b",
                    "", rest
                ).strip()

            col = _find_column(rest, columns)
            if col:
                asc_str = "True" if ascending else "False"
                return f'{df_name}.sort_values("{col}", ascending={asc_str})'

    return None


def _parse_correlation(
    query: str, columns: list[str], df_name: str
) -> str | None:
    """Parse correlation queries."""
    query_lower = query.lower().strip()

    # "correlation between X and Y"
    match = re.match(
        r"(?:correlation|corr|relationship)\s+"
        r"(?:between|of)\s+(.+?)\s+and\s+(.+)",
        query_lower,
    )
    if match:
        col1_text = match.group(1).strip()
        col2_text = match.group(2).strip()
        col1 = _find_column(col1_text, columns)
        col2 = _find_column(col2_text, columns)
        if col1 and col2:
            return f'{df_name}[["{col1}", "{col2}"]].corr()'

    # "correlation matrix" / "correlations"
    if re.match(r"(?:correlation\s+matrix|correlations|corr\s+matrix)", query_lower):
        return f"{df_name}.corr(numeric_only=True)"

    return None


def _parse_describe(
    query: str, columns: list[str], df_name: str
) -> str | None:
    """Parse summary/describe queries."""
    query_lower = query.lower().strip()

    patterns = [
        r"(?:describe|summary|summarize|statistics|stats)\s*(?:of\s+|for\s+)?(.+)?",
    ]
    for pattern in patterns:
        match = re.match(pattern, query_lower)
        if match:
            rest = (match.group(1) or "").strip()
            if rest:
                col = _find_column(rest, columns)
                if col:
                    return f'{df_name}["{col}"].describe()'
            return f"{df_name}.describe()"

    return None


def _parse_missing(
    query: str, columns: list[str], df_name: str
) -> str | None:
    """Parse missing-value queries."""
    query_lower = query.lower().strip()

    if re.search(r"missing|null|nan|na\b|empty", query_lower):
        # "missing values in X"
        match = re.search(r"(?:in|for|of)\s+(.+)", query_lower)
        if match:
            col = _find_column(match.group(1).strip(), columns)
            if col:
                return f'{df_name}["{col}"].isnull().sum()'
        return f"{df_name}.isnull().sum()"

    return None


def _parse_unique(
    query: str, columns: list[str], df_name: str
) -> str | None:
    """Parse unique-value queries."""
    query_lower = query.lower().strip()

    patterns = [
        r"(?:unique|distinct)\s+(?:values?\s+)?(?:of\s+|in\s+|for\s+)?(.+)",
    ]
    for pattern in patterns:
        match = re.match(pattern, query_lower)
        if match:
            col_text = match.group(1).strip()
            col = _find_column(col_text, columns)
            if col:
                return f'{df_name}["{col}"].unique()'

    # "how many unique X"
    match = re.match(r"how\s+many\s+unique\s+(.+)", query_lower)
    if match:
        col = _find_column(match.group(1).strip(), columns)
        if col:
            return f'{df_name}["{col}"].nunique()'

    return None


def _parse_groupby(
    query: str, columns: list[str], df_name: str
) -> str | None:
    """Parse explicit groupby queries."""
    query_lower = query.lower().strip()

    # "group by X and show Y"
    match = re.match(
        r"group\s+by\s+(.+?)\s+(?:and\s+)?(?:show|get|compute)\s+(.+)",
        query_lower,
    )
    if match:
        group_text = match.group(1).strip()
        action_text = match.group(2).strip()
        group_col = _find_column(group_text, columns)
        if group_col:
            # Try to extract aggregation from action_text
            agg_map = {
                "mean": "mean", "average": "mean", "avg": "mean",
                "sum": "sum", "count": "count",
                "min": "min", "max": "max", "median": "median",
            }
            for word, func in agg_map.items():
                if word in action_text:
                    # Try to extract column from rest
                    rest = action_text.replace(word, "").strip()
                    rest = re.sub(r"^(?:of|the)\s+", "", rest).strip()
                    val_col = _find_column(rest, columns)
                    if val_col:
                        return f'{df_name}.groupby("{group_col}")["{val_col}"].{func}()'
            # Generic groupby describe
            return f'{df_name}.groupby("{group_col}").describe()'

    return None


# ---------------------------------------------------------------------------
# Fallback message
# ---------------------------------------------------------------------------

def _fallback_message(columns: list[str]) -> str:
    """Generate a helpful fallback when the query cannot be parsed."""
    col_str = ", ".join(columns[:10])
    if len(columns) > 10:
        col_str += f", ... ({len(columns) - 10} more)"

    return (
        f"# Could not parse this query. Try one of these patterns:\n"
        f"#   'show me <column_name>'\n"
        f"#   'average of <column> by <group_column>'\n"
        f"#   'count of <column>'\n"
        f"#   'filter where <column> > <number>'\n"
        f"#   'sort by <column>'\n"
        f"#   'correlation between <column1> and <column2>'\n"
        f"#   'describe <column>'\n"
        f"#   'missing values'\n"
        f"#   'unique values of <column>'\n"
        f"#\n"
        f"# Available columns: {col_str}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def natural_language_to_pandas(
    query: str,
    df_info: dict,
) -> str:
    """Translate a natural language question into pandas code.

    Parameters
    ----------
    query : str
        A plain-English question about the data, such as "average of
        salary by department" or "show me the top 10 rows sorted by age".
    df_info : dict
        Information about the target DataFrame with keys:

        - ``"columns"`` (list[str]): column names
        - ``"dtypes"`` (dict): column name -> dtype string
        - ``"shape"`` (tuple): (n_rows, n_cols)
        - ``"name"`` (str): variable name for the DataFrame (e.g., "df")

    Returns
    -------
    str
        A pandas code snippet as a string.  If the query cannot be
        parsed, returns a comment with usage examples and available
        columns.

    Notes
    -----
    Operates in two modes:

    1. **Rule-based** (always available): uses regex patterns to match
       common query structures and resolves column references using
       fuzzy matching against ``df_info["columns"]``.
    2. **LLM-enhanced** (optional): for queries the rule-based parser
       cannot handle, falls back to Claude (if ``ANTHROPIC_API_KEY`` is
       set) to generate the pandas code.

    Examples
    --------
    >>> info = {"columns": ["salary", "department", "age"],
    ...         "dtypes": {"salary": "float64", "department": "object", "age": "int64"},
    ...         "shape": (500, 3), "name": "employees"}
    >>> natural_language_to_pandas("average salary by department", info)
    'employees.groupby("department")["salary"].mean()'
    >>> natural_language_to_pandas("sort by age descending", info)
    'employees.sort_values("age", ascending=False)'
    """
    if not query or not query.strip():
        columns = df_info.get("columns", [])
        return _fallback_message(columns)

    columns: list[str] = df_info.get("columns", [])
    df_name: str = df_info.get("name", "df")

    # Normalize query
    query_clean = query.strip()

    # Try each parser in order of specificity
    parsers = [
        _parse_correlation,
        _parse_aggregation,
        _parse_count_query,
        _parse_filter,
        _parse_sort,
        _parse_describe,
        _parse_missing,
        _parse_unique,
        _parse_groupby,
        _parse_show_me,
    ]

    for parser in parsers:
        result = parser(query_clean, columns, df_name)
        if result is not None:
            return result

    # Rule-based failed -- try LLM
    dtypes = df_info.get("dtypes", {})
    shape = df_info.get("shape", (0, 0))

    llm_prompt = (
        "You are a pandas code generator. Convert this natural language "
        "query into a SINGLE pandas expression. Return ONLY the code, "
        "no explanation.\n\n"
        f"Query: \"{query_clean}\"\n\n"
        f"DataFrame variable name: {df_name}\n"
        f"Columns: {columns}\n"
        f"Data types: {dtypes}\n"
        f"Shape: {shape[0]} rows x {shape[1]} columns\n\n"
        "Return a single line of pandas code. Use exact column names "
        "from the list above. Do not import anything."
    )

    llm_result = _try_llm(llm_prompt)
    if llm_result:
        # Clean up LLM response -- extract just the code
        code = llm_result.strip()
        # Remove markdown code fences if present
        code = re.sub(r'^```(?:python)?\s*', '', code)
        code = re.sub(r'\s*```$', '', code)
        code = code.strip()
        if code and not code.startswith("#"):
            return f"# AI-generated (verify before running):\n{code}"

    return _fallback_message(columns)
