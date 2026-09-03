"""Tests for data/transform.py."""

import numpy as np
import pandas as pd
import pytest

from pyanalytica.data.transform import (
    _validate_expr,
    add_column_arithmetic,
    add_column_binned,
    add_column_conditional,
    add_column_log,
    add_column_rank,
    add_column_zscore,
    convert_dtype,
    drop_duplicates,
    drop_missing,
    dummy_encode,
    fill_missing,
    ordinal_encode,
    rename_column,
    str_extract,
    str_lower,
    str_replace,
    str_strip,
    str_upper,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": ["  Hello  ", "WORLD", "test", "FOO"],
        "c": [10, 20, 10, 20],
    })


def test_fill_missing_mean(df):
    result, snippet = fill_missing(df, "a", "mean")
    assert result["a"].isna().sum() == 0
    assert "fillna" in snippet.code


def test_fill_missing_value(df):
    result, _ = fill_missing(df, "a", "value", value=0)
    assert result.loc[2, "a"] == 0


def test_fill_missing_ffill(df):
    result, _ = fill_missing(df, "a", "ffill")
    assert result.loc[2, "a"] == 2.0


def test_drop_missing(df):
    result, snippet = drop_missing(df, ["a"])
    assert len(result) == 3
    assert "dropna" in snippet.code


def test_convert_dtype_float(df):
    result, _ = convert_dtype(df, "c", "float")
    assert result["c"].dtype == float


def test_convert_dtype_str(df):
    result, _ = convert_dtype(df, "c", "str")
    assert pd.api.types.is_string_dtype(result["c"])


def test_drop_duplicates(df):
    result, snippet = drop_duplicates(df, ["c"])
    assert len(result) == 2
    assert "drop_duplicates" in snippet.code


def test_add_column_log(df):
    result, snippet = add_column_log(df, "c_log", "c")
    assert "c_log" in result.columns
    assert "np.log" in snippet.code


def test_add_column_zscore(df):
    result, snippet = add_column_zscore(df, "c_z", "c")
    assert "c_z" in result.columns
    assert abs(result["c_z"].mean()) < 0.01


def test_add_column_rank(df):
    result, snippet = add_column_rank(df, "c_rank", "c")
    assert "c_rank" in result.columns


def test_str_lower(df):
    result, snippet = str_lower(df, "b")
    assert result.loc[1, "b"] == "world"
    assert "str.lower" in snippet.code


def test_str_upper(df):
    result, _ = str_upper(df, "b")
    assert result.loc[2, "b"] == "TEST"


def test_str_strip(df):
    result, _ = str_strip(df, "b")
    assert result.loc[0, "b"] == "Hello"


# --- Rename column tests ---

def test_rename_column_basic(df):
    result, snippet = rename_column(df, "a", "alpha")
    assert "alpha" in result.columns
    assert "a" not in result.columns
    assert "rename" in snippet.code


def test_rename_column_preserves_data(df):
    result, _ = rename_column(df, "c", "count")
    assert list(result["count"]) == [10, 20, 10, 20]


def test_rename_column_code_snippet(df):
    _, snippet = rename_column(df, "b", "label")
    assert '"b"' in snippet.code
    assert '"label"' in snippet.code


# --- Encoding tests ---

def test_dummy_encode_basic():
    df = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
    result, snippet = dummy_encode(df, "color")
    assert "color" not in result.columns
    assert "color_red" in result.columns
    assert "color_blue" in result.columns
    assert "color_green" in result.columns
    assert "get_dummies" in snippet.code


def test_dummy_encode_drop_first():
    df = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
    result, snippet = dummy_encode(df, "color", drop_first=True)
    # One less dummy column when drop_first=True
    dummy_cols = [c for c in result.columns if c.startswith("color_")]
    assert len(dummy_cols) == 2
    assert "drop_first=True" in snippet.code


def test_dummy_encode_preserves_other_columns():
    df = pd.DataFrame({"x": [1, 2, 3], "color": ["a", "b", "a"]})
    result, _ = dummy_encode(df, "color")
    assert "x" in result.columns
    assert len(result) == 3


def test_ordinal_encode_auto_order():
    df = pd.DataFrame({"size": ["medium", "small", "large", "small"]})
    result, snippet = ordinal_encode(df, "size")
    # Sorted order: large=0, medium=1, small=2
    assert result.loc[0, "size"] == 1  # medium
    assert result.loc[1, "size"] == 2  # small
    assert result.loc[2, "size"] == 0  # large
    assert "map" in snippet.code


def test_ordinal_encode_custom_order():
    df = pd.DataFrame({"size": ["medium", "small", "large", "small"]})
    result, snippet = ordinal_encode(df, "size", order=["small", "medium", "large"])
    assert result.loc[0, "size"] == 1  # medium
    assert result.loc[1, "size"] == 0  # small
    assert result.loc[2, "size"] == 2  # large


def test_ordinal_encode_with_nan():
    df = pd.DataFrame({"size": ["small", None, "large"]})
    result, _ = ordinal_encode(df, "size")
    assert pd.isna(result.loc[1, "size"])
    assert result.loc[0, "size"] == 1  # small (sorted: large=0, small=1)
    assert result.loc[2, "size"] == 0  # large


# --- Regression tests: conversions must never silently empty a column ---


def test_convert_float_refuses_percent_strings():
    """A '%' suffix used to coerce every value to NaN and report success."""
    df = pd.DataFrame({"attendance_pct": ["96%", "95%", "88%"]})
    with pytest.raises(ValueError, match="not valid"):
        convert_dtype(df, "attendance_pct", "float")


def test_convert_float_error_names_count_and_example():
    df = pd.DataFrame({"a": ["1", "2", "oops"]})
    with pytest.raises(ValueError) as exc:
        convert_dtype(df, "a", "float")
    assert "1 of 3" in str(exc.value)
    assert "oops" in str(exc.value)


def test_convert_float_still_works_on_clean_text():
    df = pd.DataFrame({"a": ["1.5", "2.5"]})
    result, _ = convert_dtype(df, "a", "float")
    assert result["a"].tolist() == [1.5, 2.5]


def test_convert_float_keeps_existing_nulls():
    df = pd.DataFrame({"a": ["1.5", None, "2.5"]})
    result, _ = convert_dtype(df, "a", "float")
    assert result["a"].isna().sum() == 1


def test_convert_int_rejects_decimals_instead_of_raising_typeerror():
    df = pd.DataFrame({"v": [1.7, 2.2]})
    with pytest.raises(ValueError, match="decimals"):
        convert_dtype(df, "v", "int")


def test_convert_int_accepts_whole_floats():
    df = pd.DataFrame({"v": [1.0, 2.0]})
    result, _ = convert_dtype(df, "v", "int")
    assert result["v"].tolist() == [1, 2]


def test_convert_bool_maps_yes_no_by_meaning():
    """astype(bool) made every non-empty string True, including 'No'."""
    df = pd.DataFrame({"employed": ["Yes", "No", "yes", "NO"]})
    result, _ = convert_dtype(df, "employed", "bool")
    assert result["employed"].tolist() == [True, False, True, False]


def test_convert_bool_rejects_unrecognised_text():
    df = pd.DataFrame({"employed": ["Yes", "maybe"]})
    with pytest.raises(ValueError, match="not recognisable"):
        convert_dtype(df, "employed", "bool")


def test_convert_datetime_refuses_unparseable():
    df = pd.DataFrame({"d": ["2026-01-01", "not a date"]})
    with pytest.raises(ValueError, match="not valid"):
        convert_dtype(df, "d", "datetime")


# --- Regression tests: readable errors instead of tracebacks ---


def test_str_strip_on_numeric_column_gives_readable_error():
    df = pd.DataFrame({"n": [1, 2, 3]})
    with pytest.raises(ValueError, match="needs a text column"):
        str_strip(df, "n")


def test_fill_mean_on_text_column_gives_readable_error():
    df = pd.DataFrame({"t": ["a", None, "b"]})
    with pytest.raises(ValueError, match="needs a numeric column"):
        fill_missing(df, "t", "mean")


def test_zscore_on_text_column_gives_readable_error():
    df = pd.DataFrame({"t": ["a", "b"]})
    with pytest.raises(ValueError, match="needs a numeric column"):
        add_column_zscore(df, "z", "t")


def test_rank_on_text_column_is_rejected_like_zscore():
    """rank() used to succeed on text while zscore raised."""
    df = pd.DataFrame({"t": ["a", "b"]})
    with pytest.raises(ValueError, match="needs a numeric column"):
        add_column_rank(df, "r", "t")


# --- Regression tests: generated code must match behaviour ---


def test_log_refuses_non_positive_rather_than_clipping():
    """It used to clip to 1e-10 and emit code that would produce -inf."""
    df = pd.DataFrame({"x": [0.0, 1.0, 10.0]})
    with pytest.raises(ValueError, match="zero or negative"):
        add_column_log(df, "lx", "x")


def test_log_snippet_matches_actual_values():
    df = pd.DataFrame({"x": [1.0, np.e]})
    result, snippet = add_column_log(df, "lx", "x")
    assert result["lx"].round(6).tolist() == [0.0, 1.0]
    assert snippet.code == 'df["lx"] = np.log(df["x"])'


# --- Regression tests: expression validation ---


@pytest.mark.parametrize("expr", [
    "open_rate * 2",
    "direct_cost + indirect_cost",
    "exit_survey_score - 1",
    "print_volume / 12",
    "input_hours * rate",
    "revenue - cost",
])
def test_validate_expr_allows_ordinary_column_names(expr):
    """Substring matching rejected these; identifier matching must not."""
    _validate_expr(expr)


@pytest.mark.parametrize("expr", ["__import__('os')", "eval('1')", "open('f')"])
def test_validate_expr_still_blocks_dangerous_names(expr):
    with pytest.raises(ValueError):
        _validate_expr(expr)


# --- Regression tests: encoding ---


def test_ordinal_encode_rejects_incomplete_order():
    """An order missing a category silently nulled those rows."""
    df = pd.DataFrame({"size": ["small", "medium", "large"]})
    with pytest.raises(ValueError, match="does not mention"):
        ordinal_encode(df, "size", order=["small", "medium"])


def test_dummy_encode_can_keep_original_column():
    df = pd.DataFrame({"employed": ["Yes", "No"], "x": [1, 2]})
    result, _ = dummy_encode(df, "employed", keep_original=True)
    assert "employed" in result.columns
    assert "employed_Yes" in result.columns


def test_dummy_encode_still_drops_original_by_default():
    df = pd.DataFrame({"employed": ["Yes", "No"], "x": [1, 2]})
    result, _ = dummy_encode(df, "employed")
    assert "employed" not in result.columns


# --- New operations ---


def test_str_replace_strips_percent_then_converts():
    """The end-to-end path that was previously impossible in the GUI."""
    df = pd.DataFrame({"attendance_pct": ["96%", "95%", "88%"]})
    stripped, snippet = str_replace(df, "attendance_pct", "%", "")
    result, _ = convert_dtype(stripped, "attendance_pct", "float")
    assert result["attendance_pct"].tolist() == [96.0, 95.0, 88.0]
    assert "str.replace" in snippet.code


def test_str_replace_preserves_missing_values():
    df = pd.DataFrame({"a": ["9%", None]})
    result, _ = str_replace(df, "a", "%", "")
    assert result["a"].isna().sum() == 1


def test_str_extract_pulls_digits():
    df = pd.DataFrame({"a": ["96%", "88%"]})
    result, _ = str_extract(df, "num", "a", r"\d+")
    assert result["num"].tolist() == ["96", "88"]


def test_add_column_arithmetic_and_conditional():
    df = pd.DataFrame({"salary": [60000.0, 30000.0]})
    monthly, _ = add_column_arithmetic(df, "monthly", "salary / 12")
    assert monthly["monthly"].round(2).tolist() == [5000.0, 2500.0]

    flagged, _ = add_column_conditional(df, "high", "salary > 50000", 1, 0)
    assert flagged["high"].tolist() == [1, 0]
    assert flagged["high"].mean() == 0.5


def test_add_column_binned_turns_a_fact_into_a_dimension():
    df = pd.DataFrame({"score": [1, 5, 9]})
    result, _ = add_column_binned(df, "band", "score", 3, ["low", "mid", "high"])
    assert list(result["band"]) == ["low", "mid", "high"]


def test_add_column_binned_rejects_label_count_mismatch():
    df = pd.DataFrame({"score": [1, 5, 9]})
    with pytest.raises(ValueError, match="label"):
        add_column_binned(df, "band", "score", 3, ["low", "high"])
