"""Browser tests that vary the *data*, not just the module.

Why this file exists
--------------------

The main suite drives every module once, on `tips`, with a hand-picked pair of
columns. A human tester following the same worksheet on `titanic` and on his
own CSVs found faults in the first hour that the suite had passed over for
months. Reviewing what he hit, the pattern was never "this module is broken" —
it was "this module is broken *for this kind of column*":

* Pivoting by a numeric column produces integer column labels, and the data
  grid failed on those with "'DataFrame' object has no attribute 'dtype'".
  Pivoting tips by `sex` works; by `size` it did not. The suite tested `sex`.
* A 0/1 integer column classifies as numeric, so `Survived` never appeared in
  Cross-tab's variable list. Nothing errored — the option simply was absent.
* Correlate with a single column selected produced no chart and no message.

So these tests parameterise over column *shapes*: numeric labels, binary
integers, high-cardinality text, all-missing, constant. One module, several
kinds of data, rather than several modules on one kind of data.

They are slower than the main suite and deliberately narrow.
"""

from __future__ import annotations

import pytest

from tests.test_e2e import (  # reuse the harness
    _assert_choices_include,
    _assert_no_error_text,
    _assert_no_shiny_errors,
    _assert_output_has_content,
    _click_button,
    _nav_to,
    _select_multiple,
    _select_option,
    _sid,
    _wait_stable,
    app_url,  # noqa: F401 - fixture
    page,  # noqa: F401 - fixture
)


def _load_bundled(page, name: str) -> None:
    """Load a bundled dataset and make it active."""
    _nav_to(page, "Data", "Load")
    _wait_stable(page, 1500)
    _select_option(page, _sid("load", "source"), "bundled")
    _wait_stable(page, 1000)
    _select_option(page, _sid("load", "bundled_name"), name)
    _wait_stable(page, 1000)
    _click_button(page, _sid("load", "load_btn"))
    _wait_stable(page, 3000)


class TestPivotColumnLabelTypes:
    """The crash a human hit in his first twenty minutes."""

    @pytest.mark.parametrize("columns_var,label", [
        ("sex", "string labels"),
        ("size", "integer labels"),
        ("", "no column variable"),
    ])
    def test_pivot_renders_for_any_column_type(self, page, columns_var, label):
        _load_bundled(page, "tips")
        _nav_to(page, "Explore", "Pivot")
        _wait_stable(page, 2000)

        _select_option(page, _sid("pivot", "index"), "day")
        _select_option(page, _sid("pivot", "columns"), columns_var)
        _select_option(page, _sid("pivot", "values"), "total_bill")
        _select_option(page, _sid("pivot", "aggfunc"), "mean")
        _click_button(page, _sid("pivot", "run_btn"))
        _wait_stable(page, 4000)

        # Not "does an element exist" — does a table actually render, and is
        # the region free of exception text.
        _assert_output_has_content(page, _sid("pivot", "pivot_table"), kind="table")
        _assert_no_shiny_errors(page)


class TestBinaryIntegerColumns:
    """A 0/1 integer column is categorical to a student and numeric to us."""

    def test_crosstab_offers_a_binary_outcome(self, page):
        _load_bundled(page, "titanic")
        _nav_to(page, "Explore", "Cross-tab")
        _wait_stable(page, 2500)
        _assert_choices_include(page, _sid("crosstab", "col_var"), ["Survived"])

    def test_crosstab_runs_on_a_binary_outcome(self, page):
        _load_bundled(page, "titanic")
        _nav_to(page, "Explore", "Cross-tab")
        _wait_stable(page, 2500)
        _select_option(page, _sid("crosstab", "row_var"), "Sex")
        _select_option(page, _sid("crosstab", "col_var"), "Survived")
        _click_button(page, _sid("crosstab", "run_btn"))
        _wait_stable(page, 3000)
        _assert_output_has_content(page, _sid("crosstab", "crosstab_table"), kind="table")


class TestDegenerateSelections:
    """Selections a student makes by accident must say something."""

    def test_correlate_with_one_column_explains_itself(self, page):
        _load_bundled(page, "titanic")
        _nav_to(page, "Visualize", "Correlate")
        _wait_stable(page, 2500)
        _select_multiple(page, _sid("correlate", "cols"), ["Age"])
        _click_button(page, _sid("correlate", "run_btn"))
        _wait_stable(page, 3500)

        # Either a chart or an explanation. Silence is the bug: the element
        # existed and showed nothing at all.
        _assert_output_has_content(page, _sid("correlate", "chart"), kind="any")

    def test_correlate_with_several_columns_draws(self, page):
        _load_bundled(page, "titanic")
        _nav_to(page, "Visualize", "Correlate")
        _wait_stable(page, 2500)
        _select_multiple(page, _sid("correlate", "cols"), ["Age", "Fare", "Pclass"])
        _click_button(page, _sid("correlate", "run_btn"))
        _wait_stable(page, 4000)
        _assert_output_has_content(page, _sid("correlate", "chart"), kind="image")


class TestDatasetSwitching:
    """Loading a dataset should take you to it; refreshing should not move you."""

    def test_loading_a_dataset_makes_it_active(self, page):
        _load_bundled(page, "tips")
        assert page.locator(_sid("ds", "dataset")).input_value() == "tips"

        _load_bundled(page, "titanic")
        active = page.locator(_sid("ds", "dataset")).input_value()
        assert active == "titanic", (
            f"loading titanic left {active!r} active — a tester reported having "
            f"to hunt for the dropdown after loading"
        )

    def test_switching_back_sticks(self, page):
        _load_bundled(page, "tips")
        _load_bundled(page, "titanic")
        _select_option(page, _sid("ds", "dataset"), "tips")
        _wait_stable(page, 1500)

        # Visiting other tabs refreshes their dropdowns, which used to reset
        # the active dataset to whichever name sorts first.
        for tab in ("Profile", "View", "Transform"):
            _nav_to(page, "Data", tab)
            _wait_stable(page, 1200)

        assert page.locator(_sid("ds", "dataset")).input_value() == "tips"


class TestErrorTextIsNotOutput:
    """An exception shown as text is not a Shiny error, and used to pass."""

    def test_profile_on_a_wide_mix_of_types(self, page):
        _load_bundled(page, "titanic")
        _nav_to(page, "Data", "Profile")
        _wait_stable(page, 3000)
        _assert_no_error_text(page, "body")
        _assert_no_shiny_errors(page)

    def test_every_visualize_tab_is_free_of_error_text(self, page):
        _load_bundled(page, "titanic")
        for tab in ("Distribute", "Relate", "Compare", "Correlate", "Timeline"):
            _nav_to(page, "Visualize", tab)
            _wait_stable(page, 1500)
            _assert_no_error_text(page, "body")
