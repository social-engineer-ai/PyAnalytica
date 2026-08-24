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
)

# `app_url` and `page` come from tests/conftest.py, so this module gets its own
# app and browser rather than sharing test_e2e's.


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

        # Either a chart or an explanation, and the explanation lives in its
        # own output beside the plot. Silence is the bug -- the chart element
        # existed and showed nothing, with no reason given anywhere.
        guidance = page.locator(_sid("correlate", "guidance")).inner_text().strip()
        chart_images = page.locator(f"{_sid('correlate', 'chart')} img").count()

        assert guidance or chart_images, (
            "selecting one column produced no chart and no explanation"
        )
        if guidance:
            assert "two" in guidance.lower(), (
                f"the explanation should say how many columns are needed: {guidance!r}"
            )

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


class TestDateParsingOnUpload:
    """A CSV date column must become a date, so Timeline has an axis to use."""

    def test_uploaded_dates_are_recognised_and_timeline_works(self, page):
        import os

        csv = os.path.abspath("examples/tester_files/sales.csv")
        _nav_to(page, "Data", "Load")
        _wait_stable(page, 1500)

        # Switch the source to file upload. Note this is a radio group, not a
        # select -- _select_option's fallback sets .value on a <div> and does
        # nothing, which is why this clicks the radio directly.
        radio = page.locator(f"{_sid('load', 'source')} input[type=radio][value='upload']")
        if radio.count():
            radio.first.check()
            _wait_stable(page, 1200)
            # The id is on the <input> itself, not a wrapper. Note also that
            # every module's file input exists in the DOM at once, so a bare
            # input[type=file] selector picks a hidden one from another tab.
            page.locator(_sid("load", "file_upload")).set_input_files(csv)
            _wait_stable(page, 2000)
            _click_button(page, _sid("load", "load_btn"))
            _wait_stable(page, 4000)

            _nav_to(page, "Visualize", "Timeline")
            _wait_stable(page, 2500)
            options = [
                o.strip()
                for o in page.locator(f"{_sid('timeline', 'date_col')} option").all_inner_texts()
            ]
            assert "date" in options, (
                f"Timeline was offered no date column after uploading a CSV "
                f"with dates. Offered: {options}"
            )


class TestPracticeDrills:
    """Practice had never been driven in a browser, only unit-tested.

    Doing so found two faults at once: every question's feedback was registered
    under the same output id, so checking an answer showed nothing; and the
    "load the tips dataset" hint never cleared once tips was loaded.
    """

    def test_a_correct_answer_says_so(self, page):
        _load_bundled(page, "tips")
        _nav_to(page, "Practice")
        _wait_stable(page, 2500)

        page.locator(_sid("practice", "ans_rows")).fill("244")
        _click_button(page, _sid("practice", "check_rows"))
        _wait_stable(page, 2000)

        feedback = page.locator(_sid("practice", "fb_rows")).inner_text()
        assert feedback.strip(), "checking an answer produced no feedback at all"
        assert "correct" in feedback.lower()

    def test_a_wrong_answer_says_so_and_offers_the_hint(self, page):
        _load_bundled(page, "tips")
        _nav_to(page, "Practice")
        _wait_stable(page, 2500)

        page.locator(_sid("practice", "ans_mean_bill")).fill("19.79")
        _click_button(page, _sid("practice", "check_mean_bill"))
        _wait_stable(page, 2000)

        feedback = page.locator(_sid("practice", "fb_mean_bill")).inner_text().lower()
        assert "not quite" in feedback
        assert "profile" in feedback  # the hint, so a wrong answer is a route forward

    def test_each_question_has_its_own_feedback(self, page):
        """All six were writing to one output, so only one could ever show."""
        _load_bundled(page, "tips")
        _nav_to(page, "Practice")
        _wait_stable(page, 2500)

        page.locator(_sid("practice", "ans_rows")).fill("244")
        _click_button(page, _sid("practice", "check_rows"))
        _wait_stable(page, 1500)

        assert page.locator(_sid("practice", "fb_rows")).inner_text().strip()
        assert not page.locator(_sid("practice", "fb_max_bill")).inner_text().strip()

    def test_the_dataset_hint_clears_once_loaded(self, page):
        _load_bundled(page, "tips")
        _nav_to(page, "Practice")
        _wait_stable(page, 2500)
        hint = page.locator(_sid("practice", "dataset_hint")).inner_text().lower()
        assert "using the tips dataset" in hint, (
            f"the hint still tells the student to load a dataset they have: {hint!r}"
        )

    def test_score_counts_attempts_and_resets(self, page):
        _load_bundled(page, "tips")
        _nav_to(page, "Practice")
        _wait_stable(page, 2500)

        page.locator(_sid("practice", "ans_rows")).fill("244")
        _click_button(page, _sid("practice", "check_rows"))
        _wait_stable(page, 1500)
        assert "1 of 6 correct" in page.locator(_sid("practice", "score_panel")).inner_text()

        _click_button(page, _sid("practice", "reset"))
        _wait_stable(page, 1500)
        assert not page.locator(_sid("practice", "score_panel")).inner_text().strip()
