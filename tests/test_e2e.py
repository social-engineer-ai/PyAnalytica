"""
Comprehensive Playwright end-to-end test suite for the PyAnalytica Shiny app.

Run with:
    PYTHONPATH=src python -m pytest tests/test_e2e.py -v

Requirements:
    pip install pytest playwright pytest-playwright
    playwright install chromium
"""

from __future__ import annotations

import os
import random
import socket
import subprocess
import sys
import time
from typing import Generator

import pytest
from playwright.sync_api import Page, expect


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 60.0) -> None:
    """Poll *url* until it responds with 200 or *timeout* seconds elapse."""
    import urllib.request
    import urllib.error

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return
        except (urllib.error.URLError, OSError, ConnectionRefusedError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")


# ---------------------------------------------------------------------------
# Shiny module-namespaced selector helpers
#
# In Shiny for Python, a module with namespace "load" and widget id "source"
# produces an HTML element with id="load-source".  These helpers build
# the correct CSS selectors.
# ---------------------------------------------------------------------------

def _sid(module: str, widget: str) -> str:
    """Build a '#module-widget' CSS selector for a Shiny module element."""
    return f"#{module}-{widget}"


def _select_option(page: Page, selector: str, value: str, *, timeout: float = 10_000) -> None:
    """
    Set a Shiny select-input to *value*.

    Shiny uses selectize.js for some inputs and plain <select> for others.
    We try the simple select_option first and fall back to typing into
    a selectize wrapper if that fails.
    """
    locator = page.locator(selector)
    locator.wait_for(state="attached", timeout=timeout)
    try:
        locator.select_option(value, timeout=3000)
    except Exception:
        # Selectize: click the control, type value, press Enter
        wrapper = page.locator(f"{selector} + .selectize-control, {selector} ~ .selectize-control")
        if wrapper.count() > 0:
            wrapper.locator(".selectize-input").click()
            page.keyboard.type(value, delay=50)
            page.keyboard.press("Enter")
        else:
            # Last resort: try JavaScript
            page.evaluate(
                f"""(v) => {{
                    const el = document.querySelector('{selector}');
                    if (el) {{ el.value = v; el.dispatchEvent(new Event('change')); }}
                }}""",
                value,
            )


def _select_multiple(page: Page, selector: str, values: list[str], *, timeout: float = 10_000) -> None:
    """
    Set a Shiny selectize-multiple input to the given list of values.

    Works for both selectize.js and plain <select multiple> elements.
    """
    locator = page.locator(selector)
    locator.wait_for(state="attached", timeout=timeout)

    # Try plain multi-select first
    try:
        locator.select_option(values, timeout=3000)
        return
    except Exception:
        pass

    # Selectize: click and type each value
    wrapper = page.locator(f"{selector} + .selectize-control, {selector} ~ .selectize-control")
    if wrapper.count() > 0:
        selectize_input = wrapper.locator(".selectize-input")
        for val in values:
            selectize_input.click()
            page.keyboard.type(val, delay=30)
            time.sleep(0.3)
            page.keyboard.press("Enter")
            time.sleep(0.3)
    else:
        # JavaScript fallback for multi-select
        page.evaluate(
            f"""(vals) => {{
                const el = document.querySelector('{selector}');
                if (el) {{
                    Array.from(el.options).forEach(o => o.selected = vals.includes(o.value));
                    el.dispatchEvent(new Event('change'));
                }}
            }}""",
            values,
        )


def _click_button(page: Page, selector: str, *, timeout: float = 10_000) -> None:
    """Click a Shiny action button and give the server a moment to respond."""
    btn = page.locator(selector)
    btn.wait_for(state="visible", timeout=timeout)
    btn.scroll_into_view_if_needed()
    btn.click()


def _nav_to(page: Page, *tab_labels: str, timeout: float = 10_000) -> None:
    """
    Navigate through Shiny's navbar and sub-tabs by visible label text.

    Usage:
        _nav_to(page, "Data", "Load")        # top-nav "Data", sub-tab "Load"
        _nav_to(page, "Explore", "Pivot")     # top-nav "Explore", sub-tab "Pivot"
    """
    for label in tab_labels:
        link = page.locator(f"a.nav-link:has-text('{label}')")
        # There may be duplicates (navbar + sub-tab); click the first visible one
        link.first.wait_for(state="visible", timeout=timeout)
        link.first.click()
        time.sleep(0.5)


def _assert_no_shiny_errors(page: Page) -> None:
    """Assert that no Shiny output-error elements are visible on the page."""
    errors = page.locator(".shiny-output-error:visible")
    count = errors.count()
    if count > 0:
        texts = [errors.nth(i).inner_text() for i in range(count)]
        pytest.fail(f"Found {count} Shiny output error(s): {texts}")


def _wait_stable(page: Page, ms: int = 2000) -> None:
    """Wait for Shiny reactivity to settle."""
    time.sleep(ms / 1000)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")


@pytest.fixture(scope="module")
def app_url() -> Generator[str, None, None]:
    """Start the Shiny app on a random port and yield its URL.

    The process is killed after all tests in this module complete.
    """
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    # Ensure our src is on PYTHONPATH
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = SRC_DIR + (os.pathsep + existing if existing else "")

    # Use non-interactive matplotlib backend to avoid Tk issues
    env["MPLBACKEND"] = "Agg"

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "shiny", "run",
            os.path.join(SRC_DIR, "pyanalytica", "ui", "app.py"),
            "--port", str(port),
            "--host", "127.0.0.1",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_server(url, timeout=90)
    except TimeoutError:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        raise RuntimeError(
            f"Shiny app failed to start on port {port}.\n"
            f"STDOUT: {stdout.decode(errors='replace')}\n"
            f"STDERR: {stderr.decode(errors='replace')}"
        )

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture(scope="module")
def page(app_url: str) -> Generator[Page, None, None]:
    """Module-scoped page that stays open across all tests."""
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    ctx = browser.new_context()
    pg = ctx.new_page()
    pg.goto(app_url, wait_until="networkidle")
    _wait_stable(pg, 3000)
    yield pg
    ctx.close()
    browser.close()
    pw.stop()


# ============================================================================
# Test functions — ordered by naming convention (t01_, t02_, ...) so that
# pytest collects them in the intended execution sequence.
# ============================================================================


# ---------------------------------------------------------------------------
# INFRASTRUCTURE
# ---------------------------------------------------------------------------

class TestInfrastructure:
    """Tests that verify the app boots correctly and the header controls exist."""

    def test_t01_app_returns_200(self, app_url: str):
        """GET / returns HTTP 200."""
        import urllib.request
        resp = urllib.request.urlopen(app_url, timeout=10)
        assert resp.status == 200

    def test_t02_header_has_dataset_selector(self, page: Page):
        """The dataset selector dropdown is present in the header."""
        sel = page.locator(_sid("ds", "dataset"))
        expect(sel).to_be_attached()

    def test_t03_header_has_decimals_selector(self, page: Page):
        """The decimals dropdown is present in the header."""
        sel = page.locator(_sid("ds", "decimals"))
        expect(sel).to_be_attached()

    def test_t04_initial_dropdown_shows_none(self, page: Page):
        """Before loading data the dataset selector shows '(none)'."""
        sel = page.locator(_sid("ds", "dataset"))
        selected = sel.input_value()
        assert selected == "(none)", f"Expected '(none)', got '{selected}'"


# ---------------------------------------------------------------------------
# DATA > LOAD
# ---------------------------------------------------------------------------

class TestDataLoad:
    """Tests for the Data > Load module."""

    def test_t05_load_tips(self, page: Page):
        """Load the bundled 'tips' dataset."""
        _nav_to(page, "Data", "Load")
        _wait_stable(page, 1500)

        # Source should default to 'bundled'
        source = page.locator(_sid("load", "source"))
        expect(source).to_be_attached()

        # Wait for the bundled_name select to render (it is dynamic UI)
        page.wait_for_selector(_sid("load", "bundled_name"), state="attached", timeout=10_000)
        _select_option(page, _sid("load", "bundled_name"), "tips")
        _wait_stable(page)

        _click_button(page, _sid("load", "load_btn"))
        _wait_stable(page, 4000)

        # Notification should have appeared (Shiny notifications are transient;
        # check the dataset selector updated instead)
        ds_sel = page.locator(_sid("ds", "dataset"))
        expect(ds_sel).to_contain_text("tips")
        _assert_no_shiny_errors(page)

    def test_t06_load_diamonds(self, page: Page):
        """Load the bundled 'diamonds' dataset so two datasets are available."""
        _nav_to(page, "Data", "Load")
        _wait_stable(page)

        page.wait_for_selector(_sid("load", "bundled_name"), state="attached", timeout=10_000)
        _select_option(page, _sid("load", "bundled_name"), "diamonds")
        _wait_stable(page)

        _click_button(page, _sid("load", "load_btn"))
        _wait_stable(page, 5000)

        # Dataset selector should now list both
        ds_sel = page.locator(_sid("ds", "dataset"))
        html = ds_sel.inner_html()
        assert "tips" in html, "tips should still be in the selector"
        assert "diamonds" in html, "diamonds should be in the selector"
        _assert_no_shiny_errors(page)

    def test_t07_switch_back_to_tips(self, page: Page):
        """Switch the active dataset back to tips for remaining tests."""
        _select_option(page, _sid("ds", "dataset"), "tips")
        _wait_stable(page, 2000)

    def test_t08_preview_table_visible(self, page: Page):
        """After loading a dataset, the preview table on Load tab shows data."""
        _nav_to(page, "Data", "Load")
        _wait_stable(page, 2000)
        # The DataGrid renders inside a container with the output id
        table = page.locator(_sid("load", "preview_table"))
        expect(table).to_be_attached()
        # It should have some rows rendered
        page.wait_for_selector(f"{_sid('load', 'preview_table')} table, {_sid('load', 'preview_table')} .shiny-data-grid", timeout=10_000)
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# DATA > PROFILE
# ---------------------------------------------------------------------------

class TestDataProfile:
    """Tests for the Data > Profile module."""

    def test_t09_profile_overview_renders(self, page: Page):
        """Navigate to Profile tab and verify overview section renders."""
        _nav_to(page, "Data", "Profile")
        _wait_stable(page, 3000)

        overview = page.locator(_sid("profile", "overview"))
        expect(overview).to_be_attached()
        # Should show shape info (rows/columns)
        page.wait_for_selector(f"{_sid('profile', 'overview')} table", timeout=15_000)
        overview_text = overview.inner_text()
        assert "Rows" in overview_text or "rows" in overview_text.lower(), \
            f"Overview should contain 'Rows', got: {overview_text[:300]}"
        _assert_no_shiny_errors(page)

    def test_t10_profile_columns_tab(self, page: Page):
        """Navigate to the Columns sub-tab within Profile."""
        # Click the "Columns" sub-tab within the Profile panel
        columns_tab = page.locator("a.nav-link:has-text('Columns')")
        if columns_tab.count() > 0:
            columns_tab.last.click()
            _wait_stable(page, 3000)
        column_details = page.locator(_sid("profile", "column_details"))
        expect(column_details).to_be_attached()
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# DATA > VIEW
# ---------------------------------------------------------------------------

class TestDataView:
    """Tests for the Data > View module."""

    def test_t11_view_table_renders(self, page: Page):
        """Navigate to the View tab and verify the data table renders."""
        _nav_to(page, "Data", "View")
        _wait_stable(page, 3000)

        table = page.locator(_sid("view", "view_table"))
        expect(table).to_be_attached()
        _assert_no_shiny_errors(page)

    def test_t12_view_filter_info(self, page: Page):
        """Filter info text is present."""
        info = page.locator(_sid("view", "filter_info"))
        expect(info).to_be_attached()
        text = info.inner_text()
        assert "rows" in text.lower() or "Showing" in text, f"Expected row info, got: {text}"
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# DATA > TRANSFORM
# ---------------------------------------------------------------------------

class TestDataTransform:
    """Tests for the Data > Transform module."""

    def test_t13_transform_actions_available(self, page: Page):
        """Navigate to Transform tab and verify all actions are in the dropdown."""
        _nav_to(page, "Data", "Transform")
        _wait_stable(page, 2000)

        action_select = page.locator(_sid("transform", "action"))
        expect(action_select).to_be_attached()

        html = action_select.inner_html()
        expected_actions = [
            "fill_missing", "drop_missing", "drop_columns",
            "convert_dtype", "drop_duplicates", "add_log",
            "add_zscore", "add_rank", "str_lower", "str_upper", "str_strip",
        ]
        for action in expected_actions:
            assert action in html, f"Action '{action}' not found in transform dropdown"
        _assert_no_shiny_errors(page)

    def test_t14_drop_columns_shows_multiselect(self, page: Page):
        """Selecting 'Drop Column(s)' action reveals a multi-select input."""
        _select_option(page, _sid("transform", "action"), "drop_columns")
        _wait_stable(page, 2000)

        # The dynamic UI should now show a selectize input for columns to drop
        drop_cols = page.locator(_sid("transform", "drop_cols"))
        expect(drop_cols).to_be_attached(timeout=10_000)
        _assert_no_shiny_errors(page)

    def test_t15_add_log_column(self, page: Page):
        """Apply an 'Add Log Column' transform and verify preview updates."""
        _select_option(page, _sid("transform", "action"), "add_log")
        _wait_stable(page, 2000)

        # Wait for dynamic controls to render
        page.wait_for_selector(_sid("transform", "col"), state="attached", timeout=10_000)
        _select_option(page, _sid("transform", "col"), "total_bill")
        _wait_stable(page)

        _click_button(page, _sid("transform", "apply_btn"))
        _wait_stable(page, 3000)

        # Verify the preview table is still showing
        preview = page.locator(_sid("transform", "preview"))
        expect(preview).to_be_attached()
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# DATA > COMBINE
# ---------------------------------------------------------------------------

class TestDataCombine:
    """Tests for the Data > Combine module."""

    def test_t16_combine_tab_loads(self, page: Page):
        """Navigate to Combine tab and verify both dataset selectors exist."""
        _nav_to(page, "Data", "Combine")
        _wait_stable(page, 2000)

        left = page.locator(_sid("combine", "left"))
        right = page.locator(_sid("combine", "right"))
        expect(left).to_be_attached()
        expect(right).to_be_attached()

        # Both datasets should be in the selectors
        left_html = left.inner_html()
        right_html = right.inner_html()
        assert "tips" in left_html, "tips should be available as left dataset"
        assert "diamonds" in right_html or "tips" in right_html, "datasets should be in right selector"
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# DATA > EXPORT
# ---------------------------------------------------------------------------

class TestDataExport:
    """Tests for the Data > Export module."""

    def test_t17_export_tab_has_download_button(self, page: Page):
        """Navigate to Export tab and verify the download button exists."""
        _nav_to(page, "Data", "Export")
        _wait_stable(page, 2000)

        download_btn = page.locator(_sid("export", "download_btn"))
        expect(download_btn).to_be_attached()

        # Export info should show dataset info
        info = page.locator(_sid("export", "export_info"))
        expect(info).to_be_attached()
        _assert_no_shiny_errors(page)

    def test_t18_export_preview_shows_data(self, page: Page):
        """The export preview table is visible with data."""
        preview = page.locator(_sid("export", "export_preview"))
        expect(preview).to_be_attached()
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# EXPLORE > SUMMARIZE
# ---------------------------------------------------------------------------

class TestExploreSummarize:
    """Tests for the Explore > Summarize module."""

    def test_t19_summarize_select_and_run(self, page: Page):
        """Select group and value columns, run summarize, verify result table."""
        _nav_to(page, "Explore", "Summarize")
        _wait_stable(page, 2000)

        # Wait for column selectors to populate
        page.wait_for_selector(
            f"{_sid('summarize', 'group_cols')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_multiple(page, _sid("summarize", "group_cols"), ["day"])
        _wait_stable(page)
        _select_multiple(page, _sid("summarize", "value_cols"), ["total_bill"])
        _wait_stable(page)

        _click_button(page, _sid("summarize", "run_btn"))
        _wait_stable(page, 4000)

        table = page.locator(_sid("summarize", "summary_table"))
        expect(table).to_be_attached()
        # Wait for actual data to render
        page.wait_for_selector(
            f"{_sid('summarize', 'summary_table')} table, {_sid('summarize', 'summary_table')} .shiny-data-grid",
            timeout=15_000,
        )
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# EXPLORE > PIVOT
# ---------------------------------------------------------------------------

class TestExplorePivot:
    """Tests for the Explore > Pivot module."""

    def test_t20_pivot_select_and_run(self, page: Page):
        """Select row, column, value variables, create pivot, verify table."""
        _nav_to(page, "Explore", "Pivot")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('pivot', 'index')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_option(page, _sid("pivot", "index"), "day")
        _wait_stable(page)
        _select_option(page, _sid("pivot", "columns"), "time")
        _wait_stable(page)
        _select_option(page, _sid("pivot", "values"), "total_bill")
        _wait_stable(page)

        _click_button(page, _sid("pivot", "run_btn"))
        _wait_stable(page, 4000)

        table = page.locator(_sid("pivot", "pivot_table"))
        expect(table).to_be_attached()
        page.wait_for_selector(
            f"{_sid('pivot', 'pivot_table')} table, {_sid('pivot', 'pivot_table')} .shiny-data-grid",
            timeout=15_000,
        )
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# EXPLORE > CROSS-TAB
# ---------------------------------------------------------------------------

class TestExploreCrosstab:
    """Tests for the Explore > Cross-tab module."""

    def test_t21_crosstab_chi_square(self, page: Page):
        """Select variables, run cross-tab, verify chi-square result + table."""
        _nav_to(page, "Explore", "Cross-tab")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('crosstab', 'row_var')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_option(page, _sid("crosstab", "row_var"), "sex")
        _wait_stable(page)
        _select_option(page, _sid("crosstab", "col_var"), "smoker")
        _wait_stable(page)

        _click_button(page, _sid("crosstab", "run_btn"))
        _wait_stable(page, 4000)

        # Chi-square result should appear
        chi2 = page.locator(_sid("crosstab", "chi2_result"))
        expect(chi2).to_be_attached()
        page.wait_for_selector(f"{_sid('crosstab', 'chi2_result')} .alert", timeout=15_000)
        chi2_text = chi2.inner_text()
        assert "Chi-Square" in chi2_text or "chi" in chi2_text.lower(), \
            f"Expected chi-square mention, got: {chi2_text[:200]}"

        # Cross-tab table should appear
        table = page.locator(_sid("crosstab", "crosstab_table"))
        expect(table).to_be_attached()
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# VISUALIZE > DISTRIBUTE
# ---------------------------------------------------------------------------

class TestVisualizeDistribute:
    """Tests for the Visualize > Distribute module."""

    def test_t22_histogram(self, page: Page):
        """Select a column, run, verify chart appears."""
        _nav_to(page, "Visualize", "Distribute")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('distribute', 'col')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_option(page, _sid("distribute", "col"), "total_bill")
        _wait_stable(page)
        _select_option(page, _sid("distribute", "chart_type"), "histogram")
        _wait_stable(page)

        _click_button(page, _sid("distribute", "run_btn"))
        _wait_stable(page, 5000)

        # Chart should render as an <img> (Shiny render.plot outputs an image)
        chart = page.locator(_sid("distribute", "chart"))
        expect(chart).to_be_attached()
        # Wait for an img element inside the chart output
        page.wait_for_selector(
            f"{_sid('distribute', 'chart')} img",
            state="visible", timeout=15_000,
        )
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# VISUALIZE > RELATE
# ---------------------------------------------------------------------------

class TestVisualizeRelate:
    """Tests for the Visualize > Relate module."""

    def test_t23_scatter_plot(self, page: Page):
        """Select x and y, run, verify scatter plot appears."""
        _nav_to(page, "Visualize", "Relate")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('relate', 'x')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_option(page, _sid("relate", "x"), "total_bill")
        _wait_stable(page)
        _select_option(page, _sid("relate", "y"), "tip")
        _wait_stable(page)

        _click_button(page, _sid("relate", "run_btn"))
        _wait_stable(page, 5000)

        chart = page.locator(_sid("relate", "chart"))
        expect(chart).to_be_attached()
        page.wait_for_selector(
            f"{_sid('relate', 'chart')} img",
            state="visible", timeout=15_000,
        )
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# VISUALIZE > COMPARE
# ---------------------------------------------------------------------------

class TestVisualizeCompare:
    """Tests for the Visualize > Compare module."""

    def test_t24_grouped_boxplot(self, page: Page):
        """Select category and numeric, run, verify grouped plot appears."""
        _nav_to(page, "Visualize", "Compare")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('compare', 'x_cat')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_option(page, _sid("compare", "x_cat"), "day")
        _wait_stable(page)
        _select_option(page, _sid("compare", "y_num"), "total_bill")
        _wait_stable(page)

        _click_button(page, _sid("compare", "run_btn"))
        _wait_stable(page, 5000)

        chart = page.locator(_sid("compare", "chart"))
        expect(chart).to_be_attached()
        page.wait_for_selector(
            f"{_sid('compare', 'chart')} img",
            state="visible", timeout=15_000,
        )
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# VISUALIZE > CORRELATE
# ---------------------------------------------------------------------------

class TestVisualizeCorrelate:
    """Tests for the Visualize > Correlate module."""

    def test_t25_correlation_matrix(self, page: Page):
        """Select numeric columns, run, verify correlation matrix appears."""
        _nav_to(page, "Visualize", "Correlate")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('correlate', 'cols')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_multiple(page, _sid("correlate", "cols"), ["total_bill", "tip", "size"])
        _wait_stable(page)

        _click_button(page, _sid("correlate", "run_btn"))
        _wait_stable(page, 5000)

        chart = page.locator(_sid("correlate", "chart"))
        expect(chart).to_be_attached()
        page.wait_for_selector(
            f"{_sid('correlate', 'chart')} img",
            state="visible", timeout=15_000,
        )
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# ANALYZE > MEANS
# ---------------------------------------------------------------------------

class TestAnalyzeMeans:
    """Tests for the Analyze > Means module."""

    def test_t26_two_sample_ttest(self, page: Page):
        """Run a two-sample t-test and verify result, group stats, no errors."""
        _nav_to(page, "Analyze", "Means")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('means', 'value_col')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        # Select two-sample test
        _select_option(page, _sid("means", "test_type"), "two_sample")
        _wait_stable(page, 2000)

        _select_option(page, _sid("means", "value_col"), "total_bill")
        _wait_stable(page, 1500)

        # Wait for group_col to appear (dynamic UI)
        page.wait_for_selector(_sid("means", "group_col"), state="attached", timeout=10_000)
        _select_option(page, _sid("means", "group_col"), "sex")
        _wait_stable(page)

        _click_button(page, _sid("means", "run_btn"))
        _wait_stable(page, 5000)

        # Check test result alert appears
        test_result = page.locator(_sid("means", "test_result"))
        expect(test_result).to_be_attached()
        page.wait_for_selector(f"{_sid('means', 'test_result')} .alert", timeout=15_000)
        result_text = test_result.inner_text()
        assert "t-test" in result_text.lower() or "t test" in result_text.lower() or \
               "statistic" in result_text.lower() or "p-value" in result_text.lower(), \
            f"Expected test name or statistic in result, got: {result_text[:300]}"

        # Check group stats table renders
        group_stats = page.locator(_sid("means", "group_stats"))
        expect(group_stats).to_be_attached()

        _assert_no_shiny_errors(page)

    def test_t27_one_way_anova(self, page: Page):
        """Run a one-way ANOVA and verify result renders."""
        _select_option(page, _sid("means", "test_type"), "anova")
        _wait_stable(page, 2000)

        _select_option(page, _sid("means", "value_col"), "total_bill")
        _wait_stable(page, 1500)

        page.wait_for_selector(_sid("means", "group_col"), state="attached", timeout=10_000)
        _select_option(page, _sid("means", "group_col"), "day")
        _wait_stable(page)

        _click_button(page, _sid("means", "run_btn"))
        _wait_stable(page, 5000)

        test_result = page.locator(_sid("means", "test_result"))
        expect(test_result).to_be_attached()
        page.wait_for_selector(f"{_sid('means', 'test_result')} .alert", timeout=15_000)
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# ANALYZE > PROPORTIONS
# ---------------------------------------------------------------------------

class TestAnalyzeProportions:
    """Tests for the Analyze > Proportions module."""

    def test_t28_chi_square_test(self, page: Page):
        """Run proportions test and verify result + observed/expected tables."""
        _nav_to(page, "Analyze", "Proportions")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('proportions', 'row_var')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_option(page, _sid("proportions", "row_var"), "sex")
        _wait_stable(page)
        _select_option(page, _sid("proportions", "col_var"), "smoker")
        _wait_stable(page)

        _click_button(page, _sid("proportions", "run_btn"))
        _wait_stable(page, 5000)

        # Test result
        test_result = page.locator(_sid("proportions", "test_result"))
        expect(test_result).to_be_attached()
        page.wait_for_selector(f"{_sid('proportions', 'test_result')} .alert", timeout=15_000)

        # Observed table
        observed = page.locator(_sid("proportions", "observed"))
        expect(observed).to_be_attached()

        # Expected table
        expected = page.locator(_sid("proportions", "expected"))
        expect(expected).to_be_attached()

        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# ANALYZE > CORRELATION
# ---------------------------------------------------------------------------

class TestAnalyzeCorrelation:
    """Tests for the Analyze > Correlation module."""

    def test_t29_correlation_test(self, page: Page):
        """Run correlation test and verify result with r, p, CI."""
        _nav_to(page, "Analyze", "Correlation")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('correlation', 'x')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_option(page, _sid("correlation", "x"), "total_bill")
        _wait_stable(page)
        _select_option(page, _sid("correlation", "y"), "tip")
        _wait_stable(page)

        _click_button(page, _sid("correlation", "run_btn"))
        _wait_stable(page, 5000)

        test_result = page.locator(_sid("correlation", "test_result"))
        expect(test_result).to_be_attached()
        page.wait_for_selector(f"{_sid('correlation', 'test_result')} .alert", timeout=15_000)

        result_text = test_result.inner_text()
        assert "correlation" in result_text.lower() or "r =" in result_text.lower(), \
            f"Expected correlation info in result, got: {result_text[:300]}"
        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# MODEL > REGRESSION
# ---------------------------------------------------------------------------

class TestModelRegression:
    """Tests for the Model > Regression module."""

    def test_t30_linear_regression(self, page: Page):
        """Run linear regression and verify coefficients + VIF tables."""
        _nav_to(page, "Model", "Regression")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('regression', 'target')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_option(page, _sid("regression", "target"), "tip")
        _wait_stable(page)
        _select_multiple(page, _sid("regression", "features"), ["total_bill", "size"])
        _wait_stable(page)

        _click_button(page, _sid("regression", "run_btn"))
        _wait_stable(page, 6000)

        # Model summary
        summary = page.locator(_sid("regression", "model_summary"))
        expect(summary).to_be_attached()
        page.wait_for_selector(f"{_sid('regression', 'model_summary')} .alert", timeout=15_000)
        summary_text = summary.inner_text()
        assert "R" in summary_text or "regression" in summary_text.lower(), \
            f"Expected regression summary, got: {summary_text[:300]}"

        # Coefficients table
        coef = page.locator(_sid("regression", "coef_table"))
        expect(coef).to_be_attached()

        # VIF table
        vif = page.locator(_sid("regression", "vif_table"))
        expect(vif).to_be_attached()

        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# MODEL > CLASSIFY
# ---------------------------------------------------------------------------

class TestModelClassify:
    """Tests for the Model > Classify module."""

    def test_t31_logistic_regression(self, page: Page):
        """Run logistic regression and verify accuracy is displayed."""
        _nav_to(page, "Model", "Classify")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('classify', 'target')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        # For tips, use 'sex' as target (categorical binary) and numeric features
        _select_option(page, _sid("classify", "target"), "sex")
        _wait_stable(page)
        _select_multiple(page, _sid("classify", "features"), ["total_bill", "tip", "size"])
        _wait_stable(page)

        _click_button(page, _sid("classify", "run_btn"))
        _wait_stable(page, 8000)

        # Model summary should show accuracy
        summary = page.locator(_sid("classify", "model_summary"))
        expect(summary).to_be_attached()
        page.wait_for_selector(f"{_sid('classify', 'model_summary')} .alert", timeout=20_000)
        summary_text = summary.inner_text()
        assert "accuracy" in summary_text.lower() or "Logistic" in summary_text or "Train" in summary_text, \
            f"Expected accuracy info, got: {summary_text[:300]}"

        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# MODEL > CLUSTER
# ---------------------------------------------------------------------------

class TestModelCluster:
    """Tests for the Model > Cluster module."""

    def test_t32_kmeans_cluster(self, page: Page):
        """Run K-means clustering and verify cluster plot/summary appears."""
        _nav_to(page, "Model", "Cluster")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('cluster', 'features')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_multiple(page, _sid("cluster", "features"), ["total_bill", "tip"])
        _wait_stable(page)

        _click_button(page, _sid("cluster", "run_btn"))
        _wait_stable(page, 8000)

        # Cluster summary
        summary = page.locator(_sid("cluster", "cluster_summary"))
        expect(summary).to_be_attached()
        page.wait_for_selector(f"{_sid('cluster', 'cluster_summary')} .alert", timeout=20_000)

        # Scatter plot should appear
        scatter = page.locator(_sid("cluster", "scatter_plot"))
        expect(scatter).to_be_attached()

        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# MODEL > REDUCE
# ---------------------------------------------------------------------------

class TestModelReduce:
    """Tests for the Model > Reduce (PCA) module."""

    def test_t33_pca_analysis(self, page: Page):
        """Run PCA and verify scree/biplot and summary appear."""
        _nav_to(page, "Model", "Reduce")
        _wait_stable(page, 2000)

        page.wait_for_selector(
            f"{_sid('reduce', 'features')} option:not([value=''])",
            state="attached", timeout=10_000,
        )

        _select_multiple(page, _sid("reduce", "features"), ["total_bill", "tip", "size"])
        _wait_stable(page)

        _click_button(page, _sid("reduce", "run_btn"))
        _wait_stable(page, 8000)

        # PCA summary
        summary = page.locator(_sid("reduce", "pca_summary"))
        expect(summary).to_be_attached()
        page.wait_for_selector(f"{_sid('reduce', 'pca_summary')} .alert", timeout=20_000)
        summary_text = summary.inner_text()
        assert "PCA" in summary_text or "variance" in summary_text.lower(), \
            f"Expected PCA summary, got: {summary_text[:300]}"

        # Scree plot
        scree = page.locator(_sid("reduce", "scree_plot"))
        expect(scree).to_be_attached()

        _assert_no_shiny_errors(page)


# ---------------------------------------------------------------------------
# CROSS-CUTTING: DECIMALS CHANGE
# ---------------------------------------------------------------------------

class TestCrossCutting:
    """Cross-cutting tests: decimals selector, global error checks."""

    def test_t34_change_decimals(self, page: Page):
        """Change the decimals dropdown from 4 to 2 and verify page still works."""
        _select_option(page, _sid("ds", "decimals"), "2")
        _wait_stable(page, 2000)

        # Navigate to a data view to see the effect
        _nav_to(page, "Data", "View")
        _wait_stable(page, 3000)

        # Table should still render
        table = page.locator(_sid("view", "view_table"))
        expect(table).to_be_attached()
        _assert_no_shiny_errors(page)

        # Restore to 4
        _select_option(page, _sid("ds", "decimals"), "4")
        _wait_stable(page, 1000)

    def test_t35_no_errors_on_data_tabs(self, page: Page):
        """Visit each Data sub-tab and verify no Shiny errors."""
        for tab in ["Load", "Profile", "View", "Transform", "Combine", "Export"]:
            _nav_to(page, "Data", tab)
            _wait_stable(page, 1500)
            _assert_no_shiny_errors(page)

    def test_t36_no_errors_on_explore_tabs(self, page: Page):
        """Visit each Explore sub-tab and verify no Shiny errors."""
        for tab in ["Summarize", "Pivot", "Cross-tab"]:
            _nav_to(page, "Explore", tab)
            _wait_stable(page, 1500)
            _assert_no_shiny_errors(page)

    def test_t37_no_errors_on_visualize_tabs(self, page: Page):
        """Visit each Visualize sub-tab and verify no Shiny errors."""
        for tab in ["Distribute", "Relate", "Compare", "Correlate", "Timeline"]:
            _nav_to(page, "Visualize", tab)
            _wait_stable(page, 1500)
            _assert_no_shiny_errors(page)

    def test_t38_no_errors_on_analyze_tabs(self, page: Page):
        """Visit each Analyze sub-tab and verify no Shiny errors."""
        for tab in ["Means", "Proportions", "Correlation"]:
            _nav_to(page, "Analyze", tab)
            _wait_stable(page, 1500)
            _assert_no_shiny_errors(page)

    def test_t39_no_errors_on_model_tabs(self, page: Page):
        """Visit each Model sub-tab and verify no Shiny errors."""
        for tab in ["Regression", "Classify", "Evaluate", "Cluster", "Reduce"]:
            _nav_to(page, "Model", tab)
            _wait_stable(page, 1500)
            _assert_no_shiny_errors(page)

    def test_t40_dataset_selector_persists(self, page: Page):
        """The active dataset remains 'tips' after navigating across all tabs."""
        ds_sel = page.locator(_sid("ds", "dataset"))
        val = ds_sel.input_value()
        assert val == "tips", f"Expected 'tips' selected, got '{val}'"
