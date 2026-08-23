"""Shared test fixtures for PyAnalytica."""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from pyanalytica.core.state import WorkbenchState


@pytest.fixture
def sample_df():
    """Small mixed-type DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "id": range(1, n + 1),
        "name": [f"Person_{i}" for i in range(1, n + 1)],
        "age": np.random.randint(22, 65, size=n),
        "salary": np.random.normal(70000, 15000, size=n).round(0),
        "department": np.random.choice(["Sales", "Engineering", "Marketing", "HR"], size=n),
        "score": np.random.uniform(0, 100, size=n).round(2),
        "hired_date": pd.date_range("2020-01-01", periods=n, freq="W"),
        "active": np.random.choice([True, False], size=n, p=[0.8, 0.2]),
    })


@pytest.fixture
def sample_df_with_missing(sample_df):
    """DataFrame with some missing values."""
    df = sample_df.copy()
    rng = np.random.default_rng(42)
    mask = rng.choice(len(df), size=10, replace=False)
    df.loc[mask, "salary"] = np.nan
    mask2 = rng.choice(len(df), size=5, replace=False)
    df.loc[mask2, "department"] = np.nan
    return df


@pytest.fixture
def candidates_df():
    """Small candidates table for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "candidate_id": range(1, 51),
        "age": np.random.randint(22, 55, size=50),
        "seniority": np.random.choice(["Entry", "Junior", "Mid", "Senior"], size=50),
        "salary": np.random.normal(70000, 15000, size=50).round(0),
        "city": np.random.choice(["NYC", "SF", "Chicago"], size=50),
    })


@pytest.fixture
def events_df():
    """Small events table for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "event_id": range(1, 201),
        "candidate_id": np.random.randint(1, 51, size=200),
        "job_id": np.random.randint(1, 21, size=200),
        "event_type": np.random.choice(["application", "screen", "interview", "offer"], size=200),
        "event_date": pd.date_range("2025-01-01", periods=200, freq="3h"),
    })


@pytest.fixture
def jobs_df():
    """Small jobs table for testing."""
    return pd.DataFrame({
        "job_id": range(1, 21),
        "company_id": np.random.randint(1, 11, size=20),
        "title": [f"Role_{i}" for i in range(1, 21)],
        "seniority": np.random.choice(["Entry", "Junior", "Mid", "Senior"], size=20),
        "min_salary": np.random.randint(40000, 80000, size=20),
        "max_salary": np.random.randint(80000, 150000, size=20),
    })


@pytest.fixture
def companies_df():
    """Small companies table for testing."""
    return pd.DataFrame({
        "company_id": range(1, 11),
        "company_name": [f"Company_{i}" for i in range(1, 11)],
        "industry": np.random.choice(["Tech", "Finance", "Healthcare"], size=10),
        "company_size": np.random.choice(["Small", "Medium", "Large"], size=10),
    })


@pytest.fixture
def state(candidates_df, events_df):
    """Pre-loaded WorkbenchState."""
    s = WorkbenchState()
    s.load("candidates", candidates_df)
    s.load("events", events_df)
    return s


# ---------------------------------------------------------------------------
# Browser-test fixtures
#
# These live here rather than in test_e2e.py so that *each* test module gets
# its own app and its own page. A module-scoped fixture belongs to the module
# that defines it, so importing `page` from test_e2e shared one browser session
# across both browser files -- and state left by one file broke the other. They
# passed separately and failed together, which is the worst way to find out.
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

    # Isolate the app from the real home directory.
    #
    # core/session.py autosaves the workbench to ~/.pyanalytica/sessions and
    # app.py restores it on startup.  Without this the suite (a) inherits
    # whatever the previous run left behind, so results depend on run history,
    # and (b) overwrites the user's own saved session on the way out.
    fake_home = tempfile.mkdtemp(prefix="pyanalytica-e2e-home-")
    env["HOME"] = fake_home
    env["USERPROFILE"] = fake_home

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

    shutil.rmtree(fake_home, ignore_errors=True)


@pytest.fixture(scope="module")
def page(app_url: str) -> Generator[Page, None, None]:
    """Module-scoped page that stays open across all tests."""
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    ctx = browser.new_context()
    pg = ctx.new_page()
    pg.goto(app_url, wait_until="networkidle")
    time.sleep(3)  # let Shiny finish its first render
    yield pg
    ctx.close()
    browser.close()
    pw.stop()


