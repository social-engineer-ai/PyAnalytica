# PyAnalytica — End-to-End Test Plan

Status: draft (2026-08-22). Companion to `tests/test_e2e.py`.

Goal: drive the running app the way a student would, on every module, and produce evidence a human (or an agent) can audit afterwards — including the charts, which no assertion can fully judge on its own.

## 0. Scope and decisions

**In scope now:** Data, Explore, Visualize, Analyze, Model, Report, and the cross-cutting behaviours. This is 27 of the 29 UI modules and all of the analytics surface.

**Deferred to a separate plan:** `homework/mod_homework` and `ai/mod_assistant`. Both depend on unsettled product decisions — how a course is packaged, and how API access is gated to enrolled students — and testing them before those settle would encode the wrong assumptions.

**Decisions taken:**

| Question | Decision |
|---|---|
| Course content in fixtures | Course-neutral. Tests use bundled datasets and generic fixtures only; no BADM 576 assignment content, column names, or answer keys anywhere in the suite. The suite must survive the course-agnostic refactor untouched. |
| AI client | Real Anthropic client when AI testing begins (not a stub) — deferred with the AI plan. |
| CI | See below. |

**On CI:** `.github/workflows/ci.yml` already runs unit tests on Python 3.10–3.13 and explicitly does `--ignore=tests/test_e2e.py` — so today no browser test ever runs automatically. The question was where to reintroduce them. Recommendation: run the functional E2E layer on every push to `main` and every PR, on one Python version only (3.12) rather than the full matrix — browsers are slow and the matrix is about package compatibility, not UI behaviour. Run the chart-capture layer only on release tags and on demand, because its output is a pile of PNGs for a human or agent to review, which is not something a push-triggered job can act on.

---

## 1. Audit of the existing suite

`tests/test_e2e.py` — 1,128 lines, 40 tests (`t01`–`t40`), 21 test classes. It is a real foundation, not a stub: it starts the actual Shiny server on a free port, waits for HTTP 200, and drives Chromium through Playwright.

### What it already does well

| Asset | Where | Why it matters |
|---|---|---|
| Server lifecycle | `_free_port`, `_wait_for_server` | No port collisions, no fixed sleeps at startup |
| Shiny namespace helper | `_sid(module, id)` → `#module-id` | The single hardest part of testing Shiny modules is already solved |
| Nav helper | `_nav_to(page, "Visualize", "Distribute")` | Every new test is ~10 lines, not ~40 |
| Error oracle | `_assert_no_shiny_errors(page)` | Catches silent `shiny-output-error` divs that look like an empty panel |
| Settle helper | `_wait_stable(page, ms)` | Handles reactive re-render |

### Coverage by module (29 UI module files)

**Functionally tested — 20 modules**

- Data: `load` (t05–t08), `profile` (t09–t10), `view` (t11–t12), `transform` (t13–t15), `combine` (t16, smoke only), `export` (t17–t18)
- Explore: `summarize` (t19), `pivot` (t20), `crosstab` (t21)
- Visualize: `distribute` (t22), `relate` (t23), `compare` (t24), `correlate` (t25)
- Analyze: `means` (t26–t27), `proportions` (t28), `correlation` (t29)
- Model: `regression` (t30), `classify` (t31), `cluster` (t32), `reduce` (t33)

**Zero functional coverage — 9 modules**

| Module | Current state | Note |
|---|---|---|
| `explore/mod_simulate` | Not referenced at all | Newest feature (commit 9e5229f); not even in the tab smoke loop |
| `visualize/mod_timeline` | Tab-opens smoke only (t37) | No plot ever produced |
| `model/mod_evaluate` | Tab-opens smoke only (t39) | Full rewrite in Phase 6, untested through UI |
| `model/mod_predict` | Not referenced at all | Missing from the t39 tab list too |
| `report/mod_notebook` | None | |
| `report/mod_procedure` | None | Phase 6 headline feature: dynamic JS buttons, step reordering |
| `report/mod_report_builder` | None | |
| `homework/mod_homework` | None | Hash-checked grading — the part that must not silently break |
| `ai/mod_assistant` | None | Rule-based path is testable without a key |

### Depth gaps in the modules that *are* covered

1. **Charts assert existence only.** Every visualize test ends at `wait_for_selector("... img", state="visible")`. A chart that renders the wrong variable, an empty axis, or an all-grey figure passes. This is the gap the capture harness in §3 exists to close.

2. **"Show Code" is never asserted — 0 occurrences of `toggle_code` in the suite.** This is the cheapest large win available. `code_panel_ui("code")` is mounted with the *same* id in every module, so one helper (`_assert_code_contains(page, "distribute", "sns.histplot")`) works everywhere and gives a second, independent oracle per action: did the UI compute the right thing *and* emit the right pandas/sklearn code.

3. **Chart-type variants untested.** `mod_distribute` offers histogram / boxplot / violin / bar plus KDE, bins, percentages, orientation, group-by, and two facet dimensions. Exactly one combination (histogram, no options) is exercised.

4. **No error paths.** `mod_distribute` has an explicit branch that refuses a categorical column with a guidance message — never tested. The same class of branch exists across modules. These are also where the last three bug-fix commits landed (pivot index==values, optional column var, autosave signature).

5. **No edge-case data.** Everything runs on clean `tips`. No single-row frame, no all-missing column, no constant column, no high-cardinality categorical, no wide/tall extremes.

6. **State/persistence is one assertion.** t40 checks the dataset dropdown still reads `tips`. No test that a transform in Data is visible in Explore, which is the core promise of `WorkbenchState`.

7. **Smoke loops silently omit tabs.** t36 lists 3 Explore tabs (Simulate missing), t39 lists 5 Model tabs (Predict missing). A tab added without touching this list is untested by default — the loops should be generated from the app's own nav, not hand-listed.

---

## 2. Test layers

Four layers, cheapest first. Only layer 3 is new work of any size.

| Layer | Tool | Answers |
|---|---|---|
| 0. Unit | existing `tests/test_*` | Is the math right? |
| 1. Smoke | generated tab sweep | Does every tab open without a Shiny error? |
| 2. Functional E2E | Playwright | Does the happy path produce output *and* the right code snippet? |
| 3. Visual evidence | capture harness (§3) | Does the chart actually look right? |
| 4. Robustness | edge datasets | Does it fail gracefully instead of crashing? |

---

## 3. Chart capture harness (the visual-audit idea)

The design you proposed works, and it fits this codebase better than pixel-diffing would — seaborn/matplotlib output shifts slightly across versions and DPI, so byte-comparison against a golden PNG would be permanently flaky. Capturing figures *with stated expectations* and auditing them later — by eye or by an agent — sidesteps that entirely.

### Mechanism

A pytest fixture `chart_case` that a test calls after triggering a plot:

```python
chart_case(
    page,
    module="distribute",
    case_id="dist-02-boxplot-grouped",
    dataset="tips",
    inputs={"col": "total_bill", "chart_type": "boxplot", "group_by": "day"},
    expect=[
        "Four side-by-side boxplots, one per day (Thur/Fri/Sat/Sun)",
        "Y axis labelled 'total_bill', ranging roughly 3-51",
        "Sat and Sun boxes sit visibly higher than Thur and Fri",
        "Outlier points visible above the upper whiskers",
    ],
)
```

It captures via `locator.screenshot()` on the rendered `<img>` (works whether Shiny serves a data URI or a session URL) and writes:

```
tests/artifacts/<run-timestamp>/
  manifest.json          # every case, machine-readable
  index.html             # scrollable contact sheet, image + expectations side by side
  charts/
    distribute/
      dist-02-boxplot-grouped.png
      dist-02-boxplot-grouped.json   # inputs, expectations, emitted code snippet
```

Each `.json` sidecar carries the inputs, the `expect` list, and the code snippet pulled from the Show Code panel — so the auditor sees what was asked for, what was claimed, and what code allegedly produced it, next to the picture.

### Auditing pass

`manifest.json` is the agent's work list: for each case, open the PNG, read `expect`, return pass/fail plus what it actually saw. Failures come back as a list of case ids, which map straight back to the test that produced them. This runs as a separate step, not inside pytest — the test run stays fast and deterministic, and the visual audit happens on demand (pre-release, after a seaborn bump, after touching plotting code).

Writing good `expect` lines is the real work here, and it has a side benefit: stating what a chart *should* show forces the same scrutiny that catches plotting bugs in the first place.

---

## 4. Scenario matrix

Target ≈ 93 cases, excluding the deferred Homework and AI modules. The existing 40 tests are largely reusable; the count below is the end state, not net-new.

### Data (18)
- Load: bundled ×3, CSV upload, Excel upload, malformed CSV, empty file
- Profile: numeric summary, categorical summary, all-missing column, bool column (the Python 3.14 / numpy quantile gotcha)
- View: filter, sort, paginate, decimals control
- Transform: rename, retype, compute, filter, fill-missing, sample, drop columns, invalid formula
- Combine: inner/left/outer merge, concat, no-common-key error
- Export: CSV round-trip, Excel round-trip

### Explore (14)
- Summarize: single group, multi group, percent-of-total, no group var
- Pivot: rows only, rows+cols, index==values (regression guard for 5fc112c), optional column var omitted (guard for b1001fb)
- Crosstab: chi-squared, expected counts, sparse cells warning
- Simulate: each distribution, CLT demo, LLN demo, goodness-of-fit — **all new**

### Visualize (22) — all chart-capture cases
- Distribute: histogram, +KDE, bins extremes, boxplot, violin, bar, bar+percentages, horizontal, group-by, facet col, facet row, facet both, categorical-into-numeric-chart error path
- Relate: scatter, +trendline, +color, +size, log axis
- Compare: grouped box, grouped bar, violin
- Correlate: pearson matrix, spearman matrix, single-column degenerate case
- Timeline: basic line, multi-series, non-datetime x error — **all new**

### Analyze (12)
- Means: independent t, paired t, one-way ANOVA, unequal variance, n=1 group error
- Proportions: one-sample z, two-sample z, chi-squared independence
- Correlation: pearson, spearman, constant column, non-numeric error

### Model (18)
- Regression: linear, multiple predictors, categorical predictor, diagnostics plot
- Classify: logistic, kNN, SVM, tree, random forest, single-class target error
- Cluster: k-means, hierarchical, elbow plot
- Reduce: PCA, scree plot, loadings
- Evaluate: saved-model dropdown, metrics, confusion matrix — **new**
- Predict: predict from artifact, schema mismatch error — **new**

### Report (9) — all new
- Report builder: HTML export, Python script export, Jupyter export
- Notebook: cell accumulation, export round-trip
- Procedure: record steps, delete, toggle, reorder, inline comment, JSON round-trip

*(Homework and AI modules are deferred — see §0.)*

### Cross-cutting (8)
- Generated tab sweep (replaces hand-listed t35–t39)
- Transform in Data → visible in Explore (state propagation)
- Dataset switch mid-session clears stale selections
- Decimals control on all 10 modules that expose it
- Show Code non-empty and syntactically valid (`ast.parse`) on every module

---

## 5. Phasing

| Phase | Work | Why in this order |
|---|---|---|
| A | Show Code assertions across the 20 covered modules; generated tab sweep | Highest value per line; needs no new infrastructure |
| B | Chart capture harness + retrofit the 4 existing visualize tests | Unblocks all visual work |
| C | The 9 uncovered modules, functional happy paths | Closes the coverage hole |
| D | Chart-type and option variants, with expectations | Bulk of the visual matrix |
| E | Error paths and edge datasets | Guards the bug class you keep hitting |
| F | Visual audit pass over `manifest.json` | Once there is something to audit |

---

## 6. Remaining questions

1. **Edge-case fixtures.** Build them as generated frames inside `conftest.py`, or ship them as small CSVs under `tests/fixtures/`? Generated keeps the repo clean; files make failures easier to reproduce by hand.
2. **Capture run cadence.** Confirm the release-tag recommendation in §0, or run capture nightly instead.
