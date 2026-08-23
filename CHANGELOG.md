# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.1] - 2026-08-23

Everything here came from one human tester working through the first five
sessions of the test worksheet on `titanic` and his own CSVs. The automated
suite had passed all of it.

### Fixed

- **Pivoting by a numeric column crashed.** The pivot's column labels become
  integers and the data grid failed with "'DataFrame' object has no attribute
  'dtype'", which reached the student as raw text where a table should be. Not
  dataset-specific: pivoting tips by `sex` worked and by `size` did not.
- **Dropdowns reset or went stale.** `update_select` without an explicit
  `selected` snaps the input to the first choice, and between the list changing
  and the reset landing the old value is still live — merging two datasets
  raised `['carat'] not in index`, naming a column from a dataset chosen
  earlier. 49 call sites across 19 modules now preserve the current value.
- **Loading a dataset did not switch to it**, so students loaded a file and
  then had to find the dropdown themselves.
- **Binary and small-integer columns could not be grouped by.** `Survived`
  (0/1) and `Pclass` (1-3) classify as numeric, so neither appeared in any
  "group by" or "cross-tabulate by" list — on the course's main teaching
  dataset. They are offered now; their classification is unchanged, so they
  remain valid regression targets.
- **Correlate with one column selected did nothing at all** — no chart, no
  message. It now explains that a correlation needs two columns.
- `examples/hw1_tips` and its grading demo asked for the mean of `total_bill`
  with the value from the real seaborn dataset rather than the bundled one.

### Changed

- Browser tests assert what is rendered rather than that elements exist, and a
  new suite varies the *data* — numeric versus string pivot labels, binary
  outcomes, degenerate selections — instead of only the module under test.
  Unit 640 → 737; browser 36 → 47.

## [0.6.0] - 2026-08-23

### Changed

- **The app no longer grades homework.** An assignment in a student's hands now
  contains no answer material at all, because an answer their computer can
  check is an answer they can extract: multiple-choice answers fell by hashing
  the options the file already listed, and numeric answers by sweeping values
  at the stated tolerance, both in under a second. Marking happens on the
  instructor's machine against a key that never leaves it.
- **Submissions carry evidence, not verdicts.** No `correct`, no
  `points_earned`, no `auto_total` — a score computed on the student's machine
  was a claim, not a fact. The work log is now taken from the procedure
  recorder, so it carries the pandas/sklearn code behind each step rather than
  only a description, and recording starts when an assignment is opened.
- **Submissions export as HTML** with the same JSON embedded: readable in an
  LMS, parseable by a grading script, one file to upload. Format version
  `pyanalytica.submission/2`; the 0.5.x shape is not readable by this release.
- The `graded:` question flag is gone. It marked which questions had to hide
  their answers, and now they all do.

### Added

- **Practice tab** — self-check drills with instant feedback, separate from
  assignments. Drills carry no marks, so their answers are in plaintext;
  pretending to hide a derivable answer would be theatre. Two bundled drills.
- **`pyanalytica-hw`**, the instructor command line: `build` turns a master
  file into a student assignment plus an answer key, `grade` marks a folder of
  LMS downloads into a gradebook CSV, `inspect` prints one submission.
  Identity comes from the LMS filename, not from the name a student typed.
- `docs/INSTRUCTOR.md` — authoring and marking, end to end.

### Fixed

- An unanswered checkpoint scored a point: submissions now list every question,
  so testing for the presence of an id was always true. Blank counts as
  unanswered.
- A free-response answer containing `</script>` closed the embedded JSON block
  early in an exported submission.
- `examples/hw1_tips` asked for the mean of `total_bill` with 19.79, the value
  from the real seaborn dataset. The bundled data is synthetic and its mean is
  25.29, so every student would have been marked wrong.

## [0.5.1] - 2026-08-22

### Fixed

- `pyanalytica` now accepts `--port`, `--host`, `--no-browser`, and
  `--version`. It previously called `run_app()` with no argument parsing, so
  every option was silently ignored — `--port 8001` still bound 8000, and a
  mistyped option did nothing rather than reporting the mistake.
- The app opens a browser on start instead of leaving students to find the
  address themselves.
- If port 8000 is busy (usually the app already running in another window),
  a free port is chosen and reported, rather than crashing with
  "address already in use". An explicitly requested port still fails loudly.

## [0.5.0] - 2026-08-22

> Note: releases 0.2.0 through 0.4.6 were not recorded here.

### Added

- **Homework authoring** (`homework/authoring.py`) — a master file holds
  plaintext answers and stays on the author's machine; `build()` derives the
  student copy and the instructor answer key from it. Questions marked
  `graded: true` ship with no answer material at all, so their answers cannot
  be recovered from the student copy. Questions left ungraded keep a hash for
  immediate in-app feedback.
- **Authoritative re-grading** (`homework/regrade.py`) — scores collected
  submissions from their raw answers using the instructor key, ignoring the
  scores the student's app wrote into the file. Reports (but does not act on)
  disagreements between claimed and recomputed totals.
- `_assert_code()` coverage in the end-to-end suite: every tested module must
  emit a Show Code snippet that parses as valid Python and contains the
  expected call.
- Browser tests now run in CI.

### Fixed

- **Active dataset no longer resets when the selector refreshes.** Any state
  change — applying a transform, loading a second file — silently switched the
  active dataset to whichever name sorted first alphabetically.
- End-to-end suite repaired: 19 of 40 tests were failing and had never run in
  CI. Test runs are now isolated from `~/.pyanalytica`, so they neither inherit
  state from previous runs nor overwrite the user's own saved session.
- Numeric answers entered as text (`"19.790"`) are coerced before comparison
  during re-grading, instead of being marked wrong.
- Profile tests no longer read a real `ANTHROPIC_API_KEY` from the
  environment, which decided their outcome and printed the key on failure.

## [0.1.0] - 2025-02-10

### Added

#### Core & Data (Phase 1)
- Column type classification (NUMERIC, CATEGORICAL, DATETIME, ID, TEXT)
- `CodeSnippet` system — every function returns equivalent pandas/sklearn code
- Dataset loading with `load_bundled()` and `load_dataset()`
- DataFrame profiling with missing-value and type summaries
- Data transformation: rename, retype, filter, compute, combine

#### Explore & Visualize (Phase 2)
- Group summarize, pivot tables, crosstabs, frequency tables
- Histogram, boxplot, bar chart, scatter plot, line chart, heatmap
- Correlation matrix visualization

#### Analyze & Model (Phase 3)
- Compare group means and proportions with hypothesis tests
- Linear regression and logistic classification
- Model evaluation with metrics, residual plots, confusion matrices

#### Homework & Report (Phase 4)
- YAML-based homework schema with hash-checked grading
- Report export: HTML, Python script, Jupyter notebook

#### AI Agent (Phase 5)
- Rule-based and optional LLM interpretation of results
- AI-powered analysis suggestions
- Challenge questions for student engagement
- Natural-language data queries

#### Enhanced Model & Procedure Builder (Phase 6)
- Model store for saving and reusing trained models
- Prediction from saved model artifacts
- Procedure recorder: capture, replay, and export analysis workflows
- Procedure exports: JSON, Python script, Jupyter notebook
- Decimals control moved to per-module inline widget
- User profile system (`~/.pyanalytica/profile.yaml`)

#### Package & Publish (Phase 7)
- Single-source version via hatchling dynamic versioning
- `python -m pyanalytica` entry point
- Explicit sdist includes for reliable CSV bundling
- GitHub Actions CI (Python 3.10-3.13)
- Accurate README with correct API examples
- This changelog
