# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
