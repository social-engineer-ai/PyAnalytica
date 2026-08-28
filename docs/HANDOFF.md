# Where we left off — 2026-08-25, after 0.6.4

Read this first when picking the project back up. It says what changed, what is
verified, what is still unknown, and what to do next.

## State

**v0.6.4 is live on PyPI** and `docs/INSTALL.md` pins it.
`main` and PyPI agree at commit `281a7b3`.

- 829 unit tests, 65 browser tests
- Verified from a clean venv installing from PyPI: version, every fix present,
  and **zero warnings on startup**

## What this session did

Everything below was **reported by a person using the app**, not found by a
test. That pattern is the point, and it is why the test plan changed shape.

### 1. Models were never saved (reported by Ashish)

`mod_regression.py` and `mod_classify.py` only saved when the "Save Model As"
box had been typed into — and it is empty by default. Running a model saved
nothing, said nothing, and **Model > Evaluate and Model > Predict stayed
permanently empty with no explanation**.

Now always saves, named after type and target when blank
(`linear_regression_Fare`), and the confirmation says where to find it.
First automated coverage Evaluate has ever had.

### 2. Three UserWarnings on every start (reported by Ashish)

The Practice module passed `qid=qid` into per-question callbacks that are
already closed over a per-question factory; Shiny warns about a parameter it
cannot supply. **Unlike the DeprecationWarnings of the 0.6.3 episode,
`UserWarning` is shown by default** — these were the ones actually reaching
students, and 0.6.3's filter would not have caught them.

### 3. Cluster and Reduce "not working" (reported by Garv)

Both needed two or more features and enforced it with `req()`, which aborts
silently. Pick one variable, press Run: nothing at all. The Features control is
a plain multi-select, so clicking a second variable **deselects the first**
unless you hold Ctrl — and nothing said so. That makes it the *likely first
experience*, not an edge case.

Also found while probing: a refused run left the **previous** PCA on screen,
which reads as the answer to what was just asked.

### 4. The sweep behind it

Garv's report was one bug in three places, so every `req()` in the UI was
classified by what triggers its enclosing function:

| | Count | Verdict |
|---|---|---|
| Inside `@reactive.event` handlers | **48** across 21 modules | Silent refusals |
| Render guards | 53 | Correct — leave alone |

All 48 now use `require(condition, message)`. Driven against an app with an
empty workbench: **19/19 Run buttons speak, 0 silent** (was 9/19).

`tests/test_ui/test_no_silent_refusals.py` makes it unreintroducible — an AST
rule over all 30 modules, plus a check that the messages are real sentences.

### 5. First click on a Model tab leaked pandas internals

Target and Features come from the same column list, so the first column starts
selected in both. Opening Model > Regression and pressing Run gave
`Expected unique column names, got: 'Survived' 2 times`. Now the target is
dropped from the features with a note, or explained when it was the only one.

### 6. The launcher window

Now titles itself and prints a banner saying it *is* the app, can be minimised,
must not be closed. `docs/INSTALL.md` has a new "About the black window"
section and documents starting minimised via the Desktop shortcut's
**Run: Minimised** property.

Deliberately **not** hidden with `pythonw` or a `.vbs` wrapper: an invisible
server is one a student cannot stop, and reclaiming the port would mean Task
Manager.

### 7. New instructor test plan

`docs/TEST_PLAN_INSTRUCTOR.md` — weighted towards what automation is
structurally blind to, rather than repeating what CI runs on every push:

1. **Whether the numbers are right.** A test asserting "R² appears" passes on a
   wrong R². Part 2 computes 17 statistics independently in statsmodels/scipy.
2. **What the console prints.** Both of this week's bugs were visible there and
   no assertion watches it.
3. **Screens nothing has ever opened.**

Severity ladder is **S1 Wrong → S2 Lost → S3 Crash**, deliberately not
crash-first: a crash announces itself, a wrong coefficient does not.

## Still unknown — no automated and no human coverage

- **Report > Procedure, Notebook, Report Builder**
- **Explore > Simulate**
- **AI Assistant**
- Model > Predict has only minimal coverage

This is the largest remaining risk. It is Session C and part of B in
`docs/TESTER_ROUND2.md`.

## Next actions, roughly in order

1. **Ashish runs Part 1** of `docs/TEST_PLAN_INSTRUCTOR.md` against the real
   PyPI 0.6.4 (`pip install --upgrade pyanalytica`), not the dev tree. 50 min.
   Part 2 (are the numbers right?) is the highest-value thing nobody has done.
2. **Rebuild the tester pack** — `scripts/build_tester_pack.py`. The zip under
   `dist/` still says 0.6.3. Then send Garv Round 2, aimed at Report and
   Simulate.
3. **Wire the tutor client into the app.** `ai/_llm.py` still calls Anthropic
   directly with a stale model and no system prompt, and there is no UI to
   enter a token. The proxy itself works. See `docs/DECISIONS.md`.
4. AnyWare remains unverified: session persistence, clipboard, file transfer.
5. The Canvas filename parser has never been run against a real export.

## Two rules that were learned the hard way

1. **No release for a problem no user has experienced.** 0.6.3 shipped a
   warning filter that was a no-op — warnings visible under pytest were
   mistaken for warnings students see. Decision 12 in `docs/DECISIONS.md`.
2. **Verify against the artefact, not the source tree.** During 0.6.4
   verification a substring check of module source produced a false failure by
   matching a *comment*. Signature inspection and an actual startup are
   evidence; grepping source text is not.

## Two operational gotchas

- **After a PyPI upload, pip lies for a few minutes.** It reported "Could not
  find a version that satisfies pyanalytica==0.6.4" from a cached index
  response while the simple index already listed it. `--no-cache-dir` proves
  it is live. Do not mistake this for a failed upload.
- **Probe with a clean `HOME`.** A probe that launches the app inherits the
  autosaved session and finds a dataset already loaded — which silently
  destroys any "nothing is loaded yet" premise. This invalidated a first run
  of the refusal probe before it was caught.
