# Design decisions

Why things are the way they are. The code shows *what*; this records *why*, and
what was rejected. Newest last.

---

## 1. The E2E suite is the primary defence, and it must actually run

**Decision.** Browser tests run in CI on every push and PR, on one Python
version. Unit tests keep the version matrix.

**How we got there.** `tests/test_e2e.py` existed with 40 tests, but `ci.yml`
carried `--ignore=tests/test_e2e.py`, so it had never run automatically.
Running it found **19 of 40 failing** — they had been failing since they were
written.

Of the causes, one was a real user-facing bug and the rest were bugs in the
tests themselves. That ratio is the lesson: *a test suite that has never run is
mostly measuring itself.*

**The bug it found.** `dataset_selector.py` called `update_select()` without
`selected=`, which resets the input to the first choice. `dataset_names()` is
sorted alphabetically, so any state change — a transform, loading a second file
— silently moved the student's work from `tips` to `diamonds`. Every function
involved had passing unit tests. Only a browser driving the real app could see
it, because the bug lived in the reactive wiring *between* components.

**Practice that came out of it.** Establish a baseline before claiming a fix.
The first modified run showed 17 failures and there was no way to tell which
were mine, so I extracted the suite at HEAD and ran it unmodified. That is now
the habit: measure, change, compare.

---

## 2. Test runs must be isolated from the user's home directory

**Decision.** The E2E fixture points `HOME`/`USERPROFILE` at a temp directory.

**Why.** `core/session.py` autosaves the workbench to `~/.pyanalytica/sessions`
and `app.py` restores it on startup. Test runs therefore inherited whatever the
previous run left behind — results depended on run history — and, worse, **the
suite overwrote the user's own saved session.** It destroyed a real one before
this was fixed.

Same class of problem in `test_profile.py`, which read a real
`ANTHROPIC_API_KEY` from the environment: those tests passed or failed
depending on whose machine ran them, and printed the key into the failure diff.

**Rule.** A test whose result depends on ambient state is not testing anything.

---

## 3. Every emitted code snippet must parse as Python

**Decision.** `_assert_code()` opens the Show Code panel and checks the snippet
is non-empty and passes `ast.parse`, plus contains an expected call.

**Why.** Every analytics function returns `(result, CodeSnippet)`, which gives
a second, independent oracle per action: did the UI compute the right thing
*and* emit the right code. Nothing had ever asserted it — `toggle_code`
appeared zero times in the suite. A snippet a student cannot paste into a
notebook is a bug regardless of what the screen shows.

---

## 4. Distribution is public; access control belongs at the API layer

**Decision.** Ship on PyPI. Students install `pip install pyanalytica==X.Y.Z`,
pinned.

**Rejected.** Install from GitHub. It requires `git` on the machine — which
most Windows student laptops lack — and it protects nothing: the repo is public
and MIT-licensed.

**The reasoning that matters.** Part of the appeal of GitHub-only install was
limiting who could get the tool, because of the AI-key cost worry. It cannot do
that. Even a private repo gates only *download*; once a student has the code,
whatever the package can reach, they can reach. **Access control has to happen
at the API layer.** Conflating the two costs the easy install and buys nothing.

**Corollary.** Pin the version in the syllabus. Unpinned, a mid-semester
release means two students get different numbers for the same question.

---

## 5. Homework is not assessed inside the tool

**Decision.** Assignments carry no answer material. Nothing in the student's app
checks an answer. Marking happens on the instructor's machine after collection
via Canvas.

**The two facts that forced it.**

*An answer a student's computer can check is an answer they can extract.*
`hash_answer` was an unsalted 16-char SHA-256 prefix. Demonstrated against the
real `examples/hw1_tips.yaml`: multiple-choice fell by hashing the options the
file already listed, numeric by sweeping at the stated tolerance. All three
answers recovered in under a second. No hashing scheme fixes this — local
checking and local secrecy are mutually exclusive.

*A score computed on a student's machine is a claim.* The export was unsigned;
editing `auto_total` in Notepad changed the mark.

**Rejected: making the homework system more secure.** That is a fight against
Canvas, which already has enrollment-backed identity, deadlines, attempts,
gradebook integration and an appeals trail. Competing with it on assessment is
a losing position.

**What the tool uniquely has** is the thing Canvas can never have: a timestamped
record of the work, with the pandas/sklearn code each step generated. Canvas
sees `29.46` in an answer box. PyAnalytica saw them load the data, notice the
missing ages, and compute the mean. *An answer can be obtained from a classmate;
the record of arriving at it cannot.* That is both the pedagogical value and the
integrity mechanism, and it is far more useful than trying to keep answers
secret.

---

## 6. Practice is a separate feature from Homework

**Decision.** Self-check drills live in their own tab, with answers in
plaintext. Assignments carry nothing checkable.

**Why not a mode of homework.** When one feature had to serve both, each half
compromised the other: assignments carried hashes to enable feedback, which
made their answers recoverable. Split apart, each can be honest about its own
threat model.

Drills carry no marks, so their answers being visible costs nothing — and
hashing a derivable answer would be theatre. The files say so in a comment.

**Rejected.** Dropping self-check entirely. For a masters student working alone
at night, "did I compute that right?" is the most useful thing the tool can
say, and it is free of risk once decoupled from marking.

---

## 7. No AI in the homework path

**Decision.** No AI grading or AI assistance inside the homework flow. The
instructor runs AI-assisted marking themselves, after downloading from Canvas.

**Why.** It preserves a property worth more than the feature: **the tool never
sends anything anywhere.** It is a local server on `127.0.0.1`; student work
stays on the student's laptop until they upload to Canvas themselves. No
data-sharing question, no institutional review, nothing to disclose.

**Useful consequence.** The AI capability actually wanted — marking interpretive
answers — is *instructor-side*. It needs no key distribution, no proxy, no
per-student rate limiting, and no exposure to strangers who `pip install` the
package. The hardest part of the AI problem dissolves by putting it in the
right place.

**Still open.** The student-facing AI Assistant module (interpret, suggest,
challenge, query) would send data to an API. That decision has not been made.

---

## 8. Answers are computed from the bundled data, never the real dataset

**Decision.** Any answer key is verified against the data that ships with the
package. A test recomputes every bundled drill answer from the bundled data.

**Why.** The bundled datasets are *generated* with fixed seeds. Column names
match the well-known originals; the numbers do not. `hw1_tips` shipped asking
for the mean of `total_bill` with **19.79** — correct for seaborn's tips, wrong
for the synthetic data students actually have, where it is **25.29**. Every
student would have been marked wrong, and since graded questions give no
feedback, nobody would have found out until marks came back.

The failure is completely silent. Hence the test.

---

## 9. Identity comes from the LMS filename

**Decision.** Batch marking takes the student from the download filename, not
from the name inside the submission.

**Why.** The student types their own name into the app, so it is a claim. The
filename is written by Canvas from its own enrollment records. Where they
disagree, the report says so.

**Deliberate leniency.** A filename that doesn't match the expected shape is
still marked, identified by filename and flagged — losing someone's work to a
naming quirk is worse than an untidy report. Likewise an unreadable file records
its error and the batch continues, and error rows carry a **blank** score rather
than a zero, so nothing silently becomes a mark.

**Unverified.** The Canvas convention is followed from documentation and tested
against constructed examples only. **It has never been run against a real
Canvas export.** Do that before relying on it.

---

## 10. Course-neutral by construction

**Decision.** No course-specific content in the package or its tests. Course
material — assignments, keys, rosters — lives outside, in a private repo or the
LMS.

**Why.** The goal is a general Radiant-for-Python, not a BADM 576 tool. The
package is the instrument; assignments are content with a different lifetime and
a different distribution. `*.master.yaml` and `*.key.yaml` are gitignored so
answer keys cannot reach a public repository by accident, and the pack builder
refuses to ship a file containing answer material.

---

## 11. Test what a user sees, on the data they use

**Decision.** Browser tests assert rendered content, not element existence, and
parameterise over column *shapes* rather than modules. Added
`tests/test_e2e_datasets.py` alongside the main suite.

**How we got there.** A 12th-grade tester worked sessions 1-5 of the worksheet
on `titanic` and his own CSVs. In about an hour he found faults that 36 passing
browser tests had covered without noticing. Replicating each one showed the
suite was not thin on coverage -- it was thin on *assertions* and on *data*.

Three blind spots, each now an oracle:

*An error rendered as text is not a Shiny error.* Pivoting by a numeric column
crashed with `'DataFrame' object has no attribute 'dtype'`, but the module
caught it and returned the message as a string, so no `.shiny-output-error`
existed and `_assert_no_shiny_errors` passed while a stack-trace fragment sat
where a table should be. → `_assert_no_error_text`.

*Existence is not content.* `expect(x).to_be_attached()` passes on an element
that renders nothing. Correlate with one column selected showed nothing at all
and the suite was satisfied. → `_assert_output_has_content`.

*A missing option raises nothing.* `Survived` never appeared in Cross-tab
because a 0/1 integer classifies as numeric and the categorical filter dropped
it. Nothing failed; the choice simply was not offered. →
`_assert_choices_include`.

**The deeper cause was the data.** Every test ran on `tips` with one hand-picked
column pair. The pivot crash is not dataset-specific at all: pivoting tips by
`sex` works and by `size` crashes, because the second produces integer column
labels. The suite had picked the combination that works. Bugs here live in
column *shape* -- numeric vs string labels, binary integers, all-missing,
constant -- so that is what the new file varies.

**Two follow-on lessons, both learned the hard way in the same session.**

The fix for one half of a behaviour can break the other half. Stopping the
dataset selector from jumping on refresh also stopped it moving when a dataset
was loaded; fixing *that* by honouring `state.last_loaded` then dragged the user
back to it on every later change. Correct behaviour needed both halves stated
explicitly: preserve on refresh, move once per load, tracked by sequence rather
than name.

Suites that pass alone can fail together. The 16-failure regression above only
appeared when both browser files ran in one session, because the new tests
loaded datasets in a pattern the old ones never did. Run them together in CI,
not separately.

---

## 12. A warning visible under pytest is not a warning a student sees

**Decision.** Judge student-facing console output by running the code the way a
student runs it — default warning filters, plain interpreter, stderr captured —
never by what a test run prints.

**How we got there.** A student emailed on day one reporting a "setup error"
that was a successful start plus twenty-odd `ShinyDeprecationWarning` lines.
Fixing that at source (0.6.2, using shiny's renamed decorator) was correct.

Then, upgrading the development environment to match what students actually
resolve, pytest showed `MatplotlibDeprecationWarning` from inside seaborn on
every boxplot. That looked like the same problem, so 0.6.3 shipped a filter in
the launcher to suppress it.

**The filter does nothing.** Running every common operation with Python's
default filters produces no warnings at all:

| Warning | Base | Shown by default |
|---|---|---|
| `ShinyDeprecationWarning` | `RuntimeWarning`, plus shiny registers `always` for it | yes |
| `MatplotlibDeprecationWarning` | `DeprecationWarning` | no — hidden outside `__main__` |

The seaborn notices were only ever visible because **pytest enables warnings**
and because the probe used `simplefilter("always")`, which forces display. A
test artefact was mistaken for a student experience — the exact error this
project has been correcting all week, committed while correcting it.

**Kept anyway, honestly labelled.** The filter is harmless insurance for a
library that starts forcing its own category the way shiny does, or a student
running `-W always`. The changelog entry now says so instead of claiming a fix.

**The transferable rule.** Two questions, not one: *does this warning exist*,
and *does the default filter show it to anyone*. `simplefilter("always")`
answers the first and actively obscures the second.
