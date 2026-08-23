"""Instructor-side command line: build assignments, mark what comes back.

    pyanalytica-hw build hw1.master.yaml
    pyanalytica-hw grade ./downloads --key hw1.key.yaml --out grades.csv
    pyanalytica-hw inspect submission.html

Nothing here runs on a student's machine, and nothing here sends anything
anywhere -- marking happens locally against a key that never leaves this
computer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build(args: argparse.Namespace) -> int:
    from pyanalytica.homework.authoring import HomeworkBuildError, build

    try:
        student, key = build(args.master, out_dir=args.out_dir)
    except (HomeworkBuildError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Assignment for students : {student}")
    print(f"Answer key (keep private): {key}")
    print()
    print("Give students the first file. The second stays here -- it is what")
    print("'pyanalytica-hw grade' marks against.")
    return 0


def _grade(args: argparse.Namespace) -> int:
    from pyanalytica.homework.collect import grade_folder, write_gradebook_csv
    from pyanalytica.homework.regrade import load_key

    try:
        key = load_key(args.key)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: cannot read the answer key: {exc}", file=sys.stderr)
        return 1

    try:
        batch = grade_folder(args.folder, key)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not batch.graded:
        print(f"No submissions found in {args.folder}.")
        if batch.skipped:
            print(f"({len(batch.skipped)} file(s) ignored -- not .html or .json)")
        return 1

    out = write_gradebook_csv(batch, args.out)

    ok = batch.succeeded
    print(f"Marked {len(ok)} of {len(batch.graded)} submissions -> {out}")

    if ok:
        scores = [g.result.auto_total for g in ok]
        maxes = {g.result.auto_max for g in ok}
        top = max(maxes) if maxes else 0
        print(
            f"  auto-marked: mean {sum(scores) / len(scores):.1f} of {top}, "
            f"range {min(scores)}-{max(scores)}"
        )
        pending = sum(g.result.pending_review for g in ok)
        if pending:
            print(f"  {pending} points across the batch still need marking by hand")

    # Everything below needs a human. Say so plainly rather than burying it.
    if batch.failed:
        print(f"\n{len(batch.failed)} file(s) could not be read:")
        for g in batch.failed:
            print(f"  {g.file.path.name}: {g.error}")

    if batch.unrecognised:
        print(
            f"\n{len(batch.unrecognised)} file(s) did not match the expected LMS "
            f"naming, so the student was identified by filename alone:"
        )
        for g in batch.unrecognised:
            print(f"  {g.file.path.name}")

    mismatched = [
        g for g in ok
        if g.file.recognised and g.claimed_name
        and _slugify(g.claimed_name) != g.file.slug.lower()
    ]
    if mismatched:
        print(f"\n{len(mismatched)} submission(s) name someone other than the file owner:")
        for g in mismatched:
            print(f"  {g.file.path.name}: file says {g.file.slug}, content says {g.claimed_name!r}")

    warned = [g for g in ok if g.result.warnings]
    if warned:
        print(f"\n{len(warned)} submission(s) raised a warning -- see the status column.")

    if batch.skipped:
        print(f"\n{len(batch.skipped)} file(s) ignored (not .html or .json).")

    return 0


def _inspect(args: argparse.Namespace) -> int:
    from pyanalytica.homework.regrade import RegradeError, load_submission

    try:
        payload = load_submission(args.path)
    except (RegradeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Assignment : {payload.get('homework_id', '?')} "
          f"(version {payload.get('homework_version', '?')})")
    print(f"Name given : {payload.get('student_name', '?')}")
    print(f"Submitted  : {payload.get('submitted_at', '?')}")

    answers = payload.get("answers", [])
    answered = sum(1 for a in answers if str(a.get("answer", "")).strip())
    print(f"Answers    : {answered} of {len(answers)} answered")
    for a in answers:
        text = str(a.get("answer", "")).strip() or "(blank)"
        if len(text) > 70:
            text = text[:67] + "..."
        print(f"    {a.get('question_id', '?'):<6} {text}")

    work = payload.get("work", [])
    print(f"Work steps : {len(work)}")
    if args.verbose:
        for i, step in enumerate(work, start=1):
            print(f"    {i:>3}. [{step.get('action', '')}] {step.get('description', '')}")
            if step.get("code"):
                for line in str(step["code"]).splitlines():
                    print(f"         {line}")
    return 0


def _slugify(name: str) -> str:
    """Approximate the LMS filename slug for a person's name."""
    parts = [p for p in name.replace(",", " ").split() if p]
    if len(parts) < 2:
        return "".join(parts).lower()
    # Canvas writes lastname then firstname, with no separator.
    return (parts[-1] + parts[0]).lower()


def build_parser() -> argparse.ArgumentParser:
    from pyanalytica import __version__

    parser = argparse.ArgumentParser(
        prog="pyanalytica-hw",
        description="Build PyAnalytica assignments and mark what students hand in.",
    )
    parser.add_argument("--version", action="version", version=f"pyanalytica {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser(
        "build", help="turn a master file into a student assignment plus an answer key"
    )
    p_build.add_argument("master", help="the .master.yaml file holding your answers")
    p_build.add_argument("--out-dir", default=None, help="where to write (default: alongside the master)")
    p_build.set_defaults(func=_build)

    p_grade = sub.add_parser("grade", help="mark a folder of submissions downloaded from the LMS")
    p_grade.add_argument("folder", help="the unzipped download folder")
    p_grade.add_argument("--key", required=True, help="the .key.yaml produced by 'build'")
    p_grade.add_argument("--out", default="grades.csv", help="CSV to write (default: grades.csv)")
    p_grade.set_defaults(func=_grade)

    p_inspect = sub.add_parser("inspect", help="print one submission in readable form")
    p_inspect.add_argument("path", help="a submission .html or .json")
    p_inspect.add_argument("-v", "--verbose", action="store_true", help="include the code for each step")
    p_inspect.set_defaults(func=_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
