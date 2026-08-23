"""Instructor command line for the tutor server.

    pyanalytica-tutor init                      write a starter course pack and a secret
    pyanalytica-tutor serve --pack course.yaml  run the proxy
    pyanalytica-tutor issue --roster ids.csv    mint one token per student
    pyanalytica-tutor usage --pack course.yaml  who has used how much
    pyanalytica-tutor revoke STUDENT_ID         withdraw one student's access
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

SECRET_FILE = "tutor-secret.txt"


def _read_secret(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"error: no signing secret at {p}. Run 'pyanalytica-tutor init' first, "
            f"or point --secret-file at your existing one."
        )
    return p.read_text(encoding="utf-8").strip()


# Column names that mean "this is the student identifier". Used to find the
# right column in a CSV and to recognise -- and drop -- a header line in a
# plain one-id-per-line file. Missing the second case issued a token to a
# student called "student_id".
ROSTER_HEADERS = (
    "student_id", "id", "netid", "username", "login", "student", "user_id", "sis_id",
)


def read_roster(path: str | Path) -> list[str]:
    """Read student ids from a CSV export or a plain list, one per line."""
    text = Path(path).read_text(encoding="utf-8-sig")

    if "," in text or "\t" in text:
        import io as _io

        delimiter = "\t" if "\t" in text.splitlines()[0] else ","
        table = [
            row for row in csv.reader(_io.StringIO(text), delimiter=delimiter)
            if row and any(cell.strip() for cell in row)
        ]
        if not table:
            return []
        header = [cell.strip().lower() for cell in table[0]]
        column = next(
            (header.index(name) for name in ROSTER_HEADERS if name in header), 0
        )
        start = 1 if any(name in ROSTER_HEADERS for name in header) else 0
        return [
            row[column].strip()
            for row in table[start:]
            if len(row) > column and row[column].strip()
        ]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].lower() in ROSTER_HEADERS:
        lines = lines[1:]
    return lines


def _init(args: argparse.Namespace) -> int:
    from pyanalytica.tutor.pack import example_pack_yaml
    from pyanalytica.tutor.tokens import new_secret

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pack_path = out / "course-pack.yaml"
    secret_path = out / SECRET_FILE

    if pack_path.exists() and not args.force:
        print(f"error: {pack_path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    if secret_path.exists() and not args.force:
        print(
            f"error: {secret_path} already exists. Overwriting it would invalidate "
            f"every token you have issued (use --force if that is what you want).",
            file=sys.stderr,
        )
        return 1

    pack_path.write_text(example_pack_yaml(args.course_id), encoding="utf-8")
    secret_path.write_text(new_secret() + "\n", encoding="utf-8")
    try:
        os.chmod(secret_path, 0o600)
    except OSError:
        pass  # Windows; the warning below still applies

    print(f"Course pack : {pack_path}")
    print(f"Secret      : {secret_path}")
    print()
    print("Edit the course pack to change how students are guided.")
    print("Keep the secret file private and out of version control — anyone holding")
    print("it can mint tokens that spend your API budget. Replacing it invalidates")
    print("every token already issued.")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from pyanalytica.tutor.pack import CoursePackError, load_pack
    from pyanalytica.tutor.server import TutorServerError, api_key_from_environment, serve

    try:
        pack = load_pack(args.pack)
    except (CoursePackError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    api_key = args.api_key or api_key_from_environment()
    if not api_key:
        print(
            "error: no API key. Set ANTHROPIC_API_KEY in this server's environment.\n"
            "       Do not put the key in the course pack — that file is read at\n"
            "       startup and is easy to copy somewhere it should not be.",
            file=sys.stderr,
        )
        return 1

    try:
        serve(
            pack,
            _read_secret(args.secret_file),
            api_key,
            host=args.host,
            port=args.port,
            usage_path=args.usage_db,
        )
    except TutorServerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _issue(args: argparse.Namespace) -> int:
    from pyanalytica.tutor.pack import CoursePackError, load_pack
    from pyanalytica.tutor.tokens import issue_for_roster

    try:
        pack = load_pack(args.pack)
    except (CoursePackError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    roster = Path(args.roster)
    if not roster.exists():
        print(f"error: roster not found: {roster}", file=sys.stderr)
        return 1

    try:
        ids = read_roster(roster)
    except OSError as exc:
        print(f"error: cannot read roster: {exc}", file=sys.stderr)
        return 1

    if not ids:
        print(f"error: no student ids found in {roster}", file=sys.stderr)
        return 1

    tokens = issue_for_roster(
        _read_secret(args.secret_file), pack.course_id, ids, valid_days=args.valid_days
    )

    out = Path(args.out)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["student_id", "course_id", "server", "token"])
        for student_id, token in tokens.items():
            writer.writerow([student_id, pack.course_id, args.server or "", token])

    print(f"Issued {len(tokens)} token(s) for {pack.course_id} -> {out}")
    print()
    print("Give each student ONLY their own row. A token is not a password to")
    print("share: usage is counted against it, and it can be revoked on its own.")
    if not args.server:
        print()
        print("Tip: pass --server https://your-host so each row tells the student")
        print("     where to point the app.")
    return 0


def _usage(args: argparse.Namespace) -> int:
    from pyanalytica.tutor.pack import CoursePackError, load_pack
    from pyanalytica.tutor.usage import UsageStore, estimate_cost

    try:
        pack = load_pack(args.pack)
    except (CoursePackError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    store = UsageStore(args.usage_db)
    totals = store.course_totals(pack.course_id)

    if not totals["calls"]:
        print(f"No usage recorded yet for {pack.course_id}.")
        return 0

    cost = estimate_cost(totals["input_tok"], totals["output_tok"], pack.model)
    print(f"{pack.course_id} — {totals['calls']} questions from {totals['students']} students")
    print(f"  tokens : {totals['input_tok']:,} in / {totals['output_tok']:,} out")
    print(f"  cost   : about ${cost:.2f} at list price for {pack.model}")
    print(f"  budget : {totals['calls']}/{pack.limits.per_course_per_term} course-wide")
    print()

    rows = store.per_student(pack.course_id)
    revoked = set(store.revoked_students(pack.course_id))
    print(f"  {'student':<24} {'asked':>6} {'cost':>8}  last seen")
    for row in rows[: args.top]:
        row_cost = estimate_cost(row["input_tok"], row["output_tok"], pack.model)
        flag = "  (revoked)" if row["student_id"] in revoked else ""
        print(f"  {row['student_id']:<24} {row['calls']:>6} {row_cost:>7.2f}$  "
              f"{row['last_seen'][:16]}{flag}")
    if len(rows) > args.top:
        print(f"  ... and {len(rows) - args.top} more (use --top)")
    return 0


def _revoke(args: argparse.Namespace) -> int:
    from pyanalytica.tutor.pack import CoursePackError, load_pack
    from pyanalytica.tutor.usage import UsageStore

    try:
        pack = load_pack(args.pack)
    except (CoursePackError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    store = UsageStore(args.usage_db)
    if args.restore:
        store.restore(pack.course_id, args.student_id)
        print(f"Restored access for {args.student_id} on {pack.course_id}.")
    else:
        store.revoke(pack.course_id, args.student_id)
        print(f"Revoked access for {args.student_id} on {pack.course_id}.")
        print("Their existing token still verifies but every request is now refused.")
    return 0


def _redteam(args: argparse.Namespace) -> int:
    from pyanalytica.tutor.pack import CoursePackError, load_pack
    from pyanalytica.tutor.redteam import ATTACKS, evaluate, summarise
    from pyanalytica.tutor.server import api_key_from_environment, call_model

    try:
        pack = load_pack(args.pack)
    except (CoursePackError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    attacks = [a for a in ATTACKS if not args.only or args.only in a.id or args.only == a.category]
    if not attacks:
        print(f"error: no attacks match {args.only!r}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"{len(attacks)} probe(s) against {pack.course_id} (nothing sent):\n")
        for attack in attacks:
            print(f"  [{attack.category}] {attack.id}")
            print(f"      {' '.join(attack.prompt.split())[:100]}")
            if attack.note:
                print(f"      note: {attack.note}")
        print("\nDrop --dry-run to run them against the model.")
        return 0

    api_key = args.api_key or api_key_from_environment()
    if not api_key:
        print("error: set ANTHROPIC_API_KEY to run the probes.", file=sys.stderr)
        return 1

    print(f"Probing {pack.course_id} with {pack.model} — {len(attacks)} attacks, "
          f"{args.repeat}x each.\n", flush=True)

    findings = []
    for attack in attacks:
        # Repetition is itself an attack: refusals must not soften on asking again.
        history: list[dict] = []
        for round_no in range(args.repeat):
            try:
                reply, _, _ = call_model(pack, api_key, question=attack.prompt,
                                         history=history)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  {attack.id}: could not run ({type(exc).__name__}: {exc})",
                      file=sys.stderr)
                break
            finding = evaluate(attack, reply, pack.system_prompt)
            findings.append(finding)
            mark = "ok  " if finding.held else "FAIL"
            suffix = f" (ask {round_no + 1})" if args.repeat > 1 else ""
            print(f"  {mark} {attack.id}{suffix}", flush=True)
            if not finding.held:
                for failure in finding.failures:
                    print(f"         {failure}", flush=True)
            history = history + [
                {"role": "user", "content": attack.prompt},
                {"role": "assistant", "content": reply},
            ]

    print()
    print(summarise(findings))
    return 0 if all(f.held for f in findings) else 2


def build_parser() -> argparse.ArgumentParser:
    from pyanalytica import __version__

    parser = argparse.ArgumentParser(
        prog="pyanalytica-tutor",
        description="Run a course-scoped AI tutor without putting your API key on "
                    "students' machines.",
    )
    parser.add_argument("--version", action="version", version=f"pyanalytica {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write a starter course pack and a signing secret")
    p.add_argument("--course-id", default="MY-COURSE-101")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.set_defaults(func=_init)

    p = sub.add_parser("serve", help="run the tutor proxy")
    p.add_argument("--pack", default="course-pack.yaml")
    p.add_argument("--secret-file", default=SECRET_FILE)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--usage-db", default="tutor-usage.db")
    p.add_argument("--api-key", default=None, help="overrides ANTHROPIC_API_KEY")
    p.set_defaults(func=_serve)

    p = sub.add_parser("issue", help="mint one access token per student")
    p.add_argument("--pack", default="course-pack.yaml")
    p.add_argument("--secret-file", default=SECRET_FILE)
    p.add_argument("--roster", required=True, help="CSV or one id per line")
    p.add_argument("--out", default="tokens.csv")
    p.add_argument("--server", default=None, help="the address students should use")
    p.add_argument("--valid-days", type=int, default=180)
    p.set_defaults(func=_issue)

    p = sub.add_parser("usage", help="show who has used how much")
    p.add_argument("--pack", default="course-pack.yaml")
    p.add_argument("--usage-db", default="tutor-usage.db")
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(func=_usage)

    p = sub.add_parser(
        "redteam", help="attack your own course pack before students do"
    )
    p.add_argument("--pack", default="course-pack.yaml")
    p.add_argument("--only", default=None, help="run one attack id or category")
    p.add_argument("--repeat", type=int, default=1,
                   help="ask each probe N times in one conversation (persistence test)")
    p.add_argument("--dry-run", action="store_true", help="list the probes, send nothing")
    p.add_argument("--api-key", default=None)
    p.set_defaults(func=_redteam)

    p = sub.add_parser("revoke", help="withdraw one student's access")
    p.add_argument("student_id")
    p.add_argument("--pack", default="course-pack.yaml")
    p.add_argument("--usage-db", default="tutor-usage.db")
    p.add_argument("--restore", action="store_true", help="undo a revocation")
    p.set_defaults(func=_revoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
