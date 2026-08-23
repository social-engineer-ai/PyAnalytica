"""Build the zip handed to a human tester.

Assembles the pack from the canonical files in docs/ and examples/ so the two
cannot drift apart, rewrites the few paths that differ inside the pack, and
refuses to include answer keys.

Usage:
    python scripts/build_tester_pack.py
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK_NAME = "PyAnalytica-Tester-Pack"
OUT_DIR = ROOT / "dist"
STAGE = OUT_DIR / PACK_NAME
ZIP_PATH = OUT_DIR / f"{PACK_NAME}.zip"

# Anything matching these must never reach a tester.
FORBIDDEN = ("*.key.yaml", "*.master.yaml", "*.env", "*.pkl")

START_HERE = """# Start here

Thanks for testing PyAnalytica. Everything you need is in this folder.

## What's in here

| File | What it's for |
|---|---|
| **INSTALL.md** | Install instructions. Do this first. |
| **TEST-PASS.md** | The actual testing. Ten sessions, about 4 hours. |
| **BUG-LOG.csv** | Open in Excel. Record every problem here. |
| **data/** | Files you'll need part way through. Leave them where they are. |

## What to do

1. Read **INSTALL.md** and install the software. Allow about 30 minutes.
2. Open **BUG-LOG.csv** in Excel and leave it open all session.
3. Work through **TEST-PASS.md** from Session 1. Don't skip ahead.
4. Take a break between sessions — it's a long list and tired testers miss things.

## The three rules

**1. Don't fix things yourself.** If a step doesn't work, that's the point. Write
down what happened and move on. Fixing it quietly means the problem ships to
students.

**2. Write down confusion, not just breakage.** If you can't tell what a button
does, or a word means nothing to you, that's a real finding. Students will hit
the same wall. "It worked but I had no idea what it meant" is useful.

**3. Screenshot anything visual.** Name them `bug01.png`, `bug02.png` to match
the numbers in your bug log. Put them in this folder.

## When you're done

Send back:

- **BUG-LOG.csv**
- **TEST-PASS.md** with your answers filled in (or a separate document with them)
- Any screenshots

## If you get properly stuck

If something blocks you for more than 15 minutes, note where you got to, skip
that session, and carry on with the next one. A test pass with one skipped
session is far more useful than one abandoned at the halfway point.
"""

BUG_LOG_HEADER = (
    "Number,Session,What I did,What I expected,What happened,"
    "How bad (Crash/Wrong/Ugly/Confusing),Screenshot file\n"
    "1,Example - delete this row,Clicked Plot with no column chosen,"
    "An error message,The page went blank,Crash,bug01.png\n"
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "data").mkdir(parents=True)

    # --- documents, renamed for a reader who has never seen the repo ---
    install = (ROOT / "docs" / "INSTALL.md").read_text(encoding="utf-8")
    (STAGE / "INSTALL.md").write_text(install, encoding="utf-8")

    test_pass = (ROOT / "docs" / "TESTER_FULL.md").read_text(encoding="utf-8")
    # Inside the pack the data files sit in data/, not examples/tester_files/.
    test_pass = test_pass.replace(
        "They're in the `examples/tester_files` folder: **sales.csv**, "
        "**regions.csv**, **messy.csv**. Save them somewhere you can find, "
        "like your Desktop.",
        "They're in the **data** folder next to this one: **sales.csv**, "
        "**regions.csv**, **messy.csv**. You'll also need **hw1_tips.yaml** "
        "from the same place in Session 8.",
    )
    test_pass = test_pass.replace(
        "Ask for the file `hw1_tips.yaml` if you don't have it.",
        "Use `hw1_tips.yaml` from the **data** folder.",
    )
    (STAGE / "TEST-PASS.md").write_text(test_pass, encoding="utf-8")

    (STAGE / "START-HERE.md").write_text(START_HERE, encoding="utf-8")
    (STAGE / "BUG-LOG.csv").write_text(BUG_LOG_HEADER, encoding="utf-8")

    # --- data files ---
    for name in ("sales.csv", "regions.csv", "messy.csv"):
        shutil.copy2(ROOT / "examples" / "tester_files" / name, STAGE / "data" / name)
    shutil.copy2(ROOT / "examples" / "hw1_tips.yaml", STAGE / "data" / "hw1_tips.yaml")

    # --- safety net: never ship answer material ---
    for pattern in FORBIDDEN:
        found = list(STAGE.rglob(pattern))
        if found:
            fail(f"refusing to package {pattern}: {[str(p) for p in found]}")

    # Parse the YAML rather than pattern-matching it. The first version of this
    # check split on "\n  - id:", which never matches because yaml.safe_dump
    # writes list items at the parent's indent -- so the guard silently passed
    # everything, including a deliberately poisoned file. Checked by injecting
    # a hash onto a graded question and confirming the build now refuses.
    import yaml

    hw = yaml.safe_load((STAGE / "data" / "hw1_tips.yaml").read_text(encoding="utf-8"))
    for question in hw.get("questions", []):
        qid = question.get("id", "?")
        for secret in ("answer", "solution", "answer_key"):
            if secret in question:
                fail(f"question {qid} carries a plaintext '{secret}' -- that is a master file")
        if question.get("graded") and question.get("answer_hash"):
            fail(
                f"question {qid} is graded but ships an answer_hash, "
                f"which is recoverable by sweeping candidate answers"
            )

    # --- zip it ---
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(STAGE.parent))

    size_kb = ZIP_PATH.stat().st_size / 1024
    print(f"Built {ZIP_PATH.relative_to(ROOT)}  ({size_kb:.0f} KB)")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for info in zf.infolist():
            print(f"  {info.filename:<48} {info.file_size / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()
