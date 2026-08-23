"""Reading a folder of submissions downloaded from an LMS.

Identity comes from the **filename**, not from the file's contents. A student
types their own name into the app, so the name inside a submission is a claim;
the filename is written by the LMS from its own enrollment records when the
bundle is exported. Where the two disagree, the filename wins and the
disagreement is reported.

Canvas names files in its bulk download like::

    lastnamefirstname_123456_7890123_myreport.html
    lastnamefirstname_late_123456_7890123_myreport.html

The parser below follows that shape but never insists on it: an unrecognised
name yields a submission identified by its filename with a flag set, because
losing a student's work to a naming quirk is worse than an untidy report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyanalytica.homework.regrade import (
    AnswerKey,
    RegradeError,
    RegradeResult,
    load_submission,
    regrade,
)

# Extensions we will try to read. Anything else in the folder is ignored.
SUBMISSION_SUFFIXES = (".html", ".htm", ".json")

# lastnamefirstname [_late] _userid _submissionid _originalname
_CANVAS = re.compile(
    r"^(?P<slug>[^_]+)_(?:(?P<late>late)_)?(?P<user_id>\d+)_(?P<submission_id>\d+)_(?P<original>.+)$"
)


@dataclass
class SubmissionFile:
    """One file found in the download folder."""

    path: Path
    slug: str = ""              # lastnamefirstname, as the LMS wrote it
    user_id: str = ""
    submission_id: str = ""
    original_name: str = ""
    late: bool = False
    recognised: bool = False    # did the filename match the expected shape?

    @property
    def identifier(self) -> str:
        """Best available identifier for this student."""
        return self.slug or self.path.stem


@dataclass
class GradedSubmission:
    """A submission file and what marking it produced."""

    file: SubmissionFile
    result: RegradeResult | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.result is not None

    @property
    def claimed_name(self) -> str:
        return self.result.claimed_name if self.result else ""


@dataclass
class Batch:
    """Everything found and marked in one folder."""

    graded: list[GradedSubmission] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)

    @property
    def succeeded(self) -> list[GradedSubmission]:
        return [g for g in self.graded if g.ok]

    @property
    def failed(self) -> list[GradedSubmission]:
        return [g for g in self.graded if not g.ok]

    @property
    def unrecognised(self) -> list[GradedSubmission]:
        return [g for g in self.graded if not g.file.recognised]


def parse_filename(path: str | Path) -> SubmissionFile:
    """Pull student identity out of an LMS download filename."""
    p = Path(path)
    match = _CANVAS.match(p.stem)
    if not match:
        return SubmissionFile(path=p)

    return SubmissionFile(
        path=p,
        slug=match.group("slug"),
        user_id=match.group("user_id"),
        submission_id=match.group("submission_id"),
        original_name=match.group("original"),
        late=bool(match.group("late")),
        recognised=True,
    )


def find_submissions(folder: str | Path) -> list[SubmissionFile]:
    """List the submission files in *folder*, sorted by student identifier."""
    d = Path(folder)
    if not d.exists():
        raise FileNotFoundError(f"Folder not found: {d}")
    if not d.is_dir():
        raise NotADirectoryError(f"Not a folder: {d}")

    files = [
        parse_filename(p)
        for p in sorted(d.iterdir())
        if p.is_file() and p.suffix.lower() in SUBMISSION_SUFFIXES
    ]
    return sorted(files, key=lambda f: f.identifier.lower())


def grade_folder(folder: str | Path, key: AnswerKey) -> Batch:
    """Mark every submission in *folder* against *key*.

    One unreadable file never stops the batch: it is recorded with its error
    and marking continues, because finding out at file 3 of 60 that the run
    died is worse than an incomplete report you can act on.
    """
    batch = Batch()
    d = Path(folder)

    for path in sorted(d.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUBMISSION_SUFFIXES:
            batch.skipped.append(path)
            continue

        info = parse_filename(path)
        try:
            payload: dict[str, Any] = load_submission(path)
            result = regrade(payload, key, source=path.name)
            batch.graded.append(GradedSubmission(file=info, result=result))
        except (RegradeError, ValueError, OSError) as exc:
            batch.graded.append(GradedSubmission(file=info, error=str(exc)))

    batch.graded.sort(key=lambda g: g.file.identifier.lower())
    return batch


def to_gradebook_rows(batch: Batch) -> list[dict[str, Any]]:
    """Flatten a marked batch into rows for a CSV.

    Auto-marked points and points awaiting manual marking are separate
    columns: adding them would imply the free-response questions had been
    marked, which they have not.
    """
    rows: list[dict[str, Any]] = []
    for graded in batch.graded:
        row: dict[str, Any] = {
            "student": graded.file.identifier,
            "user_id": graded.file.user_id,
            "file": graded.file.path.name,
            "late": "yes" if graded.file.late else "",
            "name_in_file": graded.claimed_name,
        }
        if graded.result is None:
            row.update({
                "auto_score": "", "auto_max": "", "awaiting_marking": "",
                "total_possible": "", "status": f"ERROR: {graded.error}",
            })
        else:
            r = graded.result
            row.update({
                "auto_score": r.auto_total,
                "auto_max": r.auto_max,
                "awaiting_marking": r.pending_review,
                "total_possible": r.grand_max,
                "status": "; ".join(r.warnings) if r.warnings else "ok",
            })
        rows.append(row)
    return rows


def write_gradebook_csv(batch: Batch, path: str | Path) -> Path:
    """Write the marked batch as a CSV."""
    import csv

    rows = to_gradebook_rows(batch)
    out = Path(path)
    fields = [
        "student", "user_id", "name_in_file", "file", "late",
        "auto_score", "auto_max", "awaiting_marking", "total_possible", "status",
    ]
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out
