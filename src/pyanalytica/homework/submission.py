"""Assembling what a student submits for an assignment.

Nothing here grades anything. The student's copy of an assignment contains no
answer material, so there is nothing to grade against, and that is the point:
marking happens on the instructor's machine after collection, using
:mod:`pyanalytica.homework.regrade`.

What a submission carries instead is **evidence**: the answers, and a record
of the work that produced them -- which operations ran, in what order, on
which dataset, and the pandas/sklearn code each one generated. An answer on
its own can be obtained from a classmate. The record of arriving at it cannot,
and it is the thing an instructor actually wants to see.

Self-checking with immediate feedback lives in :mod:`pyanalytica.practice`,
which is a separate feature with nothing at stake.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyanalytica.homework.loader import Homework


@dataclass
class SubmissionAnswer:
    """One student answer. No verdict, no points -- those come later."""

    question_id: str
    answer: str
    max_points: int
    question_type: str = ""


@dataclass
class WorkStep:
    """One recorded step of the analysis behind a submission."""

    timestamp: str
    action: str
    description: str
    dataset: str
    code: str = ""


@dataclass
class Submission:
    """A complete student submission for one assignment."""

    homework_id: str
    homework_version: int
    submitted_at: str
    student_name: str
    answers: list[SubmissionAnswer] = field(default_factory=list)
    total_points: int = 0
    work: list[WorkStep] = field(default_factory=list)
    # Format version, so a grader written for one shape can recognise another.
    schema: str = "pyanalytica.submission/2"

    @property
    def answered(self) -> int:
        return sum(1 for a in self.answers if str(a.answer).strip())


def create_submission(
    homework: Homework,
    answers: dict[str, str | float],
    session_log: list[dict[str, Any]],
    student_name: str,
) -> Submission:
    """Assemble a submission from the student's answers and their work log.

    Every question in the assignment appears in the output, answered or not,
    so a grader can tell a blank from a missing question.

    Parameters
    ----------
    homework:
        The parsed assignment.
    answers:
        Mapping of ``{question_id: student_answer}``.
    session_log:
        Recorded operations. Entries may carry a ``code`` key; where the
        procedure recorder supplied one it is preserved, because the generated
        code is the most informative part of the record.
    student_name:
        What the student typed. Recorded, never trusted -- identity comes from
        the filename the LMS assigns on download.
    """
    submission_answers = [
        SubmissionAnswer(
            question_id=question.id,
            answer="" if answers.get(question.id) is None else str(answers[question.id]),
            max_points=question.points,
            question_type=question.type,
        )
        for question in homework.questions
    ]

    work = [
        WorkStep(
            timestamp=str(entry.get("timestamp", "")),
            action=str(entry.get("action", "")),
            description=str(entry.get("description", "")),
            dataset=str(entry.get("dataset", "")),
            code=str(entry.get("code", "")),
        )
        for entry in session_log
    ]

    return Submission(
        homework_id=homework.title,
        homework_version=homework.version,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        student_name=student_name,
        answers=submission_answers,
        total_points=homework.total_points,
        work=work,
    )


def export_submission_json(submission: Submission) -> str:
    """Export a submission as a pretty-printed JSON string."""
    return json.dumps(asdict(submission), indent=2, default=str)


def export_submission_bytes(submission: Submission) -> bytes:
    """Export a submission as UTF-8 bytes, for a file download."""
    return export_submission_json(submission).encode("utf-8")
