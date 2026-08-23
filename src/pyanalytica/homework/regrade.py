"""Authoritative re-grading of collected student submissions.

Nothing in the student's app grades anything: an assignment ships without
answer material, so a submission carries answers and a record of the work, and
no scores at all. This module supplies the marks, on the instructor's machine,
from a key that never leaves it.

Submissions are still treated as untrusted input -- they arrive through the
student's hands and nothing signs them -- so only the answers and question ids
are read, and everything else is recomputed.

What this does and does not defend against:

* **Defends**: editing anything in the file, and submitting against an
  assignment version that has since changed.
* **Does not defend**: a student submitting another student's correct answers.
  That is an academic-integrity matter, not a software one.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyanalytica.core.answers import answer_matches

# Status values for a single re-graded question.
STATUS_AUTO = "auto"              # scored automatically
STATUS_PENDING = "pending"        # free_response, awaiting manual marking
STATUS_UNANSWERED = "unanswered"  # no response recorded
STATUS_UNKNOWN = "unknown"        # id not present in the key


class RegradeError(Exception):
    """Raised when a key or submission cannot be interpreted at all."""


@dataclass
class KeyQuestion:
    """One question from an instructor answer key."""

    id: str
    type: str
    points: int = 1
    tolerance: float = 0.01
    answer: str | float | None = None
    answer_hash: str = ""
    rubric: str | None = None


@dataclass
class AnswerKey:
    """An instructor answer key for one assignment."""

    title: str
    version: int = 1
    questions: list[KeyQuestion] = field(default_factory=list)

    def get(self, question_id: str) -> KeyQuestion | None:
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    @property
    def grand_max(self) -> int:
        return sum(q.points for q in self.questions)


@dataclass
class QuestionOutcome:
    """The authoritative result for a single question."""

    question_id: str
    status: str
    student_answer: str
    correct: bool | None
    points_earned: int
    max_points: int
    correct_answer: str | None = None


@dataclass
class RegradeResult:
    """The authoritative result for one student's submission."""

    source: str
    claimed_name: str
    homework_title: str
    submitted_at: str
    outcomes: list[QuestionOutcome] = field(default_factory=list)
    auto_total: int = 0
    auto_max: int = 0
    pending_review: int = 0
    grand_max: int = 0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------

def parse_key(data: dict[str, Any]) -> AnswerKey:
    """Parse an answer-key dict produced by :mod:`~pyanalytica.homework.authoring`."""
    if not isinstance(data, dict):
        raise RegradeError(
            f"Answer key must be a mapping, got {type(data).__name__}."
        )
    if "questions" not in data:
        raise RegradeError("Answer key has no 'questions' section.")

    questions = []
    for raw in data.get("questions") or []:
        if not isinstance(raw, dict) or "id" not in raw:
            raise RegradeError(f"Malformed question entry in answer key: {raw!r}")
        questions.append(
            KeyQuestion(
                id=str(raw["id"]),
                type=str(raw.get("type", "numeric")),
                points=int(raw.get("points", 1)),
                tolerance=float(raw.get("tolerance", 0.01)),
                answer=raw.get("answer"),
                answer_hash=str(raw.get("answer_hash") or ""),
                rubric=raw.get("rubric"),
            )
        )

    return AnswerKey(
        title=str(data.get("title", "")),
        version=int(data.get("version", 1)),
        questions=questions,
    )


def load_submission(path: str | Path) -> dict[str, Any]:
    """Read a submission from either export format.

    Students download HTML (readable in the LMS, with the data embedded) but a
    .json export is still accepted, as are files an LMS has renamed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Submission not found: {p}")

    text = p.read_text(encoding="utf-8", errors="replace")
    stripped = text.lstrip()

    if stripped.startswith("{"):
        return json.loads(text)

    from pyanalytica.homework.export_html import extract_submission_json

    try:
        return extract_submission_json(text)
    except ValueError as exc:
        raise RegradeError(f"{p.name}: {exc}") from exc


def load_key(path: str | Path) -> AnswerKey:
    """Load an instructor answer key from YAML."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyYAML is required to load answer keys. "
            "Install it with: pip install pyyaml"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Answer key not found: {p}")
    with open(p, encoding="utf-8") as fh:
        return parse_key(yaml.safe_load(fh))


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def _is_correct(question: KeyQuestion, raw_answer: Any) -> bool:
    """Compare a student answer against the key."""
    return answer_matches(
        raw_answer,
        kind=question.type,
        expected=question.answer,
        expected_hash=question.answer_hash,
        tolerance=question.tolerance,
    )


# ---------------------------------------------------------------------------
# Re-grading
# ---------------------------------------------------------------------------

def regrade(
    submission: dict[str, Any],
    key: AnswerKey,
    *,
    source: str = "",
) -> RegradeResult:
    """Score one submission from its raw answers, ignoring its claimed scores.

    Parameters
    ----------
    submission:
        The parsed contents of a student's exported JSON.
    key:
        The instructor answer key.
    source:
        Where the submission came from (a filename), carried into the result
        for reporting.
    """
    if not isinstance(submission, dict):
        raise RegradeError(
            f"Submission must be a mapping, got {type(submission).__name__}."
        )

    warnings: list[str] = []

    # Raw answers are the ONLY thing read from the submission for scoring.
    raw_answers: dict[str, Any] = {}
    for entry in submission.get("answers") or []:
        if isinstance(entry, dict) and "question_id" in entry:
            raw_answers[str(entry["question_id"])] = entry.get("answer", "")

    if not raw_answers:
        warnings.append("Submission contains no answers.")

    submitted_version = submission.get("homework_version")
    if submitted_version is not None and int(submitted_version) != key.version:
        warnings.append(
            f"Version mismatch: submission is for version {submitted_version}, "
            f"key is version {key.version}. Scores may not be comparable."
        )

    unknown = set(raw_answers) - {q.id for q in key.questions}
    if unknown:
        warnings.append(
            f"Submission contains {len(unknown)} question id(s) absent from the "
            f"key: {sorted(unknown)}. Ignored."
        )

    outcomes: list[QuestionOutcome] = []
    auto_total = auto_max = pending = 0

    for question in key.questions:
        # Present-but-blank counts as unanswered. A submission lists every
        # question in the assignment, answered or not, so mere presence of the
        # id says nothing -- checking it alone awarded checkpoint marks to
        # students who left them untouched.
        answered = str(raw_answers.get(question.id, "") or "").strip() != "" 
        raw = raw_answers.get(question.id, "")
        answer_text = "" if raw is None else str(raw)
        correct_text = None if question.answer is None else str(question.answer)

        if question.type == "free_response":
            outcomes.append(
                QuestionOutcome(
                    question_id=question.id,
                    status=STATUS_PENDING,
                    student_answer=answer_text,
                    correct=None,
                    points_earned=0,
                    max_points=question.points,
                )
            )
            pending += question.points
            continue

        if question.type == "checkpoint":
            # Participation by design: reaching the checkpoint is the task.
            earned = question.points if answered else 0
            outcomes.append(
                QuestionOutcome(
                    question_id=question.id,
                    status=STATUS_AUTO if answered else STATUS_UNANSWERED,
                    student_answer=answer_text,
                    correct=True if answered else False,
                    points_earned=earned,
                    max_points=question.points,
                )
            )
            auto_total += earned
            auto_max += question.points
            continue

        if not answered or answer_text.strip() == "":
            outcomes.append(
                QuestionOutcome(
                    question_id=question.id,
                    status=STATUS_UNANSWERED,
                    student_answer="",
                    correct=False,
                    points_earned=0,
                    max_points=question.points,
                    correct_answer=correct_text,
                )
            )
            auto_max += question.points
            continue

        if question.answer is None and not question.answer_hash:
            warnings.append(
                f"Question '{question.id}' has no answer in the key; "
                f"cannot score it. Awarding 0 and flagging for review."
            )
            outcomes.append(
                QuestionOutcome(
                    question_id=question.id,
                    status=STATUS_UNKNOWN,
                    student_answer=answer_text,
                    correct=None,
                    points_earned=0,
                    max_points=question.points,
                )
            )
            pending += question.points
            continue

        correct = _is_correct(question, raw)
        earned = question.points if correct else 0
        outcomes.append(
            QuestionOutcome(
                question_id=question.id,
                status=STATUS_AUTO,
                student_answer=answer_text,
                correct=correct,
                points_earned=earned,
                max_points=question.points,
                correct_answer=correct_text,
            )
        )
        auto_total += earned
        auto_max += question.points

    result = RegradeResult(
        source=source,
        claimed_name=str(submission.get("student_name", "")),
        homework_title=str(submission.get("homework_id", "")),
        submitted_at=str(submission.get("submitted_at", "")),
        outcomes=outcomes,
        auto_total=auto_total,
        auto_max=auto_max,
        pending_review=pending,
        grand_max=key.grand_max,
        warnings=warnings,
    )
    return result
