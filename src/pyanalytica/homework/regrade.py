"""Authoritative re-grading of collected student submissions.

The app grades on the student's own machine and writes the outcome into the
exported JSON (``correct``, ``points_earned``, ``auto_total``).  That file then
travels through the student's hands, and nothing signs it, so every score in it
is a claim rather than a fact -- editing ``auto_total`` in a text editor is
enough to change it.

This module therefore treats the submission as **untrusted input**: it reads
only the raw ``answer`` values and the question ids, and recomputes every score
from the instructor's key.  The claimed totals are parsed for exactly one
purpose -- reporting a mismatch, which is useful signal but never affects the
mark awarded.

What this does and does not defend against:

* **Defends**: editing scores, flipping ``correct``, inflating totals,
  submitting a file for an assignment version that has since changed.
* **Does not defend**: a student submitting another student's correct answers.
  That is an academic-integrity matter, not a software one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyanalytica.homework.grader import hash_answer

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
    graded: bool = False
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
    claimed_total: int | None = None

    @property
    def score_dispute(self) -> bool:
        """True if the submission claimed a total this re-grade disagrees with.

        Not proof of tampering on its own -- an assignment edited after a
        student downloaded it produces the same signal -- but every case is
        worth a look.
        """
        return self.claimed_total is not None and self.claimed_total != self.auto_total


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
                graded=bool(raw.get("graded", False)),
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

def _canonical_hash(question: KeyQuestion, raw_answer: Any) -> str:
    """Hash a student's raw answer the same way the key's answer was hashed.

    A numeric answer arrives from a text input as a string, so "19.790" and
    19.79 would hash differently despite being the same number.  Coerce first,
    and fall back to string comparison when coercion fails -- a numeric field
    containing "twenty" is simply wrong, not a crash.
    """
    if question.type == "numeric":
        try:
            return hash_answer(float(str(raw_answer).strip()), question.tolerance)
        except (TypeError, ValueError):
            return hash_answer(str(raw_answer), 0.0)
    return hash_answer(str(raw_answer), 0.0)


def _is_correct(question: KeyQuestion, raw_answer: Any) -> bool:
    """Compare a student answer against the key.

    Prefers the plaintext answer when the key has one, because it lets numeric
    answers be re-hashed at the key's tolerance rather than trusting a hash
    that may have been generated at a different one.
    """
    if question.answer is not None:
        expected = _canonical_hash(
            question,
            question.answer if question.type != "numeric" else float(question.answer),
        )
        return _canonical_hash(question, raw_answer) == expected
    if question.answer_hash:
        return _canonical_hash(question, raw_answer) == question.answer_hash
    return False


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
        answered = question.id in raw_answers
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

    claimed = submission.get("auto_total")
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
        claimed_total=int(claimed) if isinstance(claimed, (int, float)) else None,
    )

    if result.score_dispute:
        result.warnings.append(
            f"Submission claimed {result.claimed_total} auto points; re-grading "
            f"awards {auto_total}. The re-graded score stands."
        )

    return result
