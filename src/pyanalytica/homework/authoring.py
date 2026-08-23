"""Author-side homework tooling: master files, student copies, answer keys.

Why this module exists
----------------------
In-app feedback requires an answer the student's machine can check, and an
answer the student's machine can check is an answer the student can recover.
``hash_answer`` is an unsalted 16-character SHA-256 prefix, so given a student
copy of an assignment:

* a ``multiple_choice`` answer falls by hashing the ``options`` the file
  already lists;
* a ``numeric`` answer falls by sweeping values at the stated ``tolerance``.

Both take under a second.  No hashing scheme fixes this -- local checking and
local secrecy are mutually exclusive.  The resolution is to decide per question
which one you want:

Assignments therefore carry **no answer material at all**. There is nothing
in a student's copy to recover, because nothing in the app checks an answer.
Responses are recorded and marked later from the instructor's key by
:mod:`pyanalytica.homework.regrade`.

Self-checking with instant feedback still exists -- as
:mod:`pyanalytica.practice`, a separate feature whose drills carry no marks
and whose answers are therefore in plaintext. Keeping the two apart means
neither has to compromise for the other.

The master file holds plaintext answers and never leaves the author's machine.
:func:`build` derives the student copy and the answer key from it, so the two
cannot drift apart and the dangerous step -- stripping answers -- is mechanical
rather than remembered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyanalytica.core.answers import hash_answer

# Question types whose answers are checked automatically.
_AUTO_TYPES = {"numeric", "multiple_choice"}

# Fields that must never appear in a student copy.
_SECRET_FIELDS = ("answer", "solution", "answer_key")


class HomeworkBuildError(Exception):
    """Raised when a master file is invalid or a build would leak answers."""


@dataclass
class MasterQuestion:
    """A question as written by the author, with its answer in plaintext."""

    id: str
    text: str
    type: str
    answer: str | float | None = None
    tolerance: float = 0.01
    points: int = 1
    hint: str | None = None
    options: list[str] | None = None
    rubric: str | None = None

    @property
    def needs_answer(self) -> bool:
        """True if this question type requires a plaintext answer."""
        return self.type in _AUTO_TYPES

    @property
    def answer_hash(self) -> str:
        """Hash of the plaintext answer, or ``""`` if there is none."""
        if self.answer is None:
            return ""
        tol = self.tolerance if self.type == "numeric" else 0.0
        return hash_answer(self.answer, tol)


@dataclass
class MasterHomework:
    """A homework assignment as written by the author."""

    title: str
    dataset: str
    version: int = 1
    description: str = ""
    questions: list[MasterQuestion] = field(default_factory=list)

    @property
    def total_points(self) -> int:
        return sum(q.points for q in self.questions)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_master(data: dict[str, Any]) -> MasterHomework:
    """Parse and validate a master homework dict.

    Raises
    ------
    HomeworkBuildError
        If required fields are missing, ids are duplicated, or an
        automatically-checked question has no answer.
    """
    if not isinstance(data, dict):
        raise HomeworkBuildError(
            f"Master file must be a mapping at the top level, got "
            f"{type(data).__name__}."
        )

    errors: list[str] = []
    for req in ("title", "dataset", "questions"):
        if req not in data:
            errors.append(f"Missing required top-level field: '{req}'.")

    raw_questions = data.get("questions")
    if raw_questions is not None and not isinstance(raw_questions, list):
        errors.append("'questions' must be a list.")
        raw_questions = []

    questions: list[MasterQuestion] = []
    seen: set[str] = set()

    for idx, raw in enumerate(raw_questions or []):
        prefix = f"questions[{idx}]"
        if not isinstance(raw, dict):
            errors.append(f"{prefix}: each question must be a mapping.")
            continue

        for req in ("id", "text", "type"):
            if req not in raw:
                errors.append(f"{prefix}: missing required field '{req}'.")
        if not {"id", "text", "type"} <= set(raw):
            continue

        q_id = str(raw["id"])
        if q_id in seen:
            errors.append(f"{prefix}: duplicate question id '{q_id}'.")
        seen.add(q_id)

        question = MasterQuestion(
            id=q_id,
            text=str(raw["text"]),
            type=str(raw["type"]),
            answer=raw.get("answer"),
            tolerance=float(raw.get("tolerance", 0.01)),
            points=int(raw.get("points", 1)),
            hint=raw.get("hint"),
            options=raw.get("options"),
            rubric=raw.get("rubric"),
        )

        # An auto-checked question without an answer can never be scored --
        # catch it here rather than at grading time, weeks later.
        if question.needs_answer and question.answer is None:
            errors.append(
                f"{prefix} ('{q_id}'): a '{question.type}' question needs an "
                f"'answer'. Add the correct answer in plaintext -- this file "
                f"stays on your machine."
            )

        if question.type == "multiple_choice" and not question.options:
            errors.append(f"{prefix} ('{q_id}'): multiple_choice needs 'options'.")

        if question.type == "numeric" and question.tolerance <= 0:
            errors.append(
                f"{prefix} ('{q_id}'): 'tolerance' must be greater than 0."
            )

        questions.append(question)

    if errors:
        bullets = "\n  - ".join(errors)
        raise HomeworkBuildError(
            f"Master homework is invalid ({len(errors)} problem(s)):\n  - {bullets}"
        )

    return MasterHomework(
        title=str(data["title"]),
        dataset=str(data["dataset"]),
        version=int(data.get("version", 1)),
        description=str(data.get("description", "")),
        questions=questions,
    )


def load_master(path: str | Path) -> MasterHomework:
    """Load and validate a master homework YAML file."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyYAML is required to load homework files. "
            "Install it with: pip install pyyaml"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Master homework file not found: {p}")

    with open(p, encoding="utf-8") as fh:
        return parse_master(yaml.safe_load(fh))


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def build_student_copy(master: MasterHomework) -> dict[str, Any]:
    """Derive the assignment file that students receive.

    Graded questions carry no answer material of any kind.  Practice questions
    carry ``answer_hash`` so the app can give immediate feedback.
    """
    questions: list[dict[str, Any]] = []

    for q in master.questions:
        out: dict[str, Any] = {
            "id": q.id,
            "text": q.text,
            "type": q.type,
            "points": q.points,
        }
        if q.type == "numeric":
            out["tolerance"] = q.tolerance
        if q.options:
            out["options"] = list(q.options)
        if q.hint:
            out["hint"] = q.hint
        # `rubric` describes how marks are awarded, so it stays author-side.

        questions.append(out)

    student = {
        "title": master.title,
        "dataset": master.dataset,
        "version": master.version,
        "description": master.description,
        "questions": questions,
    }

    assert_no_answers_leaked(student)
    return student


def build_answer_key(master: MasterHomework) -> dict[str, Any]:
    """Derive the instructor key used to score collected submissions."""
    return {
        "title": master.title,
        "dataset": master.dataset,
        "version": master.version,
        "questions": [
            {
                "id": q.id,
                "type": q.type,
                "points": q.points,
                "tolerance": q.tolerance,
                # Plaintext is kept alongside the hash so feedback can tell a
                # student what the answer was, not merely that they missed it.
                "answer": q.answer,
                "answer_hash": q.answer_hash,
                "rubric": q.rubric,
            }
            for q in master.questions
        ],
    }


def assert_no_answers_leaked(student: dict[str, Any]) -> None:
    """Raise if a student copy contains answer material it should not.

    This is the safety net the whole design rests on: a graded question that
    ships with a hash is silently worthless, and nothing downstream would
    notice.  Fail the build instead.
    """
    problems: list[str] = []

    for idx, q in enumerate(student.get("questions", [])):
        q_id = q.get("id", f"index {idx}")

        for secret in _SECRET_FIELDS:
            if secret in q:
                problems.append(
                    f"question '{q_id}' carries a plaintext '{secret}' field"
                )

        if q.get("answer_hash"):
            problems.append(
                f"question '{q_id}' ships an 'answer_hash'. Assignments are "
                f"marked by the instructor, so nothing in a student's copy "
                f"should be checkable -- a hash is recoverable by sweeping "
                f"candidate answers"
            )

    if problems:
        bullets = "\n  - ".join(problems)
        raise HomeworkBuildError(
            f"Refusing to write a student copy that leaks answers:\n  - {bullets}"
        )


def build(
    master_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    stem: str | None = None,
) -> tuple[Path, Path]:
    """Build the student copy and answer key from a master file.

    Returns
    -------
    tuple[Path, Path]
        ``(student_path, key_path)``.  The student copy is safe to distribute;
        the key file is not.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyYAML is required to build homework files. "
            "Install it with: pip install pyyaml"
        ) from exc

    master_p = Path(master_path)
    master = load_master(master_p)

    base = stem or master_p.name.replace(".master.yaml", "").replace(".yaml", "")
    out = Path(out_dir) if out_dir else master_p.parent
    out.mkdir(parents=True, exist_ok=True)

    student_path = out / f"{base}.yaml"
    key_path = out / f"{base}.key.yaml"

    if student_path.resolve() == master_p.resolve():
        raise HomeworkBuildError(
            f"Refusing to overwrite the master file at {master_p}. "
            f"Name it '<name>.master.yaml' or pass a different out_dir."
        )

    def _dump(data: dict[str, Any], path: Path, header: str) -> None:
        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        path.write_text(f"{header}\n{body}", encoding="utf-8")

    _dump(
        build_student_copy(master),
        student_path,
        f"# Generated from {master_p.name} -- do not edit by hand.\n"
        f"# Safe to distribute: contains no answer material of any kind.",
    )
    _dump(
        build_answer_key(master),
        key_path,
        f"# Generated from {master_p.name} -- do not edit by hand.\n"
        f"# INSTRUCTOR ONLY. Contains plaintext answers. Do not distribute or commit.",
    )

    return student_path, key_path
