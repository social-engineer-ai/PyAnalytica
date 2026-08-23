"""Practice drills: self-check exercises attached to the tool, not to a course.

A drill is a short set of questions a student can answer inside the app and
have marked immediately. Drills carry **no marks**, which is what makes the
design simple: the expected answers ship with the drill, and the fact that a
determined student could dig them out of the file costs nothing. Nobody is
being assessed.

This is deliberately separate from :mod:`pyanalytica.homework`. Assignments
are marked by the instructor after collection and contain no answer material
at all; drills are marked on the spot and contain nothing else. Keeping them
apart means neither has to compromise for the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyanalytica.core.answers import answer_matches

# Where the drills that ship with the package live.
BUNDLED_DIR = Path(__file__).parent / "bundled"

_QUESTION_KINDS = {"numeric", "multiple_choice", "short_text"}


class DrillError(Exception):
    """Raised when a drill file cannot be read or is malformed."""


@dataclass
class DrillQuestion:
    """One self-check question."""

    id: str
    text: str
    kind: str = "numeric"
    answer: str | float | None = None
    answer_hash: str = ""
    tolerance: float = 0.01
    options: list[str] | None = None
    hint: str | None = None
    explanation: str | None = None

    def check(self, submitted: Any) -> bool:
        """Return True if *submitted* is correct."""
        return answer_matches(
            submitted,
            kind=self.kind,
            expected=self.answer,
            expected_hash=self.answer_hash,
            tolerance=self.tolerance,
        )


@dataclass
class Drill:
    """A set of self-check questions on one dataset."""

    id: str
    title: str
    dataset: str
    description: str = ""
    questions: list[DrillQuestion] = field(default_factory=list)

    def get(self, question_id: str) -> DrillQuestion | None:
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    @property
    def size(self) -> int:
        return len(self.questions)


def parse_drill(data: dict[str, Any], *, drill_id: str = "") -> Drill:
    """Parse and validate a drill dict."""
    if not isinstance(data, dict):
        raise DrillError(f"A drill must be a mapping, got {type(data).__name__}.")

    errors: list[str] = []
    for required in ("title", "dataset", "questions"):
        if required not in data:
            errors.append(f"missing required field '{required}'")

    raw_questions = data.get("questions")
    if raw_questions is not None and not isinstance(raw_questions, list):
        errors.append("'questions' must be a list")
        raw_questions = []

    questions: list[DrillQuestion] = []
    seen: set[str] = set()

    for idx, raw in enumerate(raw_questions or []):
        where = f"questions[{idx}]"
        if not isinstance(raw, dict):
            errors.append(f"{where}: must be a mapping")
            continue
        if "id" not in raw or "text" not in raw:
            errors.append(f"{where}: needs both 'id' and 'text'")
            continue

        qid = str(raw["id"])
        if qid in seen:
            errors.append(f"{where}: duplicate question id '{qid}'")
        seen.add(qid)

        kind = str(raw.get("kind", "numeric"))
        if kind not in _QUESTION_KINDS:
            errors.append(
                f"{where}: unknown kind '{kind}', expected one of {sorted(_QUESTION_KINDS)}"
            )
            continue

        question = DrillQuestion(
            id=qid,
            text=str(raw["text"]),
            kind=kind,
            answer=raw.get("answer"),
            answer_hash=str(raw.get("answer_hash", "")),
            tolerance=float(raw.get("tolerance", 0.01)),
            options=raw.get("options"),
            hint=raw.get("hint"),
            explanation=raw.get("explanation"),
        )

        # A drill question with nothing to compare against would silently mark
        # every answer wrong, which is worse than refusing to load it.
        if question.answer is None and not question.answer_hash:
            errors.append(f"{where} ('{qid}'): needs an 'answer' or an 'answer_hash'")

        if kind == "multiple_choice" and not question.options:
            errors.append(f"{where} ('{qid}'): multiple_choice needs 'options'")

        questions.append(question)

    if errors:
        bullets = "\n  - ".join(errors)
        raise DrillError(f"Drill is invalid:\n  - {bullets}")

    return Drill(
        id=drill_id or str(data.get("id", data["title"])),
        title=str(data["title"]),
        dataset=str(data["dataset"]),
        description=str(data.get("description", "")),
        questions=questions,
    )


def load_drill(path: str | Path) -> Drill:
    """Load a drill from a YAML file."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyYAML is required to load drills. Install it with: pip install pyyaml"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Drill file not found: {p}")
    with open(p, encoding="utf-8") as fh:
        return parse_drill(yaml.safe_load(fh), drill_id=p.stem)


def list_bundled_drills() -> list[str]:
    """Return the ids of the drills that ship with PyAnalytica."""
    if not BUNDLED_DIR.exists():
        return []
    return sorted(p.stem for p in BUNDLED_DIR.glob("*.yaml"))


def load_bundled_drill(drill_id: str) -> Drill:
    """Load a drill that ships with PyAnalytica, by id."""
    path = BUNDLED_DIR / f"{drill_id}.yaml"
    if not path.exists():
        available = ", ".join(list_bundled_drills()) or "none"
        raise DrillError(f"No bundled drill called '{drill_id}'. Available: {available}")
    return load_drill(path)


@dataclass
class DrillProgress:
    """How far a student has got with one drill, for this session only.

    Not persisted anywhere. Drills carry no marks, so there is nothing worth
    storing, and nothing worth a student's time to tamper with.
    """

    drill_id: str
    answered: dict[str, bool] = field(default_factory=dict)

    def record(self, question_id: str, correct: bool) -> None:
        self.answered[question_id] = correct

    @property
    def attempted(self) -> int:
        return len(self.answered)

    @property
    def correct(self) -> int:
        return sum(1 for ok in self.answered.values() if ok)
