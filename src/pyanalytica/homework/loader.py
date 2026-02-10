"""Load and parse homework YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyanalytica.homework.schema import validate_homework


@dataclass
class HomeworkQuestion:
    """A single question within a homework assignment."""

    id: str
    text: str
    type: str  # "numeric", "multiple_choice", "checkpoint", "free_response"
    answer_hash: str = ""
    tolerance: float = 0.01
    points: int = 1
    hint: str | None = None
    options: list[str] | None = None
    rubric: str | None = None


@dataclass
class Homework:
    """A complete homework assignment parsed from YAML."""

    title: str
    dataset: str
    version: int = 1
    description: str = ""
    questions: list[HomeworkQuestion] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_question(self, question_id: str) -> HomeworkQuestion | None:
        """Return the question with the given *question_id*, or ``None``."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    @property
    def total_points(self) -> int:
        """Sum of all question point values."""
        return sum(q.points for q in self.questions)

    @property
    def question_ids(self) -> list[str]:
        """Ordered list of question ids."""
        return [q.id for q in self.questions]


def load_homework(yaml_path: str | Path) -> Homework:
    """Load a homework from a YAML file.

    Requires the ``PyYAML`` package (``pip install pyyaml``).

    Raises
    ------
    FileNotFoundError
        If *yaml_path* does not exist.
    ImportError
        If PyYAML is not installed.
    ValueError
        If the YAML content fails schema validation.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load homework YAML files. "
            "Install it with: pip install pyyaml"
        ) from exc

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Homework file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping at the top level, got {type(data).__name__}."
        )

    return _parse_homework(data)


def load_homework_from_dict(data: dict[str, Any]) -> Homework:
    """Load homework from a pre-parsed dict (e.g. already loaded YAML)."""
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a dict, got {type(data).__name__}."
        )
    return _parse_homework(data)


def _parse_homework(data: dict[str, Any]) -> Homework:
    """Parse a raw dict into a :class:`Homework` dataclass.

    Validates against the homework schema first and raises ``ValueError``
    with all discovered problems if validation fails.
    """
    valid, errors = validate_homework(data)
    if not valid:
        bullet_list = "\n  - ".join(errors)
        raise ValueError(
            f"Homework validation failed with {len(errors)} error(s):\n"
            f"  - {bullet_list}"
        )

    questions: list[HomeworkQuestion] = []
    for raw_q in data.get("questions", []):
        questions.append(
            HomeworkQuestion(
                id=str(raw_q["id"]),
                text=str(raw_q["text"]),
                type=str(raw_q["type"]),
                answer_hash=str(raw_q.get("answer_hash", "")),
                tolerance=float(raw_q.get("tolerance", 0.01)),
                points=int(raw_q.get("points", 1)),
                hint=raw_q.get("hint"),
                options=raw_q.get("options"),
                rubric=raw_q.get("rubric"),
            )
        )

    return Homework(
        title=str(data["title"]),
        dataset=str(data["dataset"]),
        version=int(data.get("version", 1)),
        description=str(data.get("description", "")),
        questions=questions,
    )
