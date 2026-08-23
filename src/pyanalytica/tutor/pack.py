"""Course packs: the per-course instructions that shape the tutor.

A pack is what makes the assistant *this* course's assistant. It holds the
system prompt (how to guide students — Socratic, direct, whatever the
instructor wants), the model, and the spending limits.

Packs live on the **instructor's server**, never on a student's machine. Two
consequences worth stating plainly:

* Changing how students are guided means editing one file on one machine.
  Nobody reinstalls anything.
* A student cannot read the pack out of their own copy of the app, because
  their copy never has it.

Neither makes the guidance style tamper-proof. A student can ask the assistant
to drop the act, or open claude.ai in another tab. A pack shapes the default
behaviour for cooperative students, which is nearly all of them; it is not a
lock, and building it as though it were one would be a waste of effort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Sensible for guided tutoring: cheap enough to leave on all semester, and
# well suited to short conversational turns. Instructors can override.
DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 800

DEFAULT_SYSTEM_PROMPT = """\
You are a teaching assistant for an introductory data-analysis course. \
Students are working in PyAnalytica, a point-and-click statistics workbench.

Guide, do not solve. When a student asks a question:

- Ask what they have already tried, or what they think the answer might be.
- Point them at the part of the tool that would answer it, rather than \
answering it yourself.
- If they are stuck on a concept, explain the concept with a small example \
that is NOT their dataset.
- Never give a final numeric answer to a question that looks like it came \
from an assignment. Ask them which analysis would produce it.
- If they have made an error, ask a question that would let them notice it \
themselves before you name it.

Be brief. Two or three sentences is usually enough. Use plain language and \
avoid jargon unless the student uses it first.
"""


class CoursePackError(Exception):
    """Raised when a course pack cannot be read or is invalid."""


@dataclass
class Limits:
    """Spending ceilings, all enforced server-side.

    A proxy holding an instructor's API key with no ceilings is a bill waiting
    to happen, so these are required rather than optional and default to
    something survivable.
    """

    per_student_per_day: int = 40
    per_student_per_term: int = 600
    per_course_per_term: int = 30_000

    def as_dict(self) -> dict[str, int]:
        return {
            "per_student_per_day": self.per_student_per_day,
            "per_student_per_term": self.per_student_per_term,
            "per_course_per_term": self.per_course_per_term,
        }


@dataclass
class CoursePack:
    """One course's tutor configuration."""

    course_id: str
    title: str = ""
    instructor: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    limits: Limits = field(default_factory=Limits)
    # Free text shown to students in the app, e.g. office hours or scope.
    notice: str = ""

    @property
    def cacheable(self) -> bool:
        """Whether the system prompt is long enough to be worth caching.

        Prompt caching needs roughly 1024 tokens of stable prefix. Below that
        the cache silently never engages, so there is no point paying the
        write cost.
        """
        return len(self.system_prompt) >= 4000  # ~1000 tokens at ~4 chars each


def parse_pack(data: dict[str, Any]) -> CoursePack:
    """Parse and validate a course pack dict."""
    if not isinstance(data, dict):
        raise CoursePackError(f"A course pack must be a mapping, got {type(data).__name__}.")

    course_id = str(data.get("course_id", "")).strip()
    if not course_id:
        raise CoursePackError("A course pack needs a 'course_id'.")

    raw_limits = data.get("limits") or {}
    if not isinstance(raw_limits, dict):
        raise CoursePackError("'limits' must be a mapping.")

    try:
        limits = Limits(
            per_student_per_day=int(raw_limits.get("per_student_per_day", 40)),
            per_student_per_term=int(raw_limits.get("per_student_per_term", 600)),
            per_course_per_term=int(raw_limits.get("per_course_per_term", 30_000)),
        )
    except (TypeError, ValueError) as exc:
        raise CoursePackError(f"Limits must be whole numbers: {exc}") from exc

    for name, value in limits.as_dict().items():
        if value <= 0:
            raise CoursePackError(f"limits.{name} must be greater than 0.")

    prompt = str(data.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()
    if not prompt:
        raise CoursePackError("'system_prompt' cannot be empty.")

    return CoursePack(
        course_id=course_id,
        title=str(data.get("title", "")),
        instructor=str(data.get("instructor", "")),
        system_prompt=prompt,
        model=str(data.get("model") or DEFAULT_MODEL),
        max_tokens=int(data.get("max_tokens") or DEFAULT_MAX_TOKENS),
        limits=limits,
        notice=str(data.get("notice", "")),
    )


def load_pack(path: str | Path) -> CoursePack:
    """Load a course pack from YAML."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyYAML is required to read course packs. Install: pip install pyyaml"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Course pack not found: {p}")
    with open(p, encoding="utf-8") as fh:
        return parse_pack(yaml.safe_load(fh))


def example_pack_yaml(course_id: str = "MY-COURSE-101") -> str:
    """A starter pack an instructor can edit."""
    return f"""\
# PyAnalytica tutor — course pack
#
# This file lives on YOUR server. Students never receive it, so the guidance
# style below is not visible to them and can be changed at any time without
# anybody reinstalling anything.

course_id: "{course_id}"
title: "Introduction to Business Analytics"
instructor: "Your Name"

# Shown to students in the app.
notice: "This assistant guides; it will not give you answers to assignments."

model: "{DEFAULT_MODEL}"
max_tokens: {DEFAULT_MAX_TOKENS}

limits:
  per_student_per_day: 40
  per_student_per_term: 600
  per_course_per_term: 30000

system_prompt: |
{chr(10).join("  " + line for line in DEFAULT_SYSTEM_PROMPT.strip().splitlines())}
"""
