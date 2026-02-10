"""Homework YAML schema validation."""

from __future__ import annotations

HOMEWORK_SCHEMA = {
    # JSON Schema dict for validating homework YAML
    "type": "object",
    "required": ["title", "dataset", "questions"],
    "properties": {
        "title": {"type": "string"},
        "dataset": {"type": "string"},
        "version": {"type": "integer", "default": 1},
        "description": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "text", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "numeric",
                            "multiple_choice",
                            "checkpoint",
                            "free_response",
                        ],
                    },
                    "answer_hash": {"type": "string"},
                    "tolerance": {"type": "number", "default": 0.01},
                    "points": {"type": "integer", "default": 1},
                    "hint": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "rubric": {"type": "string"},
                },
            },
        },
    },
}

# Maps JSON Schema type names to Python types for validation.
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}

_VALID_QUESTION_TYPES = {"numeric", "multiple_choice", "checkpoint", "free_response"}


def _check_type(value: object, expected: str) -> bool:
    """Return True if *value* matches the JSON Schema *expected* type string."""
    allowed = _TYPE_MAP.get(expected)
    if allowed is None:
        return True  # unknown type -- skip check
    # In JSON Schema, ``bool`` is not a valid ``integer``/``number``.
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, allowed)


def validate_homework(data: dict) -> tuple[bool, list[str]]:
    """Validate a homework dict against the schema.

    Returns ``(valid, errors)`` where *errors* is a list of human-readable
    error strings.  An empty list means the data is valid.

    This performs manual validation so that ``jsonschema`` is **not** required
    at runtime.
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        return False, ["Root element must be a mapping/dict."]

    # ------------------------------------------------------------------
    # Required top-level fields
    # ------------------------------------------------------------------
    for req in ("title", "dataset", "questions"):
        if req not in data:
            errors.append(f"Missing required top-level field: '{req}'.")

    # ------------------------------------------------------------------
    # Type checks for top-level fields
    # ------------------------------------------------------------------
    prop_schema = HOMEWORK_SCHEMA["properties"]
    for key, spec in prop_schema.items():
        if key not in data:
            continue  # already flagged above if required
        if key == "questions":
            continue  # validated separately below
        expected_type = spec.get("type")
        if expected_type and not _check_type(data[key], expected_type):
            errors.append(
                f"Field '{key}' must be of type '{expected_type}', "
                f"got {type(data[key]).__name__}."
            )

    # ------------------------------------------------------------------
    # Questions array
    # ------------------------------------------------------------------
    questions = data.get("questions")
    if questions is not None:
        if not isinstance(questions, list):
            errors.append("'questions' must be a list/array.")
        else:
            seen_ids: set[str] = set()
            q_schema = prop_schema["questions"]["items"]
            q_required: list[str] = q_schema["required"]
            q_props: dict = q_schema["properties"]

            for idx, question in enumerate(questions):
                prefix = f"questions[{idx}]"

                if not isinstance(question, dict):
                    errors.append(f"{prefix}: each question must be a mapping/dict.")
                    continue

                # Required fields inside each question
                for req in q_required:
                    if req not in question:
                        errors.append(f"{prefix}: missing required field '{req}'.")

                # Type checks for every supplied field
                for field_name, field_spec in q_props.items():
                    if field_name not in question:
                        continue
                    expected_type = field_spec.get("type")
                    if expected_type and not _check_type(
                        question[field_name], expected_type
                    ):
                        errors.append(
                            f"{prefix}.{field_name}: must be of type "
                            f"'{expected_type}', got "
                            f"{type(question[field_name]).__name__}."
                        )

                # Enum check for question type
                q_type = question.get("type")
                if isinstance(q_type, str) and q_type not in _VALID_QUESTION_TYPES:
                    errors.append(
                        f"{prefix}.type: invalid question type '{q_type}'. "
                        f"Must be one of {sorted(_VALID_QUESTION_TYPES)}."
                    )

                # Options required for multiple_choice
                if q_type == "multiple_choice" and not question.get("options"):
                    errors.append(
                        f"{prefix}: 'multiple_choice' questions should have "
                        f"an 'options' list."
                    )

                # Validate options items are strings (if present)
                if "options" in question and isinstance(question["options"], list):
                    for opt_idx, opt in enumerate(question["options"]):
                        if not isinstance(opt, str):
                            errors.append(
                                f"{prefix}.options[{opt_idx}]: each option "
                                f"must be a string."
                            )

                # Duplicate question id check
                q_id = question.get("id")
                if isinstance(q_id, str):
                    if q_id in seen_ids:
                        errors.append(
                            f"{prefix}: duplicate question id '{q_id}'."
                        )
                    seen_ids.add(q_id)

    return (len(errors) == 0, errors)
