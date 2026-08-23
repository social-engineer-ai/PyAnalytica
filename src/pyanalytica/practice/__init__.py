"""Practice drills -- self-check exercises attached to the tool.

Drills are not assignments. They carry no marks, they are never collected,
and their answers ship with them so the app can mark them instantly. See
:mod:`pyanalytica.homework` for assessed work, which contains no answer
material at all.
"""

from pyanalytica.practice.drills import (
    BUNDLED_DIR,
    Drill,
    DrillError,
    DrillProgress,
    DrillQuestion,
    list_bundled_drills,
    load_bundled_drill,
    load_drill,
    parse_drill,
)

__all__ = [
    "BUNDLED_DIR",
    "Drill",
    "DrillError",
    "DrillProgress",
    "DrillQuestion",
    "list_bundled_drills",
    "load_bundled_drill",
    "load_drill",
    "parse_drill",
]
