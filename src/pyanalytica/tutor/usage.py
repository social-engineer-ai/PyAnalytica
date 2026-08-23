"""Usage accounting and caps.

Records **counts, never content.** Student questions and the model's replies
pass through the proxy and are not written down: the row says that a student
made a request at a time and how many tokens it cost, and nothing about what
was asked. Keeping student work off the instructor's server is deliberate —
logging request bodies for debugging would quietly turn this into a store of
student work on a machine somebody has to secure.

SQLite because the whole thing is a counter: one file, no service to run, and
it survives a restart, which an in-memory tally would not.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pyanalytica.tutor.pack import Limits

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id   TEXT    NOT NULL,
    student_id  TEXT    NOT NULL,
    day         TEXT    NOT NULL,   -- YYYY-MM-DD, UTC
    at          TEXT    NOT NULL,   -- ISO 8601, UTC
    input_tok   INTEGER NOT NULL DEFAULT 0,
    output_tok  INTEGER NOT NULL DEFAULT 0,
    model       TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS calls_student_day ON calls (course_id, student_id, day);
CREATE INDEX IF NOT EXISTS calls_course      ON calls (course_id);

CREATE TABLE IF NOT EXISTS revoked (
    course_id  TEXT NOT NULL,
    student_id TEXT NOT NULL,
    at         TEXT NOT NULL,
    PRIMARY KEY (course_id, student_id)
);
"""


class CapExceeded(Exception):
    """Raised when a request would exceed a limit. Carries a student-readable reason."""

    def __init__(self, message: str, *, scope: str):
        super().__init__(message)
        self.scope = scope


@dataclass
class UsageSnapshot:
    """What one student has spent."""

    student_id: str
    today: int
    term: int
    input_tokens: int
    output_tokens: int


class UsageStore:
    """Counts calls and enforces caps."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        course_id: str,
        student_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        now: datetime | None = None,
    ) -> None:
        """Record one call. Note the absence of any content parameter."""
        moment = now or datetime.now(timezone.utc)
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO calls (course_id, student_id, day, at, input_tok, output_tok, model)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    course_id,
                    student_id,
                    moment.date().isoformat(),
                    moment.isoformat(timespec="seconds"),
                    int(input_tokens),
                    int(output_tokens),
                    model,
                ),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------

    def check(
        self,
        course_id: str,
        student_id: str,
        limits: Limits,
        *,
        now: datetime | None = None,
    ) -> None:
        """Raise :class:`CapExceeded` if another call would break a limit.

        Checked before the call is made, so a request that would exceed a cap
        costs nothing rather than being billed and then refused.
        """
        moment = now or datetime.now(timezone.utc)
        today = moment.date().isoformat()

        with closing(self._connect()) as conn:
            if conn.execute(
                "SELECT 1 FROM revoked WHERE course_id = ? AND student_id = ?",
                (course_id, student_id),
            ).fetchone():
                raise CapExceeded(
                    "This access token has been withdrawn. Contact your instructor.",
                    scope="revoked",
                )

            day_count = conn.execute(
                "SELECT COUNT(*) AS n FROM calls WHERE course_id = ? AND student_id = ? AND day = ?",
                (course_id, student_id, today),
            ).fetchone()["n"]
            if day_count >= limits.per_student_per_day:
                raise CapExceeded(
                    f"You have reached today's limit of {limits.per_student_per_day} "
                    f"questions. It resets tomorrow.",
                    scope="student_day",
                )

            term_count = conn.execute(
                "SELECT COUNT(*) AS n FROM calls WHERE course_id = ? AND student_id = ?",
                (course_id, student_id),
            ).fetchone()["n"]
            if term_count >= limits.per_student_per_term:
                raise CapExceeded(
                    f"You have used your {limits.per_student_per_term} questions for "
                    f"this course. Contact your instructor if you need more.",
                    scope="student_term",
                )

            course_count = conn.execute(
                "SELECT COUNT(*) AS n FROM calls WHERE course_id = ?", (course_id,)
            ).fetchone()["n"]
            if course_count >= limits.per_course_per_term:
                raise CapExceeded(
                    "The assistant is unavailable for this course right now. "
                    "Your instructor has been notified.",
                    scope="course_term",
                )

    # ------------------------------------------------------------------
    # Reporting and administration
    # ------------------------------------------------------------------

    def snapshot(
        self, course_id: str, student_id: str, *, now: datetime | None = None
    ) -> UsageSnapshot:
        moment = now or datetime.now(timezone.utc)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(input_tok), 0) AS i,"
                " COALESCE(SUM(output_tok), 0) AS o"
                " FROM calls WHERE course_id = ? AND student_id = ?",
                (course_id, student_id),
            ).fetchone()
            today = conn.execute(
                "SELECT COUNT(*) AS n FROM calls"
                " WHERE course_id = ? AND student_id = ? AND day = ?",
                (course_id, student_id, moment.date().isoformat()),
            ).fetchone()["n"]
        return UsageSnapshot(
            student_id=student_id,
            today=today,
            term=row["n"],
            input_tokens=row["i"],
            output_tokens=row["o"],
        )

    def course_totals(self, course_id: str) -> dict[str, int]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS calls, COUNT(DISTINCT student_id) AS students,"
                " COALESCE(SUM(input_tok), 0) AS input_tok,"
                " COALESCE(SUM(output_tok), 0) AS output_tok"
                " FROM calls WHERE course_id = ?",
                (course_id,),
            ).fetchone()
        return dict(row)

    def per_student(self, course_id: str) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT student_id, COUNT(*) AS calls,"
                " COALESCE(SUM(input_tok), 0) AS input_tok,"
                " COALESCE(SUM(output_tok), 0) AS output_tok,"
                " MAX(at) AS last_seen"
                " FROM calls WHERE course_id = ?"
                " GROUP BY student_id ORDER BY calls DESC",
                (course_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def revoke(self, course_id: str, student_id: str) -> None:
        """Withdraw one student's access without touching anybody else's."""
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO revoked (course_id, student_id, at) VALUES (?, ?, ?)",
                (course_id, student_id, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            conn.commit()

    def restore(self, course_id: str, student_id: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "DELETE FROM revoked WHERE course_id = ? AND student_id = ?",
                (course_id, student_id),
            )
            conn.commit()

    def revoked_students(self, course_id: str) -> list[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT student_id FROM revoked WHERE course_id = ? ORDER BY student_id",
                (course_id,),
            ).fetchall()
        return [r["student_id"] for r in rows]


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Rough US dollars for a token count.

    Published list prices as of 2026-06; a planning aid, not a bill.
    """
    prices = {
        "claude-haiku-4-5": (1.0, 5.0),
        "claude-sonnet-5": (3.0, 15.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-opus-5": (5.0, 25.0),
        "claude-opus-4-8": (5.0, 25.0),
    }
    per_m_in, per_m_out = prices.get(model, (1.0, 5.0))
    return input_tokens / 1_000_000 * per_m_in + output_tokens / 1_000_000 * per_m_out
