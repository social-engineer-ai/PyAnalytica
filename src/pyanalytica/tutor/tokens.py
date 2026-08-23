"""Student access tokens.

Each student gets their own token, so usage is attributable and can be capped
or revoked per person. A shared course-wide password would make every student
indistinguishable, which means one leak spends the instructor's whole budget
with nothing to trace and nothing to revoke short of locking out the class.

Tokens are **signed, not stored**. The token carries the course, the student
id and an expiry, and an HMAC over all three; the server verifies the
signature with its secret. That means no user database, no registration step,
and no state to back up — issuing 60 tokens is a loop over a roster file.

What this does and does not defend against:

* **Defends:** a stranger who finds the endpoint (no valid signature), a
  student exceeding their allowance (caps are per-token), use after the term
  ends (expiry), and one student's misuse costing everyone (revoke one token).
* **Does not defend:** a student handing their token to a friend. Nothing in
  a local app can. It is attributable and capped, which is the realistic goal.
"""

from __future__ import annotations

import base64
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256

# Bumped if the token format ever changes, so old tokens fail cleanly rather
# than being misread.
TOKEN_PREFIX = "pat1"


class TokenError(Exception):
    """Raised when a token is malformed, unsigned, expired, or for another course."""


@dataclass
class TokenClaims:
    """What a verified token asserts."""

    course_id: str
    student_id: str
    expires_at: str  # ISO 8601, UTC

    @property
    def expiry(self) -> datetime:
        return datetime.fromisoformat(self.expires_at)

    def expired(self, *, now: datetime | None = None) -> bool:
        return (now or datetime.now(timezone.utc)) > self.expiry


def new_secret() -> str:
    """Generate a signing secret for a course. Keep it on the server only."""
    return secrets.token_urlsafe(32)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(secret: str, payload: bytes) -> str:
    return _b64(hmac.new(secret.encode("utf-8"), payload, sha256).digest())


def issue_token(
    secret: str,
    course_id: str,
    student_id: str,
    *,
    valid_days: int = 180,
    now: datetime | None = None,
) -> str:
    """Mint a token for one student on one course."""
    if not secret:
        raise TokenError("A signing secret is required.")
    if not course_id or not student_id:
        raise TokenError("Both course_id and student_id are required.")

    issued = now or datetime.now(timezone.utc)
    claims = {
        "c": course_id,
        "s": student_id,
        "e": (issued + timedelta(days=valid_days)).isoformat(timespec="seconds"),
    }
    payload = json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return f"{TOKEN_PREFIX}.{_b64(payload)}.{_sign(secret, payload)}"


def verify_token(
    secret: str,
    token: str,
    *,
    course_id: str | None = None,
    now: datetime | None = None,
) -> TokenClaims:
    """Verify a token and return its claims.

    Raises :class:`TokenError` for anything wrong. Signature comparison uses
    ``hmac.compare_digest`` so a wrong token cannot be found byte by byte from
    response timings.
    """
    if not token:
        raise TokenError("No token supplied.")

    parts = token.strip().split(".")
    if len(parts) != 3 or parts[0] != TOKEN_PREFIX:
        raise TokenError("This does not look like a PyAnalytica access token.")

    _, payload_b64, signature = parts
    try:
        payload = _unb64(payload_b64)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same answer
        raise TokenError("Token is malformed.") from exc

    if not hmac.compare_digest(_sign(secret, payload), signature):
        raise TokenError("Token signature does not match. It may be for a different course.")

    try:
        claims_raw = json.loads(payload)
        claims = TokenClaims(
            course_id=str(claims_raw["c"]),
            student_id=str(claims_raw["s"]),
            expires_at=str(claims_raw["e"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise TokenError("Token contents are unreadable.") from exc

    if course_id is not None and claims.course_id != course_id:
        raise TokenError(
            f"Token is for course {claims.course_id!r}, but this server serves "
            f"{course_id!r}."
        )

    if claims.expired(now=now):
        raise TokenError(f"Token expired on {claims.expires_at}.")

    return claims


def issue_for_roster(
    secret: str,
    course_id: str,
    student_ids: list[str],
    *,
    valid_days: int = 180,
) -> dict[str, str]:
    """Mint one token per student. Returns ``{student_id: token}``."""
    seen: set[str] = set()
    issued: dict[str, str] = {}
    for raw in student_ids:
        student_id = str(raw).strip()
        if not student_id or student_id in seen:
            continue
        seen.add(student_id)
        issued[student_id] = issue_token(
            secret, course_id, student_id, valid_days=valid_days
        )
    return issued
