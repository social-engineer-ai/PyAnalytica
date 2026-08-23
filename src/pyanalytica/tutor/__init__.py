"""Tutor: a course-scoped AI assistant that keeps the API key off students' machines.

The instructor runs a small proxy holding their API key and their course pack;
students' copies of PyAnalytica hold only a signed access token and the
server's address. This is the only arrangement that actually works: a key
inside software running on someone else's computer cannot be kept secret, so
it is never sent there.
"""

from pyanalytica.tutor.pack import (
    CoursePack,
    CoursePackError,
    Limits,
    example_pack_yaml,
    load_pack,
    parse_pack,
)
from pyanalytica.tutor.tokens import (
    TokenClaims,
    TokenError,
    issue_for_roster,
    issue_token,
    new_secret,
    verify_token,
)
from pyanalytica.tutor.usage import CapExceeded, UsageStore, estimate_cost

__all__ = [
    "CapExceeded",
    "CoursePack",
    "CoursePackError",
    "Limits",
    "TokenClaims",
    "TokenError",
    "UsageStore",
    "estimate_cost",
    "example_pack_yaml",
    "issue_for_roster",
    "issue_token",
    "load_pack",
    "new_secret",
    "parse_pack",
    "verify_token",
]
