"""Map what PyAnalytica actually depends on, versus what it declares.

Three questions this answers, each of which has bitten us:

1. **What do we import but not declare?** A package that arrives only because
   something else pulls it in works until that something else drops it. The
   tutor server imports starlette and uvicorn directly; both currently arrive
   via shiny.

2. **What do we declare but never import?** Dead weight in the install, and
   100 MB already takes students a few minutes.

3. **What version actually resolves?** Floors like ``shiny>=1.0`` mean a
   student gets whatever is current while a developer keeps whatever they
   installed months ago. That gap is why a Shiny deprecation warning reached a
   student before it reached us.

Run inside the project's virtual environment::

    .venv/Scripts/python.exe scripts/audit_dependencies.py
"""

from __future__ import annotations

import ast
import importlib.metadata as md
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "pyanalytica"

# Modules in the standard library or our own package — never third-party.
LOCAL = {"pyanalytica"}


def declared() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Read runtime and optional requirements out of pyproject.toml."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]

    def split(spec: str) -> tuple[str, str]:
        for sep in (">=", "==", "~=", ">", "<"):
            if sep in spec:
                name, _, rest = spec.partition(sep)
                return name.strip(), sep + rest.strip()
        return spec.strip(), ""

    runtime = dict(split(s) for s in project.get("dependencies", []))
    extras = {
        name: dict(split(s) for s in specs)
        for name, specs in project.get("optional-dependencies", {}).items()
    }
    return runtime, extras


def imports_by_module() -> dict[str, set[Path]]:
    """Every top-level module imported anywhere under src/, and where."""
    found: dict[str, set[Path]] = defaultdict(set)

    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found[alias.name.split(".")[0]].add(path)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    found[node.module.split(".")[0]].add(path)

    stdlib = set(sys.stdlib_module_names)
    return {
        name: files
        for name, files in found.items()
        if name not in stdlib and name not in LOCAL and not name.startswith("_")
    }


def norm(name: str) -> str:
    """Canonical form for comparing distribution names (PEP 503)."""
    return name.lower().replace("_", "-").replace(".", "-")


def distribution_for(module: str) -> str | None:
    """Which installed distribution provides this importable module."""
    mapping = md.packages_distributions()
    names = mapping.get(module)
    return names[0] if names else None


def installed(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def requirements_of(name: str) -> list[str]:
    try:
        return md.requires(name) or []
    except md.PackageNotFoundError:
        return []


def main() -> int:
    runtime, extras = declared()
    used = imports_by_module()
    optional_all = {k: v for spec in extras.values() for k, v in spec.items()}

    def rel(p: Path) -> str:
        return str(p.relative_to(ROOT)).replace("\\", "/")

    print("=" * 78)
    print("DECLARED RUNTIME DEPENDENCIES")
    print("=" * 78)
    print(f"{'package':16} {'floor':12} {'resolved':12} note")
    for name, floor in sorted(runtime.items()):
        version = installed(name) or "-"
        note = ""
        if floor.startswith(">=") and version != "-":
            want = floor[2:].split(".")[0]
            have = version.split(".")[0]
            if want != have:
                note = f"major {want} -> {have}: a student's version differs from the floor"
        print(f"{name:16} {floor or '(any)':12} {version:12} {note}")

    print()
    print("=" * 78)
    print("OPTIONAL EXTRAS")
    print("=" * 78)
    for extra, specs in sorted(extras.items()):
        installed_names = [f"{n}{'' if installed(n) else ' (absent)'}" for n in sorted(specs)]
        print(f"  [{extra}]  {', '.join(installed_names)}")

    print()
    print("=" * 78)
    print("IMPORTED BUT NOT DECLARED")
    print("=" * 78)
    print("These work today only because something else installs them.")
    print()
    undeclared = []
    for module, files in sorted(used.items()):
        dist = distribution_for(module) or module
        declared_names = {norm(n) for n in list(runtime) + list(optional_all)}
        if norm(dist) in declared_names or norm(module) in declared_names:
            continue
        undeclared.append((module, dist, files))

    if not undeclared:
        print("  none")
    for module, dist, files in undeclared:
        version = installed(dist) or "?"
        pullers = [d for d in runtime if any(dist in r for r in requirements_of(d))]
        via = f"arrives via {', '.join(pullers)}" if pullers else "source unclear"
        print(f"  {module:12} (dist {dist} {version}) — {via}")
        for f in sorted(files)[:4]:
            print(f"       {rel(f)}")

    print()
    print("=" * 78)
    print("DECLARED BUT NEVER IMPORTED")
    print("=" * 78)
    modules_used = set(used)
    dists_used = {distribution_for(m) or m for m in modules_used}
    used_norm = {norm(m) for m in modules_used} | {norm(d) for d in dists_used if d}
    unused = [n for n in sorted(runtime) if norm(n) not in used_norm]
    if not unused:
        print("  none")
    for name in unused:
        # Some are needed at runtime by another library rather than by us:
        # pandas reaches for openpyxl to read .xlsx, for instance.
        needed_by = [d for d in runtime if any(norm(name) in norm(r) for r in requirements_of(d))]
        why = f"required at runtime by {', '.join(needed_by)}" if needed_by else "no importer found — candidate for removal"
        print(f"  {name:16} {why}")

    print()
    print("=" * 78)
    print("WHAT A STUDENT ACTUALLY INSTALLS")
    print("=" * 78)
    try:
        dist_names = sorted({d.metadata["Name"] for d in md.distributions() if d.metadata["Name"]})
        print(f"  {len(dist_names)} distributions in this environment")
    except Exception:  # pragma: no cover - metadata quirks
        pass
    for name in sorted(runtime):
        # Skip requirements gated behind an extra — a plain install skips them.
        reqs = [
            r.split(";")[0].split(" ")[0]
            for r in requirements_of(name)
            if "extra ==" not in r
        ]
        reqs = sorted({r for r in reqs if r})
        print(f"  {name:16} pulls {len(reqs):>2}: {', '.join(reqs[:8])}{' …' if len(reqs) > 8 else ''}")

    print()
    print("=" * 78)
    print("PYTHON VERSIONS")
    print("=" * 78)
    import tomllib as _t
    data = _t.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    print(f"  requires-python : {data['project'].get('requires-python', '(unset)')}")
    print(f"  running here    : {sys.version.split()[0]}")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    import re
    m = re.search(r'python-version:\s*\[([^\]]+)\]', ci)
    print(f"  tested in CI    : {m.group(1) if m else '(not found)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
