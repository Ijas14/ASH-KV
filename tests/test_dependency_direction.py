"""Enforce cold/hot boundary at the file level.

This is the architectural backbone. If it passes, the cold/hot
separation is real. If it fails, someone has leaked flexibility
into the hot path.

Rules:
- contracts/   may import only stdlib + numpy
- runtime/     may import only contracts/ + stdlib + numpy
- codecs/      may import contracts/ + external (triton, torch)
- compiler/    may import contracts/, codecs/, plugins/, config/
- safety/      may import contracts/, runtime/, codecs/
- telemetry/   may import only contracts/
- sglang_integration/  may import everything

The runtime/ may NEVER import compiler/, codecs/, config/, safety/,
telemetry/, or sglang_integration/. If it does, the cold/hot
boundary is broken.
"""
from __future__ import annotations

import ast
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ASHKV_ROOT = REPO_ROOT / "ashkv"

PREFIXES: dict[str, pathlib.Path] = {
    "contracts": ASHKV_ROOT / "contracts",
    "runtime": ASHKV_ROOT / "runtime",
    "compiler": ASHKV_ROOT / "compiler",
    "codecs": ASHKV_ROOT / "codecs",
    "safety": ASHKV_ROOT / "safety",
    "telemetry": ASHKV_ROOT / "telemetry",
    "sglang_integration": ASHKV_ROOT / "sglang_integration",
    "plugins": ASHKV_ROOT / "plugins",
    "config": ASHKV_ROOT / "config",
}

ALLOWED_INTERNAL_IMPORTS: dict[str, set[str]] = {
    "contracts": set(),
    "runtime": {"contracts"},
    "codecs": {"contracts"},
    # compiler binds runtime functions into closures; this is the
    # one allowed downward dependency from compiler into runtime.
    "compiler": {"contracts", "runtime", "codecs", "plugins", "config"},
    "safety": {"contracts", "runtime", "codecs"},
    "telemetry": {"contracts"},
    "sglang_integration": {
        "contracts", "runtime", "codecs", "safety",
        "compiler", "telemetry", "config",
    },
    "plugins": {"contracts"},
    "config": set(),
}

ISOLATED_MODULES: set[str] = {"sglang_integration"}

ALLOWED_EXTERNAL: dict[str, set[str]] = {
    "contracts": {"numpy", "typing", "dataclasses", "enum", "math", "__future__"},
    "runtime": {"numpy", "typing", "dataclasses", "enum", "math", "time", "__future__"},
    "codecs": set(),
    "compiler": {"numpy", "typing", "dataclasses", "enum", "math", "__future__"},
    "safety": {"numpy", "typing", "dataclasses", "enum", "math", "time", "__future__"},
    "telemetry": {"numpy", "typing", "dataclasses", "enum", "math", "time", "__future__"},
    "sglang_integration": set(),
    "plugins": {"numpy", "typing", "dataclasses", "enum", "math", "__future__"},
    "config": set(),
}


def _module_prefix_of(filepath: pathlib.Path) -> str | None:
    try:
        rel = filepath.relative_to(ASHKV_ROOT)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    return parts[0]


def _extract_imports(filepath: pathlib.Path) -> tuple[set[str], set[str]]:
    """Return (internal_imports, external_imports).

    Internal = ashkv.* top-level package names.
    External = top-level module names of non-ashkv imports.
    """
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return set(), set()

    internal: set[str] = set()
    external: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top == "ashkv":
                    parts = alias.name.split(".")
                    if len(parts) >= 2:
                        internal.add(parts[1])
                else:
                    external.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is None:
                    continue
                top = node.module.split(".")[0]
                if top == "ashkv":
                    parts = node.module.split(".")
                    if len(parts) >= 2:
                        internal.add(parts[1])
                else:
                    external.add(top)
            else:
                if _module_prefix_of(filepath) is None:
                    continue
                if node.level >= 2:
                    if node.module:
                        internal.add(node.module.split(".")[0])
                    elif node.names:
                        for alias in node.names:
                            internal.add(alias.name)
    return internal, external


def _all_python_files(module_prefix: str) -> list[pathlib.Path]:
    root = PREFIXES.get(module_prefix)
    if root is None or not root.exists():
        return []
    return [p for p in root.rglob("*.py") if p.name != "__init__.py"]


@pytest.mark.parametrize(
    "module_prefix",
    [p for p in PREFIXES if (PREFIXES[p] / "__init__.py").exists()],
)
def test_internal_dependency_direction(module_prefix: str) -> None:
    """No file in `module_prefix` may import outside its allowed set."""
    files = _all_python_files(module_prefix)
    if not files:
        pytest.skip(f"No files in {module_prefix}/")

    allowed = ALLOWED_INTERNAL_IMPORTS.get(module_prefix, set())
    isolated = ISOLATED_MODULES

    violations: list[str] = []
    for f in files:
        internal, _ = _extract_imports(f)
        internal.discard(module_prefix)
        for imp in internal:
            if imp in isolated and imp != module_prefix:
                violations.append(
                    f"{f.relative_to(REPO_ROOT)}: imports isolated module "
                    f"ashkv.{imp}"
                )
                continue
            if imp not in ALLOWED_INTERNAL_IMPORTS:
                continue
            if imp not in allowed:
                violations.append(
                    f"{f.relative_to(REPO_ROOT)}: imports ashkv.{imp} "
                    f"(allowed: {sorted(allowed) or 'stdlib only'})"
                )
    assert not violations, (
        "Dependency direction violations:\n  " + "\n  ".join(violations)
    )


def test_runtime_imports_no_external_bloat() -> None:
    """runtime/ may import only stdlib + numpy + contracts."""
    files = _all_python_files("runtime")
    if not files:
        pytest.skip("No runtime files yet")
    allowed = ALLOWED_EXTERNAL["runtime"]
    violations: list[str] = []
    for f in files:
        _, external = _extract_imports(f)
        for ext in external:
            if ext not in allowed:
                violations.append(
                    f"{f.relative_to(REPO_ROOT)}: imports external '{ext}' "
                    f"(allowed: {sorted(allowed)})"
                )
    assert not violations, (
        "runtime/ external import violations:\n  " + "\n  ".join(violations)
    )


def test_contracts_imports_minimal() -> None:
    """contracts/ may import only stdlib + numpy."""
    files = _all_python_files("contracts")
    if not files:
        pytest.skip("No contracts files")
    allowed = ALLOWED_EXTERNAL["contracts"]
    violations: list[str] = []
    for f in files:
        _, external = _extract_imports(f)
        for ext in external:
            if ext not in allowed:
                violations.append(
                    f"{f.relative_to(REPO_ROOT)}: imports external '{ext}' "
                    f"(allowed: {sorted(allowed)})"
                )
    assert not violations, (
        "contracts/ external import violations:\n  " + "\n  ".join(violations)
    )


def test_no_runtime_imports_compiler() -> None:
    """The hot path may never reach into the compiler."""
    files = _all_python_files("runtime")
    if not files:
        pytest.skip("No runtime files yet")
    violations: list[str] = []
    for f in files:
        internal, _ = _extract_imports(f)
        forbidden = {"compiler", "config", "safety", "telemetry",
                     "sglang_integration", "codecs", "plugins"}
        for imp in internal:
            if imp in forbidden:
                violations.append(
                    f"{f.relative_to(REPO_ROOT)}: runtime/ imports ashkv.{imp} "
                    f"(forbidden in runtime)"
                )
    assert not violations, (
        "runtime/ forbidden imports:\n  " + "\n  ".join(violations)
    )


def test_contracts_do_not_import_other_ashkv_modules() -> None:
    """contracts/ is the foundation. It imports nothing from ashkv."""
    files = _all_python_files("contracts")
    violations: list[str] = []
    for f in files:
        internal, _ = _extract_imports(f)
        for imp in internal:
            violations.append(
                f"{f.relative_to(REPO_ROOT)}: contracts/ imports ashkv.{imp} "
                f"(contracts must be self-contained)"
            )
    assert not violations, (
        "contracts/ self-containment violations:\n  " + "\n  ".join(violations)
    )
