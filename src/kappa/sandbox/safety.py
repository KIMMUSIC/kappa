"""AST-based path safety validator for host-executed code.

Scans generated Python code for file operations and validates that
all paths stay within allowed directories.  Also flags dangerous
patterns (os.system, subprocess, exec/eval) that could escape the
safety perimeter.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SafetyResult:
    """Outcome of a static safety scan."""

    safe: bool
    violations: list[str] = field(default_factory=list)


# Functions that perform destructive filesystem operations.
_DANGEROUS_CALLS: set[str] = {
    "os.system",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.removedirs",
    "os.rename",
    "os.renames",
    "os.replace",
    "shutil.rmtree",
    "shutil.move",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "subprocess.check_call",
    "subprocess.check_output",
}

# Built-in calls that allow arbitrary code execution.
_EXEC_BUILTINS: set[str] = {"exec", "eval", "compile", "__import__"}


def _resolve_norm(path: str, base: Path) -> str:
    """Resolve *path* against *base* and normalise case (Windows-safe)."""
    try:
        resolved = (base / path).resolve()
    except (OSError, ValueError):
        resolved = Path(path)
    return os.path.normcase(str(resolved))


def _is_under(candidate: str, allowed: list[str]) -> bool:
    """Return True if *candidate* starts with any allowed prefix."""
    return any(candidate.startswith(prefix) for prefix in allowed)


def _attr_chain(node: ast.expr) -> str | None:
    """Return dotted name for an Attribute/Name node, e.g. 'os.path.join'."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class SafetyValidator:
    """Validates generated code against allowed directories."""

    def __init__(
        self,
        allowed_dirs: list[str | Path],
        *,
        block_network: bool = True,
        block_subprocess: bool = True,
    ) -> None:
        self._allowed = [
            os.path.normcase(str(Path(d).resolve())) for d in allowed_dirs
        ]
        self._block_network = block_network
        self._block_subprocess = block_subprocess

    def validate(self, code: str) -> SafetyResult:
        """Parse *code* and return a SafetyResult."""
        violations: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Syntax errors will be caught at execution time; not a safety issue.
            return SafetyResult(safe=True)

        for node in ast.walk(tree):
            self._check_call(node, violations)
            self._check_open(node, violations)
            self._check_import(node, violations)

        return SafetyResult(safe=len(violations) == 0, violations=violations)

    # ── Visitors ──────────────────────────────────────────────────

    def _check_call(self, node: ast.AST, violations: list[str]) -> None:
        """Flag dangerous function calls."""
        if not isinstance(node, ast.Call):
            return

        name = _attr_chain(node.func) if isinstance(node.func, (ast.Attribute, ast.Name)) else None
        if name is None:
            return

        # exec / eval / compile
        if name in _EXEC_BUILTINS:
            violations.append(f"Dangerous built-in call: {name}()")

        # os.system, subprocess.*, shutil.rmtree, etc.
        if name in _DANGEROUS_CALLS:
            if self._block_subprocess or not name.startswith("subprocess"):
                violations.append(f"Dangerous call: {name}()")

    def _check_open(self, node: ast.AST, violations: list[str]) -> None:
        """Check open() / Path.write_text() etc. for out-of-bounds paths."""
        if not isinstance(node, ast.Call):
            return

        name = _attr_chain(node.func) if isinstance(node.func, (ast.Attribute, ast.Name)) else None

        # open("path", "w") — check first positional arg
        if name == "open" and node.args:
            self._check_path_arg(node.args[0], node, violations)

    def _check_path_arg(
        self, arg: ast.expr, call_node: ast.Call, violations: list[str]
    ) -> None:
        """If *arg* is a string literal, verify it's inside allowed dirs."""
        path_str = self._extract_string(arg)
        if path_str is None:
            return  # Dynamic path — can't validate statically

        # Resolve relative to CWD (which will be output_dir at execution time)
        norm = _resolve_norm(path_str, Path.cwd())
        if not _is_under(norm, self._allowed):
            violations.append(
                f"File access outside allowed directories: {path_str}"
            )

    def _check_import(self, node: ast.AST, violations: list[str]) -> None:
        """Flag imports of networking / subprocess modules when blocked."""
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif node.module:
                names = [node.module]

            for mod in names:
                if self._block_subprocess and mod.startswith("subprocess"):
                    violations.append(f"Blocked import: {mod}")
                if self._block_network and mod in ("socket", "http", "urllib", "requests", "httpx", "aiohttp"):
                    violations.append(f"Blocked network import: {mod}")

    @staticmethod
    def _extract_string(node: ast.expr) -> str | None:
        """Extract a constant string value from an AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        # f-strings and concatenations can't be resolved statically
        return None
