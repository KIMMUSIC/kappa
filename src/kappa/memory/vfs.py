"""Virtual File System manager for isolated agent memory.

Provides a sandboxed workspace directory where the agent can persist
and retrieve knowledge (e.g. LEARNINGS.md).  All file operations are
confined to the workspace root — path traversal attacks are rejected.
"""

from __future__ import annotations

import os
from pathlib import Path

from kappa.config import MemoryConfig


class VFSManager:
    """Manages an isolated virtual file system rooted at a fixed workspace.

    Guarantees:
    - All file operations are confined to ``workspace_root``.
    - Path traversal (``../``, symlink escape) is blocked with ``ValueError``.
    - Parent directories are created automatically on write.
    - Read of a non-existent file returns ``None`` (not an exception).

    Args:
        config: MemoryConfig with workspace_root path.
        base_dir: Anchor directory that ``workspace_root`` is resolved
            relative to.  Defaults to the current working directory.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        cfg = config or MemoryConfig()
        anchor = Path(base_dir) if base_dir else Path.cwd()
        self._root = (anchor / cfg.workspace_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Absolute path to the workspace root directory."""
        return self._root

    # ── Path safety ─────────────────────────────────────────────

    def _safe_path(self, relative: str) -> Path:
        """Resolve *relative* under the workspace root.

        Raises:
            ValueError: If the resolved path escapes the workspace root,
                or if ``relative`` is empty / absolute.
        """
        if not relative or not relative.strip():
            raise ValueError("Path must not be empty.")

        # Reject absolute paths on any platform
        if os.path.isabs(relative):
            raise ValueError(
                f"Absolute paths are not allowed: {relative!r}"
            )

        # Normalise separators and resolve
        target = (self._root / relative).resolve()

        # Ensure the resolved path is inside the workspace root
        try:
            target.relative_to(self._root)
        except ValueError:
            raise ValueError(
                f"Path traversal blocked: {relative!r} escapes workspace root."
            ) from None

        return target

    # ── Public API ──────────────────────────────────────────────

    def read(self, path: str) -> str | None:
        """Read file content from the workspace.

        Returns:
            File content as a string, or ``None`` if the file does not exist.

        Raises:
            ValueError: If *path* escapes the workspace root.
        """
        target = self._safe_path(path)
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8")

    def write(self, path: str, content: str) -> None:
        """Write content to a file in the workspace.

        Parent directories are created automatically.

        Raises:
            ValueError: If *path* escapes the workspace root.
        """
        target = self._safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def list(self, subdir: str = ".") -> list[str]:
        """List files in the workspace (relative paths).

        Args:
            subdir: Sub-directory to list.  Defaults to the workspace root.

        Returns:
            Sorted list of relative POSIX-style file paths.

        Raises:
            ValueError: If *subdir* escapes the workspace root.
        """
        target = self._safe_path(subdir) if subdir != "." else self._root
        if not target.is_dir():
            return []
        return sorted(
            str(p.relative_to(self._root).as_posix())
            for p in target.rglob("*")
            if p.is_file()
        )

    def exists(self, path: str) -> bool:
        """Check whether a file exists in the workspace.

        Raises:
            ValueError: If *path* escapes the workspace root.
        """
        return self._safe_path(path).is_file()

    def delete(self, path: str) -> bool:
        """Delete a file from the workspace.

        Returns:
            ``True`` if the file was deleted, ``False`` if it didn't exist.

        Raises:
            ValueError: If *path* escapes the workspace root.
        """
        target = self._safe_path(path)
        if not target.is_file():
            return False
        target.unlink()
        return True
