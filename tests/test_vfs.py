"""Tests for the VFS-based long-term memory (Task 1).

Covers:
- Path traversal attack prevention (../, absolute paths, etc.)
- Normal read / write / list / exists / delete operations
- Edge cases (non-existent files, empty content, nested directories)
- MemoryConfig integration
"""

from __future__ import annotations

import pytest

from kappa.config import MemoryConfig
from kappa.memory.vfs import VFSManager


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture()
def vfs(tmp_path):
    """Create a VFSManager rooted in a temporary directory."""
    config = MemoryConfig(workspace_root="workspace")
    return VFSManager(config=config, base_dir=tmp_path)


# ── Path traversal prevention ──────────────────────────────────


class TestPathTraversalBlocking:
    """VFS must reject any path that escapes the workspace root."""

    def test_dotdot_simple(self, vfs):
        with pytest.raises(ValueError, match="traversal"):
            vfs.read("../secret.txt")

    def test_dotdot_nested(self, vfs):
        with pytest.raises(ValueError, match="traversal"):
            vfs.read("subdir/../../secret.txt")

    def test_dotdot_write(self, vfs):
        with pytest.raises(ValueError, match="traversal"):
            vfs.write("../escape.txt", "pwned")

    def test_dotdot_deep_escape(self, vfs):
        with pytest.raises(ValueError, match="traversal"):
            vfs.read("a/b/c/../../../../etc/passwd")

    def test_absolute_path_unix(self, vfs):
        with pytest.raises(ValueError, match="Absolute"):
            vfs.read("/etc/passwd")

    def test_absolute_path_windows(self, vfs):
        with pytest.raises(ValueError, match="Absolute"):
            vfs.read("C:\\Windows\\System32\\config")

    def test_empty_path(self, vfs):
        with pytest.raises(ValueError, match="empty"):
            vfs.read("")

    def test_whitespace_only_path(self, vfs):
        with pytest.raises(ValueError, match="empty"):
            vfs.read("   ")

    def test_dotdot_exists(self, vfs):
        with pytest.raises(ValueError, match="traversal"):
            vfs.exists("../secret.txt")

    def test_dotdot_delete(self, vfs):
        with pytest.raises(ValueError, match="traversal"):
            vfs.delete("../secret.txt")

    def test_dotdot_list(self, vfs):
        with pytest.raises(ValueError, match="traversal"):
            vfs.list("../../")


# ── Normal operations ──────────────────────────────────────────


class TestReadWrite:
    """Basic read and write operations within the workspace."""

    def test_write_and_read(self, vfs):
        vfs.write("hello.txt", "world")
        assert vfs.read("hello.txt") == "world"

    def test_read_nonexistent_returns_none(self, vfs):
        assert vfs.read("does_not_exist.md") is None

    def test_overwrite_existing(self, vfs):
        vfs.write("data.txt", "v1")
        vfs.write("data.txt", "v2")
        assert vfs.read("data.txt") == "v2"

    def test_write_empty_content(self, vfs):
        vfs.write("empty.txt", "")
        assert vfs.read("empty.txt") == ""

    def test_write_multiline(self, vfs):
        content = "line1\nline2\nline3"
        vfs.write("multi.txt", content)
        assert vfs.read("multi.txt") == content

    def test_write_unicode(self, vfs):
        content = "한글 테스트 🚀 日本語"
        vfs.write("unicode.txt", content)
        assert vfs.read("unicode.txt") == content

    def test_write_creates_nested_dirs(self, vfs):
        vfs.write("deep/nested/dir/file.md", "nested content")
        assert vfs.read("deep/nested/dir/file.md") == "nested content"


class TestList:
    """File listing within the workspace."""

    def test_list_empty_workspace(self, vfs):
        assert vfs.list() == []

    def test_list_flat_files(self, vfs):
        vfs.write("a.txt", "a")
        vfs.write("b.txt", "b")
        result = vfs.list()
        assert result == ["a.txt", "b.txt"]

    def test_list_nested_files(self, vfs):
        vfs.write("root.txt", "r")
        vfs.write("sub/deep.txt", "d")
        result = vfs.list()
        assert "root.txt" in result
        assert "sub/deep.txt" in result

    def test_list_subdirectory(self, vfs):
        vfs.write("outside.txt", "o")
        vfs.write("sub/inside.txt", "i")
        result = vfs.list("sub")
        assert result == ["sub/inside.txt"]

    def test_list_nonexistent_dir_returns_empty(self, vfs):
        assert vfs.list("ghost") == []

    def test_list_returns_sorted(self, vfs):
        vfs.write("c.txt", "c")
        vfs.write("a.txt", "a")
        vfs.write("b.txt", "b")
        assert vfs.list() == ["a.txt", "b.txt", "c.txt"]


class TestExists:
    """File existence checks."""

    def test_exists_true(self, vfs):
        vfs.write("present.txt", "here")
        assert vfs.exists("present.txt") is True

    def test_exists_false(self, vfs):
        assert vfs.exists("absent.txt") is False

    def test_exists_directory_not_file(self, vfs):
        vfs.write("dir/file.txt", "content")
        assert vfs.exists("dir") is False  # dir exists but is not a file


class TestDelete:
    """File deletion."""

    def test_delete_existing(self, vfs):
        vfs.write("doomed.txt", "bye")
        assert vfs.delete("doomed.txt") is True
        assert vfs.exists("doomed.txt") is False

    def test_delete_nonexistent(self, vfs):
        assert vfs.delete("ghost.txt") is False

    def test_delete_then_read_returns_none(self, vfs):
        vfs.write("temp.txt", "temp")
        vfs.delete("temp.txt")
        assert vfs.read("temp.txt") is None


# ── Config integration ──────────────────────────────────────────


class TestMemoryConfig:
    """MemoryConfig integration with VFSManager."""

    def test_custom_workspace_root(self, tmp_path):
        config = MemoryConfig(workspace_root="custom_ws")
        vfs = VFSManager(config=config, base_dir=tmp_path)
        assert vfs.root == (tmp_path / "custom_ws").resolve()

    def test_workspace_dir_created(self, tmp_path):
        config = MemoryConfig(workspace_root="auto_created")
        vfs = VFSManager(config=config, base_dir=tmp_path)
        assert vfs.root.is_dir()

    def test_default_config(self, tmp_path):
        vfs = VFSManager(base_dir=tmp_path)
        assert vfs.root.name == ".kappa_workspace"

    def test_auto_inject_files_default(self):
        config = MemoryConfig()
        assert "LEARNINGS.md" in config.auto_inject_files
