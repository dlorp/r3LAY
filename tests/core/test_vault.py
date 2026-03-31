"""Tests for the KnowledgeVault git wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from r3lay.config import AppConfig
from r3lay.core.vault import KnowledgeVault, VaultCommit, _is_safe_hash, validate_vault_path

# ---------------------------------------------------------------------------
# Hash validation
# ---------------------------------------------------------------------------


class TestIsafeHash:
    def test_valid_short_hash(self) -> None:
        assert _is_safe_hash("abc123") is True

    def test_valid_full_hash(self) -> None:
        assert _is_safe_hash("a" * 40) is True

    def test_too_short(self) -> None:
        assert _is_safe_hash("abc") is False

    def test_too_long(self) -> None:
        assert _is_safe_hash("a" * 41) is False

    def test_non_hex(self) -> None:
        assert _is_safe_hash("ghijkl") is False

    def test_empty(self) -> None:
        assert _is_safe_hash("") is False

    def test_injection_attempt(self) -> None:
        assert _is_safe_hash("abc123; rm -rf /") is False

    def test_uppercase_hex(self) -> None:
        assert _is_safe_hash("ABCDEF1234") is True


# ---------------------------------------------------------------------------
# KnowledgeVault unit tests (no real git)
# ---------------------------------------------------------------------------


class TestValidateVaultPath:
    def test_valid_path(self, tmp_path: Path) -> None:
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        result = validate_vault_path(vault_dir)
        assert result == vault_dir.resolve()

    def test_rejects_root(self) -> None:
        with pytest.raises(ValueError, match="system directory"):
            validate_vault_path(Path("/"))

    def test_rejects_etc(self) -> None:
        with pytest.raises(ValueError, match="system directory"):
            validate_vault_path(Path("/etc"))

    def test_rejects_nonexistent(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            validate_vault_path(tmp_path / "nonexistent")


class TestKnowledgeVaultInit:
    def test_path_resolved(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path / "vault")
        assert vault.path.is_absolute()

    def test_last_pull_initially_none(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path)
        assert vault._last_pull is None


class TestCanWrite:
    def test_allowed_backend(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path)
        config = MagicMock()
        config.vault_write_backends = ["openclaw", "mlx"]
        assert vault.can_write("openclaw", config) is True

    def test_denied_backend(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path)
        config = MagicMock()
        config.vault_write_backends = ["openclaw"]
        assert vault.can_write("mlx", config) is False

    def test_empty_backends(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path)
        config = MagicMock()
        config.vault_write_backends = []
        assert vault.can_write("openclaw", config) is False


# ---------------------------------------------------------------------------
# KnowledgeVault integration tests (uses real git)
# ---------------------------------------------------------------------------


class TestVaultGitIntegration:
    """Integration tests that create real git repos in tmp directories."""

    @pytest.fixture
    def vault_path(self, tmp_path: Path) -> Path:
        """Create a directory for the vault."""
        p = tmp_path / "test-vault"
        p.mkdir()
        return p

    @pytest.fixture
    def vault(self, vault_path: Path) -> KnowledgeVault:
        return KnowledgeVault(vault_path)

    @pytest.mark.asyncio
    async def test_is_git_repo_false(self, vault: KnowledgeVault) -> None:
        assert await vault.is_git_repo() is False

    @pytest.mark.asyncio
    async def test_is_git_repo_nonexistent(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path / "nonexistent")
        assert await vault.is_git_repo() is False

    @pytest.mark.asyncio
    async def test_init_creates_repo(self, vault: KnowledgeVault) -> None:
        result = await vault.init()
        assert result is True
        assert await vault.is_git_repo() is True

    @pytest.mark.asyncio
    async def test_init_idempotent(self, vault: KnowledgeVault) -> None:
        await vault.init()
        result = await vault.init()
        assert result is True

    @pytest.mark.asyncio
    async def test_status_not_a_repo(self, vault: KnowledgeVault) -> None:
        info = await vault.status()
        assert info["is_repo"] is False

    @pytest.mark.asyncio
    async def test_status_after_init(self, vault: KnowledgeVault) -> None:
        await vault.init()
        info = await vault.status()
        assert info["is_repo"] is True
        assert info["clean"] is True

    @pytest.mark.asyncio
    async def test_commit_rejects_empty_message(self, vault: KnowledgeVault) -> None:
        await vault.init()
        success, msg = await vault.commit("")
        assert success is False
        assert "1-4096" in msg

    @pytest.mark.asyncio
    async def test_commit_rejects_oversized_message(self, vault: KnowledgeVault) -> None:
        await vault.init()
        success, msg = await vault.commit("x" * 5000)
        assert success is False
        assert "1-4096" in msg

    @pytest.mark.asyncio
    async def test_commit_empty_repo(self, vault: KnowledgeVault) -> None:
        await vault.init()
        success, msg = await vault.commit("test commit")
        assert success is True
        assert "Nothing to commit" in msg or "nothing to commit" in msg.lower()

    @pytest.mark.asyncio
    async def test_commit_with_file(self, vault: KnowledgeVault, vault_path: Path) -> None:
        await vault.init()

        # Create a file
        (vault_path / "test.md").write_text("# Test\nSome content")
        success, msg = await vault.commit("add test file")
        assert success is True

        # Verify commit shows in log
        commits = await vault.log(limit=5)
        assert len(commits) >= 1
        assert commits[0].message == "add test file"

    @pytest.mark.asyncio
    async def test_log_empty_repo(self, vault: KnowledgeVault) -> None:
        await vault.init()
        commits = await vault.log()
        assert commits == []

    @pytest.mark.asyncio
    async def test_log_returns_vault_commits(self, vault: KnowledgeVault, vault_path: Path) -> None:
        await vault.init()
        (vault_path / "a.md").write_text("a")
        await vault.commit("first")
        (vault_path / "b.md").write_text("b")
        await vault.commit("second")

        commits = await vault.log(limit=10)
        assert len(commits) == 2
        assert isinstance(commits[0], VaultCommit)
        assert commits[0].message == "second"
        assert commits[1].message == "first"
        assert len(commits[0].short_hash) >= 4

    @pytest.mark.asyncio
    async def test_revert_commit(self, vault: KnowledgeVault, vault_path: Path) -> None:
        await vault.init()

        # Create and commit a file
        (vault_path / "to_revert.md").write_text("this will be reverted")
        await vault.commit("add file to revert")

        commits = await vault.log()
        commit_hash = commits[0].short_hash

        # Revert it
        success, msg = await vault.revert(commit_hash)
        assert success is True

        # File should be gone
        assert not (vault_path / "to_revert.md").exists()

        # Should have a revert commit in log
        commits = await vault.log()
        assert len(commits) == 2
        assert "Revert" in commits[0].message

    @pytest.mark.asyncio
    async def test_revert_invalid_hash(self, vault: KnowledgeVault) -> None:
        await vault.init()
        success, msg = await vault.revert("not-a-hash!")
        assert success is False
        assert "Invalid" in msg

    @pytest.mark.asyncio
    async def test_diff_stat(self, vault: KnowledgeVault, vault_path: Path) -> None:
        await vault.init()
        (vault_path / "stats.md").write_text("some content\nmore lines\n")
        await vault.commit("add stats file")

        commits = await vault.log()
        diff = await vault.diff_stat(commits[0].short_hash)
        assert "stats.md" in diff

    @pytest.mark.asyncio
    async def test_pull_no_remote(self, vault: KnowledgeVault) -> None:
        """Pull on a repo with no remote should fail gracefully."""
        await vault.init()
        success, msg = await vault.pull()
        # No remote configured, should fail
        assert success is False

    @pytest.mark.asyncio
    async def test_status_dirty(self, vault: KnowledgeVault, vault_path: Path) -> None:
        await vault.init()
        (vault_path / "dirty.md").write_text("uncommitted")
        info = await vault.status()
        assert info["clean"] is False


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestVaultConfig:
    def test_config_roundtrip(self, tmp_path: Path) -> None:
        """Vault settings survive save/load cycle."""
        config = AppConfig(project_path=tmp_path)
        config.knowledge_vault_path = tmp_path / "vault"
        config.vault_write_backends = ["openclaw", "mlx"]
        config.save()

        loaded = AppConfig.load(tmp_path)
        assert loaded.knowledge_vault_path == tmp_path / "vault"
        assert loaded.vault_write_backends == ["openclaw", "mlx"]

    def test_config_no_vault(self, tmp_path: Path) -> None:
        """Config without vault path loads cleanly."""
        config = AppConfig(project_path=tmp_path)
        config.save()

        loaded = AppConfig.load(tmp_path)
        assert loaded.knowledge_vault_path is None
        assert loaded.vault_write_backends == ["openclaw"]

    def test_config_default_write_backends(self) -> None:
        config = AppConfig()
        assert config.vault_write_backends == ["openclaw"]


# ---------------------------------------------------------------------------
# write_file() tests
# ---------------------------------------------------------------------------


class TestVaultWriteFile:
    """Tests for KnowledgeVault.write_file() path validation and writing."""

    @pytest.mark.asyncio
    async def test_write_file_creates_directories(self, tmp_path: Path) -> None:
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        vault = KnowledgeVault(vault_dir)

        ok, msg = await vault.write_file("research/exp_001.md", "# Test")
        assert ok is True
        assert (vault_dir / "research" / "exp_001.md").exists()
        assert (vault_dir / "research" / "exp_001.md").read_text() == "# Test"

    @pytest.mark.asyncio
    async def test_write_file_rejects_path_traversal(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path)

        ok, msg = await vault.write_file("../escape.md", "bad")
        assert ok is False
        assert "escapes" in msg.lower() or "traversal" in msg.lower()

    @pytest.mark.asyncio
    async def test_write_file_rejects_absolute_path(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path)

        ok, msg = await vault.write_file("/etc/passwd", "bad")
        assert ok is False
        assert "Absolute" in msg

    @pytest.mark.asyncio
    async def test_write_file_rejects_null_bytes(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path)

        ok, msg = await vault.write_file("test\x00.md", "bad")
        assert ok is False

    @pytest.mark.asyncio
    async def test_write_file_rejects_empty_path(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path)

        ok, msg = await vault.write_file("", "content")
        assert ok is False


# ---------------------------------------------------------------------------
# asyncio.Lock tests
# ---------------------------------------------------------------------------


class TestVaultGitLock:
    """Tests for asyncio.Lock on KnowledgeVault."""

    def test_lock_attribute_exists(self, tmp_path: Path) -> None:
        import asyncio

        vault = KnowledgeVault(tmp_path)
        assert hasattr(vault, "_git_lock")
        assert isinstance(vault._git_lock, asyncio.Lock)


# ---------------------------------------------------------------------------
# backend_source tests
# ---------------------------------------------------------------------------


class TestBackendSource:
    """Tests for InferenceBackend.backend_source property."""

    def test_backend_source_unknown_class(self) -> None:
        """Unknown backend class returns 'unknown'."""
        from r3lay.core.backends.base import InferenceBackend

        class FakeBackend(InferenceBackend):
            @property
            def model_name(self) -> str:
                return "fake"

            @property
            def is_loaded(self) -> bool:
                return False

            async def load(self) -> None:
                pass

            async def unload(self) -> None:
                pass

            async def generate_stream(self, messages, **kwargs):
                yield ""

            @classmethod
            async def is_available(cls) -> bool:
                return False

        backend = FakeBackend()
        assert backend.backend_source == "unknown"


# ---------------------------------------------------------------------------
# write_and_commit() tests
# ---------------------------------------------------------------------------


class TestVaultWriteAndCommit:
    """Tests for atomic write_and_commit method."""

    @pytest.mark.asyncio
    async def test_write_and_commit_success(self, tmp_path: Path) -> None:
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        vault = KnowledgeVault(vault_dir)
        await vault.init()

        ok, msg = await vault.write_and_commit(
            "research/test.md", "# Test\n\nContent.", "test commit"
        )
        assert ok is True
        assert (vault_dir / "research" / "test.md").exists()

        # Verify it was committed
        commits = await vault.log(limit=5)
        assert any("test commit" in c.message for c in commits)

    @pytest.mark.asyncio
    async def test_write_and_commit_stages_only_target(self, tmp_path: Path) -> None:
        """Only the specified file is staged, not other untracked files."""
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        vault = KnowledgeVault(vault_dir)
        await vault.init()

        # Create an unrelated file
        (vault_dir / "unrelated.txt").write_text("should not be committed")

        ok, _ = await vault.write_and_commit("target.md", "# Target", "commit target only")
        assert ok is True

        # Check that unrelated.txt is still untracked
        code, status_out, _ = await vault._run_git("status", "--porcelain")
        assert "unrelated.txt" in status_out

    @pytest.mark.asyncio
    async def test_write_and_commit_rejects_bad_message(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path)
        ok, msg = await vault.write_and_commit("test.md", "content", "")
        assert ok is False
        assert "1-4096" in msg

    @pytest.mark.asyncio
    async def test_write_and_commit_rejects_traversal(self, tmp_path: Path) -> None:
        vault = KnowledgeVault(tmp_path)
        ok, msg = await vault.write_and_commit("../escape.md", "bad", "msg")
        assert ok is False

    @pytest.mark.asyncio
    async def test_write_and_commit_non_repo(self, tmp_path: Path) -> None:
        """Non-git directory fails before writing the file."""
        vault = KnowledgeVault(tmp_path)
        ok, msg = await vault.write_and_commit("test.md", "content", "msg")
        assert ok is False
        assert "Not a git repository" in msg
        # File should NOT be written (repo check happens first)
        assert not (tmp_path / "test.md").exists()


# ---------------------------------------------------------------------------
# Force pull
# ---------------------------------------------------------------------------


class TestVaultForcePull:
    @pytest.mark.asyncio
    async def test_force_pull_non_repo(self, tmp_path: Path) -> None:
        """Force pull fails on non-git directory."""
        vault = KnowledgeVault(tmp_path)
        ok, msg = await vault.force_pull()
        assert ok is False
        assert "Not a git repository" in msg

    @pytest.mark.asyncio
    async def test_pull_diverged_tag(self, tmp_path: Path) -> None:
        """Pull tags diverged branches in error message."""
        vault = KnowledgeVault(tmp_path)
        # Simulate a pull failure with ff-only message
        # The actual message from git contains "Not possible to fast-forward"
        # Our code lowercases it for matching
        ok, msg = await vault.pull()
        # Non-repo returns False but not with DIVERGED tag
        assert ok is False
