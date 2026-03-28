"""Comprehensive tests for r3lay.core.reranker and r3lay.core.reranker_worker modules.

Tests cover:
- RerankResult: dataclass creation and fields
- CrossEncoderReranker: init, is_loaded, load, rerank, unload
- Worker protocol: JSON message format expectations
- reranker_worker: setup_isolation, send_response, main loop command handling

All tests use mocks for heavy dependencies (sentence-transformers, subprocess)
to ensure fast, isolated testing without requiring actual models.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from r3lay.core.reranker import (
    DEFAULT_MODEL,
    MODEL_LOAD_TIMEOUT,
    RERANK_TIMEOUT,
    CrossEncoderReranker,
    RerankResult,
)

# =============================================================================
# RerankResult Tests
# =============================================================================


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a RerankResult with all fields."""
        result = RerankResult(index=0, score=0.95, passage="Python is a language.")
        assert result.index == 0
        assert result.score == 0.95
        assert result.passage == "Python is a language."

    def test_fields_types(self) -> None:
        """Test field types are preserved correctly."""
        result = RerankResult(index=3, score=0.123456789, passage="")
        assert isinstance(result.index, int)
        assert isinstance(result.score, float)
        assert isinstance(result.passage, str)

    def test_negative_score(self) -> None:
        """Test that negative scores are allowed (cross-encoders can produce them)."""
        result = RerankResult(index=1, score=-2.5, passage="irrelevant passage")
        assert result.score == -2.5

    def test_equality(self) -> None:
        """Test dataclass equality."""
        a = RerankResult(index=0, score=0.9, passage="text")
        b = RerankResult(index=0, score=0.9, passage="text")
        assert a == b

    def test_inequality(self) -> None:
        """Test dataclass inequality on different fields."""
        base = RerankResult(index=0, score=0.9, passage="text")
        assert base != RerankResult(index=1, score=0.9, passage="text")
        assert base != RerankResult(index=0, score=0.5, passage="text")
        assert base != RerankResult(index=0, score=0.9, passage="other")


# =============================================================================
# CrossEncoderReranker Initialization Tests
# =============================================================================


class TestCrossEncoderRerankerInit:
    """Tests for CrossEncoderReranker initialization."""

    def test_default_model(self) -> None:
        """Test default model name is set correctly."""
        reranker = CrossEncoderReranker()
        assert reranker.model_name == DEFAULT_MODEL
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_custom_model(self) -> None:
        """Test custom model name is preserved."""
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
        assert reranker.model_name == "cross-encoder/ms-marco-TinyBERT-L-2-v2"

    def test_initial_state_not_loaded(self) -> None:
        """Test initial state: not loaded, no process."""
        reranker = CrossEncoderReranker()
        assert not reranker.is_loaded
        assert reranker._process is None
        assert reranker._read_buffer == b""


# =============================================================================
# CrossEncoderReranker.is_loaded Tests
# =============================================================================


class TestCrossEncoderRerankerIsLoaded:
    """Tests for CrossEncoderReranker.is_loaded property."""

    def test_false_when_no_process(self) -> None:
        """Test is_loaded is False when process is None."""
        reranker = CrossEncoderReranker()
        assert not reranker.is_loaded

    def test_true_when_process_running(self) -> None:
        """Test is_loaded is True when process exists and has no returncode."""
        reranker = CrossEncoderReranker()
        mock_process = MagicMock()
        mock_process.returncode = None
        reranker._process = mock_process
        assert reranker.is_loaded
        reranker._process = None

    def test_false_when_process_exited(self) -> None:
        """Test is_loaded is False when process has exited."""
        reranker = CrossEncoderReranker()
        mock_process = MagicMock()
        mock_process.returncode = 0
        reranker._process = mock_process
        assert not reranker.is_loaded
        reranker._process = None


# =============================================================================
# CrossEncoderReranker.load() Tests
# =============================================================================


class TestCrossEncoderRerankerLoad:
    """Tests for CrossEncoderReranker.load()."""

    @pytest.mark.asyncio
    async def test_load_success(self) -> None:
        """Test successful model loading via mocked subprocess."""
        reranker = CrossEncoderReranker()

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = MagicMock()

        load_response = (
            json.dumps({"type": "loaded", "success": True, "model": DEFAULT_MODEL}).encode() + b"\n"
        )
        mock_process.stdout.read = AsyncMock(return_value=load_response)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await reranker.load()

        assert reranker.is_loaded
        # Cleanup
        reranker._process = None

    @pytest.mark.asyncio
    async def test_load_failure_error_response(self) -> None:
        """Test load failure when worker responds with success=False."""
        reranker = CrossEncoderReranker()

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdout = MagicMock()

        error_response = (
            json.dumps(
                {
                    "type": "loaded",
                    "success": False,
                    "error": "sentence-transformers not installed",
                }
            ).encode()
            + b"\n"
        )
        mock_process.stdout.read = AsyncMock(return_value=error_response)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError, match="Failed to load cross-encoder model"):
                await reranker.load()

    @pytest.mark.asyncio
    async def test_load_timeout(self) -> None:
        """Test load timeout when worker never responds."""
        reranker = CrossEncoderReranker()

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        # stdout.read always times out (returns no data)
        mock_process.stdout.read = AsyncMock(side_effect=asyncio.TimeoutError())

        # time.monotonic is called multiple times: once for start_time, then in loop.
        # Use a callable that returns 0.0 first, then always returns past-timeout.
        call_count = 0

        def fake_monotonic() -> float:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 0.0
            return float(MODEL_LOAD_TIMEOUT + 1)

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
            patch("r3lay.core.reranker.time") as mock_time_mod,
        ):
            mock_time_mod.monotonic = fake_monotonic
            with pytest.raises(RuntimeError, match="Timeout loading cross-encoder model"):
                await reranker.load()

    @pytest.mark.asyncio
    async def test_load_process_dies(self) -> None:
        """Test load failure when worker process dies during load."""
        reranker = CrossEncoderReranker()

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.read = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.wait = AsyncMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()

        # Process is dead from the start (non-None returncode)
        type(mock_process).returncode = PropertyMock(return_value=1)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError):
                await reranker.load()


# =============================================================================
# CrossEncoderReranker.rerank() Tests
# =============================================================================


class TestCrossEncoderRerankerRerank:
    """Tests for CrossEncoderReranker.rerank()."""

    def _make_loaded_reranker(self) -> tuple[CrossEncoderReranker, MagicMock]:
        """Helper: create a reranker with a mocked running process."""
        reranker = CrossEncoderReranker()
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = MagicMock()
        reranker._process = mock_process
        return reranker, mock_process

    @pytest.mark.asyncio
    async def test_rerank_not_loaded_raises(self) -> None:
        """Test rerank raises RuntimeError when model not loaded."""
        reranker = CrossEncoderReranker()
        with pytest.raises(RuntimeError, match="not loaded"):
            await reranker.rerank("query", ["passage"])

    @pytest.mark.asyncio
    async def test_rerank_empty_passages(self) -> None:
        """Test rerank with empty passages returns empty list."""
        reranker, _ = self._make_loaded_reranker()
        result = await reranker.rerank("query", [])
        assert result == []
        reranker._process = None

    @pytest.mark.asyncio
    async def test_rerank_success(self) -> None:
        """Test successful reranking with mocked worker results."""
        reranker, mock_process = self._make_loaded_reranker()

        rerank_response = (
            json.dumps(
                {
                    "type": "reranked",
                    "results": [
                        {"index": 0, "score": 0.95, "passage": "Python is a programming language."},
                        {"index": 2, "score": 0.72, "passage": "Python was created by Guido."},
                    ],
                }
            ).encode()
            + b"\n"
        )
        mock_process.stdout.read = AsyncMock(return_value=rerank_response)

        passages = [
            "Python is a programming language.",
            "Snakes are reptiles.",
            "Python was created by Guido.",
        ]
        results = await reranker.rerank("What is Python?", passages)

        assert len(results) == 2
        assert isinstance(results[0], RerankResult)
        assert results[0].index == 0
        assert results[0].score == 0.95
        assert results[0].passage == "Python is a programming language."
        assert results[1].index == 2
        assert results[1].score == 0.72
        reranker._process = None

    @pytest.mark.asyncio
    async def test_rerank_error_response(self) -> None:
        """Test rerank raises when worker returns an error."""
        reranker, mock_process = self._make_loaded_reranker()

        error_response = (
            json.dumps({"type": "error", "message": "Out of memory during scoring"}).encode()
            + b"\n"
        )
        mock_process.stdout.read = AsyncMock(return_value=error_response)

        with pytest.raises(RuntimeError, match="Reranking error: Out of memory"):
            await reranker.rerank("query", ["passage"])
        reranker._process = None

    @pytest.mark.asyncio
    async def test_rerank_timeout(self) -> None:
        """Test rerank timeout when worker never responds."""
        reranker, mock_process = self._make_loaded_reranker()

        mock_process.stdout.read = AsyncMock(side_effect=asyncio.TimeoutError())

        call_count = 0

        def fake_monotonic() -> float:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 0.0
            return float(RERANK_TIMEOUT + 1)

        with patch("r3lay.core.reranker.time") as mock_time_mod:
            mock_time_mod.monotonic = fake_monotonic
            with pytest.raises(RuntimeError, match="Timeout waiting for reranking"):
                await reranker.rerank("query", ["passage"])
        reranker._process = None

    @pytest.mark.asyncio
    async def test_rerank_process_dies(self) -> None:
        """Test rerank raises when worker process dies during reranking."""
        reranker, mock_process = self._make_loaded_reranker()

        mock_process.stdout.read = AsyncMock(side_effect=asyncio.TimeoutError())

        # Process alive initially (is_loaded check), then dies during read loop
        return_codes = [None, None, 1, 1, 1]
        rc_idx = [0]

        def get_rc() -> int | None:
            code = return_codes[min(rc_idx[0], len(return_codes) - 1)]
            rc_idx[0] += 1
            return code

        type(mock_process).returncode = PropertyMock(side_effect=get_rc)

        with patch("r3lay.core.reranker.time") as mock_time_mod:
            mock_time_mod.monotonic = MagicMock(return_value=1.0)
            with pytest.raises(RuntimeError, match="Worker process died during reranking"):
                await reranker.rerank("query", ["passage"])
        reranker._process = None

    @pytest.mark.asyncio
    async def test_rerank_sends_correct_command(self) -> None:
        """Test that rerank sends the correct JSON command to the worker."""
        reranker, mock_process = self._make_loaded_reranker()

        rerank_response = json.dumps({"type": "reranked", "results": []}).encode() + b"\n"
        mock_process.stdout.read = AsyncMock(return_value=rerank_response)

        await reranker.rerank("my query", ["p1", "p2"], top_k=5, threshold=0.5)

        # Verify the command written to stdin
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        parsed = json.loads(written_data.decode().strip())
        assert parsed["cmd"] == "rerank"
        assert parsed["query"] == "my query"
        assert parsed["passages"] == ["p1", "p2"]
        assert parsed["top_k"] == 5
        assert parsed["threshold"] == 0.5
        reranker._process = None


# =============================================================================
# CrossEncoderReranker.unload() Tests
# =============================================================================


class TestCrossEncoderRerankerUnload:
    """Tests for CrossEncoderReranker.unload()."""

    @pytest.mark.asyncio
    async def test_unload_when_not_loaded(self) -> None:
        """Test unload is a no-op when process is None."""
        reranker = CrossEncoderReranker()
        # Should not raise
        await reranker.unload()
        assert not reranker.is_loaded

    @pytest.mark.asyncio
    async def test_unload_clean_shutdown(self) -> None:
        """Test clean unload when worker exits gracefully."""
        reranker = CrossEncoderReranker()

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.wait = AsyncMock()

        # Process is alive, then exits after unload command
        return_codes = [None, None, 0]
        rc_idx = [0]

        def get_rc() -> int | None:
            code = return_codes[min(rc_idx[0], len(return_codes) - 1)]
            rc_idx[0] += 1
            return code

        type(mock_process).returncode = PropertyMock(side_effect=get_rc)

        reranker._process = mock_process
        await reranker.unload()

        assert reranker._process is None
        assert reranker._read_buffer == b""

    @pytest.mark.asyncio
    async def test_unload_force_terminate_on_timeout(self) -> None:
        """Test force termination when graceful shutdown times out."""
        reranker = CrossEncoderReranker()

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()

        # wait() always times out, process never exits
        mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError())
        type(mock_process).returncode = PropertyMock(return_value=None)

        reranker._process = mock_process
        await reranker.unload()

        mock_process.terminate.assert_called()
        assert reranker._process is None

    @pytest.mark.asyncio
    async def test_unload_already_dead_process(self) -> None:
        """Test unload when process already exited."""
        reranker = CrossEncoderReranker()

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock(side_effect=BrokenPipeError())
        mock_process.stdin.close = MagicMock()
        mock_process.stdout = MagicMock()

        reranker._process = mock_process
        await reranker.unload()

        assert reranker._process is None


# =============================================================================
# Worker Protocol Tests
# =============================================================================


class TestWorkerProtocol:
    """Tests verifying the expected JSON message format between reranker and worker."""

    def test_load_command_format(self) -> None:
        """Test the load command JSON structure."""
        cmd = {"cmd": "load", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
        serialized = json.dumps(cmd)
        parsed = json.loads(serialized)
        assert parsed["cmd"] == "load"
        assert "model" in parsed

    def test_rerank_command_format(self) -> None:
        """Test the rerank command JSON structure."""
        cmd = {
            "cmd": "rerank",
            "query": "test query",
            "passages": ["passage 1", "passage 2"],
            "top_k": 10,
            "threshold": 0.35,
        }
        serialized = json.dumps(cmd)
        parsed = json.loads(serialized)
        assert parsed["cmd"] == "rerank"
        assert isinstance(parsed["passages"], list)
        assert isinstance(parsed["top_k"], int)
        assert isinstance(parsed["threshold"], float)

    def test_unload_command_format(self) -> None:
        """Test the unload command JSON structure."""
        cmd = {"cmd": "unload"}
        serialized = json.dumps(cmd)
        parsed = json.loads(serialized)
        assert parsed["cmd"] == "unload"

    def test_loaded_success_response_format(self) -> None:
        """Test the loaded (success) response JSON structure."""
        resp = {"type": "loaded", "success": True, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
        assert resp["type"] == "loaded"
        assert resp["success"] is True

    def test_loaded_failure_response_format(self) -> None:
        """Test the loaded (failure) response JSON structure."""
        resp = {"type": "loaded", "success": False, "error": "Model not found"}
        assert resp["type"] == "loaded"
        assert resp["success"] is False
        assert "error" in resp

    def test_reranked_response_format(self) -> None:
        """Test the reranked response JSON structure."""
        resp = {
            "type": "reranked",
            "results": [
                {"index": 0, "score": 0.95, "passage": "text"},
                {"index": 2, "score": 0.72, "passage": "other"},
            ],
        }
        assert resp["type"] == "reranked"
        assert isinstance(resp["results"], list)
        for r in resp["results"]:
            assert "index" in r
            assert "score" in r
            assert "passage" in r

    def test_error_response_format(self) -> None:
        """Test the error response JSON structure."""
        resp = {"type": "error", "message": "Something went wrong"}
        assert resp["type"] == "error"
        assert "message" in resp


# =============================================================================
# CrossEncoderReranker Internal Methods Tests
# =============================================================================


class TestCrossEncoderRerankerInternals:
    """Tests for CrossEncoderReranker internal methods."""

    @pytest.mark.asyncio
    async def test_send_command_writes_json_newline(self) -> None:
        """Test _send_command writes JSON followed by newline."""
        reranker = CrossEncoderReranker()

        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        reranker._process = mock_process

        await reranker._send_command({"cmd": "test", "value": 42})

        mock_stdin.write.assert_called_once()
        written = mock_stdin.write.call_args[0][0]
        assert written.endswith(b"\n")
        parsed = json.loads(written.decode().strip())
        assert parsed == {"cmd": "test", "value": 42}
        reranker._process = None

    @pytest.mark.asyncio
    async def test_send_command_no_process_raises(self) -> None:
        """Test _send_command raises when no process exists."""
        reranker = CrossEncoderReranker()
        with pytest.raises(RuntimeError, match="Process not running"):
            await reranker._send_command({"cmd": "test"})

    @pytest.mark.asyncio
    async def test_send_command_no_stdin_raises(self) -> None:
        """Test _send_command raises when stdin is None."""
        reranker = CrossEncoderReranker()
        mock_process = MagicMock()
        mock_process.stdin = None
        reranker._process = mock_process
        with pytest.raises(RuntimeError, match="Process not running"):
            await reranker._send_command({"cmd": "test"})
        reranker._process = None

    @pytest.mark.asyncio
    async def test_read_response_complete_line(self) -> None:
        """Test _read_response returns parsed JSON when a full line is received."""
        reranker = CrossEncoderReranker()

        mock_stdout = MagicMock()
        full_response = json.dumps({"type": "test", "data": 123}).encode() + b"\n"
        mock_stdout.read = AsyncMock(return_value=full_response)

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        reranker._process = mock_process

        result = await reranker._read_response(timeout=0.1)
        assert result == {"type": "test", "data": 123}
        reranker._process = None
        reranker._read_buffer = b""

    @pytest.mark.asyncio
    async def test_read_response_partial_data(self) -> None:
        """Test _read_response handles partial reads (no complete line yet)."""
        reranker = CrossEncoderReranker()

        mock_stdout = MagicMock()
        # Partial data with no newline
        mock_stdout.read = AsyncMock(return_value=b'{"type":')

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        reranker._process = mock_process

        result = await reranker._read_response(timeout=0.1)
        assert result is None
        assert reranker._read_buffer == b'{"type":'
        reranker._process = None
        reranker._read_buffer = b""

    @pytest.mark.asyncio
    async def test_read_response_uses_buffer(self) -> None:
        """Test _read_response completes a line from buffered + new data."""
        reranker = CrossEncoderReranker()
        # Pre-fill partial buffer
        reranker._read_buffer = b'{"type":"test"'

        mock_stdout = MagicMock()
        mock_stdout.read = AsyncMock(return_value=b',"ok":true}\n')

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        reranker._process = mock_process

        result = await reranker._read_response(timeout=0.1)
        assert result == {"type": "test", "ok": True}
        reranker._process = None
        reranker._read_buffer = b""

    @pytest.mark.asyncio
    async def test_read_response_timeout_returns_none(self) -> None:
        """Test _read_response returns None on timeout."""
        reranker = CrossEncoderReranker()

        mock_stdout = MagicMock()
        mock_stdout.read = AsyncMock(side_effect=asyncio.TimeoutError())

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        reranker._process = mock_process

        result = await reranker._read_response(timeout=0.01)
        assert result is None
        reranker._process = None

    @pytest.mark.asyncio
    async def test_read_response_no_process_returns_none(self) -> None:
        """Test _read_response returns None when process is None."""
        reranker = CrossEncoderReranker()
        result = await reranker._read_response(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_read_response_buffer_from_existing_complete_line(self) -> None:
        """Test _read_response returns from buffer when it already has a complete line."""
        reranker = CrossEncoderReranker()
        reranker._read_buffer = json.dumps({"type": "buffered"}).encode() + b"\nmore"

        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        reranker._process = mock_process

        result = await reranker._read_response(timeout=0.1)
        assert result == {"type": "buffered"}
        assert reranker._read_buffer == b"more"
        reranker._process = None
        reranker._read_buffer = b""

    @pytest.mark.asyncio
    async def test_cleanup_process_state(self) -> None:
        """Test _cleanup_process_state resets all state."""
        reranker = CrossEncoderReranker()

        mock_stdin = MagicMock()
        mock_stdin.close = MagicMock()

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = MagicMock()

        reranker._process = mock_process
        reranker._read_buffer = b"leftover data"

        await reranker._cleanup_process_state()

        assert reranker._process is None
        assert reranker._read_buffer == b""


# =============================================================================
# Reranker Worker Tests (r3lay.core.reranker_worker)
# =============================================================================


class TestWorkerSetupIsolation:
    """Tests for reranker_worker.setup_isolation()."""

    def test_sets_env_vars(self) -> None:
        """Test setup_isolation sets all required environment variables."""
        from r3lay.core.reranker_worker import setup_isolation

        # Capture original stderr to restore later
        original_stderr = sys.stderr

        try:
            setup_isolation()

            assert os.environ["TERM"] == "dumb"
            assert os.environ["NO_COLOR"] == "1"
            assert os.environ["FORCE_COLOR"] == "0"
            assert os.environ["TQDM_DISABLE"] == "1"
            assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
            assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
            assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
        finally:
            sys.stderr = original_stderr

    def test_redirects_stderr(self) -> None:
        """Test setup_isolation redirects stderr to devnull."""
        from r3lay.core.reranker_worker import setup_isolation

        original_stderr = sys.stderr

        try:
            setup_isolation()
            # stderr should no longer be the original
            assert sys.stderr is not original_stderr
            assert sys.stderr.name == os.devnull
        finally:
            sys.stderr = original_stderr


class TestWorkerSendResponse:
    """Tests for reranker_worker.send_response()."""

    def test_outputs_valid_json(self) -> None:
        """Test send_response outputs valid JSON to stdout."""
        from r3lay.core.reranker_worker import send_response

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            send_response({"type": "test", "value": 42})

        output = captured.getvalue().strip()
        parsed = json.loads(output)
        assert parsed == {"type": "test", "value": 42}

    def test_outputs_single_line(self) -> None:
        """Test send_response outputs exactly one line (no embedded newlines in JSON)."""
        from r3lay.core.reranker_worker import send_response

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            send_response({"type": "loaded", "success": True, "model": "test"})

        lines = captured.getvalue().strip().split("\n")
        assert len(lines) == 1


class TestWorkerMainLoadCommand:
    """Tests for the worker main() load command handling."""

    def test_load_success(self) -> None:
        """Test worker handles load command successfully."""
        from r3lay.core.reranker_worker import main

        load_cmd = json.dumps({"cmd": "load", "model": "test-model"}) + "\n"
        unload_cmd = json.dumps({"cmd": "unload"}) + "\n"

        stdin_data = load_cmd + unload_cmd
        mock_stdin = io.StringIO(stdin_data)
        captured_stdout = io.StringIO()

        mock_cross_encoder = MagicMock()

        with (
            patch("sys.stdin", mock_stdin),
            patch("sys.stdout", captured_stdout),
            patch("sys.stderr", io.StringIO()),
        ):
            with patch.dict("sys.modules", {}):
                with patch("r3lay.core.reranker_worker.setup_isolation"):
                    # Patch the CrossEncoder import inside the function
                    mock_module = MagicMock()
                    mock_module.CrossEncoder = MagicMock(return_value=mock_cross_encoder)
                    with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
                        main()

        output_lines = captured_stdout.getvalue().strip().split("\n")
        loaded_resp = json.loads(output_lines[0])
        assert loaded_resp["type"] == "loaded"
        assert loaded_resp["success"] is True
        assert loaded_resp["model"] == "test-model"

    def test_load_import_error(self) -> None:
        """Test worker handles missing sentence-transformers."""
        from r3lay.core.reranker_worker import main

        load_cmd = json.dumps({"cmd": "load", "model": "test-model"}) + "\n"
        unload_cmd = json.dumps({"cmd": "unload"}) + "\n"

        stdin_data = load_cmd + unload_cmd
        mock_stdin = io.StringIO(stdin_data)
        captured_stdout = io.StringIO()

        with (
            patch("sys.stdin", mock_stdin),
            patch("sys.stdout", captured_stdout),
            patch("sys.stderr", io.StringIO()),
            patch("r3lay.core.reranker_worker.setup_isolation"),
        ):
            # Remove sentence_transformers to force ImportError
            with patch.dict("sys.modules", {"sentence_transformers": None}):
                # The import inside main() uses `from sentence_transformers import CrossEncoder`
                # Patching sys.modules with None causes ImportError
                main()

        output_lines = captured_stdout.getvalue().strip().split("\n")
        loaded_resp = json.loads(output_lines[0])
        assert loaded_resp["type"] == "loaded"
        assert loaded_resp["success"] is False
        assert "sentence-transformers" in loaded_resp["error"]


class TestWorkerMainRerankCommand:
    """Tests for the worker main() rerank command handling."""

    def test_rerank_short_query_uniform_scores(self) -> None:
        """Test rerank with short query returns uniform scores (smart skip)."""
        from r3lay.core.reranker_worker import MIN_QUERY_WORDS, main

        # Build a query with fewer words than MIN_QUERY_WORDS
        short_query = " ".join(["word"] * (MIN_QUERY_WORDS - 1))

        load_cmd = json.dumps({"cmd": "load", "model": "test-model"}) + "\n"
        rerank_cmd = (
            json.dumps(
                {
                    "cmd": "rerank",
                    "query": short_query,
                    "passages": ["passage one", "passage two", "passage three"],
                    "top_k": 10,
                    "threshold": 0.35,
                }
            )
            + "\n"
        )
        unload_cmd = json.dumps({"cmd": "unload"}) + "\n"

        stdin_data = load_cmd + rerank_cmd + unload_cmd
        mock_stdin = io.StringIO(stdin_data)
        captured_stdout = io.StringIO()

        mock_model = MagicMock()
        mock_module = MagicMock()
        mock_module.CrossEncoder = MagicMock(return_value=mock_model)

        with (
            patch("sys.stdin", mock_stdin),
            patch("sys.stdout", captured_stdout),
            patch("sys.stderr", io.StringIO()),
            patch("r3lay.core.reranker_worker.setup_isolation"),
            patch.dict("sys.modules", {"sentence_transformers": mock_module}),
        ):
            main()

        output_lines = captured_stdout.getvalue().strip().split("\n")
        # First response is load, second is rerank
        rerank_resp = json.loads(output_lines[1])
        assert rerank_resp["type"] == "reranked"
        results = rerank_resp["results"]
        assert len(results) == 3
        # All should have uniform score of 1.0
        for r in results:
            assert r["score"] == 1.0
        # Original order preserved
        assert results[0]["index"] == 0
        assert results[1]["index"] == 1
        assert results[2]["index"] == 2
        # model.predict should NOT have been called (smart skip)
        mock_model.predict.assert_not_called()

    def test_rerank_with_threshold_filtering(self) -> None:
        """Test rerank filters results below threshold."""
        from r3lay.core.reranker_worker import MIN_QUERY_WORDS, main

        # Build a query with enough words to trigger actual reranking
        long_query = " ".join(["word"] * (MIN_QUERY_WORDS + 2))

        load_cmd = json.dumps({"cmd": "load", "model": "test-model"}) + "\n"
        rerank_cmd = (
            json.dumps(
                {
                    "cmd": "rerank",
                    "query": long_query,
                    "passages": ["relevant passage", "irrelevant passage", "somewhat relevant"],
                    "top_k": 10,
                    "threshold": 0.5,
                }
            )
            + "\n"
        )
        unload_cmd = json.dumps({"cmd": "unload"}) + "\n"

        stdin_data = load_cmd + rerank_cmd + unload_cmd
        mock_stdin = io.StringIO(stdin_data)
        captured_stdout = io.StringIO()

        mock_model = MagicMock()
        # Scores: first above threshold, second below, third above
        import numpy as np

        mock_model.predict.return_value = np.array([0.9, 0.2, 0.6])

        mock_module = MagicMock()
        mock_module.CrossEncoder = MagicMock(return_value=mock_model)

        with (
            patch("sys.stdin", mock_stdin),
            patch("sys.stdout", captured_stdout),
            patch("sys.stderr", io.StringIO()),
            patch("r3lay.core.reranker_worker.setup_isolation"),
            patch.dict("sys.modules", {"sentence_transformers": mock_module}),
        ):
            main()

        output_lines = captured_stdout.getvalue().strip().split("\n")
        rerank_resp = json.loads(output_lines[1])
        assert rerank_resp["type"] == "reranked"
        results = rerank_resp["results"]
        # Only 2 results above threshold of 0.5
        assert len(results) == 2
        # Sorted by score descending
        assert results[0]["score"] > results[1]["score"]
        assert results[0]["index"] == 0  # score 0.9
        assert results[1]["index"] == 2  # score 0.6
        # The passage with score 0.2 should be filtered out
        filtered_indices = [r["index"] for r in results]
        assert 1 not in filtered_indices

    def test_rerank_model_not_loaded(self) -> None:
        """Test rerank command when model is not loaded returns error."""
        from r3lay.core.reranker_worker import main

        # Skip load, go straight to rerank
        rerank_cmd = (
            json.dumps(
                {
                    "cmd": "rerank",
                    "query": "test",
                    "passages": ["p1"],
                }
            )
            + "\n"
        )
        unload_cmd = json.dumps({"cmd": "unload"}) + "\n"

        stdin_data = rerank_cmd + unload_cmd
        mock_stdin = io.StringIO(stdin_data)
        captured_stdout = io.StringIO()

        with (
            patch("sys.stdin", mock_stdin),
            patch("sys.stdout", captured_stdout),
            patch("sys.stderr", io.StringIO()),
            patch("r3lay.core.reranker_worker.setup_isolation"),
        ):
            main()

        output_lines = captured_stdout.getvalue().strip().split("\n")
        error_resp = json.loads(output_lines[0])
        assert error_resp["type"] == "error"
        assert "not loaded" in error_resp["message"].lower()

    def test_rerank_empty_passages(self) -> None:
        """Test rerank with empty passages list returns empty results."""
        from r3lay.core.reranker_worker import main

        load_cmd = json.dumps({"cmd": "load", "model": "test-model"}) + "\n"
        rerank_cmd = (
            json.dumps(
                {
                    "cmd": "rerank",
                    "query": "test query",
                    "passages": [],
                }
            )
            + "\n"
        )
        unload_cmd = json.dumps({"cmd": "unload"}) + "\n"

        stdin_data = load_cmd + rerank_cmd + unload_cmd
        mock_stdin = io.StringIO(stdin_data)
        captured_stdout = io.StringIO()

        mock_model = MagicMock()
        mock_module = MagicMock()
        mock_module.CrossEncoder = MagicMock(return_value=mock_model)

        with (
            patch("sys.stdin", mock_stdin),
            patch("sys.stdout", captured_stdout),
            patch("sys.stderr", io.StringIO()),
            patch("r3lay.core.reranker_worker.setup_isolation"),
            patch.dict("sys.modules", {"sentence_transformers": mock_module}),
        ):
            main()

        output_lines = captured_stdout.getvalue().strip().split("\n")
        rerank_resp = json.loads(output_lines[1])
        assert rerank_resp["type"] == "reranked"
        assert rerank_resp["results"] == []


class TestWorkerMainUnloadCommand:
    """Tests for the worker main() unload command handling."""

    def test_unload_cleanup(self) -> None:
        """Test unload command triggers model cleanup and exits the loop."""
        from r3lay.core.reranker_worker import main

        load_cmd = json.dumps({"cmd": "load", "model": "test-model"}) + "\n"
        unload_cmd = json.dumps({"cmd": "unload"}) + "\n"
        # This should never be reached because unload breaks the loop
        extra_cmd = json.dumps({"cmd": "rerank", "query": "q", "passages": ["p"]}) + "\n"

        stdin_data = load_cmd + unload_cmd + extra_cmd
        mock_stdin = io.StringIO(stdin_data)
        captured_stdout = io.StringIO()

        mock_model = MagicMock()
        mock_module = MagicMock()
        mock_module.CrossEncoder = MagicMock(return_value=mock_model)

        with (
            patch("sys.stdin", mock_stdin),
            patch("sys.stdout", captured_stdout),
            patch("sys.stderr", io.StringIO()),
            patch("r3lay.core.reranker_worker.setup_isolation"),
            patch.dict("sys.modules", {"sentence_transformers": mock_module}),
        ):
            main()

        output_lines = captured_stdout.getvalue().strip().split("\n")
        # Only the load response should be output (unload breaks before rerank)
        assert len(output_lines) == 1
        loaded_resp = json.loads(output_lines[0])
        assert loaded_resp["type"] == "loaded"


class TestWorkerMainEdgeCases:
    """Tests for worker main() edge cases."""

    def test_invalid_json_input(self) -> None:
        """Test worker handles invalid JSON gracefully."""
        from r3lay.core.reranker_worker import main

        stdin_data = "not valid json\n" + json.dumps({"cmd": "unload"}) + "\n"
        mock_stdin = io.StringIO(stdin_data)
        captured_stdout = io.StringIO()

        with (
            patch("sys.stdin", mock_stdin),
            patch("sys.stdout", captured_stdout),
            patch("sys.stderr", io.StringIO()),
            patch("r3lay.core.reranker_worker.setup_isolation"),
        ):
            main()

        output_lines = captured_stdout.getvalue().strip().split("\n")
        error_resp = json.loads(output_lines[0])
        assert error_resp["type"] == "error"
        assert "Invalid JSON" in error_resp["message"]

    def test_unknown_command(self) -> None:
        """Test worker handles unknown commands gracefully."""
        from r3lay.core.reranker_worker import main

        unknown_cmd = json.dumps({"cmd": "unknown_action"}) + "\n"
        unload_cmd = json.dumps({"cmd": "unload"}) + "\n"

        stdin_data = unknown_cmd + unload_cmd
        mock_stdin = io.StringIO(stdin_data)
        captured_stdout = io.StringIO()

        with (
            patch("sys.stdin", mock_stdin),
            patch("sys.stdout", captured_stdout),
            patch("sys.stderr", io.StringIO()),
            patch("r3lay.core.reranker_worker.setup_isolation"),
        ):
            main()

        output_lines = captured_stdout.getvalue().strip().split("\n")
        error_resp = json.loads(output_lines[0])
        assert error_resp["type"] == "error"
        assert "Unknown command" in error_resp["message"]

    def test_empty_lines_ignored(self) -> None:
        """Test worker ignores empty lines in stdin."""
        from r3lay.core.reranker_worker import main

        stdin_data = "\n\n" + json.dumps({"cmd": "unload"}) + "\n\n"
        mock_stdin = io.StringIO(stdin_data)
        captured_stdout = io.StringIO()

        with (
            patch("sys.stdin", mock_stdin),
            patch("sys.stdout", captured_stdout),
            patch("sys.stderr", io.StringIO()),
            patch("r3lay.core.reranker_worker.setup_isolation"),
        ):
            main()

        # No output (unload produces no response, just breaks)
        output = captured_stdout.getvalue().strip()
        assert output == ""
