"""Comprehensive tests for r3lay.core.backends.mlx module.

Tests cover:
- MLXBackend initialization and properties
- Model loading with subprocess isolation
- Model unloading and process cleanup
- Streaming token generation
- Error handling (dependencies, load failures, timeouts)
- Worker communication protocol
- Vision model support

All mlx-lm imports and subprocess calls are mocked to allow testing
without Apple Silicon or MLX dependencies.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from r3lay.core.backends import DependencyError, GenerationError, ModelLoadError
from r3lay.core.backends.mlx import MODEL_LOAD_TIMEOUT, MLXBackend

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Create a temporary model directory."""
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def backend(model_path: Path) -> MLXBackend:
    """Create an MLXBackend instance for testing."""
    return MLXBackend(model_path, "TestModel")


@pytest.fixture
def vision_backend(model_path: Path) -> MLXBackend:
    """Create an MLXBackend instance for vision model testing."""
    return MLXBackend(model_path, "TestVisionModel", is_vision=True)


@pytest.fixture
def mock_process() -> MagicMock:
    """Create a mock asyncio subprocess."""
    process = MagicMock()
    process.returncode = None  # Process is running
    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdin.close = MagicMock()
    process.stdout = MagicMock()
    process.stdout.readline = AsyncMock()
    process.stdout.close = MagicMock()
    process.wait = AsyncMock()
    process.terminate = MagicMock()
    process.kill = MagicMock()
    return process


def create_json_response(data: dict) -> bytes:
    """Create a JSON line response as bytes."""
    return (json.dumps(data) + "\n").encode()


# =============================================================================
# Initialization Tests
# =============================================================================


class TestMLXBackendInit:
    """Tests for MLXBackend initialization."""

    def test_init_basic(self, model_path: Path):
        """Test basic initialization."""
        backend = MLXBackend(model_path, "TestModel")
        assert backend._path == model_path
        assert backend._name == "TestModel"
        assert backend._is_vision is False
        assert backend._process is None

    def test_init_vision_model(self, model_path: Path):
        """Test initialization with vision model flag."""
        backend = MLXBackend(model_path, "VisionModel", is_vision=True)
        assert backend._is_vision is True

    def test_model_name_property(self, backend: MLXBackend):
        """Test model_name property returns correct name."""
        assert backend.model_name == "TestModel"

    def test_is_loaded_false_initially(self, backend: MLXBackend):
        """Test is_loaded is False when no process."""
        assert backend.is_loaded is False

    def test_is_loaded_true_when_process_running(
        self, backend: MLXBackend, mock_process: MagicMock
    ):
        """Test is_loaded is True when process is running."""
        backend._process = mock_process
        mock_process.returncode = None
        assert backend.is_loaded is True

    def test_is_loaded_false_when_process_terminated(
        self, backend: MLXBackend, mock_process: MagicMock
    ):
        """Test is_loaded is False when process has exited."""
        backend._process = mock_process
        mock_process.returncode = 0
        assert backend.is_loaded is False


# =============================================================================
# Load Tests
# =============================================================================


class TestMLXBackendLoad:
    """Tests for MLXBackend.load()."""

    @pytest.mark.asyncio
    async def test_load_missing_mlx_dependency(self, backend: MLXBackend):
        """Test load raises DependencyError when mlx not installed."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = None

            with pytest.raises(DependencyError) as exc_info:
                await backend.load()

            assert "mlx is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_missing_mlx_lm_dependency(self, backend: MLXBackend):
        """Test load raises DependencyError when mlx-lm not installed."""

        def find_spec_side_effect(name):
            if name == "mlx.core":
                return MagicMock()  # mlx is installed
            return None  # mlx_lm is not

        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            with pytest.raises(DependencyError) as exc_info:
                await backend.load()

            assert "mlx-lm is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_missing_mlx_vlm_dependency(self, vision_backend: MLXBackend):
        """Test load raises DependencyError when mlx-vlm not installed for vision models."""

        def find_spec_side_effect(name):
            if name in ("mlx.core", "mlx_lm"):
                return MagicMock()
            return None  # mlx_vlm is not installed

        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            with pytest.raises(DependencyError) as exc_info:
                await vision_backend.load()

            assert "mlx-vlm is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_success(self, backend: MLXBackend, mock_process: MagicMock):
        """Test successful model loading."""
        # Mock dependencies as available
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            # Mock subprocess creation
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_process

                # Mock successful load response
                mock_process.stdout.readline = AsyncMock(
                    return_value=create_json_response({"type": "loaded", "success": True})
                )

                await backend.load()

                assert backend._process is mock_process
                mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_failure_from_worker(self, backend: MLXBackend, mock_process: MagicMock):
        """Test load raises ModelLoadError when worker reports failure."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_process

                mock_process.stdout.readline = AsyncMock(
                    return_value=create_json_response(
                        {"type": "loaded", "success": False, "error": "OOM: Not enough memory"}
                    )
                )

                with pytest.raises(ModelLoadError) as exc_info:
                    await backend.load()

                assert "OOM: Not enough memory" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_process_dies_during_load(
        self, backend: MLXBackend, mock_process: MagicMock
    ):
        """Test load raises ModelLoadError when process dies during loading."""
        call_count = 0

        async def readline_then_die():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()  # First read times out
            return b""  # Then process is dead

        with patch("importlib.util.find_spec", return_value=MagicMock()):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_process
                mock_process.stdout.readline = AsyncMock(side_effect=readline_then_die)
                mock_process.returncode = 1  # Process exited

                with pytest.raises(ModelLoadError) as exc_info:
                    await backend.load()

                assert "Worker process died" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_timeout(self, backend: MLXBackend, mock_process: MagicMock):
        """Test load raises ModelLoadError on timeout."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_process

                # Always timeout on readline
                mock_process.stdout.readline = AsyncMock(side_effect=asyncio.TimeoutError())

                # Patch time.monotonic to simulate timeout
                start_time = 0

                def mock_monotonic():
                    nonlocal start_time
                    start_time += MODEL_LOAD_TIMEOUT + 1
                    return start_time

                with patch("time.monotonic", side_effect=mock_monotonic):
                    with pytest.raises(ModelLoadError) as exc_info:
                        await backend.load()

                    assert "Timeout loading model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_sends_correct_command(self, backend: MLXBackend, mock_process: MagicMock):
        """Test load sends correct JSON command to worker."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_process
                mock_process.stdout.readline = AsyncMock(
                    return_value=create_json_response({"type": "loaded", "success": True})
                )

                await backend.load()

                # Check the command sent to stdin
                call_args = mock_process.stdin.write.call_args[0][0]
                sent_cmd = json.loads(call_args.decode().strip())

                assert sent_cmd["cmd"] == "load"
                assert sent_cmd["path"] == str(backend._path)
                assert sent_cmd["is_vision"] is False

    @pytest.mark.asyncio
    async def test_load_vision_model_sends_flag(
        self, vision_backend: MLXBackend, mock_process: MagicMock
    ):
        """Test load sends is_vision=True for vision models."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_process
                mock_process.stdout.readline = AsyncMock(
                    return_value=create_json_response({"type": "loaded", "success": True})
                )

                await vision_backend.load()

                call_args = mock_process.stdin.write.call_args[0][0]
                sent_cmd = json.loads(call_args.decode().strip())

                assert sent_cmd["is_vision"] is True


# =============================================================================
# Unload Tests
# =============================================================================


class TestMLXBackendUnload:
    """Tests for MLXBackend.unload()."""

    @pytest.mark.asyncio
    async def test_unload_no_process(self, backend: MLXBackend):
        """Test unload is safe when no process exists."""
        await backend.unload()  # Should not raise
        assert backend._process is None

    @pytest.mark.asyncio
    async def test_unload_graceful_shutdown(self, backend: MLXBackend, mock_process: MagicMock):
        """Test unload performs graceful shutdown."""
        backend._process = mock_process
        mock_process.returncode = None
        mock_process.wait = AsyncMock(return_value=0)

        # Simulate process exiting after unload command
        async def wait_and_exit():
            mock_process.returncode = 0
            return 0

        mock_process.wait = AsyncMock(side_effect=wait_and_exit)

        await backend.unload()

        assert backend._process is None
        mock_process.stdin.write.assert_called()  # Unload command sent

    @pytest.mark.asyncio
    async def test_unload_force_terminate_on_timeout(
        self, backend: MLXBackend, mock_process: MagicMock
    ):
        """Test unload force terminates if graceful shutdown times out."""
        backend._process = mock_process
        mock_process.returncode = None
        mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError())

        await backend.unload()

        mock_process.terminate.assert_called()
        assert backend._process is None

    @pytest.mark.asyncio
    async def test_unload_kills_if_terminate_fails(
        self, backend: MLXBackend, mock_process: MagicMock
    ):
        """Test unload kills process if terminate doesn't work."""
        backend._process = mock_process
        mock_process.returncode = None

        # First wait (after unload cmd) times out
        # Second wait (after terminate) also times out
        mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError())

        await backend.unload()

        mock_process.terminate.assert_called()
        mock_process.kill.assert_called()

    @pytest.mark.asyncio
    async def test_unload_idempotent(self, backend: MLXBackend, mock_process: MagicMock):
        """Test unload can be called multiple times safely."""
        backend._process = mock_process
        mock_process.wait = AsyncMock(return_value=0)

        async def exit_process():
            mock_process.returncode = 0

        mock_process.wait = AsyncMock(side_effect=exit_process)

        await backend.unload()
        await backend.unload()  # Second call should be safe
        await backend.unload()  # Third call should be safe

        assert backend._process is None


# =============================================================================
# Process Management Tests
# =============================================================================


class TestProcessManagement:
    """Tests for internal process management methods."""

    @pytest.mark.asyncio
    async def test_terminate_process(self, backend: MLXBackend, mock_process: MagicMock):
        """Test _terminate_process force terminates and cleans up."""
        backend._process = mock_process
        mock_process.returncode = None
        mock_process.wait = AsyncMock(return_value=0)

        async def set_exit():
            mock_process.returncode = 0

        mock_process.wait = AsyncMock(side_effect=set_exit)

        await backend._terminate_process()

        mock_process.terminate.assert_called()
        assert backend._process is None

    @pytest.mark.asyncio
    async def test_terminate_process_kills_on_timeout(
        self, backend: MLXBackend, mock_process: MagicMock
    ):
        """Test _terminate_process kills if terminate times out."""
        backend._process = mock_process
        mock_process.returncode = None
        mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError())

        await backend._terminate_process()

        mock_process.kill.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_process_state(self, backend: MLXBackend, mock_process: MagicMock):
        """Test _cleanup_process_state closes pipes and clears process."""
        backend._process = mock_process

        await backend._cleanup_process_state()

        mock_process.stdin.close.assert_called()
        mock_process.stdout.close.assert_called()
        assert backend._process is None

    @pytest.mark.asyncio
    async def test_cleanup_handles_close_errors(self, backend: MLXBackend, mock_process: MagicMock):
        """Test _cleanup_process_state handles errors when closing pipes."""
        backend._process = mock_process
        mock_process.stdin.close = MagicMock(side_effect=Exception("Already closed"))
        mock_process.stdout.close = MagicMock(side_effect=Exception("Already closed"))

        await backend._cleanup_process_state()  # Should not raise
        assert backend._process is None


# =============================================================================
# Communication Tests
# =============================================================================


class TestWorkerCommunication:
    """Tests for worker communication methods."""

    @pytest.mark.asyncio
    async def test_send_command_success(self, backend: MLXBackend, mock_process: MagicMock):
        """Test _send_command sends JSON line to stdin."""
        backend._process = mock_process

        cmd = {"cmd": "test", "data": "value"}
        await backend._send_command(cmd)

        expected = (json.dumps(cmd) + "\n").encode()
        mock_process.stdin.write.assert_called_with(expected)
        mock_process.stdin.drain.assert_called()

    @pytest.mark.asyncio
    async def test_send_command_no_process(self, backend: MLXBackend):
        """Test _send_command raises when no process."""
        with pytest.raises(RuntimeError) as exc_info:
            await backend._send_command({"cmd": "test"})
        assert "Process not running" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_command_no_stdin(self, backend: MLXBackend, mock_process: MagicMock):
        """Test _send_command raises when stdin is None."""
        backend._process = mock_process
        mock_process.stdin = None

        with pytest.raises(RuntimeError):
            await backend._send_command({"cmd": "test"})

    @pytest.mark.asyncio
    async def test_read_response_success(self, backend: MLXBackend, mock_process: MagicMock):
        """Test _read_response parses JSON line from stdout."""
        backend._process = mock_process
        response_data = {"type": "token", "text": "Hello"}
        mock_process.stdout.readline = AsyncMock(return_value=create_json_response(response_data))

        result = await backend._read_response()

        assert result == response_data

    @pytest.mark.asyncio
    async def test_read_response_timeout(self, backend: MLXBackend, mock_process: MagicMock):
        """Test _read_response returns None on timeout."""
        backend._process = mock_process
        mock_process.stdout.readline = AsyncMock(side_effect=asyncio.TimeoutError())

        result = await backend._read_response(timeout=0.1)

        assert result is None

    @pytest.mark.asyncio
    async def test_read_response_invalid_json(self, backend: MLXBackend, mock_process: MagicMock):
        """Test _read_response returns None for invalid JSON."""
        backend._process = mock_process
        mock_process.stdout.readline = AsyncMock(return_value=b"not valid json\n")

        result = await backend._read_response()

        assert result is None

    @pytest.mark.asyncio
    async def test_read_response_no_process(self, backend: MLXBackend):
        """Test _read_response returns None when no process."""
        result = await backend._read_response()
        assert result is None

    @pytest.mark.asyncio
    async def test_read_response_empty_line(self, backend: MLXBackend, mock_process: MagicMock):
        """Test _read_response handles empty lines."""
        backend._process = mock_process
        mock_process.stdout.readline = AsyncMock(return_value=b"")

        result = await backend._read_response()

        assert result is None


# =============================================================================
# Generate Stream Tests
# =============================================================================


class TestGenerateStream:
    """Tests for MLXBackend.generate_stream()."""

    @pytest.mark.asyncio
    async def test_generate_stream_not_loaded(self, backend: MLXBackend):
        """Test generate_stream raises when model not loaded."""
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError) as exc_info:
            async for _ in backend.generate_stream(messages):
                pass

        assert "not loaded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_stream_tokens(self, backend: MLXBackend, mock_process: MagicMock):
        """Test generate_stream yields tokens correctly."""
        backend._process = mock_process
        mock_process.returncode = None

        # Simulate token stream
        responses = [
            {"type": "token", "text": "Hello"},
            {"type": "token", "text": " "},
            {"type": "token", "text": "World"},
            {"type": "done"},
        ]
        response_iter = iter(responses)

        async def mock_readline():
            try:
                return create_json_response(next(response_iter))
            except StopIteration:
                raise asyncio.TimeoutError()

        mock_process.stdout.readline = AsyncMock(side_effect=mock_readline)

        messages = [{"role": "user", "content": "Say hello"}]
        tokens = []
        async for token in backend.generate_stream(messages):
            tokens.append(token)

        assert tokens == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_generate_stream_sends_correct_command(
        self, backend: MLXBackend, mock_process: MagicMock
    ):
        """Test generate_stream sends correct parameters to worker."""
        backend._process = mock_process
        mock_process.returncode = None
        mock_process.stdout.readline = AsyncMock(
            return_value=create_json_response({"type": "done"})
        )

        messages = [{"role": "user", "content": "Test"}]
        async for _ in backend.generate_stream(messages, max_tokens=256, temperature=0.5):
            pass

        call_args = mock_process.stdin.write.call_args[0][0]
        sent_cmd = json.loads(call_args.decode().strip())

        assert sent_cmd["cmd"] == "generate"
        assert sent_cmd["messages"] == messages
        assert sent_cmd["max_tokens"] == 256
        assert sent_cmd["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_stream_with_images(
        self, vision_backend: MLXBackend, mock_process: MagicMock, tmp_path: Path
    ):
        """Test generate_stream includes image paths for vision models."""
        vision_backend._process = mock_process
        mock_process.returncode = None
        mock_process.stdout.readline = AsyncMock(
            return_value=create_json_response({"type": "done"})
        )

        image1 = tmp_path / "test1.png"
        image2 = tmp_path / "test2.png"
        image1.touch()
        image2.touch()

        messages = [{"role": "user", "content": "Describe this image"}]
        async for _ in vision_backend.generate_stream(messages, images=[image1, image2]):
            pass

        call_args = mock_process.stdin.write.call_args[0][0]
        sent_cmd = json.loads(call_args.decode().strip())

        assert "images" in sent_cmd
        assert sent_cmd["images"] == [str(image1), str(image2)]

    @pytest.mark.asyncio
    async def test_generate_stream_error_from_worker(
        self, backend: MLXBackend, mock_process: MagicMock
    ):
        """Test generate_stream raises GenerationError on worker error."""
        backend._process = mock_process
        mock_process.returncode = None
        mock_process.stdout.readline = AsyncMock(
            return_value=create_json_response(
                {"type": "error", "message": "Generation failed: context too long"}
            )
        )

        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(GenerationError) as exc_info:
            async for _ in backend.generate_stream(messages):
                pass

        assert "context too long" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_stream_process_dies(self, backend: MLXBackend, mock_process: MagicMock):
        """Test generate_stream raises when process dies mid-generation."""
        backend._process = mock_process

        call_count = 0

        async def readline_then_die():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return create_json_response({"type": "token", "text": "Hi"})
            mock_process.returncode = 1
            raise asyncio.TimeoutError()

        mock_process.stdout.readline = AsyncMock(side_effect=readline_then_die)
        mock_process.returncode = None

        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(GenerationError) as exc_info:
            async for _ in backend.generate_stream(messages):
                pass

        assert "Worker process died" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_stream_cancellation(self, backend: MLXBackend, mock_process: MagicMock):
        """Test generate_stream handles cancellation gracefully via explicit close."""
        backend._process = mock_process
        mock_process.returncode = None

        # Infinite token stream
        async def infinite_tokens():
            return create_json_response({"type": "token", "text": "x"})

        mock_process.stdout.readline = AsyncMock(side_effect=infinite_tokens)

        messages = [{"role": "user", "content": "Test"}]

        # Get the generator and manually close it to trigger GeneratorExit
        gen = backend.generate_stream(messages)
        count = 0
        async for _ in gen:
            count += 1
            if count >= 5:
                # Explicitly close the generator to trigger stop command
                await gen.aclose()
                break

        # Verify stop command was sent
        calls = mock_process.stdin.write.call_args_list
        # Last call should be the stop command
        last_call = calls[-1][0][0]
        last_cmd = json.loads(last_call.decode().strip())
        assert last_cmd["cmd"] == "stop"

    @pytest.mark.asyncio
    async def test_generate_stream_empty_token(self, backend: MLXBackend, mock_process: MagicMock):
        """Test generate_stream handles empty tokens."""
        backend._process = mock_process
        mock_process.returncode = None

        responses = [
            {"type": "token", "text": "A"},
            {"type": "token", "text": ""},  # Empty token
            {"type": "token", "text": "B"},
            {"type": "done"},
        ]
        response_iter = iter(responses)

        async def mock_readline():
            try:
                return create_json_response(next(response_iter))
            except StopIteration:
                raise asyncio.TimeoutError()

        mock_process.stdout.readline = AsyncMock(side_effect=mock_readline)

        messages = [{"role": "user", "content": "Test"}]
        tokens = []
        async for token in backend.generate_stream(messages):
            tokens.append(token)

        assert tokens == ["A", "", "B"]


# =============================================================================
# Is Available Tests
# =============================================================================


class TestIsAvailable:
    """Tests for MLXBackend.is_available() class method."""

    @pytest.mark.asyncio
    async def test_is_available_both_installed(self):
        """Test is_available returns True when both mlx and mlx-lm installed."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            result = await MLXBackend.is_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_mlx_missing(self):
        """Test is_available returns False when mlx not installed."""

        def find_spec_side_effect(name):
            if name == "mlx.core":
                return None
            return MagicMock()

        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            result = await MLXBackend.is_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_mlx_lm_missing(self):
        """Test is_available returns False when mlx-lm not installed."""

        def find_spec_side_effect(name):
            if name == "mlx_lm":
                return None
            return MagicMock()

        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            result = await MLXBackend.is_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_both_missing(self):
        """Test is_available returns False when both missing."""
        with patch("importlib.util.find_spec", return_value=None):
            result = await MLXBackend.is_available()
            assert result is False


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestMLXBackendLifecycle:
    """Integration-style tests for full backend lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, backend: MLXBackend, mock_process: MagicMock):
        """Test complete load -> generate -> unload lifecycle."""
        # Setup mocks
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_process

                # Load phase
                load_response = create_json_response({"type": "loaded", "success": True})
                mock_process.stdout.readline = AsyncMock(return_value=load_response)

                await backend.load()
                assert backend.is_loaded

                # Generate phase
                token_responses = [
                    {"type": "token", "text": "Hello"},
                    {"type": "token", "text": "!"},
                    {"type": "done"},
                ]
                token_iter = iter(token_responses)

                async def mock_generate_readline():
                    try:
                        return create_json_response(next(token_iter))
                    except StopIteration:
                        raise asyncio.TimeoutError()

                mock_process.stdout.readline = AsyncMock(side_effect=mock_generate_readline)

                messages = [{"role": "user", "content": "Hi"}]
                output = ""
                async for token in backend.generate_stream(messages):
                    output += token

                assert output == "Hello!"

                # Unload phase
                async def exit_process():
                    mock_process.returncode = 0

                mock_process.wait = AsyncMock(side_effect=exit_process)

                await backend.unload()
                assert not backend.is_loaded

    @pytest.mark.asyncio
    async def test_reload_after_unload(self, backend: MLXBackend, mock_process: MagicMock):
        """Test backend can be reloaded after unloading."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
                # First load
                mock_process1 = MagicMock()
                mock_process1.returncode = None
                mock_process1.stdin = MagicMock()
                mock_process1.stdin.write = MagicMock()
                mock_process1.stdin.drain = AsyncMock()
                mock_process1.stdin.close = MagicMock()
                mock_process1.stdout = MagicMock()
                mock_process1.stdout.readline = AsyncMock(
                    return_value=create_json_response({"type": "loaded", "success": True})
                )
                mock_process1.stdout.close = MagicMock()
                mock_process1.wait = AsyncMock()
                mock_process1.terminate = MagicMock()

                mock_create.return_value = mock_process1
                await backend.load()
                assert backend.is_loaded

                # Unload
                async def exit1():
                    mock_process1.returncode = 0

                mock_process1.wait = AsyncMock(side_effect=exit1)
                await backend.unload()
                assert not backend.is_loaded

                # Second load with new process
                mock_process2 = MagicMock()
                mock_process2.returncode = None
                mock_process2.stdin = MagicMock()
                mock_process2.stdin.write = MagicMock()
                mock_process2.stdin.drain = AsyncMock()
                mock_process2.stdout = MagicMock()
                mock_process2.stdout.readline = AsyncMock(
                    return_value=create_json_response({"type": "loaded", "success": True})
                )

                mock_create.return_value = mock_process2
                await backend.load()
                assert backend.is_loaded
                assert backend._process is mock_process2


# =============================================================================
# Worker Module Tests (mlx_worker.py)
# =============================================================================


class TestMLXWorkerHelpers:
    """Tests for helper functions in mlx_worker module."""

    def test_setup_isolation_sets_env_vars(self):
        """Test setup_isolation sets required environment variables."""
        import os
        from unittest.mock import mock_open, patch

        original_env = os.environ.copy()

        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.open", mock_open()):
                with patch("sys.stderr"):
                    from r3lay.core.backends.mlx_worker import setup_isolation

                    setup_isolation()

                    assert os.environ.get("TERM") == "dumb"
                    assert os.environ.get("NO_COLOR") == "1"
                    assert os.environ.get("TQDM_DISABLE") == "1"

    def test_format_messages_with_chat_template(self):
        """Test format_messages uses tokenizer's chat template."""
        from r3lay.core.backends.mlx_worker import format_messages

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Formatted prompt"

        messages = [{"role": "user", "content": "Hello"}]
        result = format_messages(mock_tokenizer, messages)

        assert result == "Formatted prompt"
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_format_messages_fallback(self):
        """Test format_messages uses fallback when no chat template."""
        from r3lay.core.backends.mlx_worker import format_messages

        mock_tokenizer = MagicMock(spec=[])  # No apply_chat_template

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = format_messages(mock_tokenizer, messages)

        assert "System: You are helpful" in result
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result
        assert result.endswith("Assistant: ")

    def test_format_messages_template_exception(self):
        """Test format_messages falls back on template exception."""
        from r3lay.core.backends.mlx_worker import format_messages

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = Exception("Template error")

        messages = [{"role": "user", "content": "Test"}]
        result = format_messages(mock_tokenizer, messages)

        # Should use fallback
        assert "User: Test" in result

    def test_send_response_outputs_json(self, capsys):
        """Test send_response outputs valid JSON line."""
        from r3lay.core.backends.mlx_worker import send_response

        response = {"type": "token", "text": "Hello"}
        send_response(response)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed == response


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_load_generic_exception(self, backend: MLXBackend, mock_process: MagicMock):
        """Test load handles generic exceptions properly."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
                mock_create.side_effect = OSError("Failed to spawn process")

                with pytest.raises(ModelLoadError) as exc_info:
                    await backend.load()

                assert "Failed to load MLX model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unload_handles_send_command_error(
        self, backend: MLXBackend, mock_process: MagicMock
    ):
        """Test unload handles errors when sending unload command."""
        backend._process = mock_process
        mock_process.returncode = None
        mock_process.stdin.write = MagicMock(side_effect=BrokenPipeError())

        async def exit_on_wait():
            mock_process.returncode = 0

        mock_process.wait = AsyncMock(side_effect=exit_on_wait)

        # Should not raise
        await backend.unload()
        assert backend._process is None

    @pytest.mark.asyncio
    async def test_generate_raises_generation_error_on_unexpected(
        self, backend: MLXBackend, mock_process: MagicMock
    ):
        """Test generate_stream wraps unexpected errors in GenerationError."""
        backend._process = mock_process
        mock_process.returncode = None

        # To test unexpected errors, we need to mock _read_response to raise
        # since readline errors are caught and return None
        call_count = 0

        async def mock_readline():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return create_json_response({"type": "token", "text": "Hi"})
            # After first token, simulate process death with error
            mock_process.returncode = -9  # Killed
            raise asyncio.TimeoutError()

        mock_process.stdout.readline = AsyncMock(side_effect=mock_readline)

        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(GenerationError) as exc_info:
            async for _ in backend.generate_stream(messages):
                pass

        assert "Worker process died" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_multiple_concurrent_generate_calls(
        self, backend: MLXBackend, mock_process: MagicMock
    ):
        """Test behavior with concurrent generate calls (not recommended but should handle)."""
        backend._process = mock_process
        mock_process.returncode = None

        responses = [
            {"type": "token", "text": "A"},
            {"type": "done"},
        ]

        async def mock_readline():
            await asyncio.sleep(0.01)  # Simulate some delay
            if responses:
                return create_json_response(responses.pop(0))
            raise asyncio.TimeoutError()

        mock_process.stdout.readline = AsyncMock(side_effect=mock_readline)

        messages = [{"role": "user", "content": "Test"}]

        # Start generation
        tokens = []
        async for token in backend.generate_stream(messages):
            tokens.append(token)

        assert "A" in tokens
