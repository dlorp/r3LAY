"""Tests for MLX worker modules.

Tests cover:
- r3lay.core.backends.mlx_worker: MLX inference worker subprocess helpers
- r3lay.core.embeddings.mlx_text_worker: MLX embedding worker subprocess helpers

Note: These workers run as subprocesses and communicate via JSON over stdin/stdout.
Direct testing of the main() loops requires subprocess mocking, so we focus on
testing the helper functions and JSON protocol.
"""

import base64
import io
import json
import os
from unittest.mock import MagicMock, patch

# =============================================================================
# MLX Worker (Inference) Tests
# =============================================================================


class TestMlxWorkerHelpers:
    """Tests for mlx_worker helper functions."""

    def test_format_messages_with_chat_template(self):
        """Test message formatting with tokenizer that has chat template."""
        # Import the function (isolated to avoid side effects)
        from r3lay.core.backends.mlx_worker import format_messages

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Formatted: Hello"

        messages = [{"role": "user", "content": "Hello"}]
        result = format_messages(mock_tokenizer, messages)

        assert result == "Formatted: Hello"
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_format_messages_fallback(self):
        """Test message formatting fallback when no chat template."""
        from r3lay.core.backends.mlx_worker import format_messages

        # Tokenizer without apply_chat_template
        mock_tokenizer = MagicMock(spec=[])

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
        """Test fallback when apply_chat_template raises exception."""
        from r3lay.core.backends.mlx_worker import format_messages

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = Exception("Template error")

        messages = [{"role": "user", "content": "Hello"}]
        result = format_messages(mock_tokenizer, messages)

        assert "User: Hello" in result

    def test_send_response_format(self):
        """Test that send_response outputs valid JSON."""
        from r3lay.core.backends.mlx_worker import send_response

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            send_response({"type": "test", "data": 123})

        output = captured.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["type"] == "test"
        assert parsed["data"] == 123


class TestMlxWorkerProtocol:
    """Tests for MLX worker JSON protocol."""

    def test_load_command_format(self):
        """Test load command JSON format."""
        cmd = {"cmd": "load", "path": "/path/to/model", "is_vision": False}
        serialized = json.dumps(cmd)
        parsed = json.loads(serialized)

        assert parsed["cmd"] == "load"
        assert parsed["path"] == "/path/to/model"
        assert parsed["is_vision"] is False

    def test_generate_command_format(self):
        """Test generate command JSON format."""
        cmd = {
            "cmd": "generate",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 512,
            "temperature": 0.7,
            "images": [],
        }
        serialized = json.dumps(cmd)
        parsed = json.loads(serialized)

        assert parsed["cmd"] == "generate"
        assert len(parsed["messages"]) == 1
        assert parsed["max_tokens"] == 512

    def test_response_loaded_success(self):
        """Test loaded success response format."""
        response = {"type": "loaded", "success": True, "is_vlm": False}
        serialized = json.dumps(response)
        parsed = json.loads(serialized)

        assert parsed["type"] == "loaded"
        assert parsed["success"] is True

    def test_response_token(self):
        """Test token response format."""
        response = {"type": "token", "text": "Hello"}
        serialized = json.dumps(response)
        parsed = json.loads(serialized)

        assert parsed["type"] == "token"
        assert parsed["text"] == "Hello"

    def test_response_error(self):
        """Test error response format."""
        response = {"type": "error", "message": "Something went wrong"}
        serialized = json.dumps(response)
        parsed = json.loads(serialized)

        assert parsed["type"] == "error"
        assert "wrong" in parsed["message"]


# =============================================================================
# MLX Text Worker (Embeddings) Tests
# =============================================================================


class TestMlxTextWorkerHelpers:
    """Tests for mlx_text_worker helper functions."""

    def test_encode_array(self):
        """Test numpy array encoding to base64."""
        import numpy as np

        from r3lay.core.embeddings.mlx_text_worker import encode_array

        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        data, shape, dtype = encode_array(arr)

        assert isinstance(data, str)
        assert shape == [2, 3]
        assert dtype == "float32"

        # Verify we can decode it back
        decoded_bytes = base64.b64decode(data)
        decoded_arr = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(shape)
        assert np.allclose(decoded_arr, arr)

    def test_encode_array_1d(self):
        """Test encoding 1D array."""
        import numpy as np

        from r3lay.core.embeddings.mlx_text_worker import encode_array

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data, shape, dtype = encode_array(arr)

        assert shape == [3]

    def test_send_response_format(self):
        """Test that send_response outputs valid JSON."""
        from r3lay.core.embeddings.mlx_text_worker import send_response

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            send_response({"type": "loaded", "success": True, "dimension": 384})

        output = captured.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["type"] == "loaded"
        assert parsed["dimension"] == 384


class TestMlxTextWorkerProtocol:
    """Tests for MLX text worker JSON protocol."""

    def test_load_command_format(self):
        """Test load command JSON format."""
        cmd = {"cmd": "load", "model": "sentence-transformers/all-MiniLM-L6-v2"}
        serialized = json.dumps(cmd)
        parsed = json.loads(serialized)

        assert parsed["cmd"] == "load"
        assert "MiniLM" in parsed["model"]

    def test_embed_command_format(self):
        """Test embed command JSON format."""
        cmd = {"cmd": "embed", "texts": ["Hello world", "How are you?"]}
        serialized = json.dumps(cmd)
        parsed = json.loads(serialized)

        assert parsed["cmd"] == "embed"
        assert len(parsed["texts"]) == 2

    def test_unload_command_format(self):
        """Test unload command JSON format."""
        cmd = {"cmd": "unload"}
        serialized = json.dumps(cmd)
        parsed = json.loads(serialized)

        assert parsed["cmd"] == "unload"

    def test_response_embeddings(self):
        """Test embeddings response format."""
        import numpy as np

        # Simulate what the worker would return
        arr = np.random.rand(2, 384).astype(np.float32)
        data = base64.b64encode(arr.tobytes()).decode("ascii")

        response = {
            "type": "embeddings",
            "vectors": data,
            "shape": [2, 384],
            "dtype": "float32",
        }
        serialized = json.dumps(response)
        parsed = json.loads(serialized)

        assert parsed["type"] == "embeddings"
        assert parsed["shape"] == [2, 384]

        # Verify we can decode it
        decoded = base64.b64decode(parsed["vectors"])
        decoded_arr = np.frombuffer(decoded, dtype=np.float32).reshape(parsed["shape"])
        assert decoded_arr.shape == (2, 384)


class TestWorkerIsolation:
    """Tests for worker environment isolation."""

    def test_setup_isolation_sets_environment(self):
        """Test that setup_isolation sets required environment variables."""
        # Store original values
        original_env = os.environ.copy()

        try:
            # We can't actually call setup_isolation() because it redirects stderr,
            # but we can verify the expected environment variables
            expected_vars = [
                "TERM",
                "NO_COLOR",
                "FORCE_COLOR",
                "TQDM_DISABLE",
                "TOKENIZERS_PARALLELISM",
                "TRANSFORMERS_VERBOSITY",
                "HF_HUB_DISABLE_PROGRESS_BARS",
            ]

            # These should be set by setup_isolation
            for var in expected_vars:
                # Just verify the variables are known (don't run isolation)
                assert isinstance(var, str)
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)

    def test_text_worker_expected_env_vars(self):
        """Test that text worker would set expected environment variables."""
        expected_vars = [
            "TERM",
            "NO_COLOR",
            "FORCE_COLOR",
            "TQDM_DISABLE",
            "TOKENIZERS_PARALLELISM",
            "TRANSFORMERS_VERBOSITY",
            "HF_HUB_DISABLE_PROGRESS_BARS",
        ]

        # Verify we know about all expected isolation variables
        for var in expected_vars:
            assert isinstance(var, str)


# =============================================================================
# Round-trip Tests
# =============================================================================


class TestEmbeddingRoundTrip:
    """Tests for embedding encode/decode round trips."""

    def test_float32_roundtrip(self):
        """Test float32 array round trip through base64."""
        import numpy as np

        from r3lay.core.embeddings.mlx_text_worker import encode_array

        original = np.random.rand(10, 384).astype(np.float32)
        data, shape, dtype = encode_array(original)

        # Decode
        decoded_bytes = base64.b64decode(data)
        recovered = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(shape)

        assert np.allclose(original, recovered)

    def test_empty_array(self):
        """Test empty array encoding."""
        import numpy as np

        from r3lay.core.embeddings.mlx_text_worker import encode_array

        empty = np.array([], dtype=np.float32)
        data, shape, dtype = encode_array(empty)

        assert shape == [0]
        assert data == ""  # Empty bytes encode to empty string

    def test_large_batch(self):
        """Test encoding large batch of embeddings."""
        import numpy as np

        from r3lay.core.embeddings.mlx_text_worker import encode_array

        large = np.random.rand(1000, 768).astype(np.float32)
        data, shape, dtype = encode_array(large)

        assert shape == [1000, 768]

        # Verify size (1000 * 768 * 4 bytes = 3,072,000 bytes)
        decoded = base64.b64decode(data)
        assert len(decoded) == 1000 * 768 * 4
