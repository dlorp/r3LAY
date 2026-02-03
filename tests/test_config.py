"""Tests for r3lay.config module.

Covers:
- ModelRoles dataclass and helper methods
- AppConfig settings and environment variable support
- Configuration load/save to YAML
"""

from __future__ import annotations

from pathlib import Path

from r3lay.config import AppConfig, ModelRoles

# ============================================================================
# ModelRoles Tests
# ============================================================================


class TestModelRoles:
    """Tests for ModelRoles dataclass."""

    def test_default_values(self):
        """ModelRoles has sensible defaults."""
        roles = ModelRoles()
        assert roles.text_model is None
        assert roles.vision_model is None
        assert roles.text_embedder == "mlx-community/all-MiniLM-L6-v2-4bit"
        assert roles.vision_embedder is None

    def test_custom_values(self):
        """Can set custom model roles."""
        roles = ModelRoles(
            text_model="llama-3.2-1b",
            vision_model="llava-1.5-7b",
            text_embedder="custom/embedder",
            vision_embedder="clip-vit",
        )
        assert roles.text_model == "llama-3.2-1b"
        assert roles.vision_model == "llava-1.5-7b"
        assert roles.text_embedder == "custom/embedder"
        assert roles.vision_embedder == "clip-vit"

    def test_has_text_model(self):
        """has_text_model returns correct boolean."""
        roles_without = ModelRoles()
        assert roles_without.has_text_model() is False

        roles_with = ModelRoles(text_model="some-model")
        assert roles_with.has_text_model() is True

    def test_has_vision_model(self):
        """has_vision_model returns correct boolean."""
        roles_without = ModelRoles()
        assert roles_without.has_vision_model() is False

        roles_with = ModelRoles(vision_model="llava")
        assert roles_with.has_vision_model() is True

    def test_has_embedder(self):
        """has_embedder returns True if any embedder is configured."""
        # Default has text embedder
        roles_default = ModelRoles()
        assert roles_default.has_embedder() is True

        # No embedders
        roles_none = ModelRoles(text_embedder=None, vision_embedder=None)
        assert roles_none.has_embedder() is False

        # Only vision embedder
        roles_vision = ModelRoles(text_embedder=None, vision_embedder="clip")
        assert roles_vision.has_embedder() is True

    def test_has_text_embedder(self):
        """has_text_embedder returns correct boolean."""
        roles_with = ModelRoles()  # Has default
        assert roles_with.has_text_embedder() is True

        roles_without = ModelRoles(text_embedder=None)
        assert roles_without.has_text_embedder() is False

    def test_has_vision_embedder(self):
        """has_vision_embedder returns correct boolean."""
        roles_without = ModelRoles()
        assert roles_without.has_vision_embedder() is False

        roles_with = ModelRoles(vision_embedder="clip")
        assert roles_with.has_vision_embedder() is True


# ============================================================================
# AppConfig Tests
# ============================================================================


class TestAppConfigDefaults:
    """Tests for AppConfig default values."""

    def test_default_paths(self):
        """Default paths are set correctly."""
        config = AppConfig()
        assert config.project_path == Path.cwd()
        assert config.theme == "default"
        assert config.hf_cache_path is None
        assert config.mlx_folder is None

    def test_default_endpoints(self):
        """Default endpoint URLs are set."""
        config = AppConfig()
        assert config.ollama_endpoint == "http://localhost:11434"
        assert config.searxng_endpoint == "http://localhost:8080"

    def test_default_model_roles(self):
        """Default model roles are initialized."""
        config = AppConfig()
        assert isinstance(config.model_roles, ModelRoles)
        assert config.model_roles.text_embedder is not None

    def test_gguf_folder_default(self):
        """GGUF folder defaults to ~/.r3lay/models/."""
        config = AppConfig()
        assert config.gguf_folder == Path("~/.r3lay/models/").expanduser()


class TestAppConfigEnvironmentVariables:
    """Tests for environment variable configuration."""

    def test_project_path_from_env(self, monkeypatch, tmp_path):
        """R3LAY_PROJECT_PATH sets project_path."""
        monkeypatch.setenv("R3LAY_PROJECT_PATH", str(tmp_path))
        config = AppConfig()
        assert config.project_path == tmp_path

    def test_ollama_endpoint_from_env(self, monkeypatch):
        """R3LAY_OLLAMA_ENDPOINT sets ollama_endpoint."""
        monkeypatch.setenv("R3LAY_OLLAMA_ENDPOINT", "http://custom:8080")
        config = AppConfig()
        assert config.ollama_endpoint == "http://custom:8080"

    def test_searxng_endpoint_from_env(self, monkeypatch):
        """R3LAY_SEARXNG_ENDPOINT sets searxng_endpoint."""
        monkeypatch.setenv("R3LAY_SEARXNG_ENDPOINT", "http://searx:8888")
        config = AppConfig()
        assert config.searxng_endpoint == "http://searx:8888"

    def test_mlx_folder_from_env(self, monkeypatch, tmp_path):
        """R3LAY_MLX_FOLDER sets mlx_folder."""
        monkeypatch.setenv("R3LAY_MLX_FOLDER", str(tmp_path))
        config = AppConfig()
        assert config.mlx_folder == tmp_path

    def test_hf_cache_path_from_env(self, monkeypatch, tmp_path):
        """R3LAY_HF_CACHE_PATH sets hf_cache_path."""
        monkeypatch.setenv("R3LAY_HF_CACHE_PATH", str(tmp_path))
        config = AppConfig()
        assert config.hf_cache_path == tmp_path


class TestAppConfigLoadSave:
    """Tests for configuration file loading and saving."""

    def test_load_creates_config(self, tmp_path):
        """load() creates config even without config file."""
        config = AppConfig.load(tmp_path)
        assert config.project_path == tmp_path

    def test_save_creates_config_file(self, tmp_path):
        """save() creates .r3lay/config.yaml."""
        config = AppConfig(project_path=tmp_path)
        config.model_roles = ModelRoles(text_model="test-model")
        config.save()

        config_file = tmp_path / ".r3lay" / "config.yaml"
        assert config_file.exists()

    def test_save_creates_directory(self, tmp_path):
        """save() creates .r3lay directory if needed."""
        config = AppConfig(project_path=tmp_path)
        config.save()

        config_dir = tmp_path / ".r3lay"
        assert config_dir.exists()
        assert config_dir.is_dir()

    def test_load_reads_saved_config(self, tmp_path):
        """load() reads previously saved configuration."""
        # Save config
        config1 = AppConfig(project_path=tmp_path)
        config1.model_roles = ModelRoles(
            text_model="saved-model",
            vision_model="saved-vision",
            text_embedder="saved-embedder",
        )
        config1.save()

        # Load config
        config2 = AppConfig.load(tmp_path)
        assert config2.model_roles.text_model == "saved-model"
        assert config2.model_roles.vision_model == "saved-vision"
        assert config2.model_roles.text_embedder == "saved-embedder"

    def test_load_with_partial_config(self, tmp_path):
        """load() handles config with only some fields."""
        from ruamel.yaml import YAML

        # Create partial config
        config_dir = tmp_path / ".r3lay"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"

        yaml = YAML()
        with config_file.open("w") as f:
            yaml.dump({"model_roles": {"text_model": "partial-model"}}, f)

        config = AppConfig.load(tmp_path)
        assert config.model_roles.text_model == "partial-model"
        assert config.model_roles.vision_model is None

    def test_load_with_empty_config(self, tmp_path):
        """load() handles empty config file."""
        config_dir = tmp_path / ".r3lay"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.touch()

        # Should not raise
        config = AppConfig.load(tmp_path)
        assert config is not None

    def test_save_overwrites_existing(self, tmp_path):
        """save() overwrites existing configuration."""
        # Save first config
        config1 = AppConfig(project_path=tmp_path)
        config1.model_roles = ModelRoles(text_model="first")
        config1.save()

        # Save second config
        config2 = AppConfig(project_path=tmp_path)
        config2.model_roles = ModelRoles(text_model="second")
        config2.save()

        # Load and verify
        config3 = AppConfig.load(tmp_path)
        assert config3.model_roles.text_model == "second"


class TestAppConfigYAMLFormat:
    """Tests for YAML configuration format."""

    def test_config_yaml_structure(self, tmp_path):
        """Saved config has expected YAML structure."""
        from ruamel.yaml import YAML

        config = AppConfig(project_path=tmp_path)
        config.model_roles = ModelRoles(
            text_model="test-text",
            vision_model="test-vision",
            text_embedder="test-embedder",
            vision_embedder="test-vision-embedder",
        )
        config.save()

        yaml = YAML()
        config_file = tmp_path / ".r3lay" / "config.yaml"
        with config_file.open() as f:
            data = yaml.load(f)

        assert "model_roles" in data
        assert data["model_roles"]["text_model"] == "test-text"
        assert data["model_roles"]["vision_model"] == "test-vision"
        assert data["model_roles"]["text_embedder"] == "test-embedder"
        assert data["model_roles"]["vision_embedder"] == "test-vision-embedder"

    def test_config_handles_none_values(self, tmp_path):
        """Config properly handles None values in YAML."""
        config = AppConfig(project_path=tmp_path)
        config.model_roles = ModelRoles(text_model=None, text_embedder=None)
        config.save()

        loaded = AppConfig.load(tmp_path)
        assert loaded.model_roles.text_model is None
        assert loaded.model_roles.text_embedder is None


# ============================================================================
# Module Exports
# ============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_exports(self):
        """Module exports expected symbols."""
        from r3lay import config

        assert hasattr(config, "AppConfig")
        assert hasattr(config, "ModelRoles")
