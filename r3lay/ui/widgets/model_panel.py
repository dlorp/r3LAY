"""Model panel - model selection with role-based grouping.

Phase C: Models are grouped by capability (text, vision, embedding).
Users can assign models to specific roles via radio-button-style selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Label, OptionList, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from ...config import ModelRoles
    from ...core import R3LayState
    from ...core.models import ModelInfo, ModelScanner


class ModelPanel(Vertical):
    """Panel for selecting LLM models grouped by role.

    Displays models discovered from:
    - HuggingFace cache (safetensors, GGUF)
    - MLX models folder (direct downloads)
    - GGUF drop folder (~/.r3lay/models/)
    - Ollama API (localhost:11434)

    Models are grouped by capability:
    - TEXT MODEL: Standard LLMs for chat/completion
    - VISION MODEL: Vision-language models
    - EMBEDDING MODELS: Text and vision embedders

    Keybindings:
    - Enter on model list: Select model for the current role
    - Load button: Load selected model (toggles to Unload when loaded)
    - Scan button: Rescan all sources
    """

    # Reactive attribute for loaded model tracking
    # This ensures UI updates when model load state changes
    _loaded_model_name: reactive[str | None] = reactive(None)

    DEFAULT_CSS = """
    ModelPanel {
        width: 100%;
        height: 100%;
        padding: 0 1;
    }

    ModelPanel > Label {
        height: 1;
        margin-bottom: 0;
    }

    #model-list {
        height: 1fr;
        border: solid $primary-darken-2;
    }

    #model-list > .option-list--option {
        padding: 0 1;
    }

    #model-list > .option-list--option-highlighted {
        background: $primary-darken-1;
    }

    #model-status {
        height: auto;
        min-height: 1;
        max-height: 3;
        color: $text-muted;
        padding: 0 0;
    }

    #button-row {
        height: 3;
        width: 100%;
    }

    #scan-button {
        width: 1fr;
        margin-right: 1;
    }

    #load-button {
        width: 1fr;
    }

    #load-button:disabled {
        opacity: 0.5;
    }

    .role-header {
        color: $primary;
        text-style: bold;
    }

    .role-selected {
        color: $success;
    }

    .role-none {
        color: $text-muted;
        text-style: italic;
    }
    """

    class RoleAssigned(Message):
        """A model was assigned to a role."""

        def __init__(
            self,
            role: str,
            model_name: str | None,
            model_info: "ModelInfo | None" = None,
        ) -> None:
            self.role = role  # "text", "vision", "text_embedder", "vision_embedder"
            self.model_name = model_name
            self.model_info = model_info
            super().__init__()

    def __init__(self, state: "R3LayState", scanner: "ModelScanner | None" = None):
        """Initialize the model panel.

        Args:
            state: Shared application state.
            scanner: Model scanner instance. If None, tries to get from state.scanner.
        """
        super().__init__()
        self.state = state
        self._scanner = scanner
        self._models: dict[str, "ModelInfo"] = {}
        self._selected_model: str | None = None
        self._current_role: str | None = None  # Track which role section is selected

    @property
    def scanner(self) -> "ModelScanner | None":
        """Get the model scanner, from explicit param or state."""
        if self._scanner is not None:
            return self._scanner
        return getattr(self.state, "scanner", None)

    def compose(self) -> ComposeResult:
        yield Label("Model Roles:")
        yield OptionList(id="model-list")
        yield Static("Click Scan to discover models", id="model-status")
        with Horizontal(id="button-row"):
            yield Button("Scan", id="scan-button", variant="primary")
            yield Button("Load", id="load-button", disabled=True)

    async def on_mount(self) -> None:
        """Handle mount - display persisted role assignments."""
        # Display current role assignments from config (loaded on app startup)
        self._display_persisted_roles()

    def watch__loaded_model_name(self, old_value: str | None, new_value: str | None) -> None:
        """React to loaded model changes by refreshing the model list.

        This watcher fires AFTER the reactive attribute changes, ensuring
        the UI update uses the correct state.
        """
        # Only refresh if we have models loaded (avoid refresh on init)
        if self._models:
            self.call_later(self._refresh_model_list)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "scan-button":
            await self._scan_models()
        elif event.button.id == "load-button":
            button_text = event.button.label.plain
            if button_text == "Unload":
                await self._unload_model()
            elif button_text in ("Set Embedder", "Set Vision Embedder"):
                await self._set_embedder_role()
            else:
                await self._load_selected_model()

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle model selection from the list."""
        if event.option_list.id == "model-list":
            option_id = str(event.option.id) if event.option.id else None
            if option_id:
                self._handle_selection(option_id)

    def _handle_selection(self, option_id: str) -> None:
        """Handle selection of an option in the list.

        Option IDs are formatted as:
        - "role:text" / "role:vision" / etc. -> Role header (assign current selection)
        - "model:role:model_name" -> Model selection (role included for uniqueness)
        - "none:role" -> Clear role assignment
        """
        if option_id.startswith("model:"):
            # Format is "model:role:model_name" - extract model_name (everything after second colon)
            parts = option_id.split(":", 2)  # Split into at most 3 parts
            if len(parts) >= 3:
                model_name = parts[2]  # model_name is after "model:role:"
                # Validate model exists in scanned set
                if model_name not in self._models:
                    return  # Silently ignore invalid selection
                self._select_model(model_name)
        elif option_id.startswith("none:"):
            role = option_id[5:]  # Strip "none:" prefix
            self._clear_role(role)
        # Role headers are informational, no action needed

    async def _scan_models(self) -> None:
        """Scan for available models and display grouped by capability."""
        model_list = self.query_one("#model-list", OptionList)
        status = self.query_one("#model-status", Static)

        model_list.clear_options()
        self._models.clear()
        self._selected_model = None

        if self.scanner is None:
            status.update("No model scanner configured")
            return

        # Sync local reactive state with global state (in case app was restarted with model loaded)
        self._loaded_model_name = self.state.current_model

        status.update("Scanning...")

        try:
            models = await self.scanner.scan_all()

            # Store models by name
            for model in models:
                self._models[model.name] = model

            # Group models by capability
            # VL models go in VISION only, not TEXT
            # (they have both capabilities but are primarily vision)
            from ...core.models import ModelCapability

            vision_models = [m for m in models if ModelCapability.VISION in m.capabilities]
            text_models = [
                m
                for m in models
                if ModelCapability.TEXT in m.capabilities
                and ModelCapability.VISION not in m.capabilities
            ]
            text_embedders = [m for m in models if ModelCapability.TEXT_EMBEDDING in m.capabilities]
            vision_embedders = [
                m for m in models if ModelCapability.VISION_EMBEDDING in m.capabilities
            ]

            # Get current role assignments from app config
            roles = self._get_model_roles()

            # Build the option list with role sections
            self._add_role_section(
                model_list,
                "TEXT MODEL",
                "text",
                text_models,
                roles.text_model if roles else None,
            )

            # Add separator (disabled empty option)
            model_list.add_option(Option("", disabled=True))

            self._add_role_section(
                model_list,
                "VISION MODEL",
                "vision",
                vision_models,
                roles.vision_model if roles else None,
            )

            # Add separator
            model_list.add_option(Option("", disabled=True))

            self._add_role_section(
                model_list,
                "TEXT EMBEDDER",
                "text_embedder",
                text_embedders,
                roles.text_embedder if roles else None,
            )

            if vision_embedders:
                model_list.add_option(Option("", disabled=True))
                self._add_role_section(
                    model_list,
                    "VISION EMBEDDER",
                    "vision_embedder",
                    vision_embedders,
                    roles.vision_embedder if roles else None,
                )

            # Update status
            count = len(models)
            loaded = self.state.current_model
            if count == 0:
                status.update("No models found")
            elif loaded:
                loaded_short = loaded.split("/")[-1][:20]
                status.update(f"{count} models [Loaded: {loaded_short}]")
            else:
                status.update(f"Found {count} models")

        except Exception as e:
            status.update(f"Scan error: {e!s}")

    def _refresh_model_list(self) -> None:
        """Refresh the model list display using cached model data.

        This rebuilds the OptionList with current badge state without
        triggering a full rescan of model sources. Called by the
        _loaded_model_name watcher when load state changes.
        """

        if not self._models:
            return

        model_list = self.query_one("#model-list", OptionList)
        status = self.query_one("#model-status", Static)

        model_list.clear_options()

        # Get models by capability from cache
        # VL models go in VISION only, not TEXT
        from ...core.models import ModelCapability

        models = list(self._models.values())
        vision_models = [m for m in models if ModelCapability.VISION in m.capabilities]
        text_models = [
            m
            for m in models
            if ModelCapability.TEXT in m.capabilities
            and ModelCapability.VISION not in m.capabilities
        ]
        text_embedders = [m for m in models if ModelCapability.TEXT_EMBEDDING in m.capabilities]
        vision_embedders = [m for m in models if ModelCapability.VISION_EMBEDDING in m.capabilities]

        roles = self._get_model_roles()

        # Rebuild sections (same logic as _scan_models but from cache)
        self._add_role_section(
            model_list,
            "TEXT MODEL",
            "text",
            text_models,
            roles.text_model if roles else None,
        )
        model_list.add_option(Option("", disabled=True))

        self._add_role_section(
            model_list,
            "VISION MODEL",
            "vision",
            vision_models,
            roles.vision_model if roles else None,
        )
        model_list.add_option(Option("", disabled=True))

        self._add_role_section(
            model_list,
            "TEXT EMBEDDER",
            "text_embedder",
            text_embedders,
            roles.text_embedder if roles else None,
        )

        if vision_embedders:
            model_list.add_option(Option("", disabled=True))
            self._add_role_section(
                model_list,
                "VISION EMBEDDER",
                "vision_embedder",
                vision_embedders,
                roles.vision_embedder if roles else None,
            )

        # Update status with current count and loaded model
        count = len(models)
        loaded = self._loaded_model_name
        if count == 0:
            status.update("No models found")
        elif loaded:
            loaded_short = loaded.split("/")[-1][:20]
            status.update(f"{count} models [Loaded: {loaded_short}]")
        else:
            status.update(f"Found {count} models")

    def _add_role_section(
        self,
        option_list: OptionList,
        title: str,
        role: str,
        models: list["ModelInfo"],
        current_assignment: str | None,
    ) -> None:
        """Add a role section to the option list.

        Args:
            option_list: The OptionList widget
            title: Section title (e.g., "TEXT MODEL")
            role: Role identifier (e.g., "text", "vision")
            models: List of models with this capability
            current_assignment: Currently assigned model name (if any)
        """
        # Add section header
        header_text = f"-- {title} --"
        option_list.add_option(Option(header_text, id=f"role:{role}", disabled=True))

        if not models:
            option_list.add_option(
                Option("  (no models available)", id=f"empty:{role}", disabled=True)
            )
            return

        # Add "None" option for clearing assignment
        none_marker = "[X]" if current_assignment is None else "[ ]"
        option_list.add_option(Option(f"  {none_marker} (None - disable)", id=f"none:{role}"))

        # Add each model with selection indicator and role badges
        for model in models:
            is_selected = self._names_match(model.name, current_assignment)
            marker = "[X]" if is_selected else "[ ]"

            # Check if this model is currently loaded
            current_loaded = self.state.current_model
            is_loaded = self._names_match(current_loaded, model.name)

            # Note: Use escaped brackets for Rich markup compatibility
            # [LOADED] becomes \[LOADED] to prevent Rich interpreting it as a tag
            # Use amber/gold color to match system message border
            badges = "[#E6A817]\\[LOADED][/]" if is_loaded else ""

            # Format: [X] ModelName [TEXT] 6.2GB MLX
            # Or:     [X] ModelName        6.2GB MLX (no badges)
            # Compact layout to fit in 40-char panel width
            backend_str = model.backend.value.upper()[:3]  # Shorter backend name
            size_str = model.size_human.replace(" ", "")  # Compact size: "6.2GB"

            # Show full model name - let it wrap/scroll naturally
            # e.g., "mlx-community/Dolphin-X1-8B" -> "Dolphin-X1-8B"
            display_name = model.name.split("/")[-1]  # Get just the model name, not org

            # Format: [X] Name [LOADED] 6.2GB MLX
            if badges:
                label = f"  {marker} {display_name} {badges} {size_str} {backend_str}"
            else:
                label = f"  {marker} {display_name} {size_str} {backend_str}"

            option_list.add_option(Option(label, id=f"model:{role}:{model.name}"))

    def _get_model_roles(self) -> "ModelRoles | None":
        """Get current model role assignments from app config."""
        try:
            app = self.app
            if hasattr(app, "config") and hasattr(app.config, "model_roles"):
                return app.config.model_roles
        except Exception:
            pass
        return None

    def _names_match(self, name1: str | None, name2: str | None) -> bool:
        """Check if two model names match, handling path and quantization variations.

        Handles cases where:
        - One name might be a full path and another is short
        - Names differ by quantization suffix (4bit, 8bit, etc.)
        - Names have slight variations in format
        """
        if name1 is None or name2 is None:
            return False
        # Direct match
        if name1 == name2:
            return True

        # Normalize names for comparison
        def normalize(name: str) -> str:
            # Get last path component
            base = name.split("/")[-1].lower().strip()
            # Remove common quantization suffixes for comparison
            for suffix in ["-4bit", "-8bit", "-4b", "-8b", "-fp16", "-bf16", "-q4", "-q8", "-mlx"]:
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
            return base

        n1 = normalize(name1)
        n2 = normalize(name2)

        # Exact match after normalization
        if n1 == n2:
            return True

        # Check if one contains the other (handles partial matches)
        if n1 in n2 or n2 in n1:
            return True

        return False

    def _get_role_badges(self, model_name: str) -> str:
        """Get role badges for a model based on current assignments.

        Returns badges like [LOADED], [TEXT], [VISION], [EMBED], [V-EMBED]
        based on what the model is assigned to or if it's currently loaded.

        Args:
            model_name: Name of the model to check

        Returns:
            Space-separated badge string (e.g., "[LOADED]" or "[EMBED]")
        """
        badges = []

        # Check if this model is currently loaded in memory
        # Use local reactive attribute for guaranteed sync with UI state
        current = self._loaded_model_name

        # Note: Use escaped brackets for Rich markup compatibility
        # Use amber/gold color to match system message border
        if self._names_match(current, model_name):
            badges.append("[#E6A817]\\[LOADED][/]")

        # Check role assignments
        roles = self._get_model_roles()
        if roles is not None:
            if self._names_match(roles.text_model, model_name):
                badges.append("\\[TEXT]")
            if self._names_match(roles.vision_model, model_name):
                badges.append("\\[VISION]")
            if self._names_match(roles.text_embedder, model_name):
                badges.append("\\[EMBED]")
            if self._names_match(roles.vision_embedder, model_name):
                badges.append("\\[V-EMBED]")

        return " ".join(badges)

    def _display_persisted_roles(self) -> None:
        """Display persisted role assignments from config on startup.

        Shows which models are assigned to each role without requiring a scan.
        Users can click Scan to see all available models and change assignments.
        """
        model_list = self.query_one("#model-list", OptionList)
        status = self.query_one("#model-status", Static)

        roles = self._get_model_roles()
        if roles is None:
            status.update("Click Scan to discover models")
            return

        model_list.clear_options()

        # Badge mapping for display (escaped for Rich markup)
        badge_map = {
            "text": "\\[TEXT]",
            "vision": "\\[VISION]",
            "text_embedder": "\\[EMBED]",
            "vision_embedder": "\\[V-EMBED]",
        }

        # Show current assignments (read-only until scan)
        def add_role_display(title: str, role_id: str, assigned: str | None) -> None:
            """Add a role display row."""
            model_list.add_option(Option(f"-- {title} --", id=f"role:{role_id}", disabled=True))
            if assigned:
                # Truncate long names
                display_name = assigned
                badge = badge_map.get(role_id, "")
                if len(display_name) > 22:
                    display_name = display_name[:19] + "..."
                model_list.add_option(
                    Option(f"  [X] {display_name} {badge}", id=f"assigned:{role_id}", disabled=True)
                )
            else:
                model_list.add_option(
                    Option("  (not assigned)", id=f"empty:{role_id}", disabled=True)
                )

        add_role_display("TEXT MODEL", "text", roles.text_model)
        model_list.add_option(Option("", disabled=True))  # Separator

        add_role_display("VISION MODEL", "vision", roles.vision_model)
        model_list.add_option(Option("", disabled=True))  # Separator

        add_role_display("TEXT EMBEDDER", "text_embedder", roles.text_embedder)

        if roles.vision_embedder:
            model_list.add_option(Option("", disabled=True))  # Separator
            add_role_display("VISION EMBEDDER", "vision_embedder", roles.vision_embedder)

        # Count assigned roles
        assigned_count = sum(
            1
            for r in [
                roles.text_model,
                roles.vision_model,
                roles.text_embedder,
                roles.vision_embedder,
            ]
            if r is not None
        )

        if assigned_count > 0:
            status.update(f"{assigned_count} role(s) configured - Scan to modify")
        else:
            status.update("No roles configured - Click Scan")

    def _select_model(self, model_name: str) -> None:
        """Handle selection of a model.

        Updates status with model details and enables Load button.
        For embedding models, shows appropriate embedder button:
        - "Set Embedder" for text embedding models
        - "Set Vision Embedder" for vision embedding models
        """
        status = self.query_one("#model-status", Static)
        load_button = self.query_one("#load-button", Button)

        model = self._models.get(model_name)
        if model is None:
            status.update("Model not found")
            return

        self._selected_model = model_name

        # Determine which role this model could fill
        from ...core.models import ModelCapability

        capabilities = model.capabilities
        is_text_embedder = False
        is_vision_embedder = False

        if ModelCapability.TEXT_EMBEDDING in capabilities:
            self._current_role = "text_embedder"
            is_text_embedder = True
        elif ModelCapability.VISION_EMBEDDING in capabilities:
            self._current_role = "vision_embedder"
            is_vision_embedder = True
        elif ModelCapability.VISION in capabilities:
            self._current_role = "vision"
        elif ModelCapability.TEXT in capabilities:
            self._current_role = "text"

        # Update button based on model type and current load state
        # Only show "Unload" if the selected model IS the currently loaded model
        is_currently_loaded = (
            self.state.current_model is not None and model_name == self.state.current_model
        )

        load_button.disabled = False
        if is_currently_loaded:
            load_button.label = "Unload"
        elif is_vision_embedder:
            load_button.label = "Set Vision Embedder"
        elif is_text_embedder:
            load_button.label = "Set Embedder"
        else:
            load_button.label = "Load"

        # Show model details + loaded model indicator
        fmt = model.format.value if hasattr(model.format, "value") else str(model.format or "?")
        backend = model.backend.value if hasattr(model.backend, "value") else str(model.backend)
        caps = model.capabilities_display

        # Show both selected model details AND which model is loaded
        loaded = self.state.current_model
        if loaded:
            loaded_short = loaded.split("/")[-1][:15]
            status.update(f"{backend}|{fmt}|{model.size_human} [Loaded:{loaded_short}]")
        else:
            status.update(f"{backend} | {fmt} | {model.size_human} | {caps}")

        # Post role assignment message
        self.post_message(self.RoleAssigned(self._current_role or "text", model_name, model))

    def _clear_role(self, role: str) -> None:
        """Clear a role assignment."""
        status = self.query_one("#model-status", Static)

        # Clear the role in config
        try:
            app = self.app
            if hasattr(app, "config") and hasattr(app.config, "model_roles"):
                if role == "text":
                    app.config.model_roles.text_model = None
                elif role == "vision":
                    app.config.model_roles.vision_model = None
                elif role == "text_embedder":
                    app.config.model_roles.text_embedder = None
                elif role == "vision_embedder":
                    app.config.model_roles.vision_embedder = None
                app.config.save()
        except Exception:
            pass

        status.update(f"Cleared {role} model")
        self.post_message(self.RoleAssigned(role, None, None))

    async def _load_selected_model(self) -> None:
        """Load the currently selected LLM model (text or vision)."""
        model_info = self.get_selected_model()
        if not model_info:
            return

        load_button = self.query_one("#load-button", Button)
        status = self.query_one("#model-status", Static)

        load_button.disabled = True

        try:
            status.update(f"Loading {model_info.name}...")
            await self.state.load_model(model_info)

            # Update local reactive state - triggers watcher which refreshes UI
            self._loaded_model_name = model_info.name

            status.update(f"Loaded: {model_info.name}")
            self.app.notify(f"Model loaded: {model_info.name}")
            load_button.label = "Unload"

            # Save model role to config for model switching
            try:
                app = self.app
                if hasattr(app, "config") and hasattr(app.config, "model_roles"):
                    if self._current_role == "text":
                        app.config.model_roles.text_model = model_info.name
                    elif self._current_role == "vision":
                        app.config.model_roles.vision_model = model_info.name
                    app.config.save()
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(f"Saved {self._current_role} model role: {model_info.name}")
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to save model role: {e}")

            load_button.disabled = False

            # Refresh welcome message to show loaded model
            self._refresh_welcome_message()

        except Exception as e:
            import logging

            logging.error(f"Model load failed: {e}", exc_info=True)
            error_str = str(e)
            if len(error_str) > 50:
                error_str = error_str[:47] + "..."
            status.update(f"Error: {error_str}")
            load_button.label = "Load"
            load_button.disabled = False

    async def _set_embedder_role(self) -> None:
        """Set the selected model as the embedder (without loading it now).

        Embedding models are loaded lazily during reindex:
        - Text embedders via state.init_embedder()
        - Vision embedders via state.init_vision_embedder()

        This method only stores the model name in the config for later use.
        """
        model_info = self.get_selected_model()
        if not model_info:
            return

        load_button = self.query_one("#load-button", Button)
        status = self.query_one("#model-status", Static)

        # Determine embedder type
        from ...core.models import ModelCapability

        if ModelCapability.TEXT_EMBEDDING in model_info.capabilities:
            embedder_type = "text_embedder"
            role_display = "Text Embedder"
        elif ModelCapability.VISION_EMBEDDING in model_info.capabilities:
            embedder_type = "vision_embedder"
            role_display = "Vision Embedder"
        else:
            status.update("Not an embedding model")
            return

        # Store in app config
        try:
            app = self.app
            if hasattr(app, "config") and hasattr(app.config, "model_roles"):
                if embedder_type == "text_embedder":
                    app.config.model_roles.text_embedder = model_info.name
                    hint = "Will be loaded during reindex (Ctrl+R)"
                else:
                    app.config.model_roles.vision_embedder = model_info.name
                    hint = "Will be loaded during reindex for PDF/images"

                app.config.save()
                status.update(f"Set {role_display}: {model_info.name}")
                self.app.notify(f"{role_display} configured: {model_info.name}\n{hint}")

                # Keep button as appropriate embedder button
                if embedder_type == "vision_embedder":
                    load_button.label = "Set Vision Embedder"
                else:
                    load_button.label = "Set Embedder"
                load_button.disabled = False

                # Trigger a list refresh to update the [X] markers
                await self._scan_models()

            else:
                status.update("Config not available")

        except Exception as e:
            import logging

            logging.error(f"Failed to set embedder: {e}", exc_info=True)
            error_str = str(e)
            if len(error_str) > 50:
                error_str = error_str[:47] + "..."
            status.update(f"Error: {error_str}")

    async def _unload_model(self) -> None:
        """Unload the currently loaded model."""
        status = self.query_one("#model-status", Static)
        load_button = self.query_one("#load-button", Button)

        load_button.disabled = True

        try:
            model_name = self.state.current_model
            status.update("Unloading...")
            await self.state.unload_model()

            # Update local reactive state - triggers watcher which refreshes UI
            self._loaded_model_name = None

            status.update("Model unloaded")
            if model_name:
                self.app.notify(f"Model unloaded: {model_name}")

            # Refresh welcome message to show no model
            self._refresh_welcome_message()

            # Restore button based on currently selected model type
            if self._selected_model:
                model_info = self._models.get(self._selected_model)
                if model_info:
                    if model_info.is_vision_embedder:
                        load_button.label = "Set Vision Embedder"
                    elif model_info.is_text_embedder:
                        load_button.label = "Set Embedder"
                    else:
                        load_button.label = "Load"
                    load_button.disabled = False
                else:
                    load_button.label = "Load"
                    load_button.disabled = True
            else:
                load_button.label = "Load"
                load_button.disabled = True

        except Exception as e:
            status.update(f"Error: {e}")
            load_button.disabled = False

    def get_selected_model(self) -> "ModelInfo | None":
        """Get the currently selected model info."""
        if self._selected_model is None:
            return None
        return self._models.get(self._selected_model)

    def refresh_models(self) -> None:
        """Trigger a model rescan (convenience method)."""
        import asyncio

        asyncio.create_task(self._scan_models())

    def _refresh_welcome_message(self) -> None:
        """Refresh the welcome message in ResponsePane to reflect model state."""
        try:
            response_pane = self.screen.query_one("ResponsePane")
            if hasattr(response_pane, "refresh_welcome"):
                response_pane.refresh_welcome()
        except Exception:
            pass  # Best effort - ResponsePane may not exist


__all__ = ["ModelPanel"]
