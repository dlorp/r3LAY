"""Input pane - user input area with chat integration.

Features:
- Multi-turn conversation history via Session
- Smart routing between text/vision models (informational)
- Streaming LLM responses
- Escape key to cancel generation
- Command handling (/, /clear, /help, etc.)
- Natural language model swapping ("swap model mistral", "use llama")
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Paste
from textual.widgets import Button, Static, TextArea

from ...core.intent.parser import IntentParser
from ...core.intent.taxonomy import IntentType

if TYPE_CHECKING:
    from ...core import R3LayState
    from ...core.axioms import Axiom
    from ...core.index import RetrievalResult
    from ...core.intent.taxonomy import IntentResult
    from ...core.project_context import ProjectContext
    from ...core.router import RoutingDecision
    from ...core.sources import SourceType

logger = logging.getLogger(__name__)


def _format_trust_badge(source_type: "SourceType") -> str:
    """Return a trust indicator badge for the source type."""
    from ...core.sources import SourceType

    badges = {
        SourceType.INDEXED_CURATED: "[MANUAL]",
        SourceType.INDEXED_DOCUMENT: "[DOC]",
        SourceType.INDEXED_CODE: "[CODE]",
        SourceType.INDEXED_IMAGE: "[IMAGE]",
        SourceType.WEB_OE_FIRSTPARTY: "[OEM]",
        SourceType.WEB_TRUSTED: "[TRUSTED]",
        SourceType.WEB_GENERAL: "[WEB]",
        SourceType.WEB_COMMUNITY: "[FORUM]",
    }
    return badges.get(source_type, "[?]")


def _parse_index_query(query: str) -> tuple["SourceType | None", str]:
    """Parse /index query for optional source filter.

    Examples:
        "oil change" -> (None, "oil change")
        "manual: oil change" -> (SourceType.INDEXED_CURATED, "oil change")
    """
    from ...core.sources import SourceType

    prefixes = {
        "manual:": SourceType.INDEXED_CURATED,
        "doc:": SourceType.INDEXED_DOCUMENT,
        "code:": SourceType.INDEXED_CODE,
        "oem:": SourceType.WEB_OE_FIRSTPARTY,
        "trusted:": SourceType.WEB_TRUSTED,
        "forum:": SourceType.WEB_COMMUNITY,
    }
    for prefix, source_type in prefixes.items():
        if query.lower().startswith(prefix):
            return source_type, query[len(prefix) :].strip()
    return None, query


class InputPane(Vertical):
    """User input pane with multi-line text area.

    Features:
    - Multi-turn conversation history via Session
    - Smart routing between text/vision models (informational)
    - Streaming LLM responses
    - Escape key to cancel generation
    - Command handling (/, /clear, /help, etc.)
    """

    DEFAULT_CSS = """
    InputPane {
        width: 100%;
        height: auto;
        min-height: 6;
        max-height: 12;
        border: solid $primary-darken-2;
        border-title-color: $primary;
        padding: 1;
    }

    #input-area {
        height: auto;
        min-height: 3;
        max-height: 8;
    }

    #input-controls {
        height: 3;
        align: right middle;
    }

    #input-status {
        width: 1fr;
        color: $text-muted;
    }

    #send-button {
        min-width: 10;
    }
    """

    BORDER_TITLE = "Input"

    BINDINGS = [("escape", "cancel", "Cancel generation")]

    def __init__(self, state: "R3LayState", **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self._processing = False
        self._cancel_requested = False
        # Attachments for current message (future: file drop support)
        self._attachments: list[Path] = []
        # Intent parser for natural language commands
        self._intent_parser = IntentParser()

    def compose(self) -> ComposeResult:
        yield TextArea(id="input-area")
        with Horizontal(id="input-controls"):
            yield Static(
                "Ask about automotive, electronics, software, or home projects...",
                id="input-status",
            )
            yield Button("Send", id="send-button", variant="primary")

    def on_mount(self) -> None:
        """Set initial button state when widget mounts."""
        self._update_send_button_state()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Update placeholder status when text changes."""
        if event.text_area.id == "input-area":
            # Update button state based on validation
            self._update_send_button_state()

            if event.text_area.text.strip():
                # Only show "Ready" if validation passes
                if self._validate_can_send()[0]:
                    self.set_status("Ready")
            else:
                self.set_status("Ask about automotive, electronics, software, or home projects...")

    def _validate_can_send(self) -> tuple[bool, str]:
        """Validate if message can be sent.

        Returns:
            Tuple of (is_valid, error_message). If is_valid is True, error_message is empty.
        """
        # Check if a model is loaded
        if not hasattr(self.state, "current_backend") or self.state.current_backend is None:
            return False, "No model loaded. Load a model from the Models tab first."

        # Check if there's actual content to send
        value = self.get_value().strip()
        if not value:
            return False, "Enter a message to send."

        return True, ""

    def _update_send_button_state(self) -> None:
        """Update Send button state based on current validation.

        Disables button and shows clear error message when validation fails.
        This prevents flickering by setting state before user interaction.
        """
        if self._processing:
            # Don't change state during processing
            return

        is_valid, error_msg = self._validate_can_send()
        button = self.query_one("#send-button", Button)

        if not is_valid:
            button.disabled = True
            if error_msg:
                self.set_status(error_msg)
        else:
            button.disabled = False

    def focus_input(self) -> None:
        self.query_one("#input-area", TextArea).focus()

    def refresh_validation(self) -> None:
        """Refresh validation state and update button.

        Call this method when model state changes (e.g., after loading/unloading a model).
        """
        self._update_send_button_state()

    def set_value(self, value: str) -> None:
        self.query_one("#input-area", TextArea).text = value

    def get_value(self) -> str:
        return self.query_one("#input-area", TextArea).text

    def clear(self) -> None:
        self.query_one("#input-area", TextArea).text = ""

    def set_status(self, status: str) -> None:
        self.query_one("#input-status", Static).update(status)

    def set_processing(self, processing: bool) -> None:
        self._processing = processing
        self.query_one("#input-area", TextArea).disabled = processing

        if processing:
            # Always disable button during processing
            self.query_one("#send-button", Button).disabled = True
            self.set_status("Processing...")
        else:
            # When processing completes, update button state based on validation
            self._update_send_button_state()
            # Preserve attachment indicator when not processing
            self._update_attachment_status()

    def _update_attachment_status(self) -> None:
        """Update status to show attachment count or Ready."""
        if self._attachments:
            count = len(self._attachments)
            self.set_status(f"{count} attachment{'s' if count > 1 else ''}")
        else:
            self.set_status("Ready")

    def clear_conversation(self) -> None:
        """Clear conversation history from session."""
        if self.state.session is not None:
            self.state.session.clear()
        # Also reset router state for new conversation
        if self.state.router is not None:
            self.state.router.reset()

    def cancel_generation(self) -> None:
        """Cancel ongoing generation."""
        self._cancel_requested = True

    def action_cancel(self) -> None:
        """Handle Escape key to cancel generation."""
        if self._processing:
            self.cancel_generation()
            self.set_status("Cancelling...")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-button":
            await self.submit()

    def _check_clipboard_for_image(self) -> Path | None:
        """Check system clipboard for image data and save to temp file.

        Tries multiple approaches for macOS compatibility:
        1. Direct NSPasteboard access via PyObjC (most reliable for browser images)
        2. PIL ImageGrab.grabclipboard() as fallback

        Returns:
            Path to saved temp image file, or None if no image in clipboard.
        """
        import platform

        # Try macOS-native approach first (better browser support)
        if platform.system() == "Darwin":
            result = self._check_clipboard_macos_native()
            if result is not None:
                return result

        # Fall back to PIL ImageGrab
        return self._check_clipboard_pil()

    def _check_clipboard_macos_native(self) -> Path | None:
        """Check clipboard using macOS native NSPasteboard via PyObjC.

        This method is more reliable for browser-copied images because it
        directly accesses the pasteboard without AppleScript conversion.

        Returns:
            Path to saved temp image file, or None if no image in clipboard.
        """
        import tempfile
        import time

        try:
            from AppKit import NSPasteboard, NSPasteboardTypePNG, NSPasteboardTypeTIFF

            pb = NSPasteboard.generalPasteboard()

            # Log all available types for debugging
            available_types = pb.types()
            logger.debug(
                f"Clipboard types available: {list(available_types) if available_types else 'None'}"
            )

            # Image types to try, in order of preference
            # Browsers often provide multiple formats
            image_types = [
                NSPasteboardTypePNG,  # public.png - preferred
                NSPasteboardTypeTIFF,  # public.tiff - common fallback
                "public.jpeg",  # JPEG format
                "com.compuserve.gif",  # GIF format
                "org.webmproject.webp",  # WebP format
            ]

            for img_type in image_types:
                data = pb.dataForType_(img_type)
                if data is not None:
                    logger.debug(
                        f"Found clipboard image data: {img_type}, size: {data.length()} bytes"
                    )

                    # Determine file extension
                    ext_map = {
                        NSPasteboardTypePNG: ".png",
                        NSPasteboardTypeTIFF: ".tiff",
                        "public.jpeg": ".jpg",
                        "com.compuserve.gif": ".gif",
                        "org.webmproject.webp": ".webp",
                    }
                    ext = ext_map.get(img_type, ".png")

                    # Save to temp file
                    temp_dir = Path(tempfile.gettempdir()) / "r3lay_clipboard"
                    temp_dir.mkdir(exist_ok=True)

                    timestamp = int(time.time() * 1000)
                    temp_path = temp_dir / f"clipboard_{timestamp}{ext}"

                    # Write the raw image data
                    with open(temp_path, "wb") as f:
                        f.write(bytes(data))

                    logger.info(f"Saved clipboard image via NSPasteboard to {temp_path}")
                    return temp_path

            # Check for file URLs (dragged files)
            file_url_type = "public.file-url"
            if file_url_type in (available_types or []):
                url_string = pb.stringForType_(file_url_type)
                if url_string:
                    logger.debug(f"Clipboard contains file URL: {url_string}")
                    # Handle file:// URLs
                    if url_string.startswith("file://"):
                        path = Path(url_string[7:])  # Remove file:// prefix
                        if path.exists() and path.suffix.lower() in {
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".gif",
                            ".webp",
                            ".bmp",
                            ".tiff",
                        }:
                            if not self._is_path_allowed(path.resolve()):
                                logger.warning(f"Clipboard file outside allowed dirs: {path}")
                                return None
                            logger.info(f"Found image file in clipboard: {path}")
                            return path

            logger.debug("No image data found in clipboard via NSPasteboard")
            return None

        except ImportError as e:
            logger.debug(f"PyObjC not available for native clipboard access: {e}")
            return None
        except Exception as e:
            logger.warning(f"NSPasteboard clipboard check failed: {e}")
            return None

    def _check_clipboard_pil(self) -> Path | None:
        """Check clipboard using PIL ImageGrab (fallback method).

        Note: On macOS, this uses AppleScript internally and may miss some
        browser-copied images due to format conversion issues.

        Returns:
            Path to saved temp image file, or None if no image in clipboard.
        """
        import tempfile
        import time

        try:
            from PIL import ImageGrab

            # Try to grab image from clipboard
            image = ImageGrab.grabclipboard()

            logger.debug(
                f"PIL grabclipboard() returned: {type(image).__name__ if image else 'None'}"
            )

            if image is None:
                return None

            # Check if it's actually an image (not a file list on some systems)
            if not hasattr(image, "save"):
                # On macOS, grabclipboard can return file paths as a list
                if isinstance(image, list) and len(image) > 0:
                    logger.debug(f"PIL returned file list: {image}")
                    # It's a list of file paths
                    path = Path(image[0])
                    if path.exists() and path.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".gif",
                        ".webp",
                        ".bmp",
                    }:
                        if not self._is_path_allowed(path.resolve()):
                            logger.warning(f"Clipboard file outside allowed dirs: {path}")
                            return None
                        logger.info(f"Found image file from PIL clipboard: {path}")
                        return path
                logger.debug(f"PIL returned non-image type: {type(image)}")
                return None

            # Save to temp file with timestamp
            temp_dir = Path(tempfile.gettempdir()) / "r3lay_clipboard"
            temp_dir.mkdir(exist_ok=True)

            timestamp = int(time.time() * 1000)
            temp_path = temp_dir / f"clipboard_{timestamp}.png"

            image.save(temp_path, "PNG")
            logger.info(f"Saved clipboard image via PIL to {temp_path}")
            return temp_path

        except ImportError:
            logger.debug("PIL not available for clipboard image grab")
            return None
        except Exception as e:
            logger.warning(f"PIL clipboard image grab failed: {e}")
            return None

    async def on_paste(self, event: Paste) -> None:
        """Handle paste events for clipboard images and file paths.

        Checks for actual image data in clipboard first (screenshots, browser copies),
        then checks for single file path pastes, then falls back to multi-line handling.
        """
        from ...core.router import IMAGE_EXTENSIONS

        paste_text_preview = repr(event.text[:100] if event.text else "")
        logger.debug(f"on_paste called with text: {paste_text_preview}")

        # ALWAYS check clipboard image first (browser copies include both image AND url)
        clipboard_image = self._check_clipboard_for_image()
        if clipboard_image is not None:
            logger.info(f"Clipboard image detected and attached: {clipboard_image}")
            if clipboard_image not in self._attachments:
                self._attachments.append(clipboard_image)
                self._update_attachment_status()
                self.notify("Attached clipboard image", severity="information")
            event.stop()
            return

        # No clipboard image found - log this for debugging
        logger.debug("No clipboard image detected, falling back to text paste handling")

        # No clipboard image - check if pasted text is a file path
        text = event.text.strip()
        if not text:
            logger.debug("Empty paste text, ignoring")
            return

        # SINGLE LINE CHECK: Detect single file path paste (most common case)
        # This handles Finder drag-drop or pasting a copied path
        if "\n" not in text:
            # Clean up potential quotes or file:// prefix
            clean_path = text.replace("file://", "").strip("'\"")
            logger.debug(f"Single line paste, checking path: {repr(clean_path)}")

            try:
                path = Path(clean_path).expanduser().resolve()
                if path.exists() and path.is_file():
                    logger.debug(f"Path exists as file: {path}, suffix: {path.suffix.lower()}")
                    # Security check: ensure path is within allowed directories
                    if not self._is_path_allowed(path):
                        logger.warning(
                            f"Rejected file attachment outside allowed directories: {path}"
                        )
                        self.notify(
                            "Cannot attach files outside allowed directories",
                            severity="warning",
                        )
                        event.stop()
                        return
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        # It's an image file - attach it!
                        if path not in self._attachments:
                            self._attachments.append(path)
                            self._update_attachment_status()
                            self.notify(f"Attached: {path.name}", severity="information")
                            logger.info(f"Attached image from paste: {path}")
                        else:
                            logger.debug(f"Image already attached: {path}")
                        event.stop()
                        return
                    else:
                        # File exists but not an image - let it paste as text
                        logger.debug(
                            f"File is not an image (suffix: {path.suffix}), allowing paste as text"
                        )
                else:
                    logger.debug(f"Path does not exist or is not a file: {path}")
            except Exception as e:
                # Not a valid path - let normal paste continue
                logger.debug(f"Path parsing failed: {e}")

            # Single line that isn't an image file - let default paste handle it
            logger.debug("Single line is not an image path, allowing default paste")
            return  # Don't stop event - let TextArea handle the paste

        # MULTI-LINE: Handle multiple lines (multiple files dropped or multi-line text)
        logger.debug("Multi-line paste detected")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        attached_count = 0
        non_file_lines = []

        rejected_count = 0
        for line in lines:
            # Clean up potential file:// prefix or quotes
            clean_path = line.replace("file://", "").strip("'\"")

            try:
                path = Path(clean_path).expanduser().resolve()
                if path.exists() and path.is_file():
                    # Security check: ensure path is within allowed directories
                    if not self._is_path_allowed(path):
                        logger.warning(
                            f"Rejected file attachment outside allowed directories: {path}"
                        )
                        rejected_count += 1
                        continue
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        if path not in self._attachments:
                            self._attachments.append(path)
                            attached_count += 1
                    else:
                        # File exists but not an image - treat as text
                        non_file_lines.append(line)
                else:
                    # Not a valid file path - treat as text
                    non_file_lines.append(line)
            except Exception:
                # Path parsing failed - treat as text
                non_file_lines.append(line)

        if attached_count > 0:
            # Update status and notify
            self._update_attachment_status()
            self.notify(f"Attached {attached_count} image(s)", severity="information")
            logger.info(f"Attached {attached_count} image(s) from multi-line paste")

        if rejected_count > 0:
            self.notify(
                f"Rejected {rejected_count} file(s) outside allowed directories",
                severity="warning",
            )

        if non_file_lines:
            # Allow non-file text to be pasted into TextArea
            # Insert the non-file text into the input area
            text_area = self.query_one("#input-area", TextArea)
            text_area.insert("\n".join(non_file_lines))
            logger.debug(f"Inserted {len(non_file_lines)} non-file lines as text")

        # Prevent default paste behavior since we handled it
        event.stop()

    def _looks_like_path(self, text: str) -> bool:
        """Check if text looks like a file path rather than a command.

        File paths typically start with /Users, /home, /tmp, /var, etc.
        Commands are short words like /help, /attach, /clear.
        """
        if not text.startswith("/"):
            return False

        # Common path prefixes that indicate a file path, not a command
        path_prefixes = (
            "/Users/",
            "/home/",
            "/tmp/",
            "/var/",
            "/etc/",
            "/opt/",
            "/usr/",
            "/Library/",
            "/Applications/",
            "/Volumes/",
            "/private/",
            "/System/",
        )

        # Also check for tilde expansion
        if text.startswith("~/"):
            return True

        return text.startswith(path_prefixes)

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if a path is within allowed directories for security.

        Prevents path traversal attacks where users could attach sensitive
        system files (e.g., /etc/passwd) by pasting arbitrary paths.

        Args:
            path: The path to validate (should already be resolved).

        Returns:
            True if the path is within allowed directories, False otherwise.
        """
        allowed_dirs = [
            Path.home(),  # User's home directory
            Path.cwd(),  # Current working directory
        ]

        # Add common user media directories if they exist
        pictures_dir = Path.home() / "Pictures"
        downloads_dir = Path.home() / "Downloads"
        desktop_dir = Path.home() / "Desktop"
        documents_dir = Path.home() / "Documents"

        for extra_dir in [pictures_dir, downloads_dir, desktop_dir, documents_dir]:
            if extra_dir.exists():
                allowed_dirs.append(extra_dir)

        try:
            resolved_path = path.resolve()
            for allowed_dir in allowed_dirs:
                try:
                    if resolved_path.is_relative_to(allowed_dir.resolve()):
                        return True
                except (ValueError, OSError):
                    continue
            return False
        except (ValueError, OSError) as e:
            logger.warning(f"Path resolution failed for {path}: {e}")
            return False

    async def submit(self) -> None:
        if self._processing:
            return

        # Validate before any state changes to prevent flickering
        is_valid, error_msg = self._validate_can_send()
        if not is_valid:
            # Show error in response pane instead of just status
            screen = self.screen
            response_pane = screen.query_one("ResponsePane")
            response_pane.add_error(error_msg)
            return

        value = self.get_value().strip()
        if not value:
            return

        self.clear()
        self.set_processing(True)

        try:
            screen = self.screen
            response_pane = screen.query_one("ResponsePane")
            response_pane.add_user(value)

            # Check if this is a command (e.g., /help, /attach) vs a file path
            if value.startswith("/") and not self._looks_like_path(value):
                await self._handle_command(value, response_pane)
            else:
                # Parse intent to detect natural language commands
                intent_result = self._intent_parser.parse_sync(value)

                # Route intents to appropriate handlers based on confidence
                if intent_result.confidence >= 0.7:
                    # High confidence - route to specific handlers

                    # Handle natural language model swap commands
                    if (
                        intent_result.intent == IntentType.COMMAND
                        and intent_result.subtype == "cmd.model"
                    ):
                        await self._handle_model_swap(intent_result, response_pane)

                    # Handle maintenance logging: "Logged oil change at 98k"
                    elif (
                        intent_result.intent == IntentType.LOG
                        and intent_result.subtype == "log.maintenance"
                    ):
                        await self._handle_maintenance_intent(intent_result, response_pane)

                    # Handle maintenance queries: "When is oil change due?"
                    elif intent_result.intent == IntentType.QUERY:
                        await self._handle_query_intent(intent_result, response_pane)

                    # Handle mileage updates: "Mileage is now 98500"
                    elif (
                        intent_result.intent == IntentType.UPDATE
                        and intent_result.subtype == "update.mileage"
                    ):
                        await self._handle_mileage_update_intent(intent_result, response_pane)

                    else:
                        # Other intents fall through to chat
                        await self._handle_chat(value, response_pane)
                else:
                    # Low confidence - default to chat
                    await self._handle_chat(value, response_pane)
        finally:
            self.set_processing(False)
            self._cancel_requested = False
            # Clear attachments after processing
            self._attachments.clear()
            self.focus_input()

    async def _handle_command(self, command: str, response_pane) -> None:
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "help":
            response_pane.add_assistant(
                "## Commands\n\n"
                "**Chat & Session**\n"
                "- `/help` - Show this help\n"
                "- `/status` - Show system status\n"
                "- `/clear` - Clear chat and conversation history\n"
                "- `/session` - Show session info\n"
                "- `/save [name]` - Save current session\n"
                "- `/load <name>` - Load a saved session\n"
                "- `/sessions` - List saved sessions\n\n"
                "**Attachments**\n"
                "- `/attach <path>` - Attach image file(s) to next message\n"
                "- `/attachments` - List current attachments\n"
                "- `/detach` - Clear all attachments\n\n"
                "**Search & Research**\n"
                "- `/index <query>` - Search knowledge base\n"
                "- `/search <query>` - Web search via SearXNG\n"
                "- `/research <query>` - Deep research expedition (R3 methodology)\n\n"
                "**Knowledge Management**\n"
                "- `/axiom [category:] <statement>` - Create new axiom\n"
                "- `/axioms [category] [--disputed]` - List axioms\n"
                "- `/cite <axiom_id>` - Show provenance chain for axiom\n"
                "- `/dispute <axiom_id> <reason>` - Mark axiom as disputed\n\n"
                "**Maintenance**\n"
                "- `/log <service_type> <mileage>` - Log maintenance entry\n"
                "- `/due [mileage]` - Show due/overdue services\n"
                "- `/history [service_type] [--limit 10]` - Show maintenance history\n"
                "- `/mileage [new_value]` - Update or show current mileage\n"
            )
            self.notify("Help displayed", severity="information")
        elif cmd == "clear":
            response_pane.clear()
            self.clear_conversation()
            response_pane.add_system("Chat and conversation history cleared.")
            self.notify("Chat cleared", severity="information")
        elif cmd == "index":
            if not args:
                response_pane.add_system("Usage: `/index <query>`")
                self.notify("Missing query parameter", severity="warning")
                return
            await self._handle_index_search(args, response_pane)
        elif cmd == "session":
            self._show_session_info(response_pane)
            self.notify("Session info displayed", severity="information")
        elif cmd == "status":
            self._show_status(response_pane)
            self.notify("Status displayed", severity="information")
        elif cmd == "attach":
            if not args:
                response_pane.add_system("Usage: `/attach <path>` - attach image file(s)")
                self.notify("Missing path parameter", severity="warning")
                return
            self._handle_attach(args, response_pane)
        elif cmd == "attachments":
            self._show_attachments(response_pane)
            self.notify("Attachments listed", severity="information")
        elif cmd == "detach":
            self._clear_attachments(response_pane)
        elif cmd == "axiom":
            if not args:
                response_pane.add_system(
                    "Usage: `/axiom [category:] <statement>`\n\n"
                    "Categories: specifications, procedures, compatibility, "
                    "diagnostics, history, safety\n\n"
                    "Example: `/axiom spec: EJ25 timing belt interval is 105,000 miles`"
                )
                return
            await self._handle_create_axiom(args, response_pane)
        elif cmd == "axioms":
            await self._handle_list_axioms(args, response_pane)
        elif cmd == "cite":
            if not args:
                response_pane.add_system("Usage: `/cite <axiom_id>`")
                self.notify("Missing axiom_id parameter", severity="warning")
                return
            await self._handle_show_citations(args, response_pane)
        elif cmd == "dispute":
            parts_dispute = args.split(maxsplit=1)
            if len(parts_dispute) < 2:
                response_pane.add_system("Usage: `/dispute <axiom_id> <reason>`")
                return
            await self._handle_dispute_axiom(parts_dispute[0], parts_dispute[1], response_pane)
        elif cmd == "search":
            if not args:
                response_pane.add_system(
                    "Usage: `/search <query>`\n\n"
                    "Performs a web search via SearXNG.\n\n"
                    "Requires: SearXNG server running (default: http://localhost:8080)\n\n"
                    "Example: `/search EJ25 timing belt replacement`"
                )
                return
            await self._handle_search(args, response_pane)
        elif cmd == "research":
            if not args:
                response_pane.add_system(
                    "Usage: `/research <query>`\n\n"
                    "Starts a deep research expedition using the R3 methodology:\n"
                    "- Multi-cycle exploration with convergence detection\n"
                    "- Contradiction detection and resolution\n"
                    "- Axiom extraction with provenance tracking\n"
                    "- Synthesis report generation\n\n"
                    "Requires: Model loaded, SearXNG available (optional)\n\n"
                    "Example: `/research EJ25 head gasket failure causes`"
                )
                return
            await self._handle_research(args, response_pane)
        elif cmd == "save":
            await self._handle_save_session(args, response_pane)
        elif cmd == "load":
            if not args:
                response_pane.add_system("Usage: `/load <name>` - Load a saved session")
                self.notify("Missing session name parameter", severity="warning")
                return
            await self._handle_load_session(args, response_pane)
        elif cmd == "sessions":
            await self._handle_list_sessions(response_pane)
        elif cmd == "log":
            if not args:
                response_pane.add_system(
                    "Usage: `/log <service_type> <mileage>`\n\nExample: `/log oil_change 50000`"
                )
                return
            await self._handle_log_maintenance(args, response_pane)
        elif cmd == "due":
            await self._handle_due_services(args, response_pane)
        elif cmd == "history":
            await self._handle_maintenance_history(args, response_pane)
        elif cmd == "mileage":
            await self._handle_update_mileage(args, response_pane)
        else:
            from rich.markup import escape

            response_pane.add_system(f"Command `/{cmd}` not implemented yet.")
            self.notify(f"Unknown command: /{escape(cmd)}", severity="error")

    def _show_session_info(self, response_pane) -> None:
        """Show current session information."""
        session = self.state.session
        if session is None:
            response_pane.add_system("No active session.")
            return

        msg_count = len(session.messages)
        title = session.title or "(untitled)"
        created = session.created_at.strftime("%Y-%m-%d %H:%M")

        # Router info
        router_info = "Not initialized"
        if self.state.router is not None:
            router = self.state.router
            model_type = router.current_model_type or "none"
            router_info = f"Active ({model_type} model)"

        response_pane.add_assistant(
            f"## Session Info\n\n"
            f"- **Title**: {title}\n"
            f"- **Messages**: {msg_count}\n"
            f"- **Created**: {created}\n"
            f"- **Router**: {router_info}\n"
        )

    def _show_status(self, response_pane) -> None:
        """Show current system status (refreshed welcome message)."""
        from ...core.welcome import get_welcome_message

        status_text = get_welcome_message(self.state)
        response_pane.add_system(status_text)

    async def _handle_save_session(self, name: str, response_pane) -> None:
        """Handle /save command - save current session.

        Args:
            name: Optional name for the session (uses title if empty)
            response_pane: ResponsePane to display results
        """
        session = self.state.session
        if session is None:
            response_pane.add_system("No active session to save.")
            return

        if len(session.messages) == 0:
            response_pane.add_system("Cannot save empty session. Add some messages first.")
            return

        # Set session title if name provided
        if name.strip():
            session.title = name.strip()
        elif not session.title:
            # Auto-generate title from first user message
            for msg in session.messages:
                if msg.role == "user":
                    session.title = msg.content[:50] + ("..." if len(msg.content) > 50 else "")
                    break

        try:
            sessions_dir = self.state.get_sessions_dir()
            session.save(sessions_dir)
            response_pane.add_system(
                f"OK Session saved: **{session.title or session.id}**\n\nID: `{session.id}`"
            )
            # Refresh session panel if available
            self._refresh_session_panel()
        except IOError as e:
            response_pane.add_system(f"Failed to save session: {e}")

    async def _handle_load_session(self, name: str, response_pane) -> None:
        """Handle /load command - load a saved session.

        Args:
            name: Session name or ID to load
            response_pane: ResponsePane to display results
        """
        from ...core.session import Session

        sessions_dir = self.state.get_sessions_dir()
        if not sessions_dir.exists():
            response_pane.add_system("No saved sessions found.")
            return

        name = name.strip()

        # Find session by ID or title
        matching_session = None
        for session_file in sessions_dir.glob("*.json"):
            try:
                session = Session.load(session_file)
                # Match by ID (exact or prefix) or title (case-insensitive)
                if (
                    session.id == name
                    or session.id.startswith(name)
                    or (session.title and session.title.lower() == name.lower())
                    or (session.title and name.lower() in session.title.lower())
                ):
                    matching_session = session
                    break
            except (ValueError, IOError):
                continue

        if matching_session is None:
            response_pane.add_system(
                f"Session not found: `{name}`\n\nUse `/sessions` to list available sessions."
            )
            return

        # Replace current session
        self.state.session = matching_session
        response_pane.clear()
        response_pane.add_system(
            f"OK Loaded session: **{matching_session.title or matching_session.id}**\n\n"
            f"Messages: {len(matching_session.messages)}"
        )

        # Replay messages to response pane
        for msg in matching_session.messages:
            if msg.role == "user":
                response_pane.add_user(msg.content)
            elif msg.role == "assistant":
                response_pane.add_assistant(msg.content)
            elif msg.role == "system":
                response_pane.add_system(msg.content)

        self._refresh_session_panel()

    async def _handle_list_sessions(self, response_pane) -> None:
        """Handle /sessions command - list saved sessions."""
        from ...core.session import Session

        sessions_dir = self.state.get_sessions_dir()
        if not sessions_dir.exists():
            response_pane.add_system("No saved sessions found.")
            return

        sessions = []
        for session_file in sessions_dir.glob("*.json"):
            try:
                session = Session.load(session_file)
                sessions.append(session)
            except (ValueError, IOError):
                continue

        if not sessions:
            response_pane.add_system("No saved sessions found.")
            return

        # Sort by updated_at descending (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        lines = ["## Saved Sessions\n"]
        for session in sessions:
            title = session.title or "(untitled)"
            msg_count = len(session.messages)
            updated = session.updated_at.strftime("%Y-%m-%d %H:%M")
            short_id = session.id[:8]
            lines.append(f"- **{title}** ({msg_count} msgs) - `{short_id}` - {updated}")

        lines.append("\n\nUse `/load <name or id>` to load a session.")
        response_pane.add_assistant("\n".join(lines))

    def _refresh_session_panel(self) -> None:
        """Refresh the session panel to show updated sessions list."""
        try:
            from .session_panel import SessionPanel

            panel = self.app.query_one(SessionPanel)
            panel.refresh_sessions()
        except Exception:
            # Panel might not exist or not be visible
            pass

    def _handle_attach(self, path_str: str, response_pane) -> None:
        """Handle /attach command - attach image file(s).

        Args:
            path_str: Path to image file (can include wildcards)
            response_pane: ResponsePane to display results
        """
        from ...core.router import IMAGE_EXTENSIONS

        # Expand path (handle ~ and globs)
        path = Path(path_str).expanduser()

        # Check if it's a glob pattern
        if "*" in path_str:
            parent = path.parent
            pattern = path.name
            if parent.exists():
                matches = list(parent.glob(pattern))
            else:
                matches = []
        elif path.exists():
            matches = [path]
        else:
            response_pane.add_system(f"File not found: `{path_str}`")
            return

        # Filter to only image files
        image_files = [p for p in matches if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]

        if not image_files:
            response_pane.add_system(
                f"No image files found. Supported: {', '.join(IMAGE_EXTENSIONS)}"
            )
            return

        # Add to attachments (with security check)
        added_files = []
        for img in image_files:
            resolved_img = img.resolve()
            if not self._is_path_allowed(resolved_img):
                response_pane.add_system(f"Rejected: `{img.name}` (outside allowed directories)")
                continue
            if resolved_img not in self._attachments:
                self._attachments.append(resolved_img)
                added_files.append(img)

        # Update status to show attachment count
        self._update_attachment_status()

        # Show confirmation
        if added_files:
            names = [p.name for p in added_files]
            response_pane.add_system(
                f"Attached {len(added_files)} image(s): {', '.join(names)}\n"
                f"Total attachments: {len(self._attachments)}"
            )

    def _show_attachments(self, response_pane) -> None:
        """Handle /attachments command - list current attachments."""
        if not self._attachments:
            response_pane.add_system("No attachments. Use `/attach <path>` to add images.")
            return

        lines = ["## Current Attachments\n"]
        for i, path in enumerate(self._attachments, 1):
            size_kb = path.stat().st_size / 1024
            lines.append(f"{i}. `{path.name}` ({size_kb:.1f} KB)")

        lines.append("\n*Use `/detach` to clear all attachments.*")
        response_pane.add_system("\n".join(lines))

    def _clear_attachments(self, response_pane) -> None:
        """Handle /detach command - clear all attachments."""
        count = len(self._attachments)
        self._attachments.clear()
        self._update_attachment_status()

        if count > 0:
            response_pane.add_system(f"Cleared {count} attachment(s).")
        else:
            response_pane.add_system("No attachments to clear.")

    async def _handle_index_search(self, query: str, response_pane) -> None:
        """Handle /index command - search the hybrid RAG index.

        Args:
            query: Search query string (optionally prefixed with source filter)
            response_pane: ResponsePane to display results
        """
        # Parse query for optional source type filter
        source_filter, clean_query = _parse_index_query(query)

        # Check if index exists and has documents
        if self.state.index is None:
            response_pane.add_error(
                "Index not built. Use **Ctrl+R** or click **Reindex** in the Index tab."
            )
            self.notify("Index not built", severity="error")
            return

        stats = self.state.index.get_stats()
        if stats.get("count", 0) == 0:
            response_pane.add_error(
                "Index is empty. Use **Ctrl+R** or click **Reindex** in the Index tab "
                "to index your project files."
            )
            self.notify("Index is empty", severity="error")
            return

        self.set_status("Searching...")
        try:
            # Use async search for proper hybrid (BM25 + vector) search
            results = await self.state.index.search_async(
                clean_query, n_results=5, source_type_filter=source_filter
            )
            if not results:
                response_pane.add_system(f'No results found for: **"{query}"**')
                self.notify("No index results found", severity="information")
                return

            # Format results as markdown
            output_lines = [f"## Search Results for: {query}\n"]
            for i, result in enumerate(results, 1):
                source = Path(result.metadata.get("source", "unknown")).name
                score = result.combined_score
                badge = _format_trust_badge(result.source_type)
                # Truncate content preview to 300 chars
                content_preview = (
                    result.content[:300] + "..." if len(result.content) > 300 else result.content
                )
                output_lines.append(
                    f"### {i}. {badge} {source} (score: {score:.2f})\n```\n{content_preview}\n```\n"
                )

            response_pane.add_assistant("\n".join(output_lines))
            self.notify(f"Found {len(results)} result(s)", severity="information")

        except Exception as e:
            response_pane.add_error(f"Search error: {e}")
            self.notify("Search error", severity="error")
        finally:
            self.set_status("Ready")

    async def _handle_search(self, query: str, response_pane) -> None:
        """Handle /search command - web search via SearXNG.

        Args:
            query: Search query string
            response_pane: ResponsePane to display results
        """
        from ...core.search import SearchError, SearXNGClient

        # Initialize search client if needed
        if self.state.search_client is None:
            endpoint = self.state.config.searxng_endpoint
            self.state.search_client = SearXNGClient(endpoint=endpoint)

        self.set_status("Searching web...")
        try:
            # Check if SearXNG is available
            available = await self.state.search_client.is_available()
            if not available:
                response_pane.add_error(
                    "SearXNG is not available.\n\n"
                    "Make sure SearXNG is running at "
                    f"`{self.state.search_client.endpoint}`\n\n"
                    "You can set a custom endpoint via the `R3LAY_SEARXNG_ENDPOINT` "
                    "environment variable."
                )
                self.notify("SearXNG not available", severity="error")
                return

            # Perform search
            results = await self.state.search_client.search(query, limit=8)

            if not results:
                response_pane.add_system(f'No results found for: **"{query}"**')
                self.notify("No web results found", severity="information")
                return

            # Format results as markdown (inline, simple display)
            output_lines = [f"## Web Search: {query}\n"]
            for i, result in enumerate(results, 1):
                # Truncate snippet to 200 chars
                snippet = result.snippet
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."

                # Format: title, URL, snippet
                output_lines.append(f"**{i}. {result.title}**")
                output_lines.append(f"   {result.url}")
                if snippet:
                    output_lines.append(f"   {snippet}")
                output_lines.append("")  # blank line between results

            response_pane.add_assistant("\n".join(output_lines))
            self.notify(f"Found {len(results)} web result(s)", severity="information")

        except SearchError as e:
            response_pane.add_error(f"Search failed: {e}")
            self.notify("Web search failed", severity="error")
        except Exception as e:
            logger.exception("Web search failed")
            response_pane.add_error(f"Search error: {e}")
            self.notify("Web search error", severity="error")
        finally:
            self.set_status("Ready")

    # =========================================================================
    # Axiom Command Handlers
    # =========================================================================

    async def _handle_create_axiom(self, args: str, response_pane) -> None:
        """Handle /axiom command - create a new axiom.

        Parses optional category prefix (e.g., "spec: statement" or "specifications: statement")
        and checks for potential conflicts before creating.

        Args:
            args: The axiom statement, optionally prefixed with category
            response_pane: ResponsePane to display results
        """
        from ...core.axioms import AXIOM_CATEGORIES

        # Parse optional category prefix
        # Supports both full names and shortcuts: "spec:", "specifications:", etc.
        category_shortcuts = {
            "spec": "specifications",
            "proc": "procedures",
            "compat": "compatibility",
            "diag": "diagnostics",
            "hist": "history",
            "safe": "safety",
        }

        category = "specifications"  # Default category
        statement = args

        # Check for category prefix (e.g., "spec: statement" or "specifications: statement")
        if ":" in args:
            potential_cat, rest = args.split(":", 1)
            potential_cat = potential_cat.strip().lower()

            # Check if it's a valid category or shortcut
            if potential_cat in AXIOM_CATEGORIES:
                category = potential_cat
                statement = rest.strip()
            elif potential_cat in category_shortcuts:
                category = category_shortcuts[potential_cat]
                statement = rest.strip()

        if not statement:
            response_pane.add_system("Error: Axiom statement cannot be empty.")
            return

        # Initialize axiom manager
        axiom_mgr = self.state.init_axioms()

        # Check for potential conflicts
        conflicts = axiom_mgr.find_conflicts(statement, category)

        if conflicts:
            # Show warning about potential conflicts
            conflict_lines = ["## Potential Conflicts Found\n"]
            conflict_lines.append(
                "The following existing axioms may conflict with your statement:\n"
            )
            for ax in conflicts[:5]:  # Show max 5 conflicts
                status_icon = self._get_axiom_status_icon(ax)
                conflict_lines.append(
                    f"- {status_icon} `{ax.id}`: {ax.statement[:80]}..."
                    if len(ax.statement) > 80
                    else f"- {status_icon} `{ax.id}`: {ax.statement}"
                )
            conflict_lines.append(
                "\n*Creating axiom anyway. Use `/dispute <axiom_id> <reason>` if needed.*"
            )
            response_pane.add_system("\n".join(conflict_lines))

        # Create the axiom
        try:
            axiom = axiom_mgr.create(
                statement=statement,
                category=category,
                confidence=0.8,  # Default confidence for user-created axioms
                auto_validate=False,  # Start as PENDING for review
            )

            response_pane.add_assistant(
                f"## Axiom Created\n\n"
                f"- **ID**: `{axiom.id}`\n"
                f"- **Category**: {category}\n"
                f"- **Status**: {axiom.status.value}\n"
                f"- **Statement**: {statement}\n\n"
                f"*Use `/axioms` to view all axioms.*"
            )
            self.notify(f"Axiom created: {axiom.id}", severity="information")

        except ValueError as e:
            response_pane.add_error(f"Failed to create axiom: {e}")
            self.notify("Failed to create axiom", severity="error")
        except Exception as e:
            logger.exception("Error creating axiom")
            response_pane.add_error(f"Error creating axiom: {e}")
            self.notify("Error creating axiom", severity="error")

    async def _handle_list_axioms(self, args: str, response_pane) -> None:
        """Handle /axioms command - list axioms with optional filters.

        Supports:
        - /axioms - List all axioms
        - /axioms <category> - Filter by category
        - /axioms --disputed - Show only disputed axioms

        Args:
            args: Optional category filter or --disputed flag
            response_pane: ResponsePane to display results
        """
        from ...core.axioms import AXIOM_CATEGORIES

        # Initialize axiom manager
        axiom_mgr = self.state.init_axioms()

        # Parse arguments
        show_disputed_only = "--disputed" in args
        args_clean = args.replace("--disputed", "").strip()

        category_filter = None
        if args_clean:
            # Check if it's a valid category
            if args_clean.lower() in AXIOM_CATEGORIES:
                category_filter = args_clean.lower()
            else:
                # Check shortcuts
                category_shortcuts = {
                    "spec": "specifications",
                    "proc": "procedures",
                    "compat": "compatibility",
                    "diag": "diagnostics",
                    "hist": "history",
                    "safe": "safety",
                }
                if args_clean.lower() in category_shortcuts:
                    category_filter = category_shortcuts[args_clean.lower()]
                else:
                    response_pane.add_system(
                        f"Unknown category: `{args_clean}`\n\n"
                        f"Valid categories: {', '.join(AXIOM_CATEGORIES)}"
                    )
                    return

        # Get axioms with filters
        if show_disputed_only:
            axioms = axiom_mgr.get_disputed_axioms()
            title = "Disputed Axioms"
        elif category_filter:
            axioms = axiom_mgr.search(category=category_filter, limit=50)
            title = f"Axioms: {category_filter.title()}"
        else:
            axioms = axiom_mgr.search(limit=50)
            title = "All Axioms"

        if not axioms:
            if show_disputed_only:
                response_pane.add_system("No disputed axioms found.")
            elif category_filter:
                response_pane.add_system(f"No axioms found in category: {category_filter}")
            else:
                response_pane.add_system("No axioms yet. Create one with `/axiom <statement>`")
            return

        # Format as markdown
        lines = [f"## {title}\n"]

        # Group by category if not filtering by category
        if not category_filter:
            by_category: dict[str, list["Axiom"]] = {}
            for ax in axioms:
                by_category.setdefault(ax.category, []).append(ax)

            for cat in AXIOM_CATEGORIES:
                if cat not in by_category:
                    continue
                lines.append(f"### {cat.title()}")
                for ax in by_category[cat]:
                    status_icon = self._get_axiom_status_icon(ax)
                    confidence_pct = int(ax.confidence * 100)
                    lines.append(f"- {status_icon} `{ax.id}` ({confidence_pct}%): {ax.statement}")
                lines.append("")
        else:
            for ax in axioms:
                status_icon = self._get_axiom_status_icon(ax)
                confidence_pct = int(ax.confidence * 100)
                lines.append(f"- {status_icon} `{ax.id}` ({confidence_pct}%): {ax.statement}")

        # Add stats
        stats = axiom_mgr.get_stats()
        lines.append(
            f"\n*Total: {stats['total']} | Active: {stats['active']} | "
            f"Disputed: {stats['disputed']} | Pending: {stats['pending']}*"
        )

        response_pane.add_assistant("\n".join(lines))

    async def _handle_show_citations(self, axiom_id: str, response_pane) -> None:
        """Handle /cite command - show provenance chain for an axiom.

        Args:
            axiom_id: The axiom ID to show citations for
            response_pane: ResponsePane to display results
        """
        # Initialize managers
        axiom_mgr = self.state.init_axioms()
        signals_mgr = self.state.init_signals()

        # Get the axiom
        axiom = axiom_mgr.get(axiom_id.strip())
        if not axiom:
            response_pane.add_system(f"Axiom not found: `{axiom_id}`")
            return

        lines = [f"## Provenance: {axiom_id}\n"]
        lines.append(f"**Statement**: {axiom.statement}\n")
        lines.append(f"**Category**: {axiom.category}")
        lines.append(f"**Status**: {axiom.status.value}")
        lines.append(f"**Confidence**: {int(axiom.confidence * 100)}%")
        lines.append(f"**Created**: {axiom.created_at[:10]}")

        if axiom.validated_at:
            lines.append(f"**Validated**: {axiom.validated_at[:10]}")

        if axiom.dispute_reason:
            lines.append(f"\n**Dispute Reason**: {axiom.dispute_reason}")

        if axiom.superseded_by:
            lines.append(f"**Superseded By**: `{axiom.superseded_by}`")

        if axiom.supersedes:
            lines.append(f"**Supersedes**: `{axiom.supersedes}`")

        # Show citation chain if there are citation_ids
        if axiom.citation_ids:
            lines.append("\n### Source Citations\n")
            for cit_id in axiom.citation_ids:
                chain = signals_mgr.get_citation_chain(cit_id)
                if chain:
                    citation = chain.get("citation", {})
                    lines.append(f"**Citation** `{citation.get('id', cit_id)}`:")
                    lines.append(f"  - Statement: {citation.get('statement', 'N/A')}")
                    lines.append(f"  - Confidence: {int(citation.get('confidence', 0) * 100)}%")

                    for source in chain.get("sources", []):
                        signal = source.get("signal", {})
                        trans = source.get("transmission", {})
                        sig_type = signal.get("type", "unknown")
                        sig_title = signal.get("title", "Untitled")
                        lines.append(f"\n  **Source**: [{sig_type}] {sig_title}")
                        if signal.get("path"):
                            lines.append(f"    - Path: `{signal.get('path')}`")
                        if signal.get("url"):
                            lines.append(f"    - URL: {signal.get('url')}")
                        lines.append(f"    - Location: {trans.get('location', 'N/A')}")
                        if trans.get("excerpt"):
                            excerpt = trans.get("excerpt", "")[:200]
                            lines.append(
                                f'    - Excerpt: "{excerpt}..."'
                                if len(trans.get("excerpt", "")) > 200
                                else f'    - Excerpt: "{excerpt}"'
                            )
                else:
                    lines.append(f"- `{cit_id}` (citation data not found)")
        else:
            lines.append("\n*No citations linked to this axiom.*")

        # Show tags if present
        if axiom.tags:
            lines.append(f"\n**Tags**: {', '.join(axiom.tags)}")

        response_pane.add_assistant("\n".join(lines))

    async def _handle_dispute_axiom(self, axiom_id: str, reason: str, response_pane) -> None:
        """Handle /dispute command - mark an axiom as disputed.

        Args:
            axiom_id: The axiom ID to dispute
            reason: Explanation of why the axiom is disputed
            response_pane: ResponsePane to display results
        """
        # Initialize axiom manager
        axiom_mgr = self.state.init_axioms()

        # Get the axiom first
        axiom = axiom_mgr.get(axiom_id.strip())
        if not axiom:
            response_pane.add_system(f"Axiom not found: `{axiom_id}`")
            return

        # Check if it can be disputed
        if axiom.status.is_terminal:
            response_pane.add_system(
                f"Cannot dispute axiom `{axiom_id}`: "
                f"Status is already terminal ({axiom.status.value})"
            )
            return

        if axiom.is_disputed:
            response_pane.add_system(
                f"Axiom `{axiom_id}` is already disputed.\nCurrent reason: {axiom.dispute_reason}"
            )
            return

        # Dispute the axiom
        updated = axiom_mgr.dispute(axiom_id.strip(), reason.strip())

        if updated:
            response_pane.add_assistant(
                f"## Axiom Disputed\n\n"
                f"- **ID**: `{axiom_id}`\n"
                f"- **Statement**: {updated.statement}\n"
                f"- **Status**: {updated.status.value}\n"
                f"- **Reason**: {reason}\n\n"
                f"*Use `/axioms --disputed` to see all disputed axioms.*"
            )
            self.notify(f"Axiom disputed: {axiom_id}", severity="information")
        else:
            response_pane.add_error(
                f"Failed to dispute axiom `{axiom_id}`. "
                f"It may not be in a state that can be disputed."
            )
            self.notify("Failed to dispute axiom", severity="error")

    async def _handle_research(self, query: str, response_pane) -> None:
        """Handle /research command - run deep research expedition.

        Implements R3 (Retrospective Recursive Research) methodology:
        - Multi-cycle exploration with query generation
        - Web (SearXNG) and RAG parallel searches
        - Axiom extraction with provenance tracking
        - Contradiction detection and resolution cycles
        - Convergence detection (never with unresolved disputes)
        - Report synthesis

        Args:
            query: Research query/question
            response_pane: ResponsePane to display results
        """
        # Check prerequisites
        if self.state.current_backend is None:
            response_pane.add_error(
                "No model loaded. Load a model first to run research expeditions."
            )
            return

        # Initialize orchestrator
        try:
            orchestrator = self.state.init_research()
        except ValueError as e:
            response_pane.add_error(f"Cannot start research: {e}")
            return

        # Start streaming block for research output
        block = response_pane.start_streaming()
        block.append(f"# Research Expedition\n\n**Query:** {query}\n\n")

        self._cancel_requested = False

        try:
            self.set_status("Researching...")

            async for event in orchestrator.run(query):
                # Check for cancellation
                if self._cancel_requested:
                    orchestrator.cancel()
                    block.append("\n\n*[Research cancelled by user]*")
                    break

                # Handle event types
                event_type = event.type
                data = event.data

                if event_type == "started":
                    block.append(f"Starting expedition `{data.get('expedition_id', '???')}`...\n")

                elif event_type == "cycle_start":
                    cycle = data.get("cycle", 0)
                    cycle_type = data.get("type", "exploration")
                    block.append(f"\n## Cycle {cycle} ({cycle_type})\n")

                elif event_type == "queries_generated":
                    queries = data.get("queries", [])
                    if queries:
                        block.append("**Search queries:**\n")
                        for q in queries[:5]:
                            block.append(f"- {q}\n")

                elif event_type == "search_complete":
                    count = data.get("count", 0)
                    block.append(f"Found {count} web results\n")

                elif event_type == "rag_search_complete":
                    count = data.get("count", 0)
                    block.append(f"Found {count} local index results\n")

                elif event_type == "axiom_extracted":
                    statement = data.get("statement", "")[:80]
                    block.append(f"- Extracted: {statement}...\n")

                elif event_type == "axiom_created":
                    axiom_id = data.get("axiom_id", "")
                    block.append(f"  Created axiom: `{axiom_id}`\n")

                elif event_type == "contradiction_detected":
                    block.append("\n**[!!] Contradiction detected:**\n")
                    block.append(f"- Existing: {data.get('existing', '')[:80]}...\n")
                    block.append(f"- New finding: {data.get('new', '')[:80]}...\n")

                elif event_type == "resolution_start":
                    block.append("\n### Resolution Cycle\n")
                    block.append(
                        f"Investigating contradiction `{data.get('contradiction_id', '')}`...\n"
                    )

                elif event_type == "resolution_complete":
                    outcome = data.get("outcome", "unknown")
                    status = data.get("status", "unknown")
                    block.append(f"**Resolution:** {outcome} ({status})\n")

                elif event_type == "cycle_complete":
                    axioms = data.get("axioms", 0)
                    sources = data.get("sources", 0)
                    contradictions = data.get("contradictions", 0)
                    duration = data.get("duration", 0)
                    block.append(f"\nCycle complete: {axioms} axioms, {sources} sources")
                    if contradictions:
                        block.append(f", {contradictions} contradictions")
                    block.append(f" ({duration:.1f}s)\n")

                elif event_type == "converged":
                    reason = data.get("reason", "")
                    cycles = data.get("cycles", 0)
                    block.append(f"\n**Converged** after {cycles} cycles: {reason}\n")

                elif event_type == "synthesizing":
                    block.append("\n## Synthesizing Report...\n")

                elif event_type == "completed":
                    report = data.get("report", "")
                    block.append("\n---\n\n")
                    block.append(report)

                elif event_type == "blocked":
                    reason = data.get("reason", "")
                    pending = data.get("pending_contradictions", 0)
                    block.append(
                        f"\n**[!!] Research blocked:** {reason}\n"
                        f"*{pending} contradiction(s) require manual review.*\n"
                        f"Use `/axioms --disputed` to see disputed axioms.\n"
                    )

                elif event_type == "cancelled":
                    block.append("\n*Research expedition cancelled.*\n")

                elif event_type == "failed":
                    message = data.get("message", "Unknown error")
                    block.append(f"\n**Error:** {message}\n")

                elif event_type == "status":
                    message = data.get("message", "")
                    block.append(f"*{message}*\n")

        except Exception as e:
            logger.exception("Research expedition failed")
            block.append(f"\n**Error:** {e}\n")

        finally:
            block.finish()
            self.set_status("Ready")

    def _get_axiom_status_icon(self, axiom: "Axiom") -> str:
        """Get a status icon for an axiom.

        Args:
            axiom: The axiom to get an icon for

        Returns:
            A text-based status indicator
        """
        from ...core.axioms import AxiomStatus

        icons = {
            AxiomStatus.VALIDATED: "[OK]",
            AxiomStatus.RESOLVED: "[OK]",
            AxiomStatus.PENDING: "[??]",
            AxiomStatus.DISPUTED: "[!!]",
            AxiomStatus.SUPERSEDED: "[->]",
            AxiomStatus.REJECTED: "[XX]",
            AxiomStatus.INVALIDATED: "[XX]",
        }
        return icons.get(axiom.status, "[??]")

    def _get_routing_decision(
        self,
        message: str,
        retrieved_context: list["RetrievalResult"] | None = None,
    ) -> "RoutingDecision | None":
        """Get routing decision from SmartRouter.

        Args:
            message: User message text
            retrieved_context: Optional RAG results

        Returns:
            RoutingDecision object describing the routing decision, or None if no router
        """
        if self.state.router is None:
            return None

        try:
            decision = self.state.router.route(
                message=message,
                attachments=self._attachments,
                retrieved_context=retrieved_context,
            )

            # Log the decision for debugging
            logger.info(
                "Routing decision: %s (score: %.2f, switched: %s) - %s",
                decision.model_type,
                decision.vision_score,
                decision.switched,
                decision.reason,
            )

            return decision

        except Exception as e:
            logger.warning("Router error: %s", e)
            return None

    async def _execute_model_switch(self, decision: "RoutingDecision") -> bool:
        """Execute a model switch based on routing decision.

        Args:
            decision: RoutingDecision indicating target model type

        Returns:
            True if switch succeeded, False otherwise
        """
        self.set_status(f"Switching to {decision.model_type} model...")

        try:
            # Get configured model name from app config
            app = self.app
            if not hasattr(app, "config") or not hasattr(app.config, "model_roles"):
                logger.warning("App config not available for model switch")
                return False

            roles = app.config.model_roles
            target_model_name = (
                roles.vision_model if decision.model_type == "vision" else roles.text_model
            )

            if not target_model_name:
                logger.warning(f"No {decision.model_type} model configured")
                return False

            # Find ModelInfo in available models
            model_info = next(
                (m for m in self.state.available_models if m.name == target_model_name), None
            )

            if not model_info:
                logger.warning(f"Model not found: {target_model_name}")
                return False

            # Execute switch with timeout (30 seconds)
            await asyncio.wait_for(self.state.load_model(model_info), timeout=30.0)

            logger.info(f"Switched to {decision.model_type}: {target_model_name}")
            return True

        except asyncio.TimeoutError:
            logger.error("Model switch timed out")
            return False
        except Exception as e:
            logger.error(f"Model switch failed: {e}")
            return False

    def _find_model_by_name(self, model_name: str):
        """Find a model in available_models by exact or fuzzy name match.

        Matching priority:
        1. Exact match (case-insensitive)
        2. Prefix match (model name starts with query)
        3. Contains match (query is substring of model name)

        Args:
            model_name: User-provided model name (e.g., "mistral", "llama3")

        Returns:
            ModelInfo if found, None otherwise
        """
        if not self.state.available_models:
            return None

        model_name_lower = model_name.lower()

        # Try exact match first (case-insensitive)
        for model in self.state.available_models:
            if model.name.lower() == model_name_lower:
                return model

        # Try prefix match (e.g., "mistral" matches "mistral-7b-v0.3")
        for model in self.state.available_models:
            if model.name.lower().startswith(model_name_lower):
                return model

        # Try contains match (e.g., "qwen" matches "qwen2.5-7b-instruct")
        for model in self.state.available_models:
            if model_name_lower in model.name.lower():
                return model

        return None

    async def _handle_model_swap(self, intent_result: "IntentResult", response_pane) -> bool:
        """Handle natural language model swap command.

        Examples:
        - "swap model mistral"
        - "use llama"
        - "load qwen2.5-7b"
        - "switch to gemma"

        Args:
            intent_result: Parsed intent with model_name entity
            response_pane: ResponsePane to display results

        Returns:
            True if model was swapped successfully, False otherwise
        """
        model_name = intent_result.entities.get("model_name")

        if not model_name:
            response_pane.add_system(
                "**Model name not specified.**\n\n"
                "Usage examples:\n"
                "- `swap model mistral`\n"
                "- `use llama`\n"
                "- `load qwen2.5-7b`"
            )
            return False

        # Check if any models are available
        if not self.state.available_models:
            response_pane.add_error(
                "No models available. Check the **Models** tab to discover models."
            )
            return False

        # Find matching model
        model_info = self._find_model_by_name(model_name)

        if not model_info:
            # List available models for user
            available_names = [m.name for m in self.state.available_models[:10]]
            names_str = "\n".join(f"- `{n}`" for n in available_names)
            more = (
                f"\n- *...and {len(self.state.available_models) - 10} more*"
                if len(self.state.available_models) > 10
                else ""
            )
            response_pane.add_system(
                f"**Model not found:** `{model_name}`\n\nAvailable models:\n{names_str}{more}"
            )
            return False

        # Execute the model swap
        self.set_status(f"Loading {model_info.name}...")
        response_pane.add_system(f"Loading model: **{model_info.name}**...")

        try:
            await asyncio.wait_for(
                self.state.load_model(model_info),
                timeout=60.0,  # Longer timeout for model loading
            )

            # Success - show confirmation
            response_pane.add_assistant(
                f"**MODEL LOADED:** `{model_info.name}` "
                "░░░░░░░░░░░░░░░░░░░░░░░░░░░ READY\n\n"
                f"*{model_info.backend_type.value} backend • "
                f"{'Vision capable' if model_info.is_vision_model else 'Text only'}*"
            )
            logger.info(f"Model swapped via conversation: {model_info.name}")
            return True

        except asyncio.TimeoutError:
            response_pane.add_error(
                f"Model load timed out: `{model_info.name}`\n\n"
                "The model may still be loading. Check the **Models** tab."
            )
            logger.error(f"Model swap timed out: {model_info.name}")
            return False
        except Exception as e:
            response_pane.add_error(f"Failed to load model: {e}")
            logger.exception(f"Model swap failed: {model_info.name}")
            return False

    def _get_project_context(self) -> "ProjectContext | None":
        """Extract project context from the current project path.

        Uses the project path to determine if this is an automotive project,
        software project, etc. and extract relevant context for citations.

        Returns:
            ProjectContext if context can be extracted, None otherwise.
        """
        from ...core.project_context import extract_project_context

        if self.state.project_path is None:
            return None

        try:
            return extract_project_context(self.state.project_path)
        except Exception as e:
            logger.warning("Failed to extract project context: %s", e)
            return None

    def _get_available_source_types(self) -> list["SourceType"] | None:
        """Get list of source types present in the index.

        Scans the index to determine what types of sources are available
        (e.g., INDEXED_MANUAL, INDEXED_CODE, INDEXED_DOCUMENT).

        Returns:
            List of unique SourceType values if index exists, None otherwise.
        """

        if self.state.index is None:
            return None

        try:
            # Get unique source types from indexed chunks
            # The index stores source_type in chunk metadata
            source_types: set[SourceType] = set()

            # Check if chunks have source_type information
            for chunk in self.state.index._chunks.values():
                if hasattr(chunk, "source_type") and chunk.source_type:
                    source_types.add(chunk.source_type)

            if source_types:
                return list(source_types)
            return None
        except Exception as e:
            logger.warning("Failed to get available source types: %s", e)
            return None

    async def _handle_chat(self, message: str, response_pane) -> None:
        """Handle chat messages with LLM backend.

        Uses Session for conversation history and SmartRouter for model selection.
        Supports streaming responses and cancellation via Escape key.
        Includes source attribution with trust-based citation guidelines.
        """
        # Check if backend is loaded
        if not hasattr(self.state, "current_backend") or not self.state.current_backend:
            response_pane.add_error(
                "No model loaded. Select a model from the **Models** tab and click **Load**."
            )
            return

        # Ensure session exists
        session = self.state.session
        if session is None:
            # This shouldn't happen as R3LayState initializes session, but be safe
            from ...core.session import Session

            session = Session(project_path=self.state.project_path)
            self.state.session = session

        # Get routing decision (returns full RoutingDecision now)
        decision = self._get_routing_decision(message)

        if decision:
            # Show routing info in status
            if decision.switched:
                # Need to switch models before generating
                switch_success = await self._execute_model_switch(decision)
                if not switch_success:
                    # Warn but continue with current model
                    response_pane.add_system(
                        f"Could not switch to {decision.model_type} model. "
                        "Continuing with current model."
                    )
            else:
                # Just informational
                self.set_status(f"Router: {decision.model_type} - {decision.reason}")
                logger.debug(f"Router: {decision.model_type} - {decision.reason}")

        # Extract project context for citation customization
        project_context = self._get_project_context()

        # Collect source types from index if available (for citation guidance)
        source_types_present = self._get_available_source_types()

        # Ensure system prompt with citation instructions is set
        # Only add if this is a new conversation (no existing system message)
        has_system_prompt = any(msg.role == "system" for msg in session.messages)
        if not has_system_prompt:
            system_prompt = session.get_system_prompt_with_citations(
                project_context=project_context,
                source_types_present=source_types_present,
            )
            session.add_system_message(system_prompt)
            logger.debug("Added citation-aware system prompt")

        # Add user message to session
        session.add_user_message(
            content=message,
            images=self._attachments if self._attachments else None,
        )

        # Get formatted messages for LLM
        conversation = session.get_messages_for_llm(max_tokens=8000)

        # Start streaming block
        block = response_pane.start_streaming()
        response_text = ""

        try:
            self.set_status("Generating...")

            async for token in self.state.current_backend.generate_stream(
                conversation,
                max_tokens=1024,
                temperature=0.7,
                images=self._attachments if self._attachments else None,
            ):
                if self._cancel_requested:
                    block.append("\n\n*[Generation cancelled]*")
                    break
                block.append(token)
                response_text += token

        except Exception as e:
            block.finish()
            response_pane.add_error(f"Generation error: {e}")
            # Remove the user message from session on error
            if session.messages and session.messages[-1].role == "user":
                session.messages.pop()
            return

        finally:
            block.finish()

        # Add assistant response to session
        model_name = self.state.current_model or "unknown"
        was_cancelled = self._cancel_requested and response_text

        if response_text and not self._cancel_requested:
            session.add_assistant_message(
                content=response_text,
                model=model_name,
            )
        elif was_cancelled:
            # Add partial response with cancellation note
            session.add_assistant_message(
                content=response_text + "\n\n*[Generation cancelled]*",
                model=model_name,
                metadata={"was_cancelled": True},
            )

    async def _handle_maintenance_intent(
        self, intent_result: "IntentResult", response_pane
    ) -> None:
        """Handle natural language maintenance logging intent.

        Converts intent entities to /log command format and delegates to handler.

        Args:
            intent_result: Parsed intent with entities
            response_pane: ResponsePane to display results
        """
        entities = intent_result.entities

        # Extract mileage (required)
        mileage = entities.get("mileage")
        if mileage is None:
            response_pane.add_system(
                "Could not extract mileage from your message. "
                "Please specify mileage, e.g., 'at 98k miles' or 'at 98,500'."
            )
            return

        # Extract service type from entities or infer from part
        service_type = entities.get("service_type") or entities.get("part")
        if service_type is None:
            # Try to infer from the original text
            service_type = "general_maintenance"

        # Normalize service_type to match maintenance log expectations
        # Convert part names to service types (e.g., "engine_oil" -> "oil_change")
        service_map = {
            "engine_oil": "oil_change",
            "oil_filter": "oil_change",
            "coolant": "coolant_flush",
            "brake_fluid": "brake_fluid_flush",
            "air_filter": "air_filter_replacement",
            "fuel_filter": "fuel_filter_replacement",
            "spark_plugs": "spark_plug_replacement",
            "timing_belt": "timing_belt_replacement",
            "brake_pads": "brake_pad_replacement",
        }
        service_type = service_map.get(service_type, service_type)

        # Build args string for handler
        args = f"{service_type} {mileage}"

        # Log what we're doing for user feedback
        logger.info(
            f"Maintenance intent: service_type={service_type}, mileage={mileage}, "
            f"confidence={intent_result.confidence:.2f}"
        )

        # Delegate to existing handler
        await self._handle_log_maintenance(args, response_pane)

    async def _handle_query_intent(self, intent_result: "IntentResult", response_pane) -> None:
        """Handle natural language maintenance query intent.

        Routes to appropriate query handler based on subtype.

        Args:
            intent_result: Parsed intent with entities
            response_pane: ResponsePane to display results
        """
        subtype = intent_result.subtype
        entities = intent_result.entities

        logger.info(f"Query intent: subtype={subtype}, confidence={intent_result.confidence:.2f}")

        # Route based on query subtype
        if subtype == "query.reminder":
            # "When is oil change due?" -> show due services
            mileage = entities.get("mileage")
            args = str(mileage) if mileage else ""
            await self._handle_due_services(args, response_pane)

        elif subtype == "query.history":
            # "When did I last change oil?" -> show history
            service_type = entities.get("service_type") or entities.get("part")
            args = service_type if service_type else ""
            await self._handle_maintenance_history(args, response_pane)

        elif subtype == "query.status":
            # "What's the current mileage?" -> show mileage status
            mileage = entities.get("mileage")
            args = str(mileage) if mileage else ""
            await self._handle_update_mileage(args, response_pane)

        else:
            # Default: show due services
            mileage = entities.get("mileage")
            args = str(mileage) if mileage else ""
            await self._handle_due_services(args, response_pane)

    async def _handle_mileage_update_intent(
        self, intent_result: "IntentResult", response_pane
    ) -> None:
        """Handle natural language mileage update intent.

        Converts intent entities to /mileage command format.

        Args:
            intent_result: Parsed intent with entities
            response_pane: ResponsePane to display results
        """
        entities = intent_result.entities
        mileage = entities.get("mileage")

        if mileage is None:
            response_pane.add_system(
                "Could not extract mileage from your message. "
                "Please specify mileage, e.g., 'mileage is 98k' or 'at 98,500 miles'."
            )
            return

        logger.info(
            f"Mileage update intent: mileage={mileage}, confidence={intent_result.confidence:.2f}"
        )

        # Delegate to existing handler
        await self._handle_update_mileage(str(mileage), response_pane)

    async def _handle_log_maintenance(self, args: str, response_pane) -> None:
        """Handle /log command - log a maintenance entry.

        Args:
            args: Command arguments (service_type mileage [--parts] [--notes] [--cost] [--shop])
            response_pane: ResponsePane to display results
        """
        from pathlib import Path

        from ...core.maintenance import MaintenanceEntry, MaintenanceLog

        # Parse arguments
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            response_pane.add_system(
                "Usage: `/log <service_type> <mileage>`\n\nExample: `/log oil_change 50000`"
            )
            return

        service_type = parts[0].lower()
        try:
            mileage = int(parts[1].split()[0])
        except (ValueError, IndexError):
            response_pane.add_system(
                f"Invalid mileage: must be a positive integer. Got: {parts[1]}"
            )
            return

        # Create MaintenanceLog instance
        # Use a reasonable default project path if state.get_project_path doesn't exist
        try:
            project_path = Path.home() / ".r3lay"
        except Exception:
            project_path = Path.home() / ".r3lay"

        log = MaintenanceLog(project_path)

        # Check if service_type is valid
        if service_type not in log.intervals:
            available_services = ", ".join(sorted(log.intervals.keys())[:5])
            response_pane.add_system(
                f"Invalid service type: `{service_type}`\n\n"
                f"Available services: {available_services}... and {len(log.intervals) - 5} more\n\n"
                f"Use `/axioms` to list all available service types."
            )
            return

        # Create and add entry
        try:
            entry = MaintenanceEntry(
                service_type=service_type,
                mileage=mileage,
            )
            log.add_entry(entry)

            # Display success
            response_pane.add_assistant(
                f"✓ **Maintenance logged**\n\n"
                f"- Service: `{service_type}`\n"
                f"- Mileage: {mileage:,} miles\n"
                f"- Date: {entry.date.strftime('%Y-%m-%d %H:%M')}\n"
            )
        except Exception as e:
            response_pane.add_system(f"Failed to log maintenance: {e}")

    async def _handle_due_services(self, args: str, response_pane) -> None:
        """Handle /due command - show due/overdue services.

        Args:
            args: Optional current mileage
            response_pane: ResponsePane to display results
        """
        from pathlib import Path

        from ...core.maintenance import MaintenanceLog

        # Parse mileage if provided
        current_mileage = None
        if args and args.strip():
            try:
                current_mileage = int(args.strip())
            except ValueError:
                response_pane.add_system("Invalid mileage: must be a positive integer")
                return

        # Use default mileage if not provided
        if current_mileage is None:
            current_mileage = 100000  # Default placeholder

        # Create MaintenanceLog instance
        try:
            project_path = Path.home() / ".r3lay"
        except Exception:
            project_path = Path.home() / ".r3lay"

        log = MaintenanceLog(project_path)

        # Get upcoming services
        try:
            upcoming = log.get_upcoming(current_mileage)

            if not upcoming:
                response_pane.add_assistant(
                    f"## Services Due (at {current_mileage:,} miles)\n\n"
                    "No services found. Check service definitions."
                )
                return

            # Build table
            lines = [f"## Services Due (at {current_mileage:,} miles)\n"]
            lines.append("| Service | Miles Due | Interval | Severity |")
            lines.append("|---------|-----------|----------|----------|")

            for service_due in upcoming:
                interval = service_due.interval
                miles_text = (
                    str(service_due.miles_until_due)
                    if service_due.miles_until_due is not None
                    else "N/A"
                )

                # Highlight overdue in red
                if service_due.is_overdue:
                    miles_text = f"[red]{miles_text} ⚠️[/red]"

                lines.append(
                    f"| {interval.service_type} | {miles_text} | "
                    f"{interval.interval_miles:,} mi | {interval.severity} |"
                )

            response_pane.add_assistant("\n".join(lines))
        except Exception as e:
            response_pane.add_system(f"Failed to get due services: {e}")

    async def _handle_maintenance_history(self, args: str, response_pane) -> None:
        """Handle /history command - show maintenance history.

        Args:
            args: Optional service_type and limit flags
            response_pane: ResponsePane to display results
        """
        from pathlib import Path

        from ...core.maintenance import MaintenanceLog

        # Parse arguments
        service_type = None
        limit = 10

        if args and args.strip():
            parts = args.split()
            for i, part in enumerate(parts):
                if part == "--limit" and i + 1 < len(parts):
                    try:
                        limit = int(parts[i + 1])
                    except ValueError:
                        pass
                elif not part.startswith("--"):
                    service_type = part.lower()

        # Create MaintenanceLog instance
        try:
            project_path = Path.home() / ".r3lay"
        except Exception:
            project_path = Path.home() / ".r3lay"

        log = MaintenanceLog(project_path)

        # Get history
        try:
            history = log.get_history(limit=limit, service_type=service_type)

            if not history:
                if service_type:
                    response_pane.add_assistant(f"No history found for `{service_type}`")
                else:
                    response_pane.add_assistant("No maintenance history found.")
                return

            # Build table
            title = f"Maintenance History ({len(history)} entries)"
            if service_type:
                title += f" - {service_type}"

            lines = [f"## {title}\n"]
            lines.append("| Date | Service | Mileage | Notes |")
            lines.append("|------|---------|---------|-------|")

            for entry in history:
                date_str = entry.date.strftime("%Y-%m-%d")
                notes = (
                    entry.notes[:30] + "..."
                    if entry.notes and len(entry.notes) > 30
                    else (entry.notes or "-")
                )
                lines.append(f"| {date_str} | {entry.service_type} | {entry.mileage:,} | {notes} |")

            response_pane.add_assistant("\n".join(lines))
        except Exception as e:
            response_pane.add_system(f"Failed to get history: {e}")

    async def _handle_update_mileage(self, args: str, response_pane) -> None:
        """Handle /mileage command - update or show current mileage.

        Args:
            args: Optional new mileage value
            response_pane: ResponsePane to display results
        """
        from pathlib import Path

        from ...core.maintenance import MaintenanceLog

        # Parse mileage if provided
        new_mileage = None
        if args and args.strip():
            try:
                new_mileage = int(args.strip())
                if new_mileage < 0:
                    response_pane.add_system("Mileage must be a positive integer")
                    return
            except ValueError:
                response_pane.add_system("Invalid mileage: must be a positive integer")
                return

        # Create MaintenanceLog instance
        try:
            project_path = Path.home() / ".r3lay"
        except Exception:
            project_path = Path.home() / ".r3lay"

        log = MaintenanceLog(project_path)

        # If no specific mileage provided, use a default
        if new_mileage is None:
            new_mileage = 100000

        try:
            # Get upcoming services at this mileage
            upcoming = log.get_upcoming(new_mileage)
            overdue = log.get_overdue(new_mileage)

            lines = [f"## Current Mileage: {new_mileage:,} miles\n"]

            if overdue:
                lines.append(f"⚠️  **{len(overdue)} Service(s) Overdue**\n")
                for service_due in overdue[:3]:
                    interval = service_due.interval
                    lines.append(
                        f"- `{interval.service_type}` - {interval.interval_miles:,} mi intervals"
                    )

            # Show next due
            coming_due = [s for s in upcoming if s.miles_until_due and s.miles_until_due > 0]
            if coming_due:
                lines.append("\n**Next Services Due**\n")
                for service_due in coming_due[:3]:
                    interval = service_due.interval
                    miles_left = service_due.miles_until_due
                    lines.append(
                        f"- `{interval.service_type}` - {miles_left:,} miles away "
                        f"({interval.interval_miles:,} mi interval)"
                    )

            response_pane.add_assistant("\n".join(lines))
        except Exception as e:
            response_pane.add_system(f"Failed to get mileage info: {e}")


__all__ = ["InputPane"]
