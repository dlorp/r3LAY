"""Input pane - user input area with chat integration.

Features:
- Multi-turn conversation history via Session
- Smart routing between text/vision models (informational)
- Streaming LLM responses
- Escape key to cancel generation
- Command handling (/, /clear, /help, etc.)
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

if TYPE_CHECKING:
    from ...core import R3LayState
    from ...core.index import RetrievalResult
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

    def compose(self) -> ComposeResult:
        yield TextArea(id="input-area")
        with Horizontal(id="input-controls"):
            yield Static("Ready", id="input-status")
            yield Button("Send", id="send-button", variant="primary")

    def focus_input(self) -> None:
        self.query_one("#input-area", TextArea).focus()

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
        self.query_one("#send-button", Button).disabled = processing
        if processing:
            self.set_status("Processing...")
        else:
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
        import tempfile
        import time

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
            from Foundation import NSData

            pb = NSPasteboard.generalPasteboard()

            # Log all available types for debugging
            available_types = pb.types()
            logger.debug(f"Clipboard types available: {list(available_types) if available_types else 'None'}")

            # Image types to try, in order of preference
            # Browsers often provide multiple formats
            image_types = [
                NSPasteboardTypePNG,      # public.png - preferred
                NSPasteboardTypeTIFF,     # public.tiff - common fallback
                "public.jpeg",            # JPEG format
                "com.compuserve.gif",     # GIF format
                "org.webmproject.webp",   # WebP format
            ]

            for img_type in image_types:
                data = pb.dataForType_(img_type)
                if data is not None:
                    logger.debug(f"Found clipboard image data as type: {img_type}, size: {data.length()} bytes")

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
                        if path.exists() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'}:
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

            logger.debug(f"PIL grabclipboard() returned: {type(image).__name__ if image else 'None'}")

            if image is None:
                return None

            # Check if it's actually an image (not a file list on some systems)
            if not hasattr(image, 'save'):
                # On macOS, grabclipboard can return file paths as a list
                if isinstance(image, list) and len(image) > 0:
                    logger.debug(f"PIL returned file list: {image}")
                    # It's a list of file paths
                    path = Path(image[0])
                    if path.exists() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}:
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

        paste_text_preview = repr(event.text[:100] if event.text else '')
        logger.debug(f"on_paste called with text: {paste_text_preview}")

        # ALWAYS check for clipboard image first (browser copies put both image AND url in clipboard)
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
        logger.debug(f"No clipboard image detected, falling back to text paste handling")

        # No clipboard image - check if pasted text is a file path
        text = event.text.strip()
        if not text:
            logger.debug("Empty paste text, ignoring")
            return

        # SINGLE LINE CHECK: Detect single file path paste (most common case)
        # This handles Finder drag-drop or pasting a copied path
        if '\n' not in text:
            # Clean up potential quotes or file:// prefix
            clean_path = text.replace("file://", "").strip("'\"")
            logger.debug(f"Single line paste, checking path: {repr(clean_path)}")

            try:
                path = Path(clean_path).expanduser()
                if path.exists() and path.is_file():
                    logger.debug(f"Path exists as file: {path}, suffix: {path.suffix.lower()}")
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
                        logger.debug(f"File is not an image (suffix: {path.suffix}), allowing paste as text")
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

        for line in lines:
            # Clean up potential file:// prefix or quotes
            clean_path = line.replace("file://", "").strip("'\"")

            try:
                path = Path(clean_path).expanduser()
                if path.exists() and path.is_file():
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
            "/Users/", "/home/", "/tmp/", "/var/", "/etc/",
            "/opt/", "/usr/", "/Library/", "/Applications/",
            "/Volumes/", "/private/", "/System/"
        )

        # Also check for tilde expansion
        if text.startswith("~/"):
            return True

        return text.startswith(path_prefixes)

    async def submit(self) -> None:
        if self._processing:
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

            # Check if this is a command (e.g., /help, /attach) vs a file path (e.g., /Users/...)
            if value.startswith("/") and not self._looks_like_path(value):
                await self._handle_command(value, response_pane)
            else:
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
                "- `/session` - Show session info\n\n"
                "**Attachments**\n"
                "- `/attach <path>` - Attach image file(s) to next message\n"
                "- `/attachments` - List current attachments\n"
                "- `/detach` - Clear all attachments\n\n"
                "**Search & Research**\n"
                "- `/index <query>` - Search knowledge base\n"
                "- `/search <query>` - Web search (not implemented)\n"
                "- `/research <query>` - Deep research expedition (not implemented)\n"
            )
        elif cmd == "clear":
            response_pane.clear()
            self.clear_conversation()
            response_pane.add_system("Chat and conversation history cleared.")
        elif cmd == "index":
            if not args:
                response_pane.add_system("Usage: `/index <query>`")
                return
            await self._handle_index_search(args, response_pane)
        elif cmd == "session":
            self._show_session_info(response_pane)
        elif cmd == "status":
            self._show_status(response_pane)
        elif cmd == "attach":
            if not args:
                response_pane.add_system("Usage: `/attach <path>` - attach image file(s)")
                return
            self._handle_attach(args, response_pane)
        elif cmd == "attachments":
            self._show_attachments(response_pane)
        elif cmd == "detach":
            self._clear_attachments(response_pane)
        else:
            response_pane.add_system(f"Command `/{cmd}` not implemented yet.")

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
        image_files = [
            p for p in matches
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        ]

        if not image_files:
            response_pane.add_system(
                f"No image files found. Supported: {', '.join(IMAGE_EXTENSIONS)}"
            )
            return

        # Add to attachments
        for img in image_files:
            if img not in self._attachments:
                self._attachments.append(img)

        # Update status to show attachment count
        self._update_attachment_status()

        # Show confirmation
        names = [p.name for p in image_files]
        response_pane.add_system(
            f"Attached {len(image_files)} image(s): {', '.join(names)}\n"
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

        lines.append(f"\n*Use `/detach` to clear all attachments.*")
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
            return

        stats = self.state.index.get_stats()
        if stats.get("count", 0) == 0:
            response_pane.add_error(
                "Index is empty. Use **Ctrl+R** or click **Reindex** in the Index tab "
                "to index your project files."
            )
            return

        self.set_status("Searching...")
        try:
            # Use async search for proper hybrid (BM25 + vector) search
            results = await self.state.index.search_async(
                clean_query, n_results=5, source_type_filter=source_filter
            )
            if not results:
                response_pane.add_system(f'No results found for: **"{query}"**')
                return

            # Format results as markdown
            output_lines = [f"## Search Results for: {query}\n"]
            for i, result in enumerate(results, 1):
                source = Path(result.metadata.get("source", "unknown")).name
                score = result.combined_score
                badge = _format_trust_badge(result.source_type)
                # Truncate content preview to 300 chars
                content_preview = (
                    result.content[:300] + "..."
                    if len(result.content) > 300
                    else result.content
                )
                output_lines.append(
                    f"### {i}. {badge} {source} (score: {score:.2f})\n"
                    f"```\n{content_preview}\n```\n"
                )

            response_pane.add_assistant("\n".join(output_lines))

        except Exception as e:
            response_pane.add_error(f"Search error: {e}")
        finally:
            self.set_status("Ready")

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
                roles.vision_model if decision.model_type == "vision"
                else roles.text_model
            )

            if not target_model_name:
                logger.warning(f"No {decision.model_type} model configured")
                return False

            # Find ModelInfo in available models
            model_info = next(
                (m for m in self.state.available_models
                 if m.name == target_model_name),
                None
            )

            if not model_info:
                logger.warning(f"Model not found: {target_model_name}")
                return False

            # Execute switch with timeout (30 seconds)
            await asyncio.wait_for(
                self.state.load_model(model_info),
                timeout=30.0
            )

            logger.info(f"Switched to {decision.model_type}: {target_model_name}")
            return True

        except asyncio.TimeoutError:
            logger.error("Model switch timed out")
            return False
        except Exception as e:
            logger.error(f"Model switch failed: {e}")
            return False

    def _get_project_context(self) -> "ProjectContext | None":
        """Extract project context from the current project path.

        Uses the project path to determine if this is an automotive project,
        software project, etc. and extract relevant context for citations.

        Returns:
            ProjectContext if context can be extracted, None otherwise.
        """
        from ...core.project_context import ProjectContext, extract_project_context

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
        from ...core.sources import SourceType

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
        has_system_prompt = any(
            msg.role == "system" for msg in session.messages
        )
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


__all__ = ["InputPane"]
