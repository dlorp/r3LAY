"""Input pane - user input area with chat integration.

Features:
- Multi-turn conversation history via Session
- Smart routing between text/vision models (informational)
- Streaming LLM responses
- Escape key to cancel generation
- Command handling (/, /clear, /help, etc.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static, TextArea

if TYPE_CHECKING:
    from ...core import R3LayState
    from ...core.index import RetrievalResult
    from ...core.project_context import ProjectContext
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
        self.set_status("Processing..." if processing else "Ready")

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

            if value.startswith("/"):
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
                "- `/help` - Show this help\n"
                "- `/status` - Show system status\n"
                "- `/index <query>` - Search knowledge base\n"
                "- `/clear` - Clear chat and conversation history\n"
                "- `/session` - Show session info\n"
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
    ) -> str | None:
        """Get routing decision from SmartRouter (informational only).

        Args:
            message: User message text
            retrieved_context: Optional RAG results

        Returns:
            Status message describing the routing decision, or None if no router
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

            # Return status message
            if decision.switched:
                return f"Router: Switch to {decision.model_type} - {decision.reason}"
            else:
                return f"Router: {decision.model_type} - {decision.reason}"

        except Exception as e:
            logger.warning("Router error: %s", e)
            return None

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

        # Get routing decision (informational - we don't switch models yet)
        routing_status = self._get_routing_decision(message)
        if routing_status:
            self.set_status(routing_status)
            # Also show in response pane as a subtle indicator
            logger.debug(routing_status)

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
