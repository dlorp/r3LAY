"""Input pane - user input area."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static, TextArea

if TYPE_CHECKING:
    from ...app import R3LayState


class InputPane(Vertical):
    """User input pane with multi-line text area."""

    DEFAULT_CSS = """
    InputPane {
        width: 100%;
        height: 40%;
        min-height: 8;
        border: solid $primary-darken-2;
        border-title-color: $primary;
        padding: 1;
    }
    
    #input-area {
        height: 1fr;
        min-height: 4;
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

    def __init__(self, state: "R3LayState", **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self._processing = False

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
            self.focus_input()

    async def _handle_command(self, command: str, response_pane) -> None:
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "help":
            response_pane.add_assistant(
                "## Commands\n\n"
                "- `/help` - Show commands\n"
                "- `/search <query>` - Web search\n"
                "- `/index <query>` - Search knowledge base\n"
                "- `/research <query>` - Deep research expedition\n"
                "- `/axiom <statement>` - Add axiom\n"
                "- `/axioms [tags]` - List axioms\n"
                "- `/update <key> <value>` - Update registry\n"
                "- `/issue <desc>` - Add known issue\n"
                "- `/mileage <value>` - Update odometer\n"
                "- `/clear` - Clear chat\n"
            )
        elif cmd == "search" and args:
            await self._do_web_search(args, response_pane)
        elif cmd == "index" and args:
            await self._do_index_search(args, response_pane)
        elif cmd == "research" and args:
            await self._do_research(args, response_pane)
        elif cmd == "axiom" and args:
            axiom = self.state.axioms.create(
                statement=args, category="specifications", confidence=0.8
            )
            response_pane.add_assistant(f"✓ Added axiom `{axiom.id}`\n\n> {args}")
        elif cmd == "axioms":
            tags = [t.strip() for t in args.split(",")] if args else None
            axioms = self.state.axioms.search(tags=tags, limit=15)
            if axioms:
                lines = ["## Axioms\n"]
                for ax in axioms:
                    status = "✓" if ax.is_validated else "○"
                    lines.append(f"- [{status}] {ax.statement} ({int(ax.confidence * 100)}%)")
                response_pane.add_assistant("\n".join(lines))
            else:
                response_pane.add_assistant("No axioms found.")
        elif cmd == "update" and args:
            try:
                key, value = args.split(maxsplit=1)
                from ruamel.yaml import YAML

                yaml = YAML()
                parsed = yaml.load(io.StringIO(value))
                self.state.registry.set(key, parsed if parsed is not None else value)
                self.state.registry.save()
                response_pane.add_assistant(f"✓ Updated `{key}`")
            except ValueError:
                response_pane.add_error("Usage: `/update <key> <value>`")
        elif cmd == "issue" and args:
            self.state.registry.add_known_issue(args)
            self.state.registry.save()
            response_pane.add_assistant(f"✓ Added issue: {args}")
        elif cmd == "mileage" and args:
            try:
                val = int(args.replace(",", ""))
                self.state.registry.update_odometer(val)
                self.state.registry.save()
                response_pane.add_assistant(f"✓ Odometer: **{val:,}** miles")
            except ValueError:
                response_pane.add_error("Invalid mileage")
        elif cmd == "clear":
            response_pane.clear()
            self.state.session_manager.end_session()
            self.state.session_manager.new_session()
            response_pane.add_system("Chat cleared.")
        else:
            response_pane.add_error(f"Unknown: `{cmd}`. Try `/help`")

    async def _handle_chat(self, message: str, response_pane) -> None:
        if not self.state.llm_client:
            response_pane.add_error("No model selected. Use **Models** tab.")
            return

        rag_context = await self._get_rag_context(message)
        axiom_context = self.state.axioms.get_context_for_llm()
        system_prompt = self._build_system_prompt(rag_context, axiom_context)

        from ...core import Message

        messages = []
        session = self.state.session_manager.current
        if session:
            for msg in session.messages[-10:]:
                if msg.role != "system":
                    messages.append(Message(role=msg.role, content=msg.content))
        messages.append(Message(role="user", content=message))

        try:
            response = await self.state.llm_client.chat(
                messages=messages, system_prompt=system_prompt
            )
            response_pane.add_assistant(response.content)
        except Exception as e:
            response_pane.add_error(f"LLM error: {e}")

    async def _get_rag_context(self, query: str) -> str:
        try:
            results = self.state.index.search(query, n_results=3, min_relevance=0.3)
            if results:
                parts = ["**Context:**"]
                for r in results:
                    parts.append(f"\n*[{r.metadata.get('source', '?')}]*\n{r.content[:400]}")
                return "\n".join(parts)
        except Exception:
            pass
        return ""

    def _build_system_prompt(self, rag_context: str, axiom_context: str) -> str:
        parts = ["You are r3LAY, a research assistant. Be concise, use markdown."]
        if self.state.registry.exists():
            parts.append(f"\n## Project\n{self.state.registry.get_summary()}")
        if axiom_context:
            parts.append(f"\n{axiom_context}")
        if rag_context:
            parts.append(f"\n{rag_context}")
        return "\n".join(parts)

    async def _do_web_search(self, query: str, response_pane) -> None:
        self.set_status("Searching...")
        try:
            results = await self.state.searxng.search(query, limit=5)
            if results:
                lines = [f"## Web: {query}\n"]
                for r in results:
                    lines.append(f"**[{r.title}]({r.url})**\n{r.snippet[:150]}...\n")
                response_pane.add_assistant("\n".join(lines))
            else:
                response_pane.add_assistant(f"No results for: {query}")
        except Exception as e:
            response_pane.add_error(f"Search failed: {e}")

    async def _do_index_search(self, query: str, response_pane) -> None:
        results = self.state.index.search(query, n_results=5)
        if results:
            lines = [f"## Index: {query}\n"]
            for r in results:
                lines.append(
                    f"**{r.metadata.get('source', '?')}** ({int(r.final_score * 100)}%)\n```\n{r.content[:250]}...\n```\n"
                )
            response_pane.add_assistant("\n".join(lines))
        else:
            response_pane.add_assistant(f"No results for: {query}")

    async def _do_research(self, query: str, response_pane) -> None:
        if not self.state.llm_client:
            response_pane.add_error("Select a model first.")
            return

        from ...core import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            project_path=self.state.project_path,
            llm=self.state.llm_client,
            index=self.state.index,
            search=self.state.searxng,
            signals=self.state.signals,
            axioms=self.state.axioms,
            min_cycles=self.state.config.research.min_cycles,
            max_cycles=self.state.config.research.max_cycles,
        )

        response_pane.add_system(f"▸ Expedition: {query}")

        try:
            async for event in orchestrator.run(query):
                if event["type"] == "cycle_start":
                    self.set_status(f"Cycle {event['cycle']}...")
                elif event["type"] == "cycle_complete":
                    response_pane.add_system(
                        f"Cycle {event['cycle']}: {event['axioms']} axioms, {event['sources']} sources"
                    )
                elif event["type"] == "converged":
                    response_pane.add_system(f"◂ {event['reason']}")
                elif event["type"] == "completed":
                    response_pane.add_assistant(event["report"])
                elif event["type"] == "error":
                    response_pane.add_error(event["message"])
        except Exception as e:
            response_pane.add_error(f"Research error: {e}")
