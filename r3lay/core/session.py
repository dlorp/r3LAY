"""Session management for r3LAY.

Preserves full conversation history across model switches.
LLMs are stateless - each call receives the COMPLETE conversation.
Switching models only loses KV cache (2-3s rebuild), not history.

Features:
- Message dataclass with role, content, images, and metadata
- Session dataclass with message history and serialization
- Token budget management for LLM context windows
- JSON persistence for session recovery
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .axioms import AxiomManager
    from .project_context import ProjectContext
    from .sources import SourceType


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: Message role - user, assistant, or system
        content: Text content of the message
        images: Optional list of image paths (for vision models)
        model_used: Name of the model that generated this (assistant only)
        timestamp: When the message was created
        metadata: Additional information (e.g., was_cancelled, token_count)
    """

    role: Literal["user", "assistant", "system"]
    content: str
    images: list[Path] | None = None
    model_used: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_llm_format(self) -> dict[str, str]:
        """Convert to LLM message format (role + content).

        Note: Images are handled separately by vision-capable backends.
        """
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON persistence."""
        return {
            "role": self.role,
            "content": self.content,
            "images": [str(p) for p in self.images] if self.images else None,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Deserialize from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            images=[Path(p) for p in data["images"]] if data.get("images") else None,
            model_used=data.get("model_used"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """A conversation session with message history.

    Key insight: LLMs are stateless! Each call receives the COMPLETE
    conversation history. Switching models only loses KV cache (2-3s
    rebuild), not the actual history.

    Attributes:
        id: Unique session identifier (UUID)
        messages: List of messages in chronological order
        created_at: When the session was created
        updated_at: When the session was last modified
        title: Optional session title (auto-generated from first message)
        project_path: Path to the project this session is associated with
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    title: str | None = None
    project_path: Path | None = None

    def add_user_message(
        self,
        content: str,
        images: list[Path] | None = None,
    ) -> Message:
        """Add a user message to the session.

        Args:
            content: Text content of the message
            images: Optional list of image paths for vision models

        Returns:
            The created Message
        """
        msg = Message(
            role="user",
            content=content,
            images=images,
        )
        self.messages.append(msg)
        self.updated_at = datetime.now()

        # Auto-generate title from first user message
        if self.title is None and content:
            self.title = content[:50] + ("..." if len(content) > 50 else "")

        return msg

    def add_assistant_message(
        self,
        content: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Add an assistant message to the session.

        Args:
            content: Text content of the response
            model: Name of the model that generated this response
            metadata: Optional metadata (e.g., was_cancelled, token_count)

        Returns:
            The created Message
        """
        msg = Message(
            role="assistant",
            content=content,
            model_used=model,
            metadata=metadata or {},
        )
        self.messages.append(msg)
        self.updated_at = datetime.now()
        return msg

    def add_system_message(self, content: str) -> Message:
        """Add a system message (usually at the start).

        Args:
            content: System prompt content

        Returns:
            The created Message
        """
        msg = Message(role="system", content=content)
        self.messages.append(msg)
        self.updated_at = datetime.now()
        return msg

    def get_messages_for_llm(
        self,
        max_tokens: int = 8000,
        include_system: bool = True,
    ) -> list[dict[str, str]]:
        """Format full history for LLM (they are stateless!).

        Truncates from the beginning (oldest messages) if over budget,
        always preserving the system message if present.

        Args:
            max_tokens: Maximum estimated tokens to include
            include_system: Whether to include system messages

        Returns:
            List of {"role": str, "content": str} dicts for LLM API
        """
        if not self.messages:
            return []

        # Rough token estimation (words * 1.3)
        def estimate_tokens(text: str) -> int:
            return int(len(text.split()) * 1.3)

        # Separate system messages (always keep first one)
        system_msgs: list[Message] = []
        other_msgs: list[Message] = []

        for msg in self.messages:
            if msg.role == "system" and include_system:
                system_msgs.append(msg)
            elif msg.role != "system":
                other_msgs.append(msg)

        # Start with system message budget
        result: list[dict[str, str]] = []
        total_tokens = 0

        # Add first system message (if any)
        if system_msgs:
            sys_msg = system_msgs[0]
            result.append(sys_msg.to_llm_format())
            total_tokens += estimate_tokens(sys_msg.content)

        # Add messages from most recent, then reverse
        # This ensures we keep the most recent context
        selected: list[dict[str, str]] = []
        for msg in reversed(other_msgs):
            msg_tokens = estimate_tokens(msg.content)
            if total_tokens + msg_tokens > max_tokens:
                break
            selected.append(msg.to_llm_format())
            total_tokens += msg_tokens

        # Reverse to get chronological order
        result.extend(reversed(selected))

        return result

    def get_last_user_images(self) -> list[Path]:
        """Get images from the most recent user message (for vision routing)."""
        for msg in reversed(self.messages):
            if msg.role == "user" and msg.images:
                return msg.images
        return []

    def clear(self) -> None:
        """Clear all messages from the session."""
        self.messages = []
        self.updated_at = datetime.now()

    def get_system_prompt_with_citations(
        self,
        project_context: "ProjectContext | None" = None,
        source_types_present: list["SourceType"] | None = None,
        axiom_manager: "AxiomManager | None" = None,
    ) -> str:
        """Generate system prompt with citation instructions.

        Creates a system prompt that instructs the LLM on how to cite
        different source types with appropriate confidence levels.

        Args:
            project_context: Optional project context for customization
                           (e.g., automotive project with vehicle details).
                           Use extract_project_context(path) to create this.
            source_types_present: Optional list of source types available
                                 in the current retrieval results
            axiom_manager: Optional axiom manager to include validated
                          knowledge in the prompt context

        Returns:
            Formatted system prompt string with citation guidelines.

        Example:
            >>> from r3lay.core.project_context import extract_project_context
            >>> session = Session()
            >>> ctx = extract_project_context(Path("/projects/automotive/outback"))
            >>> prompt = session.get_system_prompt_with_citations(project_context=ctx)
        """
        base = """r³LAY — Retrospective Recursive Research, Linked Archive Yield

You are the knowledge interface for r³LAY, a system that bridges official documentation with real world community knowledge. You operate as infrastructure, not as a persona.

## Core Philosophy

Official sources (manuals, datasheets) provide specifications. Community knowledge (forums, experience) reveals what actually works. The gap between them is where real value lives.

Your role is to:
- Synthesize official specs with community-proven practices
- Track provenance — every claim links to its source
- Present confidence levels based on source quality
- Note when community experience diverges from official documentation

## System Behaviors

- Respond with factual, source-attributed information
- Use declarative statements, not conversational filler
- Reference the active project context when relevant
- Cite sources with file paths, signal IDs, or URLs
- Flag low-confidence or contested information explicitly
- When official and community sources conflict, present both with context

## Response Format

- Lead with the direct answer
- Follow with source attribution and confidence level
- Note consensus when multiple sources agree
- When sources conflict, explain the discrepancy
- Suggest related queries or commands if appropriate

## Do Not

- Use first-person emotional language ("I think", "I feel", "I'd love to")
- Add conversational filler ("Great question!", "Sure thing!")
- Apologize or hedge unnecessarily
- Pretend to have preferences or opinions
- Present community knowledge as official fact (or vice versa)

## Source Trust Hierarchy

1. **Indexed local documents (highest)** — Your project's own files
   - "According to the indexed service manual..."
   - "The local documentation states..."

2. **Official OE sources (high)** — Manufacturer documentation
   - "Per [manufacturer] specifications..."
   - "The factory service manual indicates..."

3. **Trusted community (medium)** — Forums, established guides
   - "Community consensus from [forum] suggests..."
   - "Experienced users report..."

4. **General web (lower)** — Web search results
   - "Web sources indicate (verify independently)..."

When sources agree, note the consensus. When they conflict, present the discrepancy:
- "Official spec is X, but community reports Y works better for [condition]"

## Confidence Indicators

Use these when presenting information:
- **High confidence**: Multiple official sources agree, or indexed local document
- **Medium confidence**: Single official source, or strong community consensus
- **Low confidence**: Community-only, conflicting sources, or sparse data
- **Disputed**: Direct contradiction between sources (flag for review)
"""

        # Add project-specific context
        if project_context:
            base += f"\n## Active Project Context\n"
            base += f"Project: {project_context.project_name}\n"
            base += f"Type: {project_context.project_type}\n"

            if project_context.project_type == "automotive":
                ref = project_context.project_reference
                base += f"- Reference as: {ref}\n"
                if project_context.vehicle_make:
                    base += f"- Make: {project_context.vehicle_make}\n"
                if project_context.vehicle_model:
                    base += f"- Model: {project_context.vehicle_model}\n"
                if project_context.vehicle_year:
                    base += f"- Year: {project_context.vehicle_year}\n"
                base += f'- Example: "{project_context.possessive} service manual specifies..."\n'

            elif project_context.project_type == "electronics":
                board = project_context.metadata.get("board", "")
                if board:
                    base += f"- Board/Platform: {board}\n"
                base += "- Cite datasheets with part numbers\n"
                base += "- Reference pinouts and specifications precisely\n"
                base += "- Note voltage levels and current requirements\n"

            elif project_context.project_type == "software":
                lang = project_context.metadata.get("language", "")
                if lang:
                    base += f"- Language: {lang}\n"
                base += "- Cite code snippets with file paths\n"
                base += "- Reference documentation with section names\n"
                base += "- Note API versions when relevant\n"

            elif project_context.project_type == "workshop":
                base += "- Reference materials with dimensions and specifications\n"
                base += "- Cite tool requirements and safety notes\n"
                base += "- Note measurement units (imperial/metric)\n"

            elif project_context.project_type == "home":
                base += "- Reference building codes when applicable\n"
                base += "- Cite safety requirements explicitly\n"
                base += "- Note professional requirements (licensed work)\n"

            else:  # general
                base += "- Cite sources with file paths or URLs\n"
                base += "- Reference relevant documentation sections\n"

        # Add information about available source types
        if source_types_present:

            local_sources = [s for s in source_types_present if s.is_local]
            web_sources = [s for s in source_types_present if s.is_web]

            if local_sources or web_sources:
                base += "\n## Available Sources\n"
                if local_sources:
                    source_names = ", ".join(s.value for s in local_sources)
                    base += f"- Local indexed: {source_names}\n"
                if web_sources:
                    source_names = ", ".join(s.value for s in web_sources)
                    base += f"- Web sources: {source_names}\n"

        # Add validated axioms context
        if axiom_manager:
            axiom_context = axiom_manager.get_context_for_llm(max_axioms=15)
            if axiom_context:
                base += "\n" + axiom_context

        return base

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dictionary for JSON persistence."""
        return {
            "id": self.id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "title": self.title,
            "project_path": str(self.project_path) if self.project_path else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Deserialize session from dictionary."""
        session = cls(
            id=data["id"],
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            title=data.get("title"),
            project_path=Path(data["project_path"]) if data.get("project_path") else None,
        )
        return session

    def save(self, sessions_dir: Path) -> Path:
        """Save session to JSON file.

        Args:
            sessions_dir: Directory to save session files

        Returns:
            Path to the saved session file

        Raises:
            IOError: If file cannot be written
        """
        try:
            sessions_dir.mkdir(parents=True, exist_ok=True)
            session_file = sessions_dir / f"{self.id}.json"

            # Write to temp file first for atomic save
            temp_file = session_file.with_suffix(".json.tmp")
            temp_file.write_text(json.dumps(self.to_dict(), indent=2))
            temp_file.replace(session_file)

            return session_file
        except OSError as e:
            raise IOError(f"Failed to save session to {sessions_dir}: {e}") from e

    @classmethod
    def load(cls, session_file: Path) -> "Session":
        """Load session from JSON file.

        Args:
            session_file: Path to session JSON file

        Returns:
            Loaded Session instance

        Raises:
            FileNotFoundError: If session file doesn't exist
            ValueError: If JSON is malformed or missing required fields
            IOError: If file cannot be read
        """
        try:
            data = json.loads(session_file.read_text())
            return cls.from_dict(data)
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid session file format: {e}") from e
        except KeyError as e:
            raise ValueError(f"Session file missing required field: {e}") from e
        except OSError as e:
            raise IOError(f"Failed to read session file: {e}") from e


__all__ = [
    "Message",
    "Session",
]
