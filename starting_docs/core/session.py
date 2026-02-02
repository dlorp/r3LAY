"""Session management for chat conversations."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


@dataclass
class ChatMessage:
    """A message in a chat session."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """A chat session with metadata."""

    id: str
    title: str
    created: str
    messages: list[ChatMessage] = field(default_factory=list)
    summary: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def updated(self) -> str:
        """Get the timestamp of the last message."""
        if self.messages:
            return self.messages[-1].timestamp
        return self.created


class SessionManager:
    """Manages chat sessions and their persistence."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.sessions_path = project_path / "sessions"
        self.sessions_path.mkdir(parents=True, exist_ok=True)
        self.yaml = YAML()
        self.yaml.default_flow_style = False

        self._current: Session | None = None

    @property
    def current(self) -> Session | None:
        """Get the current active session."""
        return self._current

    def new_session(self, title: str | None = None) -> Session:
        """Create a new chat session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        title = title or f"Session {session_id}"

        self._current = Session(
            id=session_id,
            title=title,
            created=datetime.now().isoformat(),
        )

        return self._current

    def add_message(self, role: str, content: str, **metadata: Any) -> ChatMessage:
        """Add a message to the current session."""
        if not self._current:
            self.new_session()

        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata,
        )

        self._current.messages.append(message)
        return message

    def save(self, session: Session | None = None) -> Path:
        """Save a session to disk."""
        session = session or self._current
        if not session:
            raise ValueError("No session to save")

        title_slug = "".join(c if c.isalnum() else "-" for c in session.title.lower()).strip("-")[
            :50
        ]
        filename = f"session_{session.id}_{title_slug}.yaml"
        session_path = self.sessions_path / filename

        data = {
            "id": session.id,
            "title": session.title,
            "created": session.created,
            "summary": session.summary,
            "tags": session.tags,
            "metadata": session.metadata,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in session.messages
            ],
        }

        with open(session_path, "w") as f:
            self.yaml.dump(data, f)

        return session_path

    def load(self, session_id: str) -> Session:
        """Load a session from disk."""
        for path in self.sessions_path.glob(f"session_{session_id}_*.yaml"):
            return self._load_file(path)
        raise FileNotFoundError(f"Session not found: {session_id}")

    def _load_file(self, path: Path) -> Session:
        """Load a session from a file path."""
        with open(path) as f:
            data = self.yaml.load(f)

        messages = [
            ChatMessage(
                role=m["role"],
                content=m["content"],
                timestamp=m.get("timestamp", ""),
                metadata=m.get("metadata", {}),
            )
            for m in data.get("messages", [])
        ]

        return Session(
            id=data["id"],
            title=data["title"],
            created=data["created"],
            messages=messages,
            summary=data.get("summary"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions with metadata."""
        sessions = []

        for path in sorted(self.sessions_path.glob("session_*.yaml"), reverse=True):
            try:
                with open(path) as f:
                    data = self.yaml.load(f)
                    sessions.append(
                        {
                            "id": data["id"],
                            "title": data["title"],
                            "created": data["created"],
                            "summary": data.get("summary"),
                            "tags": data.get("tags", []),
                            "message_count": len(data.get("messages", [])),
                            "path": str(path),
                        }
                    )
            except Exception:
                continue

            if len(sessions) >= limit:
                break

        return sessions

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search sessions by content."""
        results = []
        query_lower = query.lower()

        for path in self.sessions_path.glob("session_*.yaml"):
            try:
                with open(path) as f:
                    data = self.yaml.load(f)

                matches = False
                if query_lower in data.get("title", "").lower():
                    matches = True
                if query_lower in (data.get("summary") or "").lower():
                    matches = True

                for msg in data.get("messages", []):
                    if query_lower in msg.get("content", "").lower():
                        matches = True
                        break

                if matches:
                    results.append(
                        {
                            "id": data["id"],
                            "title": data["title"],
                            "created": data["created"],
                            "summary": data.get("summary"),
                            "path": str(path),
                        }
                    )

            except Exception:
                continue

            if len(results) >= limit:
                break

        return results

    def get_context(self, max_sessions: int = 3) -> str:
        """Get recent session context for LLM."""
        lines = ["# Recent Sessions"]

        sessions = self.list_sessions(limit=max_sessions)
        for info in sessions:
            lines.append(f"\n## {info['title']} ({info['created'][:10]})")
            if info.get("summary"):
                lines.append(info["summary"])

        return "\n".join(lines)

    def end_session(self, generate_summary: bool = True) -> Path | None:
        """End the current session and save it."""
        if not self._current:
            return None

        if generate_summary and not self._current.summary:
            # Simple extractive summary
            user_msgs = [m for m in self._current.messages if m.role == "user"]
            if user_msgs:
                first_msg = user_msgs[0].content[:200]
                self._current.summary = f"Discussion starting with: {first_msg}..."

        path = self.save()
        self._current = None
        return path
