#!/usr/bin/env python3
"""OpenClaw query processor for r3LAY.

This script is used by the OpenClaw agent (lorp) to process queries
from r3LAY's OpenClawAdapter.

Usage:
    # List pending queries
    python openclaw_processor.py list
    
    # Show a specific query
    python openclaw_processor.py show <request_id>
    
    # Respond to a query
    python openclaw_processor.py respond <request_id> "<response_content>"
    
    # Respond from file
    python openclaw_processor.py respond <request_id> --file response.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_BASE = Path.home() / ".r3lay" / "openclaw"


def list_pending(base_path: Path) -> list[dict]:
    """List all pending queries."""
    pending_dir = base_path / "pending"
    if not pending_dir.exists():
        return []
    
    queries = []
    for f in sorted(pending_dir.glob("*.json"), key=lambda x: x.stat().st_mtime):
        try:
            data = json.loads(f.read_text())
            data["_file"] = f.name
            queries.append(data)
        except (json.JSONDecodeError, IOError):
            pass
    return queries


def show_query(base_path: Path, request_id: str) -> dict | None:
    """Show a specific query."""
    query_file = base_path / "pending" / f"{request_id}.json"
    if not query_file.exists():
        return None
    return json.loads(query_file.read_text())


def respond(base_path: Path, request_id: str, content: str, tokens: int | None = None) -> bool:
    """Write a response to a query."""
    done_dir = base_path / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    
    response_file = done_dir / f"{request_id}.json"
    response_data = {
        "request_id": request_id,
        "content": content,
        "done": True,
        "timestamp": time.time(),
        "tokens": tokens,
    }
    response_file.write_text(json.dumps(response_data, indent=2))
    return True


def format_messages(messages: list[dict]) -> str:
    """Format messages for display."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"[{role}]\n{content}\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Process r3LAY OpenClaw queries")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Base path for OpenClaw files")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # List command
    subparsers.add_parser("list", help="List pending queries")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show a query")
    show_parser.add_argument("request_id", help="Request ID to show")
    
    # Respond command
    respond_parser = subparsers.add_parser("respond", help="Respond to a query")
    respond_parser.add_argument("request_id", help="Request ID to respond to")
    respond_parser.add_argument("content", nargs="?", help="Response content")
    respond_parser.add_argument("--file", "-f", type=Path, help="Read response from file")
    respond_parser.add_argument("--tokens", "-t", type=int, help="Token count (optional)")
    
    args = parser.parse_args()
    base_path = args.base
    
    if args.command == "list":
        queries = list_pending(base_path)
        if not queries:
            print("No pending queries.")
            return
        
        print(f"Found {len(queries)} pending queries:\n")
        for q in queries:
            age = time.time() - q.get("timestamp", 0)
            msgs = q.get("messages", [])
            last_msg = msgs[-1].get("content", "")[:80] if msgs else ""
            print(f"  {q['request_id']}")
            print(f"    Age: {age:.1f}s | Messages: {len(msgs)}")
            print(f"    Last: {last_msg}...")
            print()
    
    elif args.command == "show":
        query = show_query(base_path, args.request_id)
        if not query:
            print(f"Query {args.request_id} not found.", file=sys.stderr)
            sys.exit(1)
        
        print(f"Request ID: {query['request_id']}")
        print(f"Timestamp: {query.get('timestamp', 'unknown')}")
        print(f"System Prompt: {query.get('system_prompt', 'None')}")
        print(f"Temperature: {query.get('temperature', 'default')}")
        print(f"Max Tokens: {query.get('max_tokens', 'None')}")
        print("\n--- Messages ---\n")
        print(format_messages(query.get("messages", [])))
    
    elif args.command == "respond":
        if args.file:
            content = args.file.read_text()
        elif args.content:
            content = args.content
        else:
            print("Reading response from stdin (Ctrl+D to finish):")
            content = sys.stdin.read()
        
        if respond(base_path, args.request_id, content, args.tokens):
            print(f"Response written for {args.request_id}")
        else:
            print("Failed to write response.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
