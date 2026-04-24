"""PR workflow — branch management, review dispatch, commit, push, CI monitoring.

All git operations use file_read/file_write tools or the bridge API.
All commits attributed to dlorp. No AI markers. No Co-Authored-By.
NEVER push to main. NEVER auto-merge. PRs created as draft.

The 3-agent review (s3ntry, 3tch, r4bbit) is dispatched by the Hermes skill,
not by this module directly — r3LAY cannot route to other agents.
This module provides the data structures and helpers the skill needs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Result from a single review agent."""

    agent: str  # s3ntry | 3tch | r4bbit
    verdict: str  # PASS | FAIL | WARN
    findings: list[str] = field(default_factory=list)
    doc_links: list[str] = field(default_factory=list)


@dataclass
class ReviewReport:
    """Consolidated report from all review agents."""

    results: list[ReviewResult]

    @property
    def has_failures(self) -> bool:
        return any(r.verdict == "FAIL" for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.verdict == "WARN" for r in self.results)

    @property
    def all_pass(self) -> bool:
        return all(r.verdict == "PASS" for r in self.results)

    def format_summary(self) -> str:
        """Format the review report for display."""
        lines = [
            "Review Results:",
            "-" * 40,
        ]
        for r in self.results:
            label = {"s3ntry": "security", "3tch": "code", "r4bbit": "validation"}.get(
                r.agent, r.agent
            )
            lines.append(f"{r.agent} ({label}):  [{r.verdict}] -- {len(r.findings)} findings")

        fail_items = [f for r in self.results if r.verdict == "FAIL" for f in r.findings]
        warn_items = [f for r in self.results if r.verdict == "WARN" for f in r.findings]

        if fail_items:
            lines.append("")
            lines.append("Issues requiring fix:")
            for item in fail_items:
                lines.append(f"  - {item}")

        if warn_items:
            lines.append("")
            lines.append("Warnings (recommended):")
            for item in warn_items:
                lines.append(f"  - {item}")

        return "\n".join(lines)


def generate_branch_name(project_slug: str, brief_slug: str = "") -> str:
    """Generate a feature branch name.

    Args:
        project_slug: Project identifier (e.g., 'r3lay').
        brief_slug: Brief description slug (e.g., 'phase3-session').

    Returns:
        Branch name like 'feat/r3lay/2026-04-07-phase3-session'.
    """
    date = datetime.now().strftime("%Y-%m-%d")
    slug = brief_slug or "session"
    # Sanitize slug for git branch name
    slug = re.sub(r"[^a-zA-Z0-9-]", "-", slug.lower()).strip("-")
    return f"feat/{project_slug}/{date}-{slug}"


def format_commit_message(
    commit_type: str,
    scope: str,
    description: str,
) -> str:
    """Format a conventional commit message.

    No Co-Authored-By. No AI markers. dlorp attribution only.

    Args:
        commit_type: feat | fix | chore | refactor | docs | test
        scope: Component scope (e.g., 'bridge', 'conflict').
        description: Brief description of the change.

    Returns:
        Formatted commit message string.
    """
    return f"{commit_type}({scope}): {description}"


def format_pr_body(
    what: str,
    why: str,
    how: str,
    key_changes: list[tuple[str, str]],
    code_snippet: str | None = None,
    code_language: str = "python",
    testing: str = "",
    review_report: ReviewReport | None = None,
    version_info: dict | None = None,
) -> str:
    """Format a PR body.

    Args:
        what: What this PR does.
        why: Why this change was needed.
        how: Brief technical approach.
        key_changes: List of (file, description) tuples.
        code_snippet: Optional code snippet.
        code_language: Language for the code block.
        testing: What was tested.
        review_report: Consolidated review results.
        version_info: Dict with old_version, new_version.

    Returns:
        Formatted PR body string.
    """
    lines = [
        "## What",
        what,
        "",
        "## Why",
        why,
        "",
        "## How",
        how,
        "",
        "### Key changes",
    ]

    for file, desc in key_changes:
        lines.append(f"- `{file}`: {desc}")

    if code_snippet:
        lines.append("")
        lines.append("### Code snippets")
        lines.append(f"```{code_language}")
        lines.append(code_snippet)
        lines.append("```")

    lines.append("")
    lines.append("## Testing")
    lines.append(testing or "Manual verification.")

    lines.append("")
    lines.append("## Review checklist")

    if review_report:
        for r in review_report.results:
            label = {"s3ntry": "security", "3tch": "code", "r4bbit": "validation"}.get(
                r.agent, r.agent
            )
            lines.append(f"- [x] {r.agent} {label} review: {r.verdict}")
    else:
        lines.append("- [ ] s3ntry security review: pending")
        lines.append("- [ ] 3tch code review: pending")
        lines.append("- [ ] r4bbit validation: pending")

    lines.append("- [ ] CI checks: pending")

    if version_info:
        old = version_info.get("old_version", "?")
        new = version_info.get("new_version", "?")
        lines.append(f"- [x] Version bumped: {old} -> {new}")
        lines.append(f"- [x] CHANGELOG.md updated: {new}")

    return "\n".join(lines)


# =============================================================================
# Review prompts for Hermes skill dispatch
# =============================================================================

SECURITY_REVIEW_PROMPT = """Security review of the following diff. Check for:
- Injection vulnerabilities (SQL, command, path traversal)
- Secrets or API keys accidentally included
- Unsafe file operations outside sandbox
- Git safety (no force push patterns, no credential exposure)
- Known vulnerability patterns for dependencies used
Diff:
{diff}
Return: [PASS|FAIL] + findings list. FAIL requires fix before PR."""

CODE_REVIEW_PROMPT = """Code review of your own implementation. Check for:
- Logic errors, edge cases not handled
- Consistency with existing patterns in codebase
- Missing error handling
- Performance issues
- Test coverage gaps (what should have a test but doesn't)
Diff:
{diff}
Return: [PASS|FAIL|WARN] + findings list.
FAIL = must fix. WARN = should fix. PASS = good to go."""

WEB_VALIDATION_PROMPT = """Validate the APIs and library patterns used in this diff against
current documentation. Use web_search to verify:
- Each external API call exists and has the correct signature
- Library versions used match available documentation
- No deprecated methods used
- Patterns match current best practices
Diff:
{diff}
Return: [PASS|FAIL|WARN] + findings list with documentation links."""
