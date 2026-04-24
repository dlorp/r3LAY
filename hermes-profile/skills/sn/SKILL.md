---
name: sn
description: Session end -- compress transcript to sn.md, update todos/questions, bump version if code changed, update CHANGELOG, optionally create PR. THE session close command.
version: 2.0.0
author: r3LAY
license: MIT
metadata:
  hermes:
    tags: [r3lay, session, versioning, changelog, pr-workflow]
    related_skills: [r3-context, r3-plan, pr-workflow, compile]
---

# /sn -- session end + version + PR workflow

Trigger: user types /sn
This is THE session close command. Does not run silently.

## Step 1: Session compression

- Compress session transcript -> .r3lay/sn.md
  Extract: decisions made, todos completed, new todos, open questions resolved/new
- Update todos.md:
  - Mark completed items [x]
  - Add new items from session
- Update open-questions.md:
  - Mark resolved questions
  - Add new questions from session
- Confirm: "Session captured. [{N} todos, {M} questions, {K} decisions]"

## Step 2: Re-compile project context

After updating sn.md, call `mcp_r3lay_compile_project(project_id=..., write=True)`
to refresh .r3lay/compiled.md with the latest session state. This ensures
the next session cold-start has an up-to-date compiled document ready.

Report: "Compiled: {files} files, {decisions} decisions. compiled.md updated."

## Step 3: Detect code changes

- Check git status in project folder
- If no code changes: stop here. Session captured.
- If code changes detected: proceed to Step 4

## Step 4: Version bump

Grep entire repo for version references. Update ALL atomically.
Files to check: pyproject.toml, setup.py, __init__.py, package.json, Cargo.toml

Semver rules:
- Patch (0.0.X): bug fixes, minor improvements, dependency bumps
- Minor (0.X.0): new features, new endpoints, new skills
- Major (X.0.0): breaking changes, architecture shifts
- Ask user if ambiguous: "Confirm bump type? [patch/minor/major]"
- NEVER use [Unreleased]

## Step 5: CHANGELOG.md update

keepachangelog format with sections: Added, Changed, Fixed, Security.
Write above the previous version entry. No [Unreleased] section.

## Step 6: Prompt for PR

If no PR created this session:
"Code changes detected. Create a PR? [y/n]"
- yes -> proceed to PR workflow (/pr-workflow skill)
- no -> "Changes committed locally. Run /sn again to create PR later."

## Step 7: Final sn.md update

After PR created: update sn.md with "PR #{number} created -- {title} -- awaiting review"
Update plans.md: "Session closed. PR pending merge."
Mark "create PR" as complete in todos.md if applicable.
