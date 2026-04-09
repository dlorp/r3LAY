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

## Step 2: Detect code changes

- Check git status in project folder
- If no code changes: stop here. Session captured.
- If code changes detected: proceed to Step 3

## Step 3: Version bump

Grep entire repo for version references. Update ALL atomically.
Files to check: pyproject.toml, setup.py, __init__.py, package.json, Cargo.toml

Semver rules:
- Patch (0.0.X): bug fixes, minor improvements, dependency bumps
- Minor (0.X.0): new features, new endpoints, new skills
- Major (X.0.0): breaking changes, architecture shifts
- Ask user if ambiguous: "Confirm bump type? [patch/minor/major]"
- NEVER use [Unreleased]

## Step 4: CHANGELOG.md update

keepachangelog format with sections: Added, Changed, Fixed, Security.
Write above the previous version entry. No [Unreleased] section.

## Step 5: Prompt for PR

If no PR created this session:
"Code changes detected. Create a PR? [y/n]"
- yes -> proceed to PR workflow (/pr-workflow skill)
- no -> "Changes committed locally. Run /sn again to create PR later."

## Step 6: Final sn.md update

After PR created: update sn.md with "PR #{number} created -- {title} -- awaiting review"
Update plans.md: "Session closed. PR pending merge."
Mark "create PR" as complete in todos.md if applicable.
