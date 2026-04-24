# /pr-workflow -- PR creation with 3-agent review

Invoked from /sn Step 5, or explicitly by user.
All git operations via gh CLI. All commits attributed to dlorp.
NEVER push to main. NEVER auto-merge. NEVER create PR without user confirmation.

## Step 1: Branch management

- If on main: create feature branch
  branch_name = feat/{project_slug}/{date}-{brief_slug}
- If already on feature branch: use it
- Never create branch named main, master, or production

## Step 2: 3-agent review (parallel)

Dispatch three reviews simultaneously. r3LAY prepares the diff and review
prompts, then the Hermes runtime dispatches to:

**s3ntry (security):**
- Injection vulnerabilities, secrets exposure, sandbox violations
- Git safety (no force push, no credential exposure)
- Return: [PASS|FAIL] + findings

**3tch (code):**
- Logic errors, edge cases, consistency, error handling
- Performance, test coverage gaps
- Return: [PASS|FAIL|WARN] + findings

**r4bbit (validation):**
- API signatures verified against current docs via web_search
- No deprecated methods, patterns match best practices
- Return: [PASS|FAIL|WARN] + findings + doc links

## Step 3: Collect + fix

- If any FAIL: fix before proceeding, re-run failed review only
- WARN: present to user, user decides fix or proceed
- All PASS: proceed to Step 4

## Step 4: Commit

```
git add -A
git commit -m "{type}({scope}): {description}"
```
No Co-Authored-By. No Generated-with. No AI markers. dlorp only.

## Step 5: Push + PR creation

```
git push origin {branch_name}
gh pr create --title "{title}" --body "{body}" --base main --draft
```
Always draft. User promotes to ready when they want.

## Step 6: CI monitoring

Poll CI status every 30s, timeout after 10 minutes.
- CI pass: "CI passed. PR #{number} ready for review. Merge when ready."
- CI fail: fetch logs, diagnose, propose fix, re-commit

## PR body format

Sections: What, Why, How, Key changes, Code snippets, Testing, Review checklist.
Review checklist includes s3ntry/3tch/r4bbit results, CI status, version bump.
