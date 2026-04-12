---
name: r3-plan
description: Enter planning mode for an r3LAY project -- loads prior session notes, todos, questions, decisions, and generates a session plan. Full session start.
version: 2.0.0
author: r3LAY
license: MIT
metadata:
  hermes:
    tags: [r3lay, project-management, planning, session]
    related_skills: [r3-context, sn, compile]
---

# /r3-plan -- planning mode (full session start)

Trigger: user types /r3-plan or /r3-plan {project}

## Behavior

1. Identify project (from arg, active context, or ask user)
2. Check privacy level
3. Load prior context via MCP tools:
   - Call `mcp_r3lay_get_project_context(project_id=...)` -- returns
     session notes, decisions, todos, questions, conflicts in one call
   - Alternatively, if .r3lay/compiled.md exists and is recent (< 24h),
     read it directly for faster cold-start (skip the MCP call)
4. Generate session plan using configured model:
   - What we're picking up from last session (from sn.md / compiled.md)
   - Active todos
   - Open questions
   - Any conflicts or overdue items needing attention
5. Write to .r3lay/plans.md (overwrite -- plans.md is current session only)
6. Present plan to user
7. Session is now active. Track context throughout:
   - New todos mentioned -> append to todos.md
   - Decisions made -> log via POST /decision
   - Questions raised -> append to open-questions.md

## Cold-start shortcut

If the user hasn't touched a project in a while and needs full context
fast, suggest running `/compile {project}` first -- it produces a
comprehensive context document that makes the /r3-plan output richer.

## Output format

```
Session Plan -- {project}
--------------------------
Picking up: {from sn.md -- 1-2 lines}

Active todos:
  [ ] {todo 1}
  [ ] {todo 2}

Open questions:
  ? {question}

Conflicts: {N} pending -- type /conflicts to review

Ready. What are we working on?
```

## Privacy

- true: use Ollama fallback only for plan generation
- work/false: any configured model
