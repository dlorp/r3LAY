---
name: r3-context
description: Lightweight read-only project context -- lists active r3LAY projects with todo/question/conflict counts. Fast, no LLM inference needed. One MCP call.
version: 2.0.0
author: r3LAY
license: MIT
metadata:
  hermes:
    tags: [r3lay, project-management, context]
    related_skills: [r3-plan, sn, compile]
---

# /r3-context -- lightweight context (read-only, fast)

Trigger: user types /r3-context or /r3-context {project}

## Behavior

1. Call `mcp_r3lay_list_active_projects()` (one MCP call)
2. Return one-line summary per project:
   "{name} ({type}) -- {open_todos} todos, {open_questions} questions
    {conflict_flag}"
   conflict_flag: " -- conflicts" if pending_conflicts > 0
3. Privacy filter: true/work projects show name + counts only (bridge handles)
4. If a specific project is given: call
   `mcp_r3lay_get_project_context(project_id=...)` for richer detail
   (includes session notes, decisions, todos, questions, conflicts)
5. Does NOT load plans.md or full file listings -- lightweight only
6. For full context, suggest: "/compile {project}" or "/r3-plan {project}"

## When to reach for the heavier tools

- `/r3-context` = dashboard overview (1 call, fast)
- `/r3-context {project}` = project detail (1 call, moderate)
- `/compile {project}` = full compiled document with file inventory (1 call, heavy)
- `/r3-plan {project}` = planning session start (loads everything + generates plan)

## Cost

1 MCP bridge call. Fast. No LLM inference required.

## Output format

```
r3LAY Projects
--------------
944 (automotive) -- 3 todos, 1 question, 2 overdue
r3lay (software) -- 5 todos, 0 questions
homelab (homelab) -- 0 todos, 2 questions -- conflicts

Type /r3-plan {project} to start a planning session.
Type /compile {project} for a full context document.
```
