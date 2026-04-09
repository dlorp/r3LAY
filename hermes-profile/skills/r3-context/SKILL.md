# /r3-context -- lightweight context (read-only, fast)

Trigger: user types /r3-context or /r3-context {project}

## Behavior

1. Call GET /projects/active (bridge)
2. Return one-line summary per project:
   "{name} ({type}) -- {open_todos} todos, {open_questions} questions,
    {overdue} overdue{conflict_flag}"
   conflict_flag: " -- conflicts" if pending conflicts exist
3. Privacy filter: true/work projects show name + counts only (bridge handles this)
4. If project specified: call GET /project/{id}/context for richer summary
5. Does NOT load plans.md or sn.md -- lightweight only
6. Prompt: "Type /r3-plan {project} to start a planning session"

## Cost

1 bridge call. Fast. No LLM inference required.

## Output format

```
r3LAY Projects
--------------
944 (automotive) -- 3 todos, 1 question, 2 overdue
r3lay (software) -- 5 todos, 0 questions
homelab (homelab) -- 0 todos, 2 questions -- conflicts

Type /r3-plan {project} to start a planning session.
```
