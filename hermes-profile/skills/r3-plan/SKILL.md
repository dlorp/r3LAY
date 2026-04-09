# /r3-plan -- planning mode (full session start)

Trigger: user types /r3-plan or /r3-plan {project}

## Behavior

1. Identify project (from arg, active context, or ask user)
2. Check privacy level
3. Load prior context:
   - Read .r3lay/sn.md (prior session compression)
   - Read .r3lay/todos.md (active items)
   - Read .r3lay/open-questions.md
   - Call GET /project/{id}/context (bridge) for decisions + conflicts
4. Generate session plan using configured model:
   - What we're picking up from last session
   - Active todos
   - Open questions
   - Any conflicts or overdue items needing attention
5. Write to .r3lay/plans.md (overwrite -- plans.md is current session only)
6. Present plan to user
7. Session is now active. Track context throughout:
   - New todos mentioned -> append to todos.md
   - Decisions made -> log via POST /decision
   - Questions raised -> append to open-questions.md

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
