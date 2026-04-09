# session-debrief

End-of-session knowledge extraction, context compression, and write-back.

## Behavior

At session end, BEFORE write-backs:

1. Read existing .r3lay/sn.md if it exists
2. Compress current session transcript:
   - Decisions made (with confidence and source)
   - Questions raised (unresolved)
   - Work in progress (what was being done when session ended)
   - Key facts learned
3. Merge with existing sn.md content:
   - New content appended
   - Old content summarized further (compress, don't grow)
4. Write updated sn.md to .r3lay/sn.md

Then propose write-backs:

5. Extract decisions/facts/questions from the session
6. Check privacy level before choosing inference model
7. Set quality_weight on all proposed write-backs:
   - human-confirmed facts: 1.0
   - ai-extracted from session: 0.8
   - ai-compiled/summarized: 0.7
8. Wait for user approval before writing anything

## Output format

```
Session summary written to .r3lay/sn.md

Decisions to log:
  1. [statement] (confidence: X, source: session)
  2. ...

Files to update:
  1. [path] -- [what to change] (quality_weight: 0.8)
  2. ...

Open questions:
  1. [question] -- needs research
  2. ...

Approve write-backs? [y/n]
```

## sn.md is NOT embedded

sn.md is loaded as a direct context injection at session start, not stored in
the vector index. It's a compressed prompt prefix, not a searchable document.
If the user edits sn.md directly, sync.py picks it up as a human provenance
update but does NOT embed it.
