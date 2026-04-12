---
name: compile
description: Compile a project's knowledge into a single context document via mcp_r3lay_compile_project. Karpathy-style deterministic synthesis -- no LLM call.
version: 2.0.0
author: r3LAY
license: MIT
metadata:
  hermes:
    tags: [r3lay, project-management, compilation, context]
    related_skills: [r3-context, r3-plan, sn]
---

# /compile -- project knowledge compilation

Trigger: user types /compile or /compile {project}

## What it does

Calls `mcp_r3lay_compile_project` to assemble all project state into a
single structured markdown document. Deterministic -- no LLM call. One
MCP tool call returns everything:

- Project identity (name, type, language, privacy, path)
- Session context (latest sn.md)
- Active decisions with rationale and confidence
- Active todos
- Open questions
- Pending conflicts
- File inventory with quality weights

## When to use

- **Session cold-start**: user asks "what's the state of project X?" or
  starts a session with a project they haven't touched recently. One
  compile_project call gives you full context without 5 separate reads.
- **After /sn (session close)**: re-compile to keep .r3lay/compiled.md
  fresh for the next session.
- **Before handing off to another agent**: compile creates a portable
  context document that any agent can load without bridge access.
- **User explicitly asks**: "compile", "state of the project", "give me
  the full picture", "summarize everything you know about X".

## Behavior

1. Identify project (from arg, active context, or ask)
2. Call `mcp_r3lay_compile_project(project_id=..., write=True)`
3. Present the key stats from the response:
   "{name}: {files} files, {chunks} chunks, {decisions} decisions,
    {todos} todos, {questions} questions, {conflicts} conflicts"
4. If write=True (default): mention the document was saved to
   .r3lay/compiled.md for future cold-start loading
5. If user wants the full document: show it (it's in the `document` field)

## Output format

```
Compiled: compileproj
45 files, 380 chunks, 12 decisions, 3 todos, 1 question, 0 conflicts
Written to .r3lay/compiled.md

Key context:
- Working on compile endpoint (from sn.md)
- 12 active decisions, most recent: "Use Qwen3 embeddings" (0.95 confidence)
- 3 open todos, 1 open question
```

## Cost

1 bridge call via MCP. Fast. No LLM inference required.

## Future: LLM distillation pass

The current compile step is deterministic assembly. If compiled output grows
too large or too noisy to parse in one context load (hundreds of decisions,
sprawling cross-project state, session histories overwritten dozens of times),
revisit adding a **distill** step: Hermes reads the compiled output and
synthesizes a narrative summary with cross-referencing, contradiction
detection, and pattern recognition. The compile endpoint stays fast and
deterministic; the distill layer runs on top as an LLM pass. Not needed at
current scale -- revisit when the raw compiled.md stops being directly
digestible.
