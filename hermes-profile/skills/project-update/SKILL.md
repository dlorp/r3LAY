# project-update

Natural language update to a project file with privacy and conflict enforcement.

## Behavior

1. Load project metadata via GET /project/{id}
2. Check privacy level:
   - true: use Ollama only for any inference needed
   - work: any model ok, content marked work-restricted in responses
   - false: full pipeline
3. Run conflict check via POST /project/update
4. If conflict detected:
   - Display full conflict report
   - Wait for user choice: [1] Override, [2] Cancel
   - If override: POST /decision with new statement, supersede old via /conflicts/resolve
   - If cancel: abort, log conflict as rejected
5. If no conflict:
   - Atomic write pattern: write to tmp -> DB transaction -> rename
   - Log decision if the update contains a factual assertion
   - Set quality_weight based on provenance (human=1.0, ai-updated=0.8)

## Conflict report format

```
CONFLICT (hard -- structural)
----------------------------------------
Proposed:  {proposed_change}
Conflicts: Decision {id} ({decided_at})
           "{statement}"
           Rationale: "{rationale}"

[1] Override -- proceed and log new decision
[2] Cancel
Choice: _
```
