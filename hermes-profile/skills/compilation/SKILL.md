# compilation

Karpathy-style knowledge compilation loop.

## Behavior

1. Read all recent notes and session debriefs for the project
2. Identify clusters of related information
3. For each cluster:
   a. Check if a wiki page already exists for this topic
   b. If yes: update it with new information, cite sources
   c. If no: create a new page with proper structure
4. Update the project's knowledge graph (edges table)
5. Flag any contradictions discovered during compilation

## Trigger Conditions

- Manual: user invokes compilation
- Automatic: after N session debriefs accumulate (configurable)
- Scheduled: via Hermes cron (e.g., weekly compilation)

## Output

- Updated/created wiki pages in the project folder
- New edges in the knowledge graph
- Compilation report (what changed, what was created)
