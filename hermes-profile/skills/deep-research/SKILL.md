# deep-research

Multi-source synthesis with citation tracking and privacy awareness.

## Behavior

1. Check project privacy level first:
   - true: local search only (r3LAY index via Ollama), no web search
   - work: local + web search ok
   - false: full pipeline
2. Search local r3LAY index for existing knowledge
3. If gaps remain and privacy allows: use Hermes web_search tool
4. Synthesize findings into a cited research note
5. Check for contradictions against existing decisions
6. Propose new decisions from validated findings

## Citation format

Every factual claim must cite its source:
- [LOCAL:chunk_id] for r3LAY index results
- [WEB:url] for web search results
- [SESSION:timestamp] for conversation-derived facts

## Escalation

If local model can't synthesize adequately:
- privacy:true -> escalate to larger local model (e.g., qwen3:32b via Ollama)
- privacy:work/false -> escalate to research_model or remote model per routing config

## Quality weights

- Facts from official docs: 1.0
- Facts from community sources: 0.75
- Facts from web search: 0.7
- AI-synthesized conclusions: 0.6
