# Clean Code Rules

Apply to all Python and TypeScript code written in this project.

## Naming

- Names must describe intent — `search_results` not `data`, `is_search_needed` not `flag`
- Functions are verbs: `fetch_sources`, `classify_intent`, `render_message`
- Classes are nouns: `OrchestratorAgent`, `ChatMessage`, `TavilyService`
- Boolean variables and functions start with `is_`, `has_`, `can_`, `should_`
- Avoid abbreviations unless universally known (`url`, `api`, `id` are fine; `orch`, `srvc` are not)
- No single-letter variables except loop indices `i`, `j` and short lambdas

## Functions

- One function = one responsibility. If the name needs "and", split it.
- Maximum ~20 lines per function. Extract helpers freely.
- No more than 3 positional parameters. Use keyword-only args or a dataclass for more.
- Functions that can fail return `X | None` or raise a specific exception — never return sentinel values like `-1` or `""`
- Pure functions preferred; side effects must be explicit in the name (`save_`, `send_`, `delete_`)

## Comments

- Never comment *what* the code does — the code should be self-explanatory
- Comment *why* when the reason is non-obvious: unusual algorithm, business rule, API quirk
- Remove dead code and commented-out code — use git history instead
- TODO comments must include a ticket or owner: `# TODO(vivian): handle rate limit #42`

## Files and Modules

- One primary concept per file
- File length: soft limit 200 lines, hard limit 400 lines — split if exceeded
- No circular imports — if A imports B and B imports A, extract a shared module C
- Keep `__init__.py` files minimal — only re-export the public API

## General

- No magic numbers — use named constants or config values
- No deeply nested code (> 3 levels) — early returns and guard clauses flatten nesting
- Prefer immutable data where possible (`frozen=True`, `const`, `readonly`)
- Delete unused variables, imports, and functions immediately
