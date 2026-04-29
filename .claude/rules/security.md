# Security Rules

These rules are always active. Violations must be fixed before merging.

## Secrets & Configuration

- API keys (`ANTHROPIC_API_KEY`, `TAVILY_API_KEY`) live only in `.env` — never in code
- `.env` is in `.gitignore` — never commit it
- Access all secrets via `settings.*` (pydantic-settings) — never `os.environ["KEY"]` inline
- Never log, print, or include secrets in error messages or API responses

## Input Validation

- All request bodies are validated by Pydantic before any business logic runs
- Maximum message length: 4000 characters (enforced in `ChatRequest`)
- Reject requests with empty messages at the schema level (`min_length=1`)
- Never pass raw user input directly to shell commands, SQL, or file paths

## Prompt Injection

- User message content passed to Claude agents must be clearly delimited from system instructions
- The Search Agent wraps user content inside `User question: <content>` — never interpolate it into the system prompt
- Never allow user input to modify the system prompt at runtime
- Treat all content returned from web search (Tavily results) as untrusted — do not execute or eval it

## SSRF Prevention

- The Tavily service constructs its own URLs — user input is passed as a `query` string, never as a URL
- Never accept a URL from the user and fetch it server-side without allowlist validation
- The `httpx.AsyncClient` should not follow redirects to private IP ranges

## CORS

- `allow_origins` in `app/core/config.py` defaults to `["http://localhost:5173"]`
- In production, set `CORS_ORIGINS` to the exact frontend domain — never `["*"]`

## Authentication (future)

When auth is added:
- Use short-lived JWT tokens (≤ 1 hour expiry)
- Validate token on every request in middleware — not inside route handlers
- Never store tokens in `localStorage` — use `httpOnly` cookies

## Dependency Security

- Run `uv audit` (or `pip-audit`) before releases
- Pin major versions in `pyproject.toml`; use `uv lock` to pin transitive deps
- Frontend: run `npm audit` before releases

## React XSS

- Never use `dangerouslySetInnerHTML` with user-generated or LLM-generated content
- Source URLs from Tavily results must use `rel="noopener noreferrer"` on all `<a target="_blank">` links
- Do not render markdown as raw HTML without a sanitiser (use `DOMPurify` if markdown rendering is added)
