# Logging Rules

## Never use `print()`

All diagnostic output goes through Python's `logging` module. `print()` is reserved for CLI tools only.

## Logger Setup

Each module gets its own logger named by `__name__`:
```python
import logging
logger = logging.getLogger(__name__)
```

Never configure logging inside a module — only in `app/core/logging.py` at startup.

## Log Levels

| Level | When to use |
|---|---|
| `DEBUG` | Detailed tracing — agent prompts, raw API responses, timing breakdowns |
| `INFO` | Normal operations — request received, search triggered, response sent |
| `WARNING` | Recoverable unexpected state — search returned 0 results, retrying |
| `ERROR` | Operation failed, request could not be completed |
| `EXCEPTION` | Unexpected error with full traceback — use `logger.exception(...)` |

## Structured Context

Always pass relevant context as `extra` kwargs — never interpolate into the message string:
```python
# Good
logger.info("Search completed", extra={"query": query, "result_count": len(results), "duration_ms": elapsed})

# Bad
logger.info(f"Search completed for '{query}', got {len(results)} results")
```

## What to Log

**Always log:**
- Incoming chat requests (message length, conversation_id — never the full message text)
- Whether Orchestrator decided to search or answer directly
- Search queries sent to Tavily (not the raw results)
- Agent errors and retries
- Request duration at the end of each `/api/chat` call

**Never log:**
- Full user messages (privacy)
- API keys or auth tokens (security)
- Full LLM responses (cost / noise)
- SSE token chunks (too noisy)

## Log Format

Configured once in `app/core/logging.py`:
```python
logging.basicConfig(
    stream=sys.stdout,
    level=settings.log_level,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
```

In production, output structured JSON using `python-json-logger` or similar — never ad-hoc `json.dumps`.
