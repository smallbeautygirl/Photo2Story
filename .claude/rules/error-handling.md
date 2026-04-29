# Error Handling Rules

## Python

**Never swallow exceptions silently:**
```python
# Bad
try:
    result = await tavily.search(query)
except Exception:
    pass

# Good
try:
    result = await tavily.search(query)
except httpx.TimeoutException as e:
    logger.warning("Tavily search timed out", extra={"query": query})
    raise SearchError("Search service timed out") from e
```

**Always chain exceptions** to preserve the original traceback:
```python
raise SearchError("Failed to fetch results") from e   # not: raise SearchError(...)
```

**Custom exception hierarchy** — one base class per domain:
```python
class OrbitrieveError(Exception): ...
class SearchError(OrbitrieveError): ...
class AgentError(OrbitrieveError): ...
```

**FastAPI error responses** — always structured, never plain strings:
```python
raise HTTPException(
    status_code=502,
    detail={"error": "Search service unavailable", "code": "SEARCH_FAILED"},
)
```

**Catch specific exceptions**, ordered from most to least specific:
```python
except httpx.TimeoutException:   ...
except httpx.HTTPStatusError:    ...
except httpx.RequestError:       ...
```

## TypeScript / React

**Typed error boundaries** — wrap async operations with try/catch and type the error:
```typescript
try {
    await streamChat(...)
} catch (err: unknown) {
    if (err instanceof Error && err.name !== 'AbortError') {
        setError(err.message)
    }
}
```

**Never expose raw error messages to users** — map to user-friendly strings:
```typescript
const USER_MESSAGES: Record<string, string> = {
    SEARCH_FAILED: "Couldn't search the web right now. Please try again.",
    RATE_LIMITED: "Too many requests. Please wait a moment.",
}
```

**Propagate SSE stream errors** — the backend sends `{ type: "error", code: "..." }` events; the frontend must handle them and show a visible error state, never silently drop them.

## API Contract for Errors

All backend error responses follow this shape:
```json
{ "error": "Human-readable message", "code": "MACHINE_READABLE_CODE" }
```

Standard error codes used in this project:
| Code | HTTP | Meaning |
|---|---|---|
| `SEARCH_FAILED` | 502 | Tavily API failed |
| `AGENT_FAILED` | 502 | Claude API failed |
| `INVALID_REQUEST` | 422 | Validation error |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
