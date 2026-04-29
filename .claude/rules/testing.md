# Testing Rules

## Principles

- **One test per behaviour**, not per function — a single function may have many tests
- **Test names describe the scenario**: `test_orchestrator_triggers_search_for_stock_price_query`
- **No implementation detail testing** — test what the code does, not how
- **Tests are independent** — no shared mutable state between tests
- **Tests must be deterministic** — no sleeps, no random values without a seed

## Python — pytest

File structure mirrors `app/`:
```
tests/
├── conftest.py
├── api/
│   └── test_chat.py
├── agents/
│   ├── test_orchestrator.py
│   └── test_search_agent.py
└── services/
    └── test_tavily.py
```

Always mark async tests:
```python
import pytest

@pytest.mark.asyncio
async def test_search_agent_returns_sources():
    ...
```

Set `asyncio_mode = "auto"` in `pyproject.toml` to avoid repeating the mark.

Mock external APIs — never make real HTTP calls in unit tests:
```python
from unittest.mock import AsyncMock, patch

async def test_tavily_search_returns_results(monkeypatch):
    monkeypatch.setattr(
        "app.services.tavily.TavilyService.search",
        AsyncMock(return_value=[{"title": "AAPL", "url": "https://example.com", "content": "..."}])
    )
```

Use `pytest.raises` for exception testing:
```python
with pytest.raises(SearchError, match="timed out"):
    await tavily.search("query")
```

## TypeScript — Vitest + React Testing Library

Test files co-located: `src/components/Message.test.tsx`

Test React components through user interactions, not internal state:
```typescript
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

test('sends message on Enter key', async () => {
    const onSend = vi.fn()
    render(<ChatInput onSend={onSend} />)
    await userEvent.type(screen.getByRole('textbox'), 'Hello{Enter}')
    expect(onSend).toHaveBeenCalledWith('Hello')
})
```

Mock `fetch` for SSE streaming tests:
```typescript
vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true,
    body: new ReadableStream({ ... }),
}))
```

## Coverage Targets

| Layer | Target |
|---|---|
| Agent logic (orchestrator, search agent) | 90% |
| API routes | 80% |
| Services (Tavily, Claude wrapper) | 70% |
| React hooks | 80% |
| React components | 60% |

Run coverage:
```bash
uv run pytest --cov=app --cov-report=term-missing
```

## What NOT to Test

- Framework internals (FastAPI routing, Pydantic validation)
- Third-party SDK behaviour (Anthropic, Tavily)
- Trivial getters / one-liner functions with no logic
- UI pixel-perfect rendering (use Storybook for visual review)
