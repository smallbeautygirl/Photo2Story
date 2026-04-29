# Database Rules

This project does not currently use a persistent database. Apply these rules when a database is added (e.g. PostgreSQL via SQLAlchemy or Supabase for conversation history).

## Query Design

- **Never `SELECT *`** — always name the columns you need
- **Use parameterised queries** — never f-string or `.format()` SQL strings (SQL injection)
- **Limit all list queries** — always include `LIMIT` / pagination; never fetch unbounded rows
- **Index foreign keys and frequently filtered columns** — add index in migration, not ad-hoc

## Async

- Use an async driver (`asyncpg`, `databases`, or SQLAlchemy async engine) — never block the event loop with synchronous DB calls
- Use a connection pool; do not open a new connection per request

## Schema / Migrations

- All schema changes via migration files (Alembic or similar) — never `ALTER TABLE` manually
- Migrations must be reversible — always provide `downgrade()` alongside `upgrade()`
- Run migrations in CI before tests

## Data Access Pattern

Follow the Repository pattern (see `docs/skills/python-backend.md`):

```python
class ConversationRepository:
    async def get(self, conversation_id: str) -> Conversation | None: ...
    async def save(self, conversation: Conversation) -> None: ...
    async def list_recent(self, limit: int = 20) -> list[Conversation]: ...
```

- Business logic (agents) calls repository methods — never raw SQL
- Repository methods take and return Pydantic models or dataclasses — never raw dicts

## Conversation Storage (planned)

When storing chat history:
- Store `conversation_id`, `role`, `content`, `created_at`, `sources` (JSONB)
- Index on `conversation_id` + `created_at DESC`
- Limit history to last 50 messages when loading context for agents
- Do not store raw LLM prompts or internal agent reasoning in user-visible history
