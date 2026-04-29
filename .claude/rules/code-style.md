# Code Style Rules

Enforced by tooling (`ruff` for Python, ESLint + Prettier for TypeScript). Follow these even when the linter is not running.

## Python

- Formatter: `ruff format` (double quotes, 100-char line length)
- Linter: `ruff check` with rules `E, F, I, UP, B, SIM, TCH`
- Type checker: `mypy --strict`
- `from __future__ import annotations` at the top of every file
- Use built-in generics: `list[str]`, `dict[str, int]`, `X | None` — never `List`, `Dict`, `Optional` from `typing`
- f-strings for all string interpolation — no `%` formatting or `.format()`
- `pathlib.Path` for all file paths — never raw string concatenation
- `Enum` / `StrEnum` for fixed sets of string values — never bare string literals as identifiers

Import order (ruff/isort enforces this):
```
from __future__ import annotations
# 1. stdlib
# 2. third-party
# 3. local (app.*)
```

## TypeScript / React

- Prettier: 2-space indent, single quotes, trailing commas, 100-char line length
- ESLint: `@typescript-eslint/recommended` + `react-hooks` rules
- No `any` — use `unknown` and narrow it, or define a proper interface
- Explicit return types on all exported functions and hooks
- `interface` for object shapes passed between components; `type` for unions and aliases
- `const` by default; `let` only when reassignment is necessary; never `var`
- Named exports only — no default exports except for route-level page components
- Destructure props at the function signature: `function Foo({ title, onClick }: FooProps)`
- Event handler names: `handle<Event>` (local) or `on<Event>` (prop)

## File Naming

| Context | Convention | Example |
|---|---|---|
| Python modules | `snake_case.py` | `search_agent.py` |
| React components | `PascalCase.tsx` | `MessageList.tsx` |
| React hooks | `camelCase.ts` starting with `use` | `useChat.ts` |
| TypeScript types | `PascalCase.ts` | `chat.ts` |
| Test files (Python) | `test_<module>.py` | `test_orchestrator.py` |
| Test files (TS) | `<Component>.test.tsx` | `Message.test.tsx` |
