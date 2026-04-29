# Git Commit Rules

All commits in this repository follow the **Conventional Commits** specification.

## Format

```
<type>(<scope>): <emoji> <description>

[optional body — explain WHY]

[optional footers]
```

## Types

| Type | Emoji | Use for |
|---|---|---|
| `feat` | ✨ | New feature or capability |
| `fix` | 🐛 | Bug fix |
| `refactor` | ♻️ | Code restructure without behaviour change |
| `perf` | ⚡️ | Performance improvement |
| `test` | ✅ | Adding or fixing tests |
| `docs` | 📝 | Documentation only |
| `style` | 🎨 | Formatting, linting (no logic change) |
| `chore` | 🔧 | Config, dependencies, tooling |
| `ci` | 👷 | CI/CD pipeline changes |
| `build` | 📦 | Build system or packaging |
| `revert` | ⏪️ | Reverting a previous commit |

## Scopes (project-specific)

Use one of these scopes that matches the area changed:

| Scope | Area |
|---|---|
| `agents` | Orchestrator or Search Agent logic |
| `orchestrator` | Orchestrator Agent specifically |
| `search` | Search Agent or Tavily service |
| `api` | FastAPI routes |
| `config` | App config, env vars |
| `chat` | Chat UI components |
| `stream` | SSE streaming logic |
| `store` | Zustand store |
| `hooks` | React hooks |
| `ui` | Generic UI components |
| `deps` | Dependency updates |
| `ci` | CI configuration |

## Rules

- Description: imperative mood, max 72 chars, **no period** at the end
- Body: separated by one blank line, explains motivation (not "what")
- Breaking changes: add `!` before colon AND a `BREAKING CHANGE:` footer
- Use `/commit` command to auto-generate messages from staged diff

## Examples

```
feat(agents): ✨ add orchestrator intent classification

The orchestrator now classifies user messages before deciding
whether to trigger a web search, reducing unnecessary Tavily calls.

feat(search): ✨ add Tavily source citations to SSE stream

fix(stream): 🐛 handle partial SSE data chunks in frontend parser

refactor(orchestrator)!: ♻️ replace string classifier with tool use

BREAKING CHANGE: _classify now uses Claude tool_use instead of text output.
Callers expecting a "yes"/"no" string must update to boolean return.

chore(deps): 🔧 upgrade anthropic SDK to 0.45
```

## What NOT to Do

- No `fix: fix bug` — be specific
- No `feat: update code` — name the feature
- No present tense: `adds` → `add`
- No committing `.env`, `__pycache__/`, `node_modules/`
- Never `git commit --no-verify` to skip hooks
