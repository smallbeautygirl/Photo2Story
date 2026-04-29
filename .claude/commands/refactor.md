Refactor the specified code without changing its external behaviour.

Steps:

1. Ask the user which file or function to refactor if not specified in $ARGUMENTS.
2. Read the target file(s) in full before making any changes.
3. Use the Agent tool with `model: "haiku"` to identify all refactoring opportunities. Pass the file content and these instructions:

   Analyse this code and list refactoring opportunities. For each, note:
   - Location (file:line)
   - Issue (e.g. duplicated logic, long function, unclear name, missing type hint)
   - Suggested fix (one sentence)
   Do NOT rewrite code — only list issues.

4. Present the list to the user and confirm which improvements to apply.
5. Apply only the confirmed changes. Do not add features, change behaviour, or touch unrelated code.
6. After editing, verify the changes with `git diff` and summarise what was changed and why.
7. Run any existing tests relevant to the refactored code to confirm nothing broke:
   - Python: `uv run pytest tests/ -x -q`
   - Frontend: `npm test --run` (from `frontend/`)
