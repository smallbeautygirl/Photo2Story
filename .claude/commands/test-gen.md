Generate tests for the specified file or function.

Steps:

1. Identify the target from $ARGUMENTS. If not provided, ask the user which file or function to test.
2. Read the target source file in full.
3. Check if a corresponding test file already exists:
   - Python: `tests/` mirroring `app/` structure (e.g. `app/agents/orchestrator.py` → `tests/agents/test_orchestrator.py`)
   - Frontend: co-located `*.test.tsx` or `*.spec.ts` next to the component
4. Use the Agent tool with `model: "haiku"` to generate tests. Pass the source code with:

   Generate comprehensive tests for this code following these rules:
   - One test per behaviour, not per function
   - Test names describe the scenario: `test_<function>_<condition>_<expected>`
   - Cover: happy path, edge cases, error cases, boundary values
   - For Python: use `pytest` + `pytest-asyncio`; mock external calls with `unittest.mock.AsyncMock` or `monkeypatch`
   - For TypeScript/React: use Vitest + React Testing Library; mock fetch with `vi.fn()`
   - Never test implementation details — test observable behaviour
   - Include a `conftest.py` fixture section if shared setup is needed
   - Return only the test file content, no explanation.

5. Show the generated tests to the user for review.
6. On confirmation, write the file to the correct path.
7. Run the new tests to confirm they pass:
   - Python: `uv run pytest <test_file> -v`
   - Frontend: `npm test <test_file>` (from `frontend/`)
8. If tests fail, diagnose and fix before finishing.
