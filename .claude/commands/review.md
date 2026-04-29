Perform a code review on the specified file or the current staged diff.

Steps:

1. If $ARGUMENTS specifies a file path, read that file. Otherwise run `git diff --cached` (fall back to `git diff HEAD~1` if nothing staged).
2. Use the Agent tool with `model: "haiku"` to review the code. Pass the content with these instructions:

   Review this code and report issues in the following categories. For each issue, include the file:line, severity (🔴 critical / 🟡 warning / 🔵 suggestion), and a one-line explanation.

   Categories:
   - **Correctness** — logic errors, off-by-one, wrong async usage, race conditions
   - **Security** — injection, hardcoded secrets, missing auth checks, unvalidated input
   - **Type Safety** — missing type hints, use of `any`, unchecked `None`
   - **Error Handling** — bare `except`, swallowed exceptions, missing error propagation
   - **Performance** — blocking I/O in async context, N+1 queries, unnecessary allocations
   - **Readability** — unclear names, functions > 20 lines, missing docstrings on public APIs
   - **Test Coverage** — untested branches, missing edge cases

   End with a short overall summary (2-3 sentences).

3. Present the review output to the user.
4. Ask: "Would you like me to fix any of these issues?"
5. If yes, apply only the fixes the user selects. Do not touch anything else.
