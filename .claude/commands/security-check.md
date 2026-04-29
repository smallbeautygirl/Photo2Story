Run a security audit on the codebase or a specified file.

Steps:

1. Determine scope from $ARGUMENTS:
   - If a file path is given, read only that file.
   - Otherwise, scan the entire `backend/app/` and `frontend/src/` directories.

2. For Python files, run static analysis tools if available:
   ```bash
   uv run bandit -r backend/app/ -f text 2>/dev/null || echo "bandit not installed"
   ```

3. Use the Agent tool with `model: "haiku"` to perform a manual security review. Pass file contents with these instructions:

   Perform a security review focused on the OWASP Top 10 and common API vulnerabilities.
   Report each finding with: file:line | severity (Critical/High/Medium/Low) | vulnerability type | description | recommended fix.

   Check for:
   - **Injection** — SQL, command, prompt injection via unsanitised user input
   - **Broken Auth** — missing auth middleware, weak token validation, exposed API keys
   - **Sensitive Data** — secrets/keys in code or logs, unencrypted sensitive fields
   - **SSRF** — user-controlled URLs passed to internal HTTP clients (e.g. Tavily service)
   - **Insecure Deserialization** — untrusted JSON/pickle parsed without validation
   - **Missing Rate Limiting** — endpoints accepting unbounded user input without throttle
   - **XSS** — unsanitised content rendered as HTML in the React frontend
   - **CORS Misconfiguration** — overly permissive `allow_origins`
   - **Dependency Vulnerabilities** — note any obviously outdated or known-vuln packages
   - **LLM-specific** — prompt injection, user content passed raw to Claude without sanitisation

4. Display the full report.
5. For Critical or High findings, ask the user if they want fixes applied immediately.
