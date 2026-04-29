Generate a Conventional Commits message for the staged changes in this repository.

Steps:

1. Run `git diff --cached` to see what is staged. If nothing is staged, run `git diff` and tell the user to stage changes first.
2. Use the Agent tool with `model: "haiku"` to analyse the diff and generate the commit message. Pass the full diff output to the agent with these instructions:

   Analyse this git diff and produce a Conventional Commits message following these rules:
   - Choose the appropriate `type`: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `style`, `chore`, `ci`, `build`, or `revert`
   - Add a `scope` — a short noun for the area changed (e.g. `agents`, `search`, `chat`, `ui`, `api`)
   - Mark breaking changes with `!` before the colon and/or a `BREAKING CHANGE` footer
   - Add the matching gitmoji emoji after the colon and space
   - Format:

     ```text
     <type>(<scope>): <emoji> <description>

     <body — explain WHY, not what. Only include if non-trivial.>

     <footers — BREAKING CHANGE, Refs, etc. Only include if relevant.>
     ```

   - Description: imperative mood, max 72 chars, no period
   - Body: one blank line after description, explain motivation
   - Return only the commit message text, nothing else.

   Gitmoji reference:
   - feat → ✨
   - fix → 🐛
   - refactor → ♻️
   - perf → ⚡️
   - test → ✅
   - docs → 📝
   - style → 🎨
   - chore → 🔧
   - ci → 👷
   - build → 📦
   - revert → ⏪️

3. Show the generated commit message to the user and ask for confirmation before running `git commit`.
4. On confirmation, run `git commit -m "<message>"`.
