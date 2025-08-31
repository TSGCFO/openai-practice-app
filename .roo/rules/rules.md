# Roo Code instructions for this repo (openai-practice-app)

Purpose and scope

- This repo is docs-first. The only runnable code is a minimal Python example in `openai-practice/example.py`. System architecture and patterns live under `docs/` (see `docs/system_architecture_flask.md`).
- Architecture described: Flask frontend (SSR + WebSocket/SSE), FastAPI orchestrator (LLM + MCP orchestration), optional MCP servers; Postgres/Redis/Vector DB as supporting infra.

Where to look first

- `openai-practice/example.py`: Shows how we call OpenAI Responses API and (via an orchestration layer) fetch Gmail messages. Use this file to prototype SDK usage and result-shaping.
- `docs/system_architecture_flask.md`: Source of truth for component responsibilities and data flows (frontend-as-proxy vs client-direct patterns).
- `openai-practice/requirements.txt`: Dependencies used by the example and prospective services.

Non-obvious conventions and gotchas

- Never print the raw OpenAI SDK response objects. They can embed large orchestration traces (tool discovery, plans, logs) that explode token/console output. Always extract and print compact, task-specific fields.
- The example already implements safe serialization and selective extraction of Gmail data:
  - Prefer `response.to_dict()` (or `.dict()`), then recursively locate `messages` and `nextPageToken`.
  - Summarize each message with: `messageId/id`, `threadId`, `messageTimestamp/internalDate`, `sender/from`, `subject`, `preview/snippet`.
- Treat external connectors (e.g., Rube MCP) as remote services. Do not hardcode secrets; rely on env vars (`OPENAI_API_KEY`) via `python-dotenv`.
- Documentation style (for edits to `docs/`): use ATX headings, 1 blank line around headings, no leading spaces on top-level bullets, and 2-space indents for nested lists. Mermaid diagrams are allowed.

Developer workflow (current repo state)

- Quick run of the example (conda):

  ```bash
  # (Optional first time) create env with desired Python version
  # conda create -n myenv python=3.11
  conda activate myenv
  pip install -r openai-practice/requirements.txt
  export OPENAI_API_KEY=sk-...  # or use .env with python-dotenv
  python openai-practice/example.py
  ```

- Expected output: human readable responses.

Patterns to follow when adding code here

- Responses: build small, schema-like JSON shapes for UI/consumers; avoid echoing full upstream payloads.
- Pagination: if tool outputs include `nextPageToken`, expose a simple `page_token` param and return both `messages` and `nextPageToken`.
- Logging: prefer structured logs and avoid dumping entire SDK objects.

Key files/directories

- `openai-practice/example.py` — OpenAI Responses call + compact result extraction
- `openai-practice/requirements.txt` — Python deps (openai, pydantic, dotenv, httpx, etc.)
- `docs/system_architecture_flask.md` — Flask-oriented system architecture and contracts

If you need deeper context

- Start from `docs/README.md` and `README.md` at the repo root. They describe the intended multi-service setup even if those services are not yet scaffolded here.
