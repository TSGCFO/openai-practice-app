# System architecture and technical design — Flask frontend edition

## Overview

This document describes a Python-based AI Assistant that uses OpenAI as its core LLM and integrates Model Context Protocol (MCP) servers for extensible tool functionality. The frontend uses Flask (server-side rendered templates with optional client-side enhancements) and supports streaming via WebSocket (Flask-SocketIO) or Server-Sent Events (SSE). The orchestrator is an async Python service (FastAPI recommended) responsible for LLM calls, MCP discovery/execution, state, auth, and observability.

## Checklist (what this doc contains)

- Component architecture with Flask frontend integration
- API endpoints and streaming shapes (WS/SSE) and data shapes
- MCP server discovery/registration and execution patterns
- Authentication and authorization flows (user & service-to-service)
- State management (session, vector DB, persistence) and prompt-building
- Error handling, retries, idempotency, and circuit breaker strategies
- Deployment considerations (Docker/Kubernetes/CI), secrets, and cost control
- Example tool implementation contract and Flask integration notes

## High-level architecture

- Frontend: Flask app serving Jinja2 templates and static assets. Uses Flask-SocketIO for WebSocket streaming or provides SSE endpoints for token streams. Can optionally act as a reverse-proxy for auth convenience.
- Orchestrator: FastAPI (async) service orchestrating OpenAI calls and MCP servers. Exposes REST and WebSocket endpoints consumed by the frontend.
- MCP Servers: Independently-deployable services offering tool discovery and execution endpoints (HTTP/gRPC). The orchestrator discovers and calls MCP servers.
- Supporting infra: Postgres (users, audits), Redis (sessions, locks), Vector DB (embeddings), secrets manager, object store, observability (OpenTelemetry, Prometheus, Grafana).

## High-level flow

```mermaid
flowchart LR
  Browser[Browser] -->|HTTP / WebSocket| FlaskApp[Flask Frontend (Flask-SocketIO)]
  FlaskApp -->|REST| Orch[Orchestrator (FastAPI)]
  Orch -->|OpenAI API (stream)| OpenAI[OpenAI / LLM]
  Orch -->|HTTP/gRPC| MCP[MCP Servers]
  Orch --> Postgres
  Orch --> Redis
  Orch --> VectorDB
  MCP -->|optional| Sandbox[Remote Workbench/Sandbox]
```

## Why Flask for frontend (short rationale)

- Familiar, lightweight, and Python-native templating with Jinja2.
- Simplifies server-side rendering for authenticated UIs and reduces CORS complexity.
- Integrates easily with Flask-SocketIO for streaming tokens.
- Good fit when teams want Python across presentation and glue code or prefer server-rendered pages.

## Frontend design (Flask) — responsibilities

- Serve HTML templates (Jinja2) that render the chat UI and helper pages (login, settings, tools list).
- Host static JS and CSS to connect WebSocket, render streaming tokens, and show tool cards/confirmations.
- Provide endpoints for login flows, tool registration UI (admin), and optional proxy endpoints to the orchestrator.
- Streaming choices:
  - WebSocket (preferred): Flask-SocketIO (uses eventlet/gevent/ASGI bridge). Server pushes JSON chunks: {type: 'token' | 'tool_request' | 'tool_result' | 'final' | 'error'}.
- Serve HTML templates (Jinja2) that render the chat UI and helper pages (login, settings, tools list).
- Host static JS bundle and CSS (minimal client) to: connect WebSocket, render streaming tokens incrementally, and show tool cards/confirmations.
- Provide endpoints for: login flows, tool registration UI (admin), and optionally proxy endpoints for fetches to the orchestrator.
- Realtime streaming choices:
  - WebSocket (preferred): Flask-SocketIO (uses eventlet/uvicorn with ASGI bridge or gevent). Client opens socket, server pushes JSON chunks {type:'token'|'tool_request'|'tool_result'|'final'|'error'}.
  - SSE: Simpler for server→client streaming but only one-way and less interactive.
- Security: CSRF protection for form POSTs, secure cookies, and ensure socket auth tokens (JWT) are validated server-side.

## Frontend flow (user interaction)

 1. User authenticates via OAuth2 (Flask route triggers provider redirect).
 2. Flask saves user session (server cookie) and obtains user_id to send to orchestrator.
 3. User types message in the chat UI. Client issues AJAX POST to Flask: POST /chat/send or emits a WebSocket event.
 4. Flask forwards the request to Orchestrator (server-to-server call with service token) or instructs the client to open a WS directly to orchestrator (if you prefer separation). Flask then subscribes to orchestrator run_id channel and streams tokens back to the user via socket.

## Integration patterns between Flask frontend and Orchestrator

 Two common deployment patterns:

 A) Frontend-as-Proxy (Flask mediates orchestration):

- Flask handles auth and sessions; it forwards user requests (with user context) to orchestrator REST endpoints. Flask subscribes to orchestrator WebSocket/notifications and emits streaming updates to the browser via Flask-SocketIO.
- Benefit: Simplifies CORS and keeps token management inside Flask sessions.
- Downside: Slight added latency and more coupling between Flask and Orchestrator.

 B) Frontend-as-client (Flask serves static UI only):

- Flask serves JS/CSS and authentication endpoints. The client (browser) directly opens a WebSocket to Orchestrator using a short-lived token (obtained from Flask). Streaming happens directly between Orchestrator and browser.
- Benefit: Lower latency, simpler orchestrator scaling, clearer separation of concerns.
- Downside: Requires robust cross-origin and token issuance logic.

## API endpoints and shapes (orchestrator-focused)

 These endpoints are implemented in the orchestrator (FastAPI). Flask either proxies or facilitates requests to them.

### REST endpoints

- POST /api/v1/respond
  - Body: { session_id?: string, input: string, options?: {max_output_tokens, tool_overrides}}
  - Returns: { run_id }
- GET /api/v1/history?session_id=...
  - Returns conversation history, structured tool outputs.
- POST /api/v1/tools/register (admin)
  - Register an MCP server or tool manually.

### WebSocket channel

- /ws?session_id=...&run_id=...
  - Messages streamed as JSON chunks with types: token, tool_request, tool_result, final, error.
  - Example token message: { type: "token", text: "Hello" }
  - Example tool_request (LLM asks orchestrator to call a tool): { type: "tool_request", tool_slug: "GMAIL_FETCH_EMAILS", args: {...}, require_confirmation: true }

## MCP contract (discovery & execute)

 MCP servers expose:

- GET /mcp/discover or GET /.well-known/mcp
  - Returns server metadata, allowed_tools and JSON Schema for each tool.
- POST /mcp/execute
  - Input: multi-execute envelope
  
    ```json
    {
      "tools": [ { "tool_slug": "GMAIL_FETCH_EMAILS", "arguments": { ... } } ],
      "sync_response_to_workbench": false,
      "thought": "Fetch latest 10 emails",
      "current_step": "FETCHING_EMAILS"
    }
    ```

  - Output:
  
    ```json
    {
      "requestId": "uuid",
      "successful": true,
      "data": { "results": [ {"tool_slug":"...","response":{...}} ]},
      "error": null
    }
    ```

## Important design notes for MCP servers

- Validate arguments against the tool's JSON Schema.
- Return bounded-size responses; if result is large, persist to object store and return a pointer or enable workbench sync.
- Support idempotent execution by honoring requestId.
- Provide health and metrics endpoints.

## Authentication & authorization

### User flows (recommended)

- Use OAuth2 / OpenID Connect for user login (Flask initiates and handles callback). Store user record and encrypted refresh tokens server-side (Postgres + secrets manager).
- Issue a short-lived session cookie to the browser (Flask session cookie). For direct orchestrator WS access, Flask exchanges the session for a short-lived JWT scoped to the run_id.

### Service-to-service

- Orchestrator ↔ MCP servers: use mTLS or signed JWT (service account tokens). In Kubernetes, prefer service accounts plus SPIFFE/SPIRE if available.
- Orchestrator ↔ OpenAI: store keys in secrets manager and rotate them periodically.

### Consent & action confirmation

- For any side-effect tool (send email, modify repo), require explicit recorded consent from the user (UI modal). Orchestrator must verify consent before executing.

## State management & context

### Short-term state

- Redis stores: session buffers, current run state (run_id -> progress, partial tokens), locking (per-session), rate-limiting counters.
- Keep only last N messages in prompt builder to control token sizes.

### Long-term state

- Postgres: user profiles, audit logs, tool registration, per-user preferences.
- Vector DB: embeddings for RAG (long-term memory). Store metadata linking embeddings back to transcripts.

### Prompt building

- Combine: system prompt, sanitized short conversation window, relevant tool schema snippets (only necessary fields), retrieved memories if applicable.
- Avoid including entire tool catalogs in the prompt (send only relevant schema slices) to reduce token usage.

## Error handling and reliability strategies

- Validation errors: respond 400 with schema error details to the caller.
- Transient errors: retry with exponential backoff for transient network or 5xx errors.
- Circuit breaker: monitor failure rates to MCP endpoints and short-circuit calls when unhealthy, with cooling period.
- Timeouts: set per-call timeouts for LLM and MCP executions. If a stream stalls, return a partial response and surface a retry action in the UI.
- Large payloads: route large outputs to workbench / object store and return a short summary + link.
- Logging: structured logs with correlationId (run_id). Implement tracing with OpenTelemetry.

## Idempotency & dedup

- Use requestId for tool executes and store results keyed by requestId to dedupe retries.
- For user actions with side-effects, surface a confirmation and store action tokens; prevent duplicate execution if the user resubmits.

## Deployment & infra notes

- Dockerize all components (Flask app, Orchestrator, MCP servers if self-hosted).
- Kubernetes recommended for production with separate namespaces for frontend, orchestrator, MCPs, and infra.
- Use an ingress (nginx/contour) and handle TLS at the edge; use mTLS within the mesh for service-to-service auth if required.
- CI: run lint, unit tests, contract tests (MCP mock), build images, run integration tests on ephemeral environment, and deploy via GitHub Actions / GitLab CI.

## Cost & usage control

- Monitor token usage per-session and provide per-user quotas.
- For large workloads, offload to the remote workbench to avoid streaming enormous content inline.

## Example tool contract: GMAIL_FETCH_EMAILS

### Input schema (JSON Schema)

```json
{
  "type": "object",
  "properties": {
    "user_id": {"type":"string"},
    "label_ids": {"type":"array","items":{"type":"string"}},
    "max_results": {"type":"integer","minimum":1,"maximum":100},
    "include_payload": {"type":"boolean"},
    "query": {"type":"string"},
    "page_token": {"type":"string"}
  },
  "required":["user_id"]
}
```

### Output shape

```json
{
  "messages": [ {"messageId":"...","threadId":"...","messageTimestamp":"ISO8601","sender":"...","subject":"...","snippet":"..."} ],
  "nextPageToken": "...",
  "resultSizeEstimate": 201
}
```

### Sample implementer notes

- Validate input against the schema before performing external API calls.
- Use stored refresh tokens from orchestrator for per-user Gmail access (do not ask user to provide raw credentials to MCP).
- Return bounded previews; if full bodies needed and large, store to object store and set sync_response_to_workbench:true.

## Flask-specific code pattern (client-to-orchestrator streaming)

1. User submits message to Flask route (POST /chat/send). Flask validates session, then forwards to orchestrator POST /api/v1/respond with user context. Orchestrator returns run_id.
2. Flask subscribes to orchestrator's WebSocket or SSE (or opens a socket as a client) for that run_id.
3. Flask emits token events to the browser via Flask-SocketIO (socketio.emit with namespace room=run_id) as the orchestrator streams.
4. Alternatively, Flask returns short-lived JWT to the browser so the browser can directly open WS to orchestrator and receive stream.

Security: ensure runs and sockets are scoped to the logged-in user to avoid leaking data.

## Developer workflow and local dev

- Repo layout (suggestion):
  - /frontend-flask: Flask app (templates, static, socket handlers)
  - /orchestrator: FastAPI app (LLM + MCP orchestration)
  - /mcp-examples: example MCP servers (Gmail mock, simple tools)
  - /docs: system docs
  - docker-compose.yml for local dev (postgres, redis, mock vector DB or sqlite, Flask, Orchestrator, a mock MCP server)

- Recommended steps to run locally:
  1. Copy .env.example to .env and fill simple dev values.
  2. docker-compose up --build
  3. Visit `http://localhost:5000` (Flask) and login via dev OAuth or mocked user.

## Testing & contract tests

- Provide a mocked MCP server for contract tests: it must respond to /mcp/discover and /mcp/execute.
- Orchestrator should include unit tests that validate schema enforcement and that tool outputs are handled correctly.

## Best practices summary (avoid over-engineering)

- Keep UI simple; use Flask templates and a small JS bundle for streaming rendering.
- Keep orchestration logic in one service (FastAPI) that is language & framework neutral — Flask focuses on presentation and session handling.
- Limit token sizes in prompts; offload heavy data handling to the workbench/sandbox.
- Prefer simple REST + JSON Schema contracts between orchestrator and MCP servers; add webhooks only when necessary for long-running jobs.
- Instrument everything from day one: logs, traces, and metrics.

## Next steps I can implement for you

- Create a minimal `frontend-flask` skeleton (Flask + Flask-SocketIO) that proxies a POST to orchestrator and streams tokens to the browser.
- Produce an `orchestrator` FastAPI skeleton with `/api/v1/respond` and a mock MCP discover/execute client.
- Add `docker-compose.yml` and `requirements.txt` for local dev.

Which artifact should I create next? (I can scaffold the Flask frontend skeleton now.)

---
