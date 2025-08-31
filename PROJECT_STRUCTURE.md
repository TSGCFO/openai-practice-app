# AI Assistant Project Structure

```mermaid
ai-assistant/
├── frontend/                     # Flask Frontend Service
│   ├── app.py                   # Main Flask application
│   ├── config.py                # Configuration management
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Container definition
│   ├── auth/                   # Authentication modules
│   │   ├── __init__.py
│   │   ├── oauth.py            # OAuth2 integration
│   │   └── middleware.py       # Auth middleware
│   ├── routes/                 # Flask routes
│   │   ├── __init__.py
│   │   ├── main.py            # Main routes
│   │   ├── chat.py            # Chat endpoints
│   │   └── admin.py           # Admin routes
│   ├── sockets/                # WebSocket handlers
│   │   ├── __init__.py
│   │   ├── chat.py            # Chat socket events
│   │   └── handlers.py        # Event handlers
│   ├── templates/              # Jinja2 templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── chat.html
│   │   └── login.html
│   ├── static/                 # Static assets
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── tests/                  # Frontend tests
│
├── orchestrator/                # FastAPI Orchestrator Service
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Container definition
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── conversations.py   # Conversation endpoints
│   │   ├── messages.py        # Message endpoints
│   │   ├── tools.py           # Tool management
│   │   └── health.py          # Health checks
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── openai_service.py  # OpenAI integration
│   │   ├── mcp_manager.py     # MCP coordination
│   │   ├── session_manager.py # Session handling
│   │   └── conversation.py    # Conversation logic
│   ├── models/                 # Data models
│   │   ├── __init__.py
│   │   ├── conversation.py    # Conversation models
│   │   ├── message.py         # Message models
│   │   └── tool.py            # Tool models
│   ├── database/               # Database operations
│   │   ├── __init__.py
│   │   ├── connection.py      # DB connection
│   │   └── repositories.py    # Data repositories
│   ├── streaming/              # Streaming handlers
│   │   ├── __init__.py
│   │   ├── websocket.py       # WebSocket handler
│   │   └── sse.py             # SSE handler
│   └── tests/                  # Orchestrator tests
│
├── mcp_servers/                 # MCP Server Examples
│   ├── email_tools/            # Email MCP server
│   │   ├── main.py
│   │   ├── tools.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── database_tools/         # Database query MCP
│   │   ├── main.py
│   │   ├── tools.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── web_search/             # Web search MCP
│       ├── main.py
│       ├── tools.py
│       ├── requirements.txt
│       └── Dockerfile
│
├── shared/                      # Shared libraries
│   ├── __init__.py
│   ├── auth/                   # Shared auth utilities
│   │   ├── __init__.py
│   │   ├── jwt_service.py     # JWT handling
│   │   └── validators.py      # Auth validators
│   ├── errors/                 # Error handling
│   │   ├── __init__.py
│   │   ├── handlers.py        # Error handlers
│   │   └── exceptions.py      # Custom exceptions
│   ├── state/                  # State management
│   │   ├── __init__.py
│   │   ├── redis_store.py     # Redis operations
│   │   └── context_manager.py # Context handling
│   └── monitoring/             # Monitoring utilities
│       ├── __init__.py
│       ├── metrics.py         # Prometheus metrics
│       └── tracing.py         # OpenTelemetry
│
├── database/                    # Database schemas
│   ├── migrations/             # Alembic migrations
│   │   └── versions/
│   ├── schemas/                # SQL schemas
│   │   ├── postgres.sql       # PostgreSQL schema
│   │   └── init.sql           # Initial data
│   └── alembic.ini            # Alembic config
│
├── monitoring/                  # Monitoring configs
│   ├── prometheus/
│   │   └── prometheus.yml     # Prometheus config
│   ├── grafana/
│   │   └── dashboards/        # Grafana dashboards
│   └── alerts/                 # Alert rules
│
├── k8s/                        # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── deployments/
│   │   ├── frontend.yaml
│   │   ├── orchestrator.yaml
│   │   └── mcp-servers.yaml
│   ├── services/
│   │   ├── frontend-svc.yaml
│   │   ├── orchestrator-svc.yaml
│   │   └── mcp-svc.yaml
│   ├── ingress.yaml
│   └── hpa.yaml               # Horizontal Pod Autoscaler
│
├── scripts/                    # Utility scripts
│   ├── init_db.py             # Database initialization
│   ├── migrate.py             # Run migrations
│   ├── seed_data.py           # Seed test data
│   └── health_check.py        # Health check script
│
├── tests/                      # Integration tests
│   ├── integration/
│   │   ├── test_flow.py       # End-to-end tests
│   │   └── test_mcp.py        # MCP integration tests
│   ├── contract/              # Contract tests
│   │   └── test_contracts.py
│   └── load/                  # Load tests
│       └── locustfile.py
│
├── docker-compose.yml          # Local development
├── docker-compose.prod.yml     # Production config
├── .env.example               # Environment template
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
├── Makefile                   # Build automation
└── requirements-dev.txt       # Dev dependencies
```

## Key Directory Descriptions

### `/frontend`

Flask-based web application providing:

- Server-side rendered UI with Jinja2 templates
- WebSocket support via Flask-SocketIO
- OAuth2 authentication flow
- Static asset management

### `/orchestrator`

FastAPI service handling:

- OpenAI API integration
- MCP server discovery and tool execution
- Session and context management
- Real-time streaming responses

### `/mcp_servers`

Example MCP (Model Context Protocol) servers:

- Modular tool implementations
- Standard discovery/execute endpoints
- Independent deployment units

### `/shared`

Common libraries used across services:

- Authentication utilities
- Error handling
- State management
- Monitoring instrumentation

### `/database`

Database-related files:

- Schema definitions
- Migration scripts
- Initial seed data

### `/k8s`

Kubernetes deployment manifests:

- Service definitions
- Deployment configurations
- Ingress rules
- Auto-scaling policies

### `/monitoring`

Observability configurations:

- Prometheus metrics collection
- Grafana dashboards
- Alert rules and notifications

### `/tests`

Comprehensive test suites:

- Unit tests (in each service directory)
- Integration tests
- Contract tests
- Load/performance tests

## Development Workflow

1. **Local Development**: Use `docker-compose.yml` to spin up all services
2. **Testing**: Run tests with `make test`
3. **Building**: Build containers with `make build`
4. **Deployment**: Deploy to Kubernetes with `make deploy`

## Configuration Management

- Environment variables via `.env` files
- Kubernetes ConfigMaps for cluster deployments
- Secrets management via Kubernetes Secrets or external vault
