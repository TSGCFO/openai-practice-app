# AI Assistant System

A production-ready AI Assistant system leveraging OpenAI's GPT models with extensible Model Context Protocol (MCP) servers for tool functionality.

## Features

- 🤖 **OpenAI Integration**: Powered by GPT-4 with streaming responses
- 🔧 **Extensible Tools**: MCP servers for email, database queries, and more
- 💬 **Real-time Chat**: WebSocket-based streaming for instant responses
- 🔐 **Secure Authentication**: OAuth2/OIDC with JWT tokens
- 📊 **Monitoring**: Built-in Prometheus metrics and Grafana dashboards
- 🚀 **Production Ready**: Docker, Kubernetes, and CI/CD pipelines

## Architecture

- **Frontend**: Flask with server-side rendering and WebSocket support
- **Orchestrator**: FastAPI for async OpenAI and MCP coordination
- **MCP Servers**: Modular tool servers following Model Context Protocol
- **Data Layer**: PostgreSQL, Redis, and optional Vector DB

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Node.js 18+ (for frontend assets)
- OpenAI API key

### Local Development

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-assistant
   ```

2. Copy environment configuration:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Start services with Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Initialize the database:
   ```bash
   docker-compose exec orchestrator python scripts/init_db.py
   ```

5. Access the application:
   - Frontend: http://localhost:5000
   - Orchestrator API: http://localhost:8000/docs
   - MCP Email Server: http://localhost:8001/mcp/discover
   - MCP Database Server: http://localhost:8002/mcp/discover

## Project Structure

```
ai-assistant/
├── frontend/           # Flask web application
├── orchestrator/       # FastAPI service
├── mcp_servers/        # MCP tool servers
├── shared/            # Shared libraries
├── database/          # Database schemas and migrations
├── k8s/              # Kubernetes manifests
├── monitoring/        # Monitoring configurations
└── tests/            # Test suites
```

## Development

### Running Tests

```bash
# Unit tests
make test-unit

# Integration tests
make test-integration

# All tests
make test
```

### Building Docker Images

```bash
# Build all images
make build

# Build specific service
make build-frontend
make build-orchestrator
```

### Database Migrations

```bash
# Create new migration
make migration NAME="add_user_table"

# Run migrations
make migrate
```

## API Documentation

- **Orchestrator API**: http://localhost:8000/docs (FastAPI Swagger UI)
- **MCP Protocol**: See `docs/mcp_integration.md`
- **WebSocket Events**: See `docs/api_reference.md`

## Configuration

All configuration is managed through environment variables. See `.env.example` for available options.

Key configurations:
- `OPENAI_API_KEY`: Your OpenAI API key
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `JWT_SECRET_KEY`: Secret for JWT token signing

## Deployment

### Docker Compose (Development)

```bash
docker-compose up -d
```

### Kubernetes (Production)

```bash
# Apply configurations
kubectl apply -f k8s/

# Check status
kubectl get pods -n ai-assistant
```

## Monitoring

- **Metrics**: Prometheus metrics at `/metrics` endpoints
- **Dashboards**: Grafana dashboards in `monitoring/grafana/`
- **Logs**: Structured JSON logs to stdout
- **Tracing**: OpenTelemetry integration (optional)

## Security

- OAuth2/OIDC authentication with Google/GitHub
- JWT tokens for API access
- Rate limiting per user and endpoint
- Input validation and sanitization
- Encrypted data at rest and in transit

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run `make lint` and `make test`
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- Documentation: `docs/`
- Issues: GitHub Issues
- Discord: [Join our community](https://discord.gg/example)

## Acknowledgments

- OpenAI for GPT models
- Model Context Protocol specification
- Flask and FastAPI communities