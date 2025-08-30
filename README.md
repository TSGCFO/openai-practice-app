# AI Assistant with OpenAI & MCP Integration

A production-ready Python-based AI Assistant that leverages OpenAI's API as its core language model and implements Model Context Protocol (MCP) servers for extensible functionality. Features a modern web interface with real-time streaming responses, automatic tool discovery, and modular architecture.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- OpenAI API Key
- PostgreSQL 14+ (or use Docker)
- Redis 7+ (or use Docker)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ai-assistant.git
cd ai-assistant

# Copy environment configuration
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...

# Start with Docker Compose (recommended)
docker-compose up -d

# OR manual setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application

```bash
# Using Docker Compose (recommended)
docker-compose up

# OR run services individually
# Terminal 1: Start PostgreSQL and Redis
docker-compose up postgres redis

# Terminal 2: Start Orchestrator
cd orchestrator
uvicorn main:app --reload --port 8000

# Terminal 3: Start Frontend
cd frontend
python app.py

# Access the application
# Frontend: http://localhost:5000
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## 🏗️ Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │────▶│   Frontend  │────▶│ Orchestrator│
└─────────────┘     │   (Flask)   │     │  (FastAPI)  │
                    └─────────────┘     └─────────────┘
                                              │
                                              ▼
                                        ┌──────────┐
                                        │ OpenAI   │
                                        │   API    │
                                        └──────────┘
                                              │
                                              ▼
                                        ┌──────────┐
                                        │   MCP    │
                                        │ Servers  │
                                        └──────────┘
```

### Core Components

- **Frontend (Flask)**: Server-rendered web interface with real-time WebSocket streaming
- **Orchestrator (FastAPI)**: Core service managing LLM interactions and tool coordination
- **MCP Servers**: Extensible tool servers following Model Context Protocol
- **PostgreSQL**: Primary database for users, conversations, and audit logs
- **Redis**: Session storage, caching, and real-time pub/sub

## ✨ Key Features

### Core Capabilities

- 🤖 **OpenAI Integration**: GPT-5/GPT-5 with streaming responses
- 🔧 **Extensible Tools**: MCP protocol for adding custom tools
- 🔄 **Real-time Streaming**: WebSocket/SSE for instant responses
- 🔐 **Secure Authentication**: OAuth2/JWT with multiple providers
- 📊 **Session Management**: Conversation history and context
- ⚡ **High Performance**: Async architecture with caching
- 🛡️ **Production Ready**: Error handling, monitoring, and scaling

### Supported Features

- Multi-turn conversations with context management
- Tool discovery and automatic registration
- Rate limiting and quota management
- Comprehensive error handling and retry logic
- Distributed state management with Redis
- Vector database integration for RAG
- Full audit logging and compliance

## 📚 Documentation

### Complete Documentation

- 📖 [System Architecture](docs/system_architecture.md) - Complete system design and components
- 🔌 [API Reference](docs/api_reference.md) - REST and WebSocket API documentation
- 🔧 [MCP Integration Guide](docs/mcp_integration.md) - Creating and integrating MCP servers
- 🔐 [Authentication & Security](docs/authentication_security.md) - Security implementation details
- 💾 [State Management](docs/state_management.md) - Session and error handling
- 🚀 [Deployment Guide](docs/deployment_guide.md) - Production deployment instructions
- 💡 [Implementation Examples](docs/implementation_examples.md) - Code examples and patterns

### Quick Links

- [Getting Started Guide](#-quick-start)
- [Configuration](#-configuration)
- [API Examples](#-api-examples)
- [Creating MCP Tools](#-creating-mcp-tools)
- [Troubleshooting](#-troubleshooting)

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-5
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.7

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/ai_assistant
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-change-in-production

# OAuth Providers (optional)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# MCP Servers
MCP_DISCOVERY_URLS=http://localhost:8001,http://localhost:8002
```

### Configuration Files

- `docker-compose.yml` - Docker services configuration
- `orchestrator/config.py` - Orchestrator settings
- `frontend/config.py` - Frontend settings
- `k8s/` - Kubernetes deployment configs

## 💻 API Examples

### Create a Conversation

```bash
curl -X POST http://localhost:8000/api/v1/conversations \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "New Chat",
    "model": "gpt-5",
    "system_prompt": "You are a helpful assistant"
  }'
```

### Send a Message

```bash
curl -X POST http://localhost:8000/api/v1/conversations/{conversation_id}/messages \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hello, how can you help me today?"
  }'
```

### Stream Response (SSE)

```javascript
const eventSource = new EventSource(
  `/api/v1/stream/${conversationId}`,
  { headers: { 'Authorization': `Bearer ${token}` } }
);

eventSource.addEventListener('token', (e) => {
  const data = JSON.parse(e.data);
  console.log('Token:', data.content);
});
```

### WebSocket Connection

```javascript
const socket = io('http://localhost:5000');

socket.emit('send_message', {
  conversation_id: conversationId,
  message: 'Hello!',
  token: authToken
});

socket.on('token', (data) => {
  console.log('Received token:', data.content);
});
```

## 🔧 Creating MCP Tools

### Basic MCP Server Template

```python
# mcp_servers/my_tool/main.py
from fastapi import FastAPI
from typing import Dict, Any

app = FastAPI(title="My Tool MCP Server")

TOOLS = {
    "my_tool": {
        "name": "my_tool",
        "description": "Description of what your tool does",
        "input_schema": {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "integer"}
            },
            "required": ["param1"]
        }
    }
}

@app.get("/mcp/discover")
async def discover():
    return {
        "server_name": "my_tool_server",
        "version": "1.0.0",
        "tools": list(TOOLS.values())
    }

@app.post("/mcp/execute")
async def execute(request: Dict[str, Any]):
    # Implement tool execution logic
    return {
        "request_id": request.get("request_id"),
        "successful": True,
        "data": {"results": []}
    }
```

### Registering Your MCP Server

1. Add your server URL to `.env`:
```env
MCP_DISCOVERY_URLS=http://localhost:8001,http://localhost:8002,http://localhost:8003
```

2. Start your MCP server:
```bash
cd mcp_servers/my_tool
uvicorn main:app --port 8003
```

3. The orchestrator will automatically discover and register your tools!

## 🐳 Docker Deployment

### Development

```bash
# Build and start all services
docker-compose up --build

# Scale orchestrator service
docker-compose up -d --scale orchestrator=3

# View logs
docker-compose logs -f orchestrator

# Stop all services
docker-compose down
```

### Production

```bash
# Build production images
docker build -t ai-assistant/orchestrator:latest ./orchestrator
docker build -t ai-assistant/frontend:latest ./frontend

# Deploy with Docker Swarm
docker stack deploy -c docker-compose.prod.yml ai-assistant

# OR deploy to Kubernetes
kubectl apply -f k8s/
```

## ☸️ Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace ai-assistant

# Apply configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Deploy services
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/services/
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl get pods -n ai-assistant
kubectl get svc -n ai-assistant
```

## 📊 Monitoring

### Prometheus Metrics

The application exposes metrics at `/metrics`:

- Request count and duration
- Token usage by model
- Active connections
- Error rates
- Cache hit ratios

### Health Checks

- Health endpoint: `/health`
- Ready endpoint: `/ready`
- Metrics endpoint: `/metrics`

### Grafana Dashboards

Import the provided dashboards from `monitoring/grafana/dashboards/`:

- System Overview
- API Performance
- Token Usage
- Error Analysis

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py

# Run integration tests
pytest tests/integration/
```

### Test Categories

- **Unit Tests**: `tests/unit/`
- **Integration Tests**: `tests/integration/`
- **End-to-End Tests**: `tests/e2e/`
- **Load Tests**: `tests/load/`

## 🔍 Troubleshooting

### Common Issues

#### Connection Refused
```bash
# Check if services are running
docker-compose ps

# Check logs
docker-compose logs orchestrator

# Verify port configuration
netstat -tulpn | grep 8000
```

#### OpenAI API Errors
```bash
# Verify API key
echo $OPENAI_API_KEY

# Test API connection
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### Database Connection Issues
```bash
# Test database connection
psql $DATABASE_URL -c "SELECT 1"

# Check database logs
docker-compose logs postgres
```

### Debug Mode

Enable debug logging:

```env
# .env
DEBUG=true
LOG_LEVEL=DEBUG
```

### Getting Help

- 📖 Check the [documentation](docs/)
- 🐛 Report issues on [GitHub Issues](https://github.com/your-org/ai-assistant/issues)
- 💬 Join our [Discord community](https://discord.gg/your-community)
- 📧 Email support: support@your-domain.com

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/ai-assistant.git

# Create a feature branch
git checkout -b feature/your-feature

# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Make your changes and run tests
pytest

# Submit a pull request
```

### Code Style

- Python: Follow PEP 8
- JavaScript: Use ESLint configuration
- Format code with Black and Prettier
- Write comprehensive docstrings
- Add unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the GPT models
- Model Context Protocol community
- FastAPI and Flask communities
- All contributors and supporters

## 📈 Roadmap

### Version 1.1 (Q1 2025)
- [ ] Multi-modal support (images, audio)
- [ ] Advanced RAG implementation
- [ ] Plugin marketplace
- [ ] Mobile app

### Version 1.2 (Q2 2025)
- [ ] Fine-tuning integration
- [ ] Advanced analytics dashboard
- [ ] Team collaboration features
- [ ] Enterprise SSO

### Version 2.0 (Q3 2025)
- [ ] Multi-model support (Claude, Gemini)
- [ ] Workflow automation
- [ ] Advanced memory systems
- [ ] Self-hosted LLM support

## 📞 Contact

- **Project Lead**: [Your Name](mailto:your.email@example.com)
- **Documentation**: [docs.ai-assistant.com](https://docs.ai-assistant.com)
- **Security Issues**: security@your-domain.com
- **Community**: [Discord](https://discord.gg/your-community) | [Twitter](https://twitter.com/your-handle)

---

**Built with ❤️ by the AI Assistant Team**

*Last Updated: December 2024 | Version: 1.0.0*