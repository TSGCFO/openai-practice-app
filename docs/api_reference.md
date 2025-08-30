# API Reference Documentation

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [REST API Endpoints](#rest-api-endpoints)
4. [WebSocket API](#websocket-api)
5. [Server-Sent Events (SSE)](#server-sent-events-sse)
6. [Data Models](#data-models)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [API Versioning](#api-versioning)
10. [OpenAPI Specification](#openapi-specification)

## Overview

The AI Assistant API provides programmatic access to all system functionality through RESTful endpoints and real-time WebSocket connections. The API follows REST principles and uses JSON for request and response payloads.

### Base URLs

```text
Production: https://api.ai-assistant.com
Staging:    https://api-staging.ai-assistant.com
Development: http://localhost:8000
```

### API Conventions

- **JSON API**: All requests and responses use JSON format
- **UTF-8 Encoding**: All text must be UTF-8 encoded
- **ISO 8601 Dates**: All timestamps use ISO 8601 format
- **UUID Identifiers**: All resource IDs are UUIDs v4
- **HTTP Status Codes**: Standard HTTP status codes indicate success/failure

## Authentication

### OAuth2 Flow

```mermaid
sequenceDiagram
    participant User
    participant Client
    participant AuthServer
    participant API
    
    User->>Client: Login Request
    Client->>AuthServer: Redirect to OAuth Provider
    AuthServer->>User: Login Form
    User->>AuthServer: Credentials
    AuthServer->>Client: Authorization Code
    Client->>AuthServer: Exchange Code for Token
    AuthServer->>Client: Access Token + Refresh Token
    Client->>API: API Request + Bearer Token
    API->>Client: API Response
```

### Authentication Methods

#### 1. Bearer Token (Recommended)

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### 2. API Key (Service Accounts)

```http
X-API-Key: your-api-key-here-replace-with-actual-key
```

#### 3. Session Cookie (Web Frontend)

```http
Cookie: session=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Token Management

```python
# Token refresh example
POST /api/v1/auth/refresh
Content-Type: application/json

{
    "refresh_token": "rt_1234567890abcdef"
}

# Response
{
    "access_token": "at_0987654321fedcba",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "rt_1234567890abcdef"
}
```

## REST API Endpoints

### Conversation Management

#### Create New Conversation

```http
POST /api/v1/conversations
Content-Type: application/json
Authorization: Bearer {token}

{
    "title": "New Conversation",
    "model": "gpt-5",
    "system_prompt": "You are a helpful assistant",
    "temperature": 0.7,
    "max_tokens": 4096
}

# Response: 201 Created
{
    "id": "conv_123e4567-e89b-12d3-a456-426614174000",
    "title": "New Conversation",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
    "model": "gpt-5",
    "message_count": 0,
    "token_usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
}
```

#### List Conversations

```http
GET /api/v1/conversations?page=1&limit=20&sort=-created_at
Authorization: Bearer {token}

# Response: 200 OK
{
    "data": [
        {
            "id": "conv_123e4567-e89b-12d3-a456-426614174000",
            "title": "Project Discussion",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "message_count": 15,
            "last_message": "Let me help you with that..."
        }
    ],
    "meta": {
        "page": 1,
        "limit": 20,
        "total": 45,
        "pages": 3
    }
}
```

#### Get Conversation Details

```http
GET /api/v1/conversations/{conversation_id}
Authorization: Bearer {token}

# Response: 200 OK
{
    "id": "conv_123e4567-e89b-12d3-a456-426614174000",
    "title": "Project Discussion",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
    "model": "gpt-5",
    "system_prompt": "You are a helpful assistant",
    "temperature": 0.7,
    "max_tokens": 4096,
    "messages": [
        {
            "id": "msg_123",
            "role": "user",
            "content": "Hello, I need help with my project",
            "created_at": "2024-01-01T00:00:00Z"
        },
        {
            "id": "msg_124",
            "role": "assistant",
            "content": "I'd be happy to help with your project...",
            "created_at": "2024-01-01T00:00:01Z",
            "tool_calls": []
        }
    ],
    "token_usage": {
        "prompt_tokens": 150,
        "completion_tokens": 250,
        "total_tokens": 400
    }
}
```

#### Delete Conversation

```http
DELETE /api/v1/conversations/{conversation_id}
Authorization: Bearer {token}

# Response: 204 No Content
```

### Message Operations

#### Send Message

```http
POST /api/v1/conversations/{conversation_id}/messages
Content-Type: application/json
Authorization: Bearer {token}

{
    "content": "Can you help me understand async programming?",
    "attachments": [
        {
            "type": "image",
            "url": "https://example.com/diagram.png"
        }
    ],
    "tools_enabled": true,
    "stream": true
}

# Response: 201 Created
{
    "id": "msg_456",
    "conversation_id": "conv_123",
    "role": "user",
    "content": "Can you help me understand async programming?",
    "created_at": "2024-01-01T00:00:00Z",
    "run_id": "run_789"
}
```

#### Get Message

```http
GET /api/v1/messages/{message_id}
Authorization: Bearer {token}

# Response: 200 OK
{
    "id": "msg_456",
    "conversation_id": "conv_123",
    "role": "assistant",
    "content": "Async programming allows...",
    "created_at": "2024-01-01T00:00:00Z",
    "tool_calls": [
        {
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "search_web",
                "arguments": "{\"query\": \"async programming tutorials\"}"
            }
        }
    ],
    "metadata": {
        "model": "gpt-5",
        "finish_reason": "stop",
        "tokens_used": 250
    }
}
```

#### Edit Message

```http
PATCH /api/v1/messages/{message_id}
Content-Type: application/json
Authorization: Bearer {token}

{
    "content": "Can you help me understand async/await in Python?"
}

# Response: 200 OK
{
    "id": "msg_456",
    "content": "Can you help me understand async/await in Python?",
    "edited_at": "2024-01-01T00:01:00Z"
}
```

### Tool Management

#### List Available Tools

```http
GET /api/v1/tools
Authorization: Bearer {token}

# Response: 200 OK
{
    "tools": [
        {
            "id": "tool_email",
            "name": "email_sender",
            "description": "Send emails on behalf of the user",
            "category": "communication",
            "mcp_server": "mcp_communication",
            "schema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "format": "email"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            },
            "requires_confirmation": true,
            "enabled": true
        }
    ],
    "total": 15
}
```

#### Register MCP Server

```http
POST /api/v1/tools/servers
Content-Type: application/json
Authorization: Bearer {token}

{
    "name": "custom_tools",
    "url": "https://mcp.example.com",
    "description": "Custom tools for project management",
    "auth": {
        "type": "bearer",
        "token": "mcp_token_123"
    }
}

# Response: 201 Created
{
    "id": "server_xyz",
    "name": "custom_tools",
    "url": "https://mcp.example.com",
    "status": "discovering",
    "tools_count": 0,
    "created_at": "2024-01-01T00:00:00Z"
}
```

#### Execute Tool

```http
POST /api/v1/tools/{tool_id}/execute
Content-Type: application/json
Authorization: Bearer {token}

{
    "conversation_id": "conv_123",
    "arguments": {
        "to": "user@example.com",
        "subject": "Meeting Tomorrow",
        "body": "Don't forget about our meeting tomorrow at 2 PM."
    },
    "require_confirmation": true
}

# Response: 200 OK
{
    "execution_id": "exec_789",
    "tool_id": "tool_email",
    "status": "pending_confirmation",
    "confirmation_token": "confirm_abc123",
    "expires_at": "2024-01-01T00:05:00Z"
}
```

#### Confirm Tool Execution

```http
POST /api/v1/tools/executions/{execution_id}/confirm
Content-Type: application/json
Authorization: Bearer {token}

{
    "confirmation_token": "confirm_abc123",
    "approved": true
}

# Response: 200 OK
{
    "execution_id": "exec_789",
    "status": "completed",
    "result": {
        "success": true,
        "message_id": "msg_sent_123",
        "details": "Email sent successfully"
    }
}
```

### User Management

#### Get User Profile

```http
GET /api/v1/users/me
Authorization: Bearer {token}

# Response: 200 OK
{
    "id": "user_123",
    "email": "user@example.com",
    "name": "John Doe",
    "created_at": "2024-01-01T00:00:00Z",
    "preferences": {
        "model": "gpt-5",
        "language": "en",
        "timezone": "America/New_York",
        "theme": "dark"
    },
    "quota": {
        "tokens_used": 15000,
        "tokens_limit": 100000,
        "reset_at": "2024-02-01T00:00:00Z"
    }
}
```

#### Update User Preferences

```http
PATCH /api/v1/users/me/preferences
Content-Type: application/json
Authorization: Bearer {token}

{
    "model": "gpt-5",
    "temperature": 0.8,
    "theme": "light",
    "notifications": {
        "email": true,
        "push": false
    }
}

# Response: 200 OK
{
    "preferences": {
        "model": "gpt-5",
        "temperature": 0.8,
        "theme": "light",
        "notifications": {
            "email": true,
            "push": false
        }
    },
    "updated_at": "2024-01-01T00:00:00Z"
}
```

## WebSocket API

### Connection Establishment

```javascript
// WebSocket connection example
const ws = new WebSocket('wss://api.ai-assistant.com/ws');

ws.onopen = () => {
    // Authenticate
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
    }));
    
    // Subscribe to conversation
    ws.send(JSON.stringify({
        type: 'subscribe',
        conversation_id: 'conv_123'
    }));
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};
```

### WebSocket Message Types

#### Client → Server Messages

```typescript
// Authentication
{
    type: 'auth',
    token: string
}

// Subscribe to conversation
{
    type: 'subscribe',
    conversation_id: string
}

// Send message
{
    type: 'message',
    conversation_id: string,
    content: string,
    stream: boolean
}

// Cancel streaming
{
    type: 'cancel',
    run_id: string
}

// Heartbeat
{
    type: 'ping'
}
```

#### Server → Client Messages

```typescript
// Authentication result
{
    type: 'auth_result',
    success: boolean,
    user_id?: string,
    error?: string
}

// Streaming token
{
    type: 'token',
    run_id: string,
    content: string,
    index: number
}

// Tool request
{
    type: 'tool_request',
    run_id: string,
    tool_id: string,
    tool_name: string,
    arguments: object,
    requires_confirmation: boolean
}

// Tool result
{
    type: 'tool_result',
    run_id: string,
    tool_id: string,
    result: object,
    error?: string
}

// Stream complete
{
    type: 'complete',
    run_id: string,
    message_id: string,
    finish_reason: 'stop' | 'length' | 'tool_calls' | 'error',
    usage: {
        prompt_tokens: number,
        completion_tokens: number,
        total_tokens: number
    }
}

// Error
{
    type: 'error',
    code: string,
    message: string,
    details?: object
}

// Heartbeat response
{
    type: 'pong'
}
```

### WebSocket Flow Example

```mermaid
sequenceDiagram
    participant Client
    participant WebSocket
    participant Orchestrator
    participant OpenAI
    
    Client->>WebSocket: Connect
    Client->>WebSocket: Authenticate
    WebSocket->>Client: Auth Success
    
    Client->>WebSocket: Send Message
    WebSocket->>Orchestrator: Process Message
    Orchestrator->>OpenAI: Stream Completion
    
    loop Streaming
        OpenAI-->>Orchestrator: Token
        Orchestrator-->>WebSocket: Token
        WebSocket-->>Client: Token Event
    end
    
    Orchestrator-->>WebSocket: Complete
    WebSocket-->>Client: Complete Event
```

## Server-Sent Events (SSE)

### SSE Endpoint

```http
GET /api/v1/stream/{conversation_id}
Authorization: Bearer {token}
Accept: text/event-stream

# Response: 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

event: token
data: {"content": "Hello", "index": 0}

event: token
data: {"content": " world", "index": 1}

event: tool_request
data: {"tool": "search", "args": {"query": "latest news"}}

event: tool_result
data: {"tool": "search", "result": {"items": [...]}}

event: complete
data: {"message_id": "msg_123", "tokens": 150}
```

### SSE Client Example

```javascript
// SSE client implementation
const eventSource = new EventSource('/api/v1/stream/conv_123', {
    headers: {
        'Authorization': 'Bearer token_here'
    }
});

eventSource.addEventListener('token', (e) => {
    const data = JSON.parse(e.data);
    appendToMessage(data.content);
});

eventSource.addEventListener('complete', (e) => {
    const data = JSON.parse(e.data);
    console.log('Message complete:', data);
    eventSource.close();
});

eventSource.addEventListener('error', (e) => {
    console.error('SSE error:', e);
    eventSource.close();
});
```

## Data Models

### Core Models

#### User Model

```typescript
interface User {
    id: string;
    email: string;
    name: string;
    created_at: string;
    updated_at: string;
    role: 'user' | 'admin' | 'developer';
    status: 'active' | 'suspended' | 'deleted';
    preferences: UserPreferences;
    quota: UserQuota;
}

interface UserPreferences {
    model: string;
    temperature: number;
    max_tokens: number;
    language: string;
    timezone: string;
    theme: 'light' | 'dark' | 'auto';
    notifications: {
        email: boolean;
        push: boolean;
        sms: boolean;
    };
}

interface UserQuota {
    tokens_used: number;
    tokens_limit: number;
    requests_used: number;
    requests_limit: number;
    reset_at: string;
}
```

#### Conversation Model

```typescript
interface Conversation {
    id: string;
    user_id: string;
    title: string;
    created_at: string;
    updated_at: string;
    model: string;
    system_prompt?: string;
    temperature: number;
    max_tokens: number;
    messages: Message[];
    token_usage: TokenUsage;
    metadata: Record<string, any>;
}

interface Message {
    id: string;
    conversation_id: string;
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string;
    created_at: string;
    edited_at?: string;
    tool_calls?: ToolCall[];
    attachments?: Attachment[];
    metadata: Record<string, any>;
}

interface TokenUsage {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    cost_usd: number;
}
```

#### Tool Models

```typescript
interface Tool {
    id: string;
    name: string;
    description: string;
    category: string;
    mcp_server: string;
    schema: JSONSchema;
    requires_confirmation: boolean;
    enabled: boolean;
    rate_limit?: RateLimit;
    permissions?: string[];
}

interface ToolCall {
    id: string;
    type: 'function';
    function: {
        name: string;
        arguments: string;
    };
    result?: ToolResult;
}

interface ToolResult {
    success: boolean;
    output: any;
    error?: string;
    execution_time_ms: number;
}

interface ToolExecution {
    id: string;
    tool_id: string;
    conversation_id: string;
    message_id: string;
    status: 'pending' | 'pending_confirmation' | 'executing' | 'completed' | 'failed';
    arguments: Record<string, any>;
    result?: ToolResult;
    created_at: string;
    completed_at?: string;
}
```

### MCP Models

```typescript
interface MCPServer {
    id: string;
    name: string;
    url: string;
    description: string;
    status: 'discovering' | 'active' | 'inactive' | 'error';
    tools: Tool[];
    auth: MCPAuth;
    health_check_url?: string;
    created_at: string;
    updated_at: string;
}

interface MCPDiscoveryResponse {
    server_name: string;
    version: string;
    tools: MCPTool[];
    capabilities: string[];
}

interface MCPTool {
    slug: string;
    name: string;
    description: string;
    input_schema: JSONSchema;
    output_schema?: JSONSchema;
    requires_auth: boolean;
    rate_limit?: RateLimit;
}

interface MCPExecuteRequest {
    tools: Array<{
        tool_slug: string;
        arguments: Record<string, any>;
    }>;
    sync_response_to_workbench?: boolean;
    thought?: string;
    current_step?: string;
}

interface MCPExecuteResponse {
    request_id: string;
    successful: boolean;
    data?: {
        results: Array<{
            tool_slug: string;
            response: any;
            error?: string;
        }>;
    };
    error?: string;
}
```

## Error Handling

### Error Response Format

```json
{
    "error": {
        "type": "validation_error",
        "message": "Invalid request parameters",
        "code": "ERR_VALIDATION",
        "details": {
            "field": "temperature",
            "reason": "Must be between 0 and 2"
        },
        "request_id": "req_123abc",
        "documentation_url": "https://docs.ai-assistant.com/errors/ERR_VALIDATION"
    }
}
```

### Error Types

| Error Type | HTTP Status | Description |
|------------|-------------|-------------|
| `authentication_error` | 401 | Invalid or expired authentication |
| `authorization_error` | 403 | Insufficient permissions |
| `validation_error` | 400 | Invalid request parameters |
| `not_found_error` | 404 | Resource not found |
| `rate_limit_error` | 429 | Rate limit exceeded |
| `quota_exceeded_error` | 402 | Usage quota exceeded |
| `server_error` | 500 | Internal server error |
| `service_unavailable` | 503 | Service temporarily unavailable |

### Error Codes

```typescript
enum ErrorCode {
    // Authentication
    ERR_AUTH_INVALID_TOKEN = 'ERR_AUTH_INVALID_TOKEN',
    ERR_AUTH_TOKEN_EXPIRED = 'ERR_AUTH_TOKEN_EXPIRED',
    ERR_AUTH_INVALID_CREDENTIALS = 'ERR_AUTH_INVALID_CREDENTIALS',
    
    // Validation
    ERR_VALIDATION_REQUIRED_FIELD = 'ERR_VALIDATION_REQUIRED_FIELD',
    ERR_VALIDATION_INVALID_FORMAT = 'ERR_VALIDATION_INVALID_FORMAT',
    ERR_VALIDATION_OUT_OF_RANGE = 'ERR_VALIDATION_OUT_OF_RANGE',
    
    // Resources
    ERR_RESOURCE_NOT_FOUND = 'ERR_RESOURCE_NOT_FOUND',
    ERR_RESOURCE_ALREADY_EXISTS = 'ERR_RESOURCE_ALREADY_EXISTS',
    ERR_RESOURCE_LOCKED = 'ERR_RESOURCE_LOCKED',
    
    // Rate Limiting
    ERR_RATE_LIMIT_EXCEEDED = 'ERR_RATE_LIMIT_EXCEEDED',
    ERR_QUOTA_EXCEEDED = 'ERR_QUOTA_EXCEEDED',
    
    // Tools
    ERR_TOOL_NOT_FOUND = 'ERR_TOOL_NOT_FOUND',
    ERR_TOOL_EXECUTION_FAILED = 'ERR_TOOL_EXECUTION_FAILED',
    ERR_TOOL_TIMEOUT = 'ERR_TOOL_TIMEOUT',
    
    // System
    ERR_INTERNAL_SERVER = 'ERR_INTERNAL_SERVER',
    ERR_SERVICE_UNAVAILABLE = 'ERR_SERVICE_UNAVAILABLE',
    ERR_DEPENDENCY_FAILURE = 'ERR_DEPENDENCY_FAILURE'
}
```

## Rate Limiting

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 75
X-RateLimit-Reset: 1640995200
X-RateLimit-Reset-After: 3600
X-RateLimit-Bucket: api_general
```

### Rate Limit Configuration

```yaml
rate_limits:
  global:
    requests_per_minute: 100
    requests_per_hour: 1000
    
  per_user:
    requests_per_minute: 30
    requests_per_hour: 500
    tokens_per_hour: 100000
    
  per_endpoint:
    /api/v1/conversations:
      POST: 10/minute
      GET: 100/minute
    /api/v1/tools/*/execute:
      POST: 5/minute
```

### Rate Limit Response

```http
429 Too Many Requests
Retry-After: 60

{
    "error": {
        "type": "rate_limit_error",
        "message": "Rate limit exceeded. Please retry after 60 seconds.",
        "code": "ERR_RATE_LIMIT_EXCEEDED",
        "details": {
            "limit": 100,
            "remaining": 0,
            "reset_at": "2024-01-01T00:01:00Z"
        }
    }
}
```

## API Versioning

### Version Strategy

- **URL Path Versioning**: `/api/v1/`, `/api/v2/`
- **Backward Compatibility**: Minimum 6 months deprecation notice
- **Version Sunset**: Clear communication of EOL dates

### Version Headers

```http
# Request specific version
API-Version: 2024-01-01

# Response includes version info
X-API-Version: 2024-01-01
X-API-Deprecated: false
X-API-Sunset: 2025-01-01
```

### Deprecation Notice

```json
{
    "data": {...},
    "warnings": [
        {
            "type": "deprecation",
            "message": "This endpoint will be removed in v2",
            "deprecated_at": "2024-01-01",
            "sunset_at": "2024-07-01",
            "migration_guide": "https://docs.ai-assistant.com/migration/v1-to-v2"
        }
    ]
}
```

## OpenAPI Specification

### OpenAPI Document Structure

```yaml
openapi: 3.1.0
info:
  title: AI Assistant API
  version: 1.0.0
  description: API for AI Assistant with OpenAI and MCP integration
  contact:
    email: api@ai-assistant.com
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: https://api.ai-assistant.com
    description: Production server
  - url: https://api-staging.ai-assistant.com
    description: Staging server
  - url: http://localhost:8000
    description: Development server

security:
  - BearerAuth: []
  - ApiKeyAuth: []

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

  schemas:
    User:
      type: object
      required: [id, email, name]
      properties:
        id:
          type: string
          format: uuid
        email:
          type: string
          format: email
        name:
          type: string

    Conversation:
      type: object
      required: [id, user_id, title]
      properties:
        id:
          type: string
          format: uuid
        user_id:
          type: string
          format: uuid
        title:
          type: string
        messages:
          type: array
          items:
            $ref: '#/components/schemas/Message'

    Message:
      type: object
      required: [id, role, content]
      properties:
        id:
          type: string
          format: uuid
        role:
          type: string
          enum: [system, user, assistant, tool]
        content:
          type: string

    Error:
      type: object
      required: [error]
      properties:
        error:
          type: object
          required: [type, message, code]
          properties:
            type:
              type: string
            message:
              type: string
            code:
              type: string
            details:
              type: object

paths:
  /api/v1/conversations:
    get:
      summary: List conversations
      operationId: listConversations
      tags: [Conversations]
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Conversation'
                  meta:
                    type: object
        '401':
          $ref: '#/components/responses/Unauthorized'
        '429':
          $ref: '#/components/responses/RateLimited'
```

### API Client Generation

```bash
# Generate TypeScript client
openapi-generator generate \
  -i openapi.yaml \
  -g typescript-axios \
  -o ./client/typescript

# Generate Python client
openapi-generator generate \
  -i openapi.yaml \
  -g python \
  -o ./client/python

# Generate documentation
openapi-generator generate \
  -i openapi.yaml \
  -g html2 \
  -o ./docs/api
```

## API Testing

### Example cURL Commands

```bash
# Authenticate
curl -X POST https://api.ai-assistant.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Create conversation
curl -X POST https://api.ai-assistant.com/api/v1/conversations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "New Chat", "model": "gpt-5"}'

# Send message
curl -X POST https://api.ai-assistant.com/api/v1/conversations/$CONV_ID/messages \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello, world!"}'

# Stream response
curl -N https://api.ai-assistant.com/api/v1/stream/$CONV_ID \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: text/event-stream"
```

### Postman Collection

```json
{
  "info": {
    "name": "AI Assistant API",
    "version": "1.0.0",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "auth": {
    "type": "bearer",
    "bearer": [
      {
        "key": "token",
        "value": "{{access_token}}",
        "type": "string"
      }
    ]
  },
  "variable": [
    {
      "key": "base_url",
      "value": "https://api.ai-assistant.com",
      "type": "string"
    },
    {
      "key": "access_token",
      "value": "",
      "type": "string"
    }
  ],
  "item": [
    {
      "name": "Authentication",
      "item": [
        {
          "name": "Login",
          "request": {
            "method": "POST",
            "header": [
              {
                "key": "Content-Type",
                "value": "application/json"
              }
            ],
            "body": {
              "mode": "raw",
              "raw": "{\"email\": \"{{email}}\", \"password\": \"{{password}}\"}"
            },
            "url": {
              "raw": "{{base_url}}/api/v1/auth/login",
              "host": ["{{base_url}}"],
              "path": ["api", "v1", "auth", "login"]
            }
          }
        }
      ]
    }
  ]
}
```

## SDK Examples

### Python SDK

```python
from ai_assistant import Client, StreamHandler

# Initialize client
client = Client(
    api_key="sk_live_...",
    base_url="https://api.ai-assistant.com"
)

# Create conversation
conversation = client.conversations.create(
    title="Python SDK Test",
    model="gpt-5",
    temperature=0.7
)

# Send message with streaming
class MyStreamHandler(StreamHandler):
    def on_token(self, token: str):
        print(token, end="", flush=True)
    
    def on_tool_request(self, tool: str, args: dict):
        print(f"\nTool request: {tool}")
    
    def on_complete(self, message_id: str):
        print(f"\nComplete: {message_id}")

response = client.messages.create(
    conversation_id=conversation.id,
    content="Explain quantum computing",
    stream=True,
    stream_handler=MyStreamHandler()
)
```

### JavaScript/TypeScript SDK

```typescript
import { AIAssistant, StreamEventType } from '@ai-assistant/sdk';

// Initialize client
const client = new AIAssistant({
    apiKey: 'sk_live_...',
    baseURL: 'https://api.ai-assistant.com'
});

// Create conversation
const conversation = await client.conversations.create({
    title: 'TypeScript SDK Test',
    model: 'gpt-5',
    temperature: 0.7
});

// Send message with streaming
const stream = await client.messages.stream(conversation.id, {
    content: 'Explain quantum computing'
});

stream.on(StreamEventType.Token, (token) => {
    process.stdout.write(token);
});

stream.on(StreamEventType.ToolRequest, (tool, args) => {
    console.log(`Tool request: ${tool}`);
});

stream.on(StreamEventType.Complete, (messageId) => {
    console.log(`Complete: ${messageId}`);
});
```

## Best Practices

### API Design Guidelines

1. **Consistent Naming**: Use consistent naming conventions across all endpoints
2. **Idempotency**: Support idempotency keys for critical operations
3. **Pagination**: Always paginate list endpoints
4. **Filtering**: Provide comprehensive filtering options
5. **Sorting**: Support multiple sort fields
6. **Field Selection**: Allow clients to specify returned fields
7. **Bulk Operations**: Support batch operations where appropriate

### Security Best Practices

1. **Rate Limiting**: Implement per-user and per-IP rate limits
2. **Authentication**: Use short-lived tokens with refresh mechanism
3. **Authorization**: Implement fine-grained permissions
4. **Input Validation**: Validate all inputs against schemas
5. **Output Sanitization**: Sanitize responses to prevent XSS
6. **Audit Logging**: Log all API calls for security analysis
7. **CORS**: Configure CORS policies appropriately

### Performance Guidelines

1. **Response Compression**: Use gzip/brotli compression
2. **Caching**: Implement appropriate cache headers
3. **Connection Pooling**: Reuse connections
4. **Async Processing**: Use async for long-running operations
5. **Database Optimization**: Use appropriate indexes
6. **CDN**: Serve static content via CDN
7. **Monitoring**: Track API performance metrics

---

**Document Version**: 1.0.0  
**Last Updated**: December 2024  
**API Version**: v1  
**Status**: Stable