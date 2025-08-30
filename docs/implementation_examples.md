# Implementation Examples & Best Practices

## Table of Contents

1. [Overview](#overview)
2. [Orchestrator Implementation](#orchestrator-implementation)
3. [MCP Server Examples](#mcp-server-examples)
4. [Frontend Implementation](#frontend-implementation)
5. [Tool Implementation Examples](#tool-implementation-examples)
6. [Streaming Response Handling](#streaming-response-handling)
7. [Production Best Practices](#production-best-practices)
8. [Common Patterns](#common-patterns)
9. [Performance Monitoring](#performance-monitoring)
10. [Troubleshooting Guide](#troubleshooting-guide)

## Overview

This document provides practical implementation examples and best practices for building and maintaining the AI Assistant system. All examples follow production-ready patterns and include error handling, logging, and performance considerations.

## Orchestrator Implementation

### Basic Orchestrator Setup

```python
# orchestrator/main.py
from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from .config import settings
from .services import OpenAIService, MCPManager, SessionManager
from .middleware import AuthMiddleware, ErrorHandlerMiddleware, RateLimitMiddleware
from .database import init_db, close_db
from .redis_client import init_redis, close_redis

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager"""
    # Startup
    logger.info("Starting AI Assistant Orchestrator")
    await init_db()
    await init_redis()
    await app.state.mcp_manager.discover_tools()
    yield
    # Shutdown
    logger.info("Shutting down AI Assistant Orchestrator")
    await close_db()
    await close_redis()

# Create FastAPI app
app = FastAPI(
    title="AI Assistant Orchestrator",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
app.state.openai_service = OpenAIService(api_key=settings.OPENAI_API_KEY)
app.state.mcp_manager = MCPManager(settings.MCP_SERVERS)
app.state.session_manager = SessionManager()

# Import routers
from .routers import conversations, messages, tools, auth, health

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(conversations.router, prefix="/api/v1/conversations", tags=["conversations"])
app.include_router(messages.router, prefix="/api/v1/messages", tags=["messages"])
app.include_router(tools.router, prefix="/api/v1/tools", tags=["tools"])
app.include_router(health.router, tags=["health"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Assistant Orchestrator",
        "version": "1.0.0",
        "status": "running"
    }
```

### Conversation Handler Implementation

```python
# orchestrator/services/conversation_service.py
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import uuid
import logging
from openai import AsyncOpenAI

from ..models import Conversation, Message, ToolCall
from ..database import get_db
from ..redis_client import get_redis
from ..services.mcp_manager import MCPManager

logger = logging.getLogger(__name__)

class ConversationService:
    """Service for managing conversations"""
    
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        mcp_manager: MCPManager
    ):
        self.openai = openai_client
        self.mcp = mcp_manager
        
    async def create_conversation(
        self,
        user_id: str,
        title: str = None,
        model: str = "gpt-5",
        system_prompt: str = None
    ) -> Conversation:
        """Create new conversation"""
        conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title or "New Conversation",
            model=model,
            system_prompt=system_prompt,
            created_at=datetime.utcnow()
        )
        
        # Save to database
        async with get_db() as db:
            await db.conversations.insert(conversation.dict())
        
        logger.info(f"Created conversation {conversation.id} for user {user_id}")
        return conversation
    
    async def process_message(
        self,
        conversation_id: str,
        content: str,
        user_id: str,
        stream: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process user message and generate response"""
        
        # Load conversation
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Add user message
        user_message = await self.add_message(
            conversation_id,
            role="user",
            content=content
        )
        
        # Build prompt
        messages = await self.build_prompt(conversation)
        
        # Get available tools
        tools = await self.mcp.get_tools_for_conversation(conversation_id)
        
        try:
            # Call OpenAI with streaming
            if stream:
                async for chunk in self.stream_completion(messages, tools, conversation.model):
                    yield chunk
            else:
                response = await self.get_completion(messages, tools, conversation.model)
                yield {"type": "complete", "content": response}
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            yield {"type": "error", "error": str(e)}
    
    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream completion from OpenAI"""
        
        # Prepare OpenAI request
        request_params = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"
        
        # Stream response
        stream = await self.openai.chat.completions.create(**request_params)
        
        full_response = ""
        tool_calls = []
        
        async for chunk in stream:
            delta = chunk.choices[0].delta
            
            # Stream content tokens
            if delta.content:
                full_response += delta.content
                yield {
                    "type": "token",
                    "content": delta.content,
                    "index": len(full_response)
                }
            
            # Handle tool calls
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    tool_calls.append(tool_call)
                    yield {
                        "type": "tool_request",
                        "tool_id": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
        
        # Execute tool calls if any
        if tool_calls:
            for tool_call in tool_calls:
                result = await self.execute_tool(
                    tool_call.function.name,
                    tool_call.function.arguments
                )
                yield {
                    "type": "tool_result",
                    "tool_id": tool_call.function.name,
                    "result": result
                }
                
                # Continue conversation with tool result
                messages.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call.id
                })
                
                # Get follow-up response
                async for chunk in self.stream_completion(messages, tools, model):
                    yield chunk
        
        # Save assistant message
        await self.add_message(
            conversation_id=messages[0].get("conversation_id"),
            role="assistant",
            content=full_response,
            tool_calls=tool_calls
        )
        
        yield {
            "type": "complete",
            "message_id": str(uuid.uuid4()),
            "content": full_response
        }
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: str
    ) -> Dict[str, Any]:
        """Execute tool via MCP"""
        import json
        
        try:
            args = json.loads(arguments)
            result = await self.mcp.execute_tool(tool_name, args)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": str(e)}
    
    async def build_prompt(
        self,
        conversation: Conversation
    ) -> List[Dict[str, Any]]:
        """Build prompt from conversation history"""
        messages = []
        
        # Add system prompt
        if conversation.system_prompt:
            messages.append({
                "role": "system",
                "content": conversation.system_prompt
            })
        
        # Add conversation messages
        async with get_db() as db:
            history = await db.messages.find(
                {"conversation_id": conversation.id},
                sort=[("created_at", 1)],
                limit=20  # Keep last 20 messages
            ).to_list()
        
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return messages
```

## MCP Server Examples

### Email Tools MCP Server

```python
# mcp_servers/email_tools/main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

app = FastAPI(title="Email Tools MCP Server")
logger = logging.getLogger(__name__)

# Tool definitions
TOOLS = {
    "send_email": {
        "name": "send_email",
        "description": "Send an email to specified recipients",
        "category": "communication",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string", "format": "email"},
                    "description": "Email recipients"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject"
                },
                "body": {
                    "type": "string",
                    "description": "Email body (HTML supported)"
                },
                "cc": {
                    "type": "array",
                    "items": {"type": "string", "format": "email"},
                    "description": "CC recipients",
                    "default": []
                },
                "attachments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "content": {"type": "string"},
                            "content_type": {"type": "string"}
                        }
                    },
                    "default": []
                }
            },
            "required": ["to", "subject", "body"]
        },
        "requires_confirmation": True
    },
    "search_emails": {
        "name": "search_emails",
        "description": "Search emails in inbox",
        "category": "communication",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "folder": {
                    "type": "string",
                    "description": "Email folder to search",
                    "enum": ["inbox", "sent", "drafts", "trash"],
                    "default": "inbox"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}

class EmailService:
    """Email service implementation"""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_host = smtp_config.get("host", "smtp.gmail.com")
        self.smtp_port = smtp_config.get("port", 587)
        self.smtp_user = smtp_config.get("user")
        self.smtp_password = smtp_config.get("password")
    
    async def send_email(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: List[str] = None,
        attachments: List[Dict] = None
    ) -> Dict[str, Any]:
        """Send email implementation"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_user
            msg['To'] = ', '.join(to)
            if cc:
                msg['Cc'] = ', '.join(cc)
            
            # Add body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Add attachments if any
            if attachments:
                for attachment in attachments:
                    # Process attachment
                    pass
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                
                recipients = to + (cc or [])
                server.send_message(msg, to_addrs=recipients)
            
            logger.info(f"Email sent to {recipients}")
            
            return {
                "status": "sent",
                "message_id": f"msg_{datetime.utcnow().timestamp()}",
                "recipients": recipients,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def search_emails(
        self,
        query: str,
        folder: str = "inbox",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search emails implementation"""
        # This would typically connect to an email service API
        # For example, using Gmail API or IMAP
        
        results = []
        
        # Mock implementation
        import random
        for i in range(min(limit, random.randint(1, 5))):
            results.append({
                "id": f"email_{i}",
                "subject": f"Email about {query}",
                "from": f"sender{i}@example.com",
                "date": datetime.utcnow().isoformat(),
                "snippet": f"This email contains information about {query}..."
            })
        
        return results

# Initialize email service
email_service = EmailService({
    "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
    "port": int(os.getenv("SMTP_PORT", "587")),
    "user": os.getenv("SMTP_USER"),
    "password": os.getenv("SMTP_PASSWORD")
})

@app.get("/mcp/discover")
async def discover():
    """MCP discovery endpoint"""
    return {
        "server_name": "email_tools",
        "version": "1.0.0",
        "description": "Email management tools",
        "tools": list(TOOLS.values()),
        "capabilities": ["email", "communication"]
    }

@app.post("/mcp/execute")
async def execute(request: Dict[str, Any]):
    """MCP execution endpoint"""
    tools_to_execute = request.get("tools", [])
    results = []
    
    for tool_request in tools_to_execute:
        tool_slug = tool_request.get("tool_slug")
        arguments = tool_request.get("arguments", {})
        
        if tool_slug == "send_email":
            result = await email_service.send_email(**arguments)
        elif tool_slug == "search_emails":
            result = await email_service.search_emails(**arguments)
        else:
            return {
                "request_id": request.get("request_id"),
                "successful": False,
                "error": f"Unknown tool: {tool_slug}"
            }
        
        results.append({
            "tool_slug": tool_slug,
            "response": result
        })
    
    return {
        "request_id": request.get("request_id"),
        "successful": True,
        "data": {"results": results}
    }
```

### Database Query MCP Server

```python
# mcp_servers/database_tools/main.py
from fastapi import FastAPI, HTTPException
import asyncpg
from typing import Dict, Any, List
import json
import logging

app = FastAPI(title="Database Tools MCP Server")
logger = logging.getLogger(__name__)

class DatabaseService:
    """Database query service"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=20
        )
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def execute_query(
        self,
        query: str,
        parameters: List[Any] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Execute database query"""
        
        # Validate query (basic safety checks)
        if not self._is_safe_query(query):
            raise ValueError("Query contains potentially unsafe operations")
        
        async with self.pool.acquire() as connection:
            try:
                # Add limit if not present
                if "LIMIT" not in query.upper():
                    query += f" LIMIT {limit}"
                
                # Execute query
                if parameters:
                    rows = await connection.fetch(query, *parameters)
                else:
                    rows = await connection.fetch(query)
                
                # Convert to list of dicts
                results = [dict(row) for row in rows]
                
                return {
                    "success": True,
                    "rows": results,
                    "count": len(results)
                }
                
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
    
    def _is_safe_query(self, query: str) -> bool:
        """Check if query is safe to execute"""
        # Disallow destructive operations
        unsafe_keywords = [
            "DROP", "DELETE", "TRUNCATE", "ALTER",
            "CREATE", "INSERT", "UPDATE"
        ]
        
        query_upper = query.upper()
        for keyword in unsafe_keywords:
            if keyword in query_upper:
                return False
        
        return True

# Tool definition
TOOLS = {
    "query_database": {
        "name": "query_database",
        "description": "Execute a read-only database query",
        "category": "data",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute (read-only)"
                },
                "parameters": {
                    "type": "array",
                    "description": "Query parameters",
                    "items": {"type": ["string", "number", "boolean", "null"]},
                    "default": []
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of rows to return",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100
                }
            },
            "required": ["query"]
        }
    }
}

# Initialize service
db_service = None

@app.on_event("startup")
async def startup():
    global db_service
    db_service = DatabaseService(os.getenv("DATABASE_URL"))
    await db_service.initialize()

@app.on_event("shutdown")
async def shutdown():
    if db_service:
        await db_service.close()

@app.get("/mcp/discover")
async def discover():
    """MCP discovery endpoint"""
    return {
        "server_name": "database_tools",
        "version": "1.0.0",
        "description": "Database query tools",
        "tools": list(TOOLS.values()),
        "capabilities": ["data", "query"]
    }

@app.post("/mcp/execute")
async def execute(request: Dict[str, Any]):
    """MCP execution endpoint"""
    tools_to_execute = request.get("tools", [])
    results = []
    
    for tool_request in tools_to_execute:
        tool_slug = tool_request.get("tool_slug")
        arguments = tool_request.get("arguments", {})
        
        if tool_slug == "query_database":
            result = await db_service.execute_query(**arguments)
        else:
            return {
                "request_id": request.get("request_id"),
                "successful": False,
                "error": f"Unknown tool: {tool_slug}"
            }
        
        results.append({
            "tool_slug": tool_slug,
            "response": result
        })
    
    return {
        "request_id": request.get("request_id"),
        "successful": True,
        "data": {"results": results}
    }
```

## Frontend Implementation

### Flask Frontend with WebSocket

```python
# frontend/app.py
from flask import Flask, render_template, request, session, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import httpx
import asyncio
import json
import logging
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

logger = logging.getLogger(__name__)

# Configuration
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

@app.route('/')
def index():
    """Render main chat interface"""
    return render_template('index.html')

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get user conversations"""
    user_token = session.get('user_token')
    if not user_token:
        return jsonify({"error": "Not authenticated"}), 401
    
    # Fetch from orchestrator
    response = httpx.get(
        f"{ORCHESTRATOR_URL}/api/v1/conversations",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    return jsonify(response.json())

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join_conversation')
def handle_join_conversation(data):
    """Join conversation room"""
    conversation_id = data.get('conversation_id')
    join_room(conversation_id)
    emit('joined', {'conversation_id': conversation_id}, room=conversation_id)

@socketio.on('send_message')
def handle_message(data):
    """Handle incoming message"""
    conversation_id = data.get('conversation_id')
    message = data.get('message')
    user_token = data.get('token')
    
    if not all([conversation_id, message, user_token]):
        emit('error', {'error': 'Missing required fields'})
        return
    
    # Start async task to process message
    socketio.start_background_task(
        process_message_async,
        conversation_id,
        message,
        user_token
    )

def process_message_async(conversation_id, message, user_token):
    """Process message asynchronously"""
    asyncio.run(stream_response(conversation_id, message, user_token))

async def stream_response(conversation_id, message, user_token):
    """Stream response from orchestrator"""
    async with httpx.AsyncClient() as client:
        try:
            # Send message to orchestrator
            async with client.stream(
                'POST',
                f"{ORCHESTRATOR_URL}/api/v1/conversations/{conversation_id}/messages",
                json={"content": message},
                headers={"Authorization": f"Bearer {user_token}"}
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        
                        # Emit to client based on event type
                        if data['type'] == 'token':
                            socketio.emit('token', data, room=conversation_id)
                        elif data['type'] == 'tool_request':
                            socketio.emit('tool_request', data, room=conversation_id)
                        elif data['type'] == 'tool_result':
                            socketio.emit('tool_result', data, room=conversation_id)
                        elif data['type'] == 'complete':
                            socketio.emit('complete', data, room=conversation_id)
                        elif data['type'] == 'error':
                            socketio.emit('error', data, room=conversation_id)
                            
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            socketio.emit('error', {'error': str(e)}, room=conversation_id)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
```

### Frontend JavaScript Client

```javascript
// frontend/static/js/chat.js
class ChatClient {
    constructor() {
        this.socket = null;
        this.conversationId = null;
        this.messageBuffer = '';
        this.isStreaming = false;
    }
    
    initialize() {
        // Connect to WebSocket
        this.socket = io({
            transports: ['websocket'],
            upgrade: false
        });
        
        // Set up event handlers
        this.setupEventHandlers();
        
        // Load conversations
        this.loadConversations();
    }
    
    setupEventHandlers() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus('connected');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus('disconnected');
        });
        
        // Message events
        this.socket.on('token', (data) => {
            this.handleToken(data);
        });
        
        this.socket.on('tool_request', (data) => {
            this.handleToolRequest(data);
        });
        
        this.socket.on('tool_result', (data) => {
            this.handleToolResult(data);
        });
        
        this.socket.on('complete', (data) => {
            this.handleComplete(data);
        });
        
        this.socket.on('error', (data) => {
            this.handleError(data);
        });
    }
    
    async loadConversations() {
        try {
            const response = await fetch('/api/conversations');
            const conversations = await response.json();
            this.displayConversations(conversations);
        } catch (error) {
            console.error('Failed to load conversations:', error);
        }
    }
    
    createConversation(title) {
        return fetch('/api/conversations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ title })
        })
        .then(response => response.json())
        .then(conversation => {
            this.conversationId = conversation.id;
            this.joinConversation(conversation.id);
            return conversation;
        });
    }
    
    joinConversation(conversationId) {
        this.conversationId = conversationId;
        this.socket.emit('join_conversation', {
            conversation_id: conversationId
        });
    }
    
    sendMessage(message) {
        if (!this.conversationId) {
            console.error('No conversation selected');
            return;
        }
        
        if (this.isStreaming) {
            console.warn('Already streaming a response');
            return;
        }
        
        // Display user message
        this.displayMessage('user', message);
        
        // Clear input
        document.getElementById('message-input').value = '';
        
        // Start streaming
        this.isStreaming = true;
        this.messageBuffer = '';
        
        // Create placeholder for assistant message
        const messageElement = this.createMessageElement('assistant', '');
        document.getElementById('messages').appendChild(messageElement);
        
        // Send message via WebSocket
        this.socket.emit('send_message', {
            conversation_id: this.conversationId,
            message: message,
            token: this.getAuthToken()
        });
    }
    
    handleToken(data) {
        // Append token to buffer
        this.messageBuffer += data.content;
        
        // Update message display
        const messageElement = document.querySelector('.message.assistant:last-child .content');
        if (messageElement) {
            messageElement.innerHTML = this.renderMarkdown(this.messageBuffer);
        }
    }
    
    handleToolRequest(data) {
        // Display tool request UI
        const toolCard = this.createToolCard('request', data);
        document.getElementById('messages').appendChild(toolCard);
    }
    
    handleToolResult(data) {
        // Update tool card with result
        const lastToolCard = document.querySelector('.tool-card:last-child');
        if (lastToolCard) {
            lastToolCard.classList.add('completed');
            const resultElement = lastToolCard.querySelector('.result');
            resultElement.textContent = JSON.stringify(data.result, null, 2);
        }
    }
    
    handleComplete(data) {
        // Mark streaming as complete
        this.isStreaming = false;
        
        // Add completion indicator
        const messageElement = document.querySelector('.message.assistant:last-child');
        if (messageElement) {
            messageElement.classList.add('complete');
        }
        
        // Save to conversation history
        this.saveToHistory(data.message_id, this.messageBuffer);
    }
    
    handleError(data) {
        console.error('Error:', data.error);
        this.isStreaming = false;
        
        // Display error message
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = `Error: ${data.error}`;
        document.getElementById('messages').appendChild(errorElement);
    }
    
    displayMessage(role, content) {
        const messageElement = this.createMessageElement(role, content);
        document.getElementById('messages').appendChild(messageElement);
        this.scrollToBottom();
    }
    
    createMessageElement(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar';
        avatarDiv.textContent = role === 'user' ? 'U' : 'A';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        contentDiv.innerHTML = this.renderMarkdown(content);
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        return messageDiv;
    }
    
    createToolCard(type, data) {
        const card = document.createElement('div');
        card.className = `tool-card ${type}`;
        
        const header = document.createElement('div');
        header.className = 'tool-header';
        header.innerHTML = `
            <span class="tool-icon">🔧</span>
            <span class="tool-name">${data.tool_id}</span>
            <span class="tool-status">${type === 'request' ? 'Executing...' : 'Complete'}</span>
        `;
        
        const body = document.createElement('div');
        body.className = 'tool-body';
        
        if (type === 'request') {
            body.innerHTML = `
                <div class="arguments">
                    <strong>Arguments:</strong>
                    <pre>${JSON.stringify(data.arguments, null, 2)}</pre>
                </div>
                <div class="result"></div>
            `;
        }
        
        card.appendChild(header);
        card.appendChild(body);
        
        return card;
    }
    
    renderMarkdown(text) {
        // Basic markdown rendering
        return marked.parse(text);
    }
    
    scrollToBottom() {
        const messagesContainer = document.getElementById('messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    getAuthToken() {
        // Get auth token from localStorage or session
        return localStorage.getItem('auth_token');
    }
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        statusElement.className = `status ${status}`;
        statusElement.textContent = status === 'connected' ? '🟢 Connected' : '🔴 Disconnected';
    }
}

// Initialize chat client
document.addEventListener('DOMContentLoaded', () => {
    const chatClient = new ChatClient();
    chatClient.initialize();
    
    // Set up event listeners
    document.getElementById('send-button').addEventListener('click', () => {
        const input = document.getElementById('message-input');
        if (input.value.trim()) {
            chatClient.sendMessage(input.value);
        }
    });
    
    document.getElementById('message-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const input = e.target;
            if (input.value.trim()) {
                chatClient.sendMessage(input.value);
            }
        }
    });
    
    document.getElementById('new-conversation').addEventListener('click', () => {
        const title = prompt('Enter conversation title:');
        if (title) {
            chatClient.createConversation(title);
        }
    });
});
```

## Tool Implementation Examples

### Web Search Tool

```python
# tools/web_search_tool.py
from typing import Dict, Any, List
import httpx
from bs4 import BeautifulSoup
import asyncio
import logging

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Web search tool implementation"""
    
    def __init__(self, search_api_key: str = None):
        self.search_api_key = search_api_key
        self.session = None
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def search(
        self,
        query: str,
        num_results: int = 5,
        search_type: str = "web"
    ) -> List[Dict[str, Any]]:
        """Perform web search"""
        
        if search_type == "web":
            return await self._web_search(query, num_results)
        elif search_type == "news":
            return await self._news_search(query, num_results)
        elif search_type == "images":
            return await self._image_search(query, num_results)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    async def _web_search(
        self,
        query: str,
        num_results: int
    ) -> List[Dict[str, Any]]:
        """Perform web search using search API"""
        
        # Using a search API (e.g., Serper, SerpAPI, or custom)
        if self.search_api_key:
            return await self._api_search(query, num_results)
        else:
            # Fallback to scraping (be respectful of robots.txt)
            return await self._scrape_search(query, num_results)
    
    async def _api_search(
        self,
        query: str,
        num_results: int
    ) -> List[Dict[str, Any]]:
        """Search using API"""
        
        url = "https://api.serper.dev/search"
        headers = {
            "X-API-KEY": self.search_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": num_results
        }
        
        try:
            response = await self.session.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("organic", [])[:num_results]:
                results.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "snippet": item.get("snippet"),
                    "date": item.get("date")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Search API error: {e}")
            return []
    
    async def fetch_page_content(
        self,
        url: str,
        max_length: int = 5000
    ) -> Dict[str, Any]:
        """Fetch and extract content from a web page"""
        
        try:
            response = await self.session.get(
                url,
                follow_redirects=True,
                timeout=10.0
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Truncate if needed
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            # Extract metadata
            title = soup.find('title')
            title = title.string if title else ""
            
            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description.get('content', '') if meta_description else ""
            
            return {
                "url": url,
                "title": title,
                "description": description,
                "content": text,
                "word_count": len(text.split())
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return {
                "url": url,
                "error": str(e)
            }
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for MCP"""
        return {
            "name": "web_search",
            "description": "Search the web and fetch page content",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "fetch"],
                        "description": "Action to perform"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for search action)"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to fetch (for fetch action)"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results",
                        "default": 5
                    }
                },
                "required": ["action"]
            }
        }
```

## Streaming Response Handling

### Server-Sent Events (SSE) Implementation

```python
# orchestrator/streaming/sse.py
from fastapi import Response
from typing import AsyncGenerator, Dict, Any
import json
import asyncio

class SSEResponse(Response):
    """Server-Sent Events response"""
    
    def __init__(self, generator: AsyncGenerator[str, None]):
        super().__init__(
            content=self.generate(generator),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable Nginx buffering
            }
        )
    
    async def generate(self, generator: AsyncGenerator[str, None]):
        """Generate SSE stream"""
        try:
            async for data in generator:
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            yield "event: close\ndata: {}\n\n"

async def stream_chat_completion(
    messages: List[Dict[str, Any]],
    model: str,
    openai_client
) -> AsyncGenerator[str, None]:
    """Stream chat completion as SSE"""
    
    try:
        stream = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        async for chunk in stream:
            delta = chunk.choices[0].delta
            
            if delta.content:
                event = {
                    "type": "token",
                    "content": delta.content,
                    "timestamp": datetime.utcnow().isoformat()
                }
                yield json.dumps(event)
            
            if chunk.choices[0].finish_reason:
                event = {
                    "type": "complete",
                    "finish_reason": chunk.choices[0].finish_reason,
                    "timestamp": datetime.utcnow().isoformat()
                }
                yield json.dumps(event)
                
    except Exception as e:
        event = {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        yield json.dumps(event)

# Usage in FastAPI endpoint
@app.post("/api/v1/chat/stream")
async def stream_chat(request: ChatRequest):
    """Stream chat completion"""
    
    generator = stream_chat_completion(
        request.messages,
        request.model,
        openai_client
    )
    
    return SSEResponse(generator)
```

## Production Best Practices

### Error Handling and Logging

```python
# best_practices/error_handling.py
import logging
import traceback
from functools import wraps
from typing import Callable
import sentry_sdk

# Configure structured logging
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def handle_errors(func: Callable):
    """Decorator for comprehensive error handling"""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request_id = kwargs.get('request_id', 'unknown')
        
        try:
            # Log function entry
            logger.info(
                "function_called",
                function=func.__name__,
                request_id=request_id,
                args=str(args)[:100],  # Truncate for safety
                kwargs=str(kwargs)[:100]
            )
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Log success
            logger.info(
                "function_completed",
                function=func.__name__,
                request_id=request_id
            )
            
            return result
            
        except ValidationError as e:
            # Client error - don't log full stack trace
            logger.warning(
                "validation_error",
                function=func.__name__,
                request_id=request_id,
                error=str(e)
            )
            raise
            
        except RateLimitError as e:
            # Rate limit - expected behavior
            logger.info(
                "rate_limit_exceeded",
                function=func.__name__,
                request_id=request_id,
                error=str(e)
            )
            raise
            
        except ExternalServiceError as e:
            # External service failure
            logger.error(
                "external_service_error",
                function=func.__name__,
                request_id=request_id,
                service=e.service,
                error=str(e),
                traceback=traceback.format_exc()
            )
            
            # Report to Sentry
            sentry_sdk.capture_exception(e)
            
            # Return degraded response if possible
            if hasattr(e, 'fallback_response'):
                return e.fallback_response
            raise
            
        except Exception as e:
            # Unexpected error
            logger.error(
                "unexpected_error",
                function=func.__name__,
                request_id=request_id,
                error=str(e),
                traceback=traceback.format_exc()
            )
            
            # Report to Sentry
            sentry_sdk.capture_exception(e)
            
            raise
    
    return wrapper
```

### Performance Monitoring

```python
# best_practices/performance.py
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
import asyncio

# Define metrics
request_count = Counter(
    'app_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'app_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

active_requests = Gauge(
    'app_active_requests',
    'Number of active requests'
)

token_usage = Counter(
    'openai_tokens_total',
    'Total OpenAI tokens used',
    ['model', 'type']
)

def track_performance(endpoint: str):
    """Decorator to track performance metrics"""
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Track active requests
            active_requests.inc()
            
            # Start timer
            start_time = time.time()
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Track success
                request_count.labels(
                    method=kwargs.get('method', 'unknown'),
                    endpoint=endpoint,
                    status='success'
                ).inc()
                
                return result
                
            except Exception as e:
                # Track error
                request_count.labels(
                    method=kwargs.get('method', 'unknown'),
                    endpoint=endpoint,
                    status='error'
                ).inc()
                
                raise
                
            finally:
                # Track duration
                duration = time.time() - start_time
                request_duration.labels(
                    method=kwargs.get('method', 'unknown'),
                    endpoint=endpoint
                ).observe(duration)
                
                # Update active requests
                active_requests.dec()
        
        return wrapper
    return decorator

class PerformanceMonitor:
    """Performance monitoring service"""
    
    def __init__(self):
        self.slow_query_threshold = 1.0  # seconds
        self.memory_threshold = 500 * 1024 * 1024  # 500MB
        
    async def monitor_query(self, query: str, func: Callable):
        """Monitor database query performance"""
        start_time = time.time()
        
        try:
            result = await func()
            duration = time.time() - start_time
            
            if duration > self.slow_query_threshold:
                logger.warning(
                    "slow_query_detected",
                    query=query[:100],
                    duration=duration
                )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "query_failed",
                query=query[:100],
                duration=duration,
                error=str(e)
            )
            raise
    
    def check_memory_usage(self):
        """Check memory usage"""
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        if memory_info.rss > self.memory_threshold:
            logger.warning(
                "high_memory_usage",
                rss=memory_info.rss,
                vms=memory_info.vms,
                threshold=self.memory_threshold
            )
            
            # Trigger garbage collection
            import gc
            gc.collect()
```

## Common Patterns

### Repository Pattern for Data Access

```python
# patterns/repository.py
from typing import Generic, TypeVar, List, Optional, Dict, Any
from abc import ABC, abstractmethod
import uuid
from datetime import datetime

T = TypeVar('T')

class Repository(ABC, Generic[T]):
    """Abstract repository pattern"""
    
    @abstractmethod
    async def get(self, id: str) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def list(
        self,
        filters: Dict[str, Any] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[T]:
        """List entities with filters"""
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create new entity"""
        pass
    
    @abstractmethod
    async def update(self, id: str, entity: T) -> T:
        """Update entity"""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity"""
        pass

class ConversationRepository(Repository[Conversation]):
    """Conversation repository implementation"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def get(self, id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        query = "SELECT * FROM conversations WHERE id = $1"
        row = await self.db.fetchrow(query, id)
        
        if row:
            return Conversation(**dict(row))
        return None
    
    async def list(
        self,
        filters: Dict[str, Any] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Conversation]:
        """List conversations"""
        query = "SELECT * FROM conversations WHERE 1=1"
        params = []
        
        if filters:
            if 'user_id' in filters:
                query += f" AND user_id = ${len(params) + 1}"
                params.append(filters['user_id'])
            
            if 'created_after' in filters:
                query += f" AND created_at > ${len(params) + 1}"
                params.append(filters['created_after'])
        
        query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
        params.extend([limit, offset])
        
        rows = await self.db.fetch(query, *params)
        return [Conversation(**dict(row)) for row in rows]
    
    async def create(self, entity: Conversation) -> Conversation:
        """Create conversation"""
        entity.id = str(uuid.uuid4())
        entity.created_at = datetime.utcnow()
        
        query = """
            INSERT INTO conversations (id, user_id, title, model, system_prompt, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
        """
        
        row = await self.db.fetchrow(
            query,
            entity.id,
            entity.user_id,
            entity.title,
            entity.model,
            entity.system_prompt,
            entity.created_at
        )
        
        return Conversation(**dict(row))
    
    async def update(self, id: str, entity: Conversation) -> Conversation:
        """Update conversation"""
        entity.updated_at = datetime.utcnow()
        
        query = """
            UPDATE conversations
            SET title = $2, model = $3, system_prompt = $4, updated_at = $5
            WHERE id = $1
            RETURNING *
        """
        
        row = await self.db.fetchrow(
            query,
            id,
            entity.title,
            entity.model,
            entity.system_prompt,
            entity.updated_at
        )
        
        if row:
            return Conversation(**dict(row))
        raise ValueError(f"Conversation {id} not found")
    
    async def delete(self, id: str) -> bool:
        """Delete conversation"""
        query = "DELETE FROM conversations WHERE id = $1"
        result = await self.db.execute(query, id)
        return result == "DELETE 1"
```

## Troubleshooting Guide

### Common Issues and Solutions

```python
# troubleshooting/common_issues.py

class TroubleshootingGuide:
    """Common issues and their solutions"""
    
    ISSUES = {
        "connection_refused": {
            "symptoms": [
                "Connection refused errors",
                "Cannot connect to service"
            ],
            "causes": [
                "Service not running",
                "Firewall blocking connection",
                "Wrong host/port configuration"
            ],
            "solutions": [
                "Check if service is running: docker ps",
                "Verify port configuration in .env",
                "Check firewall rules",
                "Ensure services are in same network"
            ]
        },
        "rate_limit_exceeded": {
            "symptoms": [
                "429 status codes",
                "Rate limit error messages"
            ],
            "causes": [
                "Too many requests",
                "Burst traffic",
                "Misconfigured rate limits"
            ],
            "solutions": [
                "Implement exponential backoff",
                "Use request queuing",
                "Adjust rate limit configuration",
                "Add caching layer"
            ]
        },
        "memory_leak": {
            "symptoms": [
                "Increasing memory usage",
                "OOM kills",
                "Slow performance over time"
            ],
            "causes": [
                "Unclosed connections",
                "Large objects in memory",
                "Circular references"
            ],
            "solutions": [
                "Profile memory usage",
                "Implement connection pooling",
                "Use weak references",
                "Regular garbage collection"
            ]
        },
        "slow_responses": {
            "symptoms": [
                "High latency",
                "Timeouts",
                "Poor user experience"
            ],
            "causes": [
                "Unoptimized queries",
                "Missing indexes",
                "Network latency",
                "Cold starts"
            ],
            "solutions": [
                "Add database indexes",
                "Implement caching",
                "Use connection pooling",
                "Optimize queries",
                "Pre-warm services"
            ]
        }
    }
    
    @classmethod
    def diagnose(cls, symptom: str) -> Dict[str, Any]:
        """Diagnose issue based on symptom"""
        matches = []
        
        for issue_key, issue_data in cls.ISSUES.items():
            for issue_symptom in issue_data["symptoms"]:
                if symptom.lower() in issue_symptom.lower():
                    matches.append({
                        "issue": issue_key,
                        "data": issue_data
                    })
        
        return matches
    
    @classmethod
    def get_solution(cls, issue: str) -> Dict[str, Any]:
        """Get solution for specific issue"""
        return cls.ISSUES.get(issue, {})

# Health check implementation
async def diagnose_system_health():
    """Comprehensive system health diagnosis"""
    
    health_checks = {
        "database": check_database_health(),
        "redis": check_redis_health(),
        "openai": check_openai_health(),
        "mcp_servers": check_mcp_servers_health(),
        "memory": check_memory_health(),
        "disk": check_disk_health()
    }
    
    results = {}
    for name, check in health_checks.items():
        try:
            result = await check
            results[name] = {
                "status": "healthy" if result else "unhealthy",
                "details": result
            }
        except Exception as e:
            results[name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Determine overall health
    overall_status = "healthy"
    for check_result in results.values():
        if check_result["status"] == "unhealthy":
            overall_status = "unhealthy"
            break
    
    return {
        "status": overall_status,
        "checks": results,
        "timestamp": datetime.utcnow().isoformat()
    }
```

---

**Document Version**: 1.0.0  
**Last Updated**: December 2024  
**Code Standards**: PEP 8, ES6+, Production-ready