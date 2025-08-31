"""
AI Assistant Orchestrator Service
FastAPI application for managing OpenAI interactions and MCP tool execution
"""

from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
import os
from dotenv import load_dotenv

from .config import settings
from .api import conversations, messages, tools, health
from .services.session_manager import SessionManager
from .services.mcp_manager import MCPManager
from .services.openai_service import OpenAIService
from .database.connection import init_db, close_db
from .middleware import AuthMiddleware, ErrorHandlerMiddleware, RateLimitMiddleware

# Load environment variables
load_dotenv()

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
    
    # Initialize services
    app.state.session_manager = SessionManager()
    app.state.mcp_manager = MCPManager(settings.MCP_SERVERS)
    app.state.openai_service = OpenAIService(api_key=settings.OPENAI_API_KEY)
    
    # Discover MCP tools
    await app.state.mcp_manager.discover_tools()
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Assistant Orchestrator")
    await close_db()

# Create FastAPI app
app = FastAPI(
    title="AI Assistant Orchestrator",
    version="1.0.0",
    description="Orchestrator service for AI Assistant system",
    docs_url="/docs",
    redoc_url="/redoc",
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

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(conversations.router, prefix="/api/v1/conversations", tags=["conversations"])
app.include_router(messages.router, prefix="/api/v1/messages", tags=["messages"])
app.include_router(tools.router, prefix="/api/v1/tools", tags=["tools"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Assistant Orchestrator",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    await websocket.accept()
    session_id = None
    
    try:
        # Authenticate WebSocket connection
        auth_message = await websocket.receive_json()
        if auth_message.get("type") != "auth":
            await websocket.send_json({"type": "error", "message": "Authentication required"})
            await websocket.close()
            return
        
        # Validate token and get session
        token = auth_message.get("token")
        session = await app.state.session_manager.validate_token(token)
        if not session:
            await websocket.send_json({"type": "error", "message": "Invalid token"})
            await websocket.close()
            return
        
        session_id = session.id
        await websocket.send_json({"type": "auth_success", "session_id": session_id})
        
        # Handle messages
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                # Process message through OpenAI and stream response
                conversation_id = data.get("conversation_id")
                content = data.get("content")
                
                async for chunk in app.state.openai_service.stream_completion(
                    conversation_id=conversation_id,
                    content=content,
                    session=session,
                    mcp_manager=app.state.mcp_manager
                ):
                    await websocket.send_json(chunk)
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif data.get("type") == "close":
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if session_id:
            await app.state.session_manager.cleanup_session(session_id)
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )