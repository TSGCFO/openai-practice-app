"""
Configuration management for the Orchestrator service
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = Field(default="AI Assistant Orchestrator", env="APP_NAME")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_EXPIRATION_MINUTES: int = Field(default=60, env="JWT_EXPIRATION_MINUTES")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:5000", "http://localhost:3000"],
        env="ALLOWED_ORIGINS"
    )
    
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=40, env="DATABASE_MAX_OVERFLOW")
    
    # Redis
    REDIS_URL: str = Field(..., env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_SSL: bool = Field(default=False, env="REDIS_SSL")
    
    # OpenAI
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4", env="OPENAI_MODEL")
    OPENAI_MAX_TOKENS: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    OPENAI_TEMPERATURE: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    OPENAI_STREAM: bool = Field(default=True, env="OPENAI_STREAM")
    
    # MCP Servers
    MCP_SERVERS: List[str] = Field(
        default=[
            "http://mcp-email:8001",
            "http://mcp-database:8002"
        ],
        env="MCP_DISCOVERY_URLS"
    )
    MCP_SERVICE_TOKEN: Optional[str] = Field(default=None, env="MCP_SERVICE_TOKEN")
    MCP_TIMEOUT: int = Field(default=30, env="MCP_TIMEOUT")
    MCP_RETRY_COUNT: int = Field(default=3, env="MCP_RETRY_COUNT")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    # Session Management
    SESSION_TTL: int = Field(default=3600, env="SESSION_TTL")  # 1 hour
    SESSION_MAX_TTL: int = Field(default=86400, env="SESSION_MAX_TTL")  # 24 hours
    
    # Context Management
    MAX_CONTEXT_LENGTH: int = Field(default=10, env="MAX_CONTEXT_LENGTH")
    MAX_TOKEN_CONTEXT: int = Field(default=4000, env="MAX_TOKEN_CONTEXT")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    JAEGER_AGENT_HOST: Optional[str] = Field(default=None, env="JAEGER_AGENT_HOST")
    JAEGER_AGENT_PORT: Optional[int] = Field(default=None, env="JAEGER_AGENT_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT.lower() == "development"
    
    def get_database_url(self, async_mode: bool = True) -> str:
        """Get database URL with async support"""
        if async_mode and self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        return self.DATABASE_URL
    
    def get_redis_url(self) -> str:
        """Get Redis URL with password if provided"""
        if self.REDIS_PASSWORD:
            # Parse and add password to URL
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.REDIS_URL)
            netloc = f":{self.REDIS_PASSWORD}@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return urlunparse((
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
        return self.REDIS_URL

# Create settings instance
settings = Settings()

# Export common configurations
OPENAI_CONFIG = {
    "api_key": settings.OPENAI_API_KEY,
    "model": settings.OPENAI_MODEL,
    "max_tokens": settings.OPENAI_MAX_TOKENS,
    "temperature": settings.OPENAI_TEMPERATURE,
    "stream": settings.OPENAI_STREAM
}

REDIS_CONFIG = {
    "url": settings.get_redis_url(),
    "ssl": settings.REDIS_SSL,
    "decode_responses": True
}

DATABASE_CONFIG = {
    "url": settings.get_database_url(),
    "pool_size": settings.DATABASE_POOL_SIZE,
    "max_overflow": settings.DATABASE_MAX_OVERFLOW,
    "echo": settings.DEBUG
}