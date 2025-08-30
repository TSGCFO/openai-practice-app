# State Management & Error Handling Documentation

## Table of Contents

1. [Overview](#overview)
2. [Session State Management](#session-state-management)
3. [Conversation Context Management](#conversation-context-management)
4. [Distributed State with Redis](#distributed-state-with-redis)
5. [Vector Database Integration](#vector-database-integration)
6. [Error Handling Strategies](#error-handling-strategies)
7. [Retry Mechanisms](#retry-mechanisms)
8. [Circuit Breaker Pattern](#circuit-breaker-pattern)
9. [Graceful Degradation](#graceful-degradation)
10. [Monitoring & Alerting](#monitoring--alerting)

## Overview

This document outlines the state management and error handling strategies for the AI Assistant system. It covers session management, conversation context, distributed state, error recovery, and resilience patterns.

### State Management Principles

1. **Stateless Services**: Keep services stateless for scalability
2. **Externalized State**: Store state in dedicated systems (Redis, PostgreSQL)
3. **Eventual Consistency**: Accept eventual consistency for performance
4. **Idempotency**: Ensure operations are idempotent
5. **Fault Tolerance**: Design for failure and recovery

### Error Handling Principles

1. **Fail Fast**: Detect and report errors quickly
2. **Graceful Degradation**: Provide partial functionality when possible
3. **Clear Error Messages**: Provide actionable error information
4. **Automatic Recovery**: Implement self-healing mechanisms
5. **Comprehensive Logging**: Log all errors with context

## Session State Management

### Session Store Architecture

```python
# state/session_manager.py
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import uuid
import redis
from dataclasses import dataclass, asdict

@dataclass
class Session:
    """User session data model"""
    id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    data: Dict[str, Any]
    active_conversation_id: Optional[str] = None
    
class SessionManager:
    """Manages user sessions with Redis backend"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
        self.max_ttl = 86400  # 24 hours
        
    async def create_session(
        self,
        user_id: str,
        initial_data: Dict[str, Any] = None
    ) -> Session:
        """Create new session"""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        session = Session(
            id=session_id,
            user_id=user_id,
            created_at=now,
            last_accessed=now,
            expires_at=now + timedelta(seconds=self.default_ttl),
            data=initial_data or {}
        )
        
        # Store in Redis
        await self._save_session(session)
        
        # Add to user's session index
        await self._add_to_user_sessions(user_id, session_id)
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session by ID"""
        key = f"session:{session_id}"
        data = await self.redis.get(key)
        
        if not data:
            return None
        
        session_dict = json.loads(data)
        session = self._deserialize_session(session_dict)
        
        # Update last accessed time
        session.last_accessed = datetime.utcnow()
        await self._save_session(session)
        
        return session
    
    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update session data"""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        # Update data
        session.data.update(updates)
        session.last_accessed = datetime.utcnow()
        
        # Save
        await self._save_session(session)
        return True
    
    async def extend_session(
        self,
        session_id: str,
        additional_seconds: int = None
    ) -> bool:
        """Extend session TTL"""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        additional = additional_seconds or self.default_ttl
        new_expiry = datetime.utcnow() + timedelta(seconds=additional)
        
        # Don't exceed max TTL
        max_expiry = session.created_at + timedelta(seconds=self.max_ttl)
        session.expires_at = min(new_expiry, max_expiry)
        
        await self._save_session(session)
        return True
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        # Remove from Redis
        key = f"session:{session_id}"
        await self.redis.delete(key)
        
        # Remove from user's session index
        await self._remove_from_user_sessions(session.user_id, session_id)
        
        return True
    
    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user"""
        key = f"user_sessions:{user_id}"
        session_ids = await self.redis.smembers(key)
        
        sessions = []
        for session_id in session_ids:
            session = await self.get_session(session_id.decode())
            if session:
                sessions.append(session)
            else:
                # Clean up stale reference
                await self.redis.srem(key, session_id)
        
        return sessions
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        # This should run periodically
        pattern = "session:*"
        cursor = 0
        
        while True:
            cursor, keys = await self.redis.scan(
                cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                data = await self.redis.get(key)
                if data:
                    session_dict = json.loads(data)
                    expires_at = datetime.fromisoformat(session_dict['expires_at'])
                    
                    if expires_at < datetime.utcnow():
                        session_id = key.decode().split(':')[1]
                        await self.delete_session(session_id)
            
            if cursor == 0:
                break
    
    async def _save_session(self, session: Session):
        """Save session to Redis"""
        key = f"session:{session.id}"
        ttl = int((session.expires_at - datetime.utcnow()).total_seconds())
        
        if ttl > 0:
            session_dict = self._serialize_session(session)
            await self.redis.setex(
                key,
                ttl,
                json.dumps(session_dict)
            )
    
    def _serialize_session(self, session: Session) -> Dict[str, Any]:
        """Serialize session to dictionary"""
        return {
            'id': session.id,
            'user_id': session.user_id,
            'created_at': session.created_at.isoformat(),
            'last_accessed': session.last_accessed.isoformat(),
            'expires_at': session.expires_at.isoformat(),
            'data': session.data,
            'active_conversation_id': session.active_conversation_id
        }
    
    def _deserialize_session(self, data: Dict[str, Any]) -> Session:
        """Deserialize session from dictionary"""
        return Session(
            id=data['id'],
            user_id=data['user_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            data=data['data'],
            active_conversation_id=data.get('active_conversation_id')
        )
    
    async def _add_to_user_sessions(self, user_id: str, session_id: str):
        """Add session to user's session index"""
        key = f"user_sessions:{user_id}"
        await self.redis.sadd(key, session_id)
        await self.redis.expire(key, self.max_ttl)
    
    async def _remove_from_user_sessions(self, user_id: str, session_id: str):
        """Remove session from user's session index"""
        key = f"user_sessions:{user_id}"
        await self.redis.srem(key, session_id)
```

## Conversation Context Management

### Context Manager Implementation

```python
# state/context_manager.py
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class ConversationContext:
    """Manages conversation context and history"""
    
    def __init__(self, redis_client, max_context_length: int = 10):
        self.redis = redis_client
        self.max_context_length = max_context_length
        
    async def initialize_context(
        self,
        conversation_id: str,
        system_prompt: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Initialize new conversation context"""
        context = {
            'conversation_id': conversation_id,
            'created_at': datetime.utcnow().isoformat(),
            'messages': [],
            'system_prompt': system_prompt,
            'metadata': metadata or {},
            'token_count': 0,
            'turn_count': 0
        }
        
        await self._save_context(conversation_id, context)
        return context
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: List[Dict] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Add message to conversation context"""
        context = await self.get_context(conversation_id)
        if not context:
            context = await self.initialize_context(conversation_id)
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow().isoformat(),
            'tool_calls': tool_calls,
            'metadata': metadata or {}
        }
        
        # Add to messages
        context['messages'].append(message)
        context['turn_count'] += 1
        
        # Update token count (approximate)
        context['token_count'] += self._estimate_tokens(content)
        
        # Trim context if needed
        await self._trim_context(context)
        
        # Save
        await self._save_context(conversation_id, context)
        
        return message
    
    async def get_context(
        self,
        conversation_id: str,
        include_system: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get conversation context"""
        key = f"context:{conversation_id}"
        data = await self.redis.get(key)
        
        if not data:
            return None
        
        context = json.loads(data)
        
        if not include_system and context.get('system_prompt'):
            # Remove system prompt if not needed
            context = context.copy()
            del context['system_prompt']
        
        return context
    
    async def get_messages_for_prompt(
        self,
        conversation_id: str,
        max_tokens: int = 4000
    ) -> List[Dict[str, Any]]:
        """Get messages formatted for LLM prompt"""
        context = await self.get_context(conversation_id)
        if not context:
            return []
        
        messages = []
        
        # Add system prompt if exists
        if context.get('system_prompt'):
            messages.append({
                'role': 'system',
                'content': context['system_prompt']
            })
        
        # Add conversation messages
        total_tokens = self._estimate_tokens(context.get('system_prompt', ''))
        
        # Add messages in reverse order (most recent first)
        for message in reversed(context['messages']):
            message_tokens = self._estimate_tokens(message['content'])
            
            if total_tokens + message_tokens > max_tokens:
                break
            
            messages.insert(1 if context.get('system_prompt') else 0, {
                'role': message['role'],
                'content': message['content']
            })
            
            total_tokens += message_tokens
        
        return messages
    
    async def summarize_context(
        self,
        conversation_id: str,
        summary: str
    ):
        """Summarize and compress conversation context"""
        context = await self.get_context(conversation_id)
        if not context:
            return
        
        # Store original messages in archive
        archive_key = f"context_archive:{conversation_id}:{datetime.utcnow().isoformat()}"
        await self.redis.setex(
            archive_key,
            86400 * 7,  # Keep for 7 days
            json.dumps(context['messages'])
        )
        
        # Replace messages with summary
        context['messages'] = [{
            'role': 'system',
            'content': f"Previous conversation summary: {summary}",
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': {'type': 'summary'}
        }]
        
        context['token_count'] = self._estimate_tokens(summary)
        
        await self._save_context(conversation_id, context)
    
    async def _save_context(self, conversation_id: str, context: Dict[str, Any]):
        """Save context to Redis"""
        key = f"context:{conversation_id}"
        await self.redis.setex(
            key,
            86400,  # 24 hours
            json.dumps(context)
        )
    
    async def _trim_context(self, context: Dict[str, Any]):
        """Trim context to maximum length"""
        if len(context['messages']) > self.max_context_length:
            # Keep only the most recent messages
            context['messages'] = context['messages'][-self.max_context_length:]
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: 1 token per 4 characters
        return len(text) // 4

class ContextWindowManager:
    """Manages sliding context window for conversations"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.reserved_tokens = 500  # Reserve for system prompt
        
    def build_prompt(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = None,
        tools: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Build prompt with context window management"""
        prompt_messages = []
        total_tokens = 0
        
        # Add system prompt
        if system_prompt:
            prompt_messages.append({
                'role': 'system',
                'content': system_prompt
            })
            total_tokens += self._estimate_tokens(system_prompt)
        
        # Add tool descriptions if provided
        if tools:
            tools_description = self._format_tools(tools)
            if prompt_messages and prompt_messages[0]['role'] == 'system':
                prompt_messages[0]['content'] += f"\n\n{tools_description}"
            else:
                prompt_messages.insert(0, {
                    'role': 'system',
                    'content': tools_description
                })
            total_tokens += self._estimate_tokens(tools_description)
        
        # Add messages within token limit
        available_tokens = self.max_tokens - total_tokens - self.reserved_tokens
        
        for message in reversed(messages):
            message_tokens = self._estimate_tokens(message['content'])
            
            if message_tokens > available_tokens:
                # Add truncated message
                truncated_content = self._truncate_message(
                    message['content'],
                    available_tokens
                )
                prompt_messages.append({
                    'role': message['role'],
                    'content': truncated_content
                })
                break
            
            prompt_messages.append(message)
            available_tokens -= message_tokens
        
        return prompt_messages
    
    def _format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format tool descriptions for prompt"""
        tool_descriptions = []
        for tool in tools:
            desc = f"- {tool['name']}: {tool['description']}"
            if 'parameters' in tool:
                desc += f" Parameters: {json.dumps(tool['parameters'])}"
            tool_descriptions.append(desc)
        
        return "Available tools:\n" + "\n".join(tool_descriptions)
    
    def _truncate_message(self, content: str, max_tokens: int) -> str:
        """Truncate message to fit token limit"""
        # Rough truncation based on character count
        max_chars = max_tokens * 4
        if len(content) > max_chars:
            return content[:max_chars] + "... [truncated]"
        return content
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text) // 4
```

## Distributed State with Redis

### Redis State Store

```python
# state/redis_store.py
import redis.asyncio as redis
from typing import Any, Dict, List, Optional, Set
import json
import pickle
from datetime import timedelta

class RedisStateStore:
    """Distributed state store using Redis"""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        
    # Key-Value Operations
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = "state"
    ) -> bool:
        """Set value with optional TTL"""
        full_key = f"{namespace}:{key}"
        serialized = self._serialize(value)
        
        if ttl:
            return await self.redis.setex(full_key, ttl, serialized)
        return await self.redis.set(full_key, serialized)
    
    async def get(
        self,
        key: str,
        namespace: str = "state"
    ) -> Optional[Any]:
        """Get value by key"""
        full_key = f"{namespace}:{key}"
        value = await self.redis.get(full_key)
        
        if value:
            return self._deserialize(value)
        return None
    
    async def delete(
        self,
        key: str,
        namespace: str = "state"
    ) -> bool:
        """Delete key"""
        full_key = f"{namespace}:{key}"
        return await self.redis.delete(full_key) > 0
    
    async def exists(
        self,
        key: str,
        namespace: str = "state"
    ) -> bool:
        """Check if key exists"""
        full_key = f"{namespace}:{key}"
        return await self.redis.exists(full_key) > 0
    
    # Hash Operations
    
    async def hset(
        self,
        key: str,
        field: str,
        value: Any,
        namespace: str = "state"
    ) -> bool:
        """Set hash field"""
        full_key = f"{namespace}:{key}"
        serialized = self._serialize(value)
        return await self.redis.hset(full_key, field, serialized)
    
    async def hget(
        self,
        key: str,
        field: str,
        namespace: str = "state"
    ) -> Optional[Any]:
        """Get hash field"""
        full_key = f"{namespace}:{key}"
        value = await self.redis.hget(full_key, field)
        
        if value:
            return self._deserialize(value)
        return None
    
    async def hgetall(
        self,
        key: str,
        namespace: str = "state"
    ) -> Dict[str, Any]:
        """Get all hash fields"""
        full_key = f"{namespace}:{key}"
        data = await self.redis.hgetall(full_key)
        
        return {
            field: self._deserialize(value)
            for field, value in data.items()
        }
    
    # List Operations
    
    async def lpush(
        self,
        key: str,
        value: Any,
        namespace: str = "state"
    ) -> int:
        """Push to list head"""
        full_key = f"{namespace}:{key}"
        serialized = self._serialize(value)
        return await self.redis.lpush(full_key, serialized)
    
    async def rpush(
        self,
        key: str,
        value: Any,
        namespace: str = "state"
    ) -> int:
        """Push to list tail"""
        full_key = f"{namespace}:{key}"
        serialized = self._serialize(value)
        return await self.redis.rpush(full_key, serialized)
    
    async def lrange(
        self,
        key: str,
        start: int,
        stop: int,
        namespace: str = "state"
    ) -> List[Any]:
        """Get list range"""
        full_key = f"{namespace}:{key}"
        values = await self.redis.lrange(full_key, start, stop)
        
        return [self._deserialize(value) for value in values]
    
    # Set Operations
    
    async def sadd(
        self,
        key: str,
        value: Any,
        namespace: str = "state"
    ) -> int:
        """Add to set"""
        full_key = f"{namespace}:{key}"
        serialized = self._serialize(value)
        return await self.redis.sadd(full_key, serialized)
    
    async def smembers(
        self,
        key: str,
        namespace: str = "state"
    ) -> Set[Any]:
        """Get set members"""
        full_key = f"{namespace}:{key}"
        values = await self.redis.smembers(full_key)
        
        return {self._deserialize(value) for value in values}
    
    # Atomic Operations
    
    async def increment(
        self,
        key: str,
        amount: int = 1,
        namespace: str = "state"
    ) -> int:
        """Atomic increment"""
        full_key = f"{namespace}:{key}"
        return await self.redis.incrby(full_key, amount)
    
    async def decrement(
        self,
        key: str,
        amount: int = 1,
        namespace: str = "state"
    ) -> int:
        """Atomic decrement"""
        full_key = f"{namespace}:{key}"
        return await self.redis.decrby(full_key, amount)
    
    # Distributed Locking
    
    async def acquire_lock(
        self,
        resource: str,
        timeout: int = 10,
        namespace: str = "lock"
    ) -> Optional[str]:
        """Acquire distributed lock"""
        import uuid
        
        lock_key = f"{namespace}:{resource}"
        lock_value = str(uuid.uuid4())
        
        acquired = await self.redis.set(
            lock_key,
            lock_value,
            nx=True,
            ex=timeout
        )
        
        if acquired:
            return lock_value
        return None
    
    async def release_lock(
        self,
        resource: str,
        lock_value: str,
        namespace: str = "lock"
    ) -> bool:
        """Release distributed lock"""
        lock_key = f"{namespace}:{resource}"
        
        # Use Lua script for atomic check and delete
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        result = await self.redis.eval(
            lua_script,
            1,
            lock_key,
            lock_value
        )
        
        return result == 1
    
    # Pub/Sub Operations
    
    async def publish(
        self,
        channel: str,
        message: Any
    ) -> int:
        """Publish message to channel"""
        serialized = self._serialize(message)
        return await self.redis.publish(channel, serialized)
    
    async def subscribe(
        self,
        channels: List[str]
    ):
        """Subscribe to channels"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(*channels)
        return pubsub
    
    # Serialization
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage"""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value)
        return json.dumps(value, default=str)
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage"""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
```

## Vector Database Integration

### Vector Store for Long-term Memory

```python
# state/vector_store.py
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import hashlib

class VectorMemoryStore:
    """Vector database for long-term memory and RAG"""
    
    def __init__(self, vector_client, embedding_service):
        self.vector_client = vector_client
        self.embedding_service = embedding_service
        self.collection_name = "conversation_memory"
        
    async def store_memory(
        self,
        user_id: str,
        conversation_id: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store memory in vector database"""
        # Generate embedding
        embedding = await self.embedding_service.generate_embedding(content)
        
        # Create memory ID
        memory_id = self._generate_memory_id(user_id, conversation_id, content)
        
        # Prepare metadata
        full_metadata = {
            'user_id': user_id,
            'conversation_id': conversation_id,
            'timestamp': datetime.utcnow().isoformat(),
            'content_hash': hashlib.md5(content.encode()).hexdigest(),
            **(metadata or {})
        }
        
        # Store in vector database
        await self.vector_client.upsert(
            collection=self.collection_name,
            points=[{
                'id': memory_id,
                'vector': embedding,
                'payload': {
                    'content': content,
                    'metadata': full_metadata
                }
            }]
        )
        
        return memory_id
    
    async def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search memories using semantic similarity"""
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query)
        
        # Build filter
        filter_conditions = []
        if user_id:
            filter_conditions.append({
                'key': 'metadata.user_id',
                'match': {'value': user_id}
            })
        if conversation_id:
            filter_conditions.append({
                'key': 'metadata.conversation_id',
                'match': {'value': conversation_id}
            })
        
        # Search
        results = await self.vector_client.search(
            collection=self.collection_name,
            query_vector=query_embedding,
            filter={'must': filter_conditions} if filter_conditions else None,
            limit=limit,
            score_threshold=min_similarity
        )
        
        # Format results
        memories = []
        for result in results:
            memories.append({
                'id': result['id'],
                'content': result['payload']['content'],
                'metadata': result['payload']['metadata'],
                'similarity': result['score']
            })
        
        return memories
    
    async def get_context_memories(
        self,
        user_id: str,
        current_message: str,
        limit: int = 3
    ) -> List[str]:
        """Get relevant memories for context"""
        memories = await self.search_memories(
            query=current_message,
            user_id=user_id,
            limit=limit
        )
        
        # Format for inclusion in prompt
        formatted_memories = []
        for memory in memories:
            timestamp = memory['metadata'].get('timestamp', 'Unknown')
            content = memory['content']
            formatted_memories.append(
                f"[Memory from {timestamp}]: {content}"
            )
        
        return formatted_memories
    
    async def summarize_conversation(
        self,
        conversation_id: str,
        summary: str
    ):
        """Store conversation summary for future reference"""
        # Get conversation metadata
        metadata = {
            'type': 'conversation_summary',
            'conversation_id': conversation_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store summary as memory
        await self.store_memory(
            user_id='system',
            conversation_id=conversation_id,
            content=summary,
            metadata=metadata
        )
    
    def _generate_memory_id(
        self,
        user_id: str,
        conversation_id: str,
        content: str
    ) -> str:
        """Generate unique memory ID"""
        data = f"{user_id}:{conversation_id}:{content}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()
```

## Error Handling Strategies

### Global Error Handler

```python
# errors/error_handler.py
from typing import Dict, Any, Optional, Type
from enum import Enum
import traceback
import logging
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

class ErrorType(Enum):
    """Error type classification"""
    VALIDATION = "validation_error"
    AUTHENTICATION = "authentication_error"
    AUTHORIZATION = "authorization_error"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit_exceeded"
    EXTERNAL_SERVICE = "external_service_error"
    INTERNAL = "internal_error"
    TIMEOUT = "timeout_error"
    CONFLICT = "conflict_error"

class ApplicationError(Exception):
    """Base application error"""
    
    def __init__(
        self,
        error_type: ErrorType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 400
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)

class GlobalErrorHandler:
    """Global error handling and recovery"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_mappings = {
            ValidationError: (ErrorType.VALIDATION, 400),
            HTTPException: (ErrorType.INTERNAL, 500),
            TimeoutError: (ErrorType.TIMEOUT, 504),
            ConnectionError: (ErrorType.EXTERNAL_SERVICE, 503),
        }
    
    async def handle_error(
        self,
        request: Request,
        error: Exception
    ) -> JSONResponse:
        """Handle and format errors"""
        # Determine error type and status
        error_type, status_code = self._classify_error(error)
        
        # Log error
        self._log_error(request, error, error_type)
        
        # Format response
        response = self._format_error_response(
            error_type,
            error,
            request
        )
        
        # Send alerts for critical errors
        if status_code >= 500:
            await self._send_error_alert(request, error)
        
        return JSONResponse(
            status_code=status_code,
            content=response
        )
    
    def _classify_error(
        self,
        error: Exception
    ) -> tuple[ErrorType, int]:
        """Classify error type and determine status code"""
        if isinstance(error, ApplicationError):
            return error.error_type, error.status_code
        
        for error_class, (error_type, status_code) in self.error_mappings.items():
            if isinstance(error, error_class):
                return error_type, status_code
        
        return ErrorType.INTERNAL, 500
    
    def _format_error_response(
        self,
        error_type: ErrorType,
        error: Exception,
        request: Request
    ) -> Dict[str, Any]:
        """Format error response"""
        response = {
            'error': {
                'type': error_type.value,
                'message': str(error),
                'request_id': request.state.request_id if hasattr(request.state, 'request_id') else None,
                'path': str(request.url.path),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        # Add details for application errors
        if isinstance(error, ApplicationError):
            response['error']['details'] = error.details
        
        # Add debug info in development
        if self._is_development():
            response['error']['debug'] = {
                'exception': error.__class__.__name__,
                'traceback': traceback.format_exc()
            }
        
        return response
    
    def _log_error(
        self,
        request: Request,
        error: Exception,
        error_type: ErrorType
    ):
        """Log error with context"""
        log_data = {
            'error_type': error_type.value,
            'error_message': str(error),
            'request_path': str(request.url.path),
            'request_method': request.method,
            'user_agent': request.headers.get('user-agent'),
            'client_ip': request.client.host,
            'exception_class': error.__class__.__name__,
            'traceback': traceback.format_exc()
        }
        
        if error_type in [ErrorType.INTERNAL, ErrorType.EXTERNAL_SERVICE]:
            self.logger.error(f"Error occurred: {log_data}")
        else:
            self.logger.warning(f"Client error: {log_data}")
    
    async def _send_error_alert(
        self,
        request: Request,
        error: Exception
    ):
        """Send alert for critical errors"""
        # Implement alerting logic (e.g., send to Slack, PagerDuty)
        pass
    
    def _is_development(self) -> bool:
        """Check if running in development mode"""
        import os
        return os.getenv('ENVIRONMENT', 'production') == 'development'
```

## Retry Mechanisms

### Retry Decorator

```python
# errors/retry.py
import asyncio
from typing import Callable, Type, Tuple, Optional
from functools import wraps
import random
import logging

class RetryConfig:
    """Retry configuration"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

def retry_async(config: Optional[RetryConfig] = None):
    """Async retry decorator"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        # Last attempt failed
                        logging.error(
                            f"All retry attempts failed for {func.__name__}: {e}"
                        )
                        raise
                    
                    # Calculate delay
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter
                    if config.jitter:
                        delay *= (0.5 + random.random())
                    
                    logging.warning(
                        f"Retry attempt {attempt + 1}/{config.max_attempts} "
                        f"for {func.__name__} after {delay:.2f}s delay. "
                        f"Error: {e}"
                    )
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator

class RetryManager:
    """Centralized retry management"""
    
    def __init__(self):
        self.default_config = RetryConfig()
        self.service_configs = {
            'openai': RetryConfig(
                max_attempts=5,
                initial_delay=2.0,
                retryable_exceptions=(
                    ConnectionError,
                    TimeoutError,
                )
            ),
            'mcp': RetryConfig(
                max_attempts=3,
                initial_delay=1.0,
                retryable_exceptions=(
                    ConnectionError,
                    HTTPException,
                )
            ),
            'database': RetryConfig(
                max_attempts=3,
                initial_delay=0.5,
                retryable_exceptions=(
                    ConnectionError,
                )
            )
        }
    
    def get_config(self, service: str) -> RetryConfig:
        """Get retry config for service"""
        return self.service_configs.get(service, self.default_config)
    
    async def execute_with_retry(
        self,
        func: Callable,
        service: str = 'default',
        *args,
        **kwargs
    ):
        """Execute function with retry"""
        config = self.get_config(service)
        
        @retry_async(config)
        async def wrapped():
            return await func(*args, **kwargs)
        
        return await wrapped()
```

## Circuit Breaker Pattern

### Circuit Breaker Implementation

```python
# errors/circuit_breaker.py
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, Any
import asyncio
import logging

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        
    async def call(
        self,
        func: Callable,
        *args,
        **kwargs
    ):
        """Execute function through circuit breaker"""
        if self.state == CircuitState.OPEN:
            if await self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except self.expected_exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            # Need multiple successes to fully close
            if self.success_count >= 3:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logging.info(f"Circuit breaker {self.name} is now CLOSED")
    
    async def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state reopens circuit
            self.state = CircuitState.OPEN
            logging.warning(f"Circuit breaker {self.name} is now OPEN (half-open failure)")
            
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logging.warning(
                f"Circuit breaker {self.name} is now OPEN "
                f"(threshold {self.failure_threshold} reached)"
            )
    
    async def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure > timedelta(seconds=self.recovery_timeout)
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
        }

class CircuitBreakerManager:
    """Manage multiple circuit breakers"""
    
    def __init__(self):
        self.breakers = {}
        
    def get_breaker(
        self,
        name: str,
        **config
    ) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, **config)
        return self.breakers[name]
    
    async def call(
        self,
        service: str,
        func: Callable,
        *args,
        **kwargs
    ):
        """Call function through circuit breaker"""
        breaker = self.get_breaker(service)
        return await breaker.call(func, *args, **kwargs)
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all circuit breaker states"""
        return {
            name: breaker.get_state()
            for name, breaker in self.breakers.items()
        }
```

## Graceful Degradation

### Fallback Strategies

```python
# errors/fallback.py
from typing import Callable, Any, Optional, Dict
import logging

class FallbackStrategy:
    """Fallback strategy for graceful degradation"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.fallback_handlers = {}
        
    def register_fallback(
        self,
        service: str,
        fallback_func: Callable
    ):
        """Register fallback handler for service"""
        self.fallback_handlers[service] = fallback_func
    
    async def execute_with_fallback(
        self,
        primary_func: Callable,
        service: str,
        *args,
        **kwargs
    ):
        """Execute with fallback on failure"""
        try:
            return await primary_func(*args, **kwargs)
            
        except Exception as e:
            self.logger.warning(
                f"Primary function failed for {service}: {e}, "
                f"attempting fallback"
            )
            
            if service in self.fallback_handlers:
                fallback_func = self.fallback_handlers[service]
                return await fallback_func(*args, **kwargs)
            
            raise

class CacheBasedFallback:
    """Fallback to cached responses"""
    
    def __init__(self, cache_store):
        self.cache = cache_store
        
    async def with_cache_fallback(
        self,
        func: Callable,
        cache_key: str,
        ttl: int = 3600,
        *args,
        **kwargs
    ):
        """Execute with cache fallback"""
        try:
            # Try primary function
            result = await func(*args, **kwargs)
            
            # Update cache on success
            await self.cache.set(cache_key, result, ttl=ttl)
            
            return result
            
        except Exception as e:
            # Fallback to cached value
            cached_result = await self.cache.get(cache_key)
            
            if cached_result is not None:
                logging.warning(
                    f"Using cached result for {cache_key} due to error: {e}"
                )
                return cached_result
            
            raise

class DefaultValueFallback:
    """Fallback to default values"""
    
    def __init__(self):
        self.defaults = {
            'user_preferences': {
                'theme': 'light',
                'language': 'en',
                'notifications': True
            },
            'model_config': {
                'model': 'gpt-3.5-turbo',
                'temperature': 0.7,
                'max_tokens': 1000
            }
        }
    
    def get_with_default(
        self,
        key: str,
        value: Optional[Any] = None
    ) -> Any:
        """Get value with default fallback"""
        if value is not None:
            return value
        
        return self.defaults.get(key, {})
```

## Monitoring & Alerting

### Health Check System

```python
# monitoring/health_check.py
from typing import Dict, Any, List
from enum import Enum
from datetime import datetime
import asyncio

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck:
    """Health check system"""
    
    def __init__(self):
        self.checks = {}
        
    def register_check(
        self,
        name: str,
        check_func: Callable
    ):
        """Register health check"""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, check_func in self.checks.items():
            try:
                result = await asyncio.wait_for(
                    check_func(),
                    timeout=5.0
                )
                results[name] = {
                    'status': result.get('status', HealthStatus.HEALTHY.value),
                    'details': result.get('details', {}),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Update overall status
                if result.get('status') == HealthStatus.UNHEALTHY.value:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.get('status') == HealthStatus.DEGRADED.value and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except asyncio.TimeoutError:
                results[name] = {
                    'status': HealthStatus.UNHEALTHY.value,
                    'error': 'Health check timeout',
                    'timestamp': datetime.utcnow().isoformat()
                }
                overall_status = HealthStatus.UNHEALTHY
                
            except Exception as e:
                results[name] = {
                    'status': HealthStatus.UNHEALTHY.value,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                overall_status = HealthStatus.UNHEALTHY
        
        return {
            'status': overall_status.value,
            'checks': results,
            'timestamp': datetime.utcnow().isoformat()
        }

# Example health checks
async def check_database() -> Dict[str, Any]:
    """Check database health"""
    try:
        # Perform database query
        # await db.execute("SELECT 1")
        return {'status': HealthStatus.HEALTHY.value}
    except Exception as e:
        return {
            'status': HealthStatus.UNHEALTHY.value,
            'details': {'error': str(e)}
        }

async def check_redis() -> Dict[str, Any]:
    """Check Redis health"""
    try:
        # Check Redis connection
        # await redis.ping()
        return {'status': HealthStatus.HEALTHY.value}
    except Exception as e:
        return {
            'status': HealthStatus.UNHEALTHY.value,
            'details': {'error': str(e)}
        }

async def check_openai() -> Dict[str, Any]:
    """Check OpenAI API health"""
    try:
        # Test OpenAI API
        # await openai_client.models.list()
        return {'status': HealthStatus.HEALTHY.value}
    except Exception as e:
        return {
            'status': HealthStatus.DEGRADED.value,
            'details': {'error': str(e)}
        }
```

---

**Document Version**: 1.0.0  
**Last Updated**: December 2024  
**State Management Standard**: Redis-backed distributed state