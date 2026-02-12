"""
Conversation session management for multi-turn dialogue.

Tracks chat history, context, and follow-up question support.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from shared.utils import get_logger

logger = get_logger(__name__)


class Message(BaseModel):
    """A single message in a conversation."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: Optional[str] = None
    citations: List[Dict] = Field(default_factory=list)


class ConversationSession(BaseModel):
    """A conversation session with message history."""

    session_id: UUID = Field(default_factory=uuid4)
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)

    def add_message(self, role: str, content: str, trace_id: Optional[str] = None, citations: Optional[List] = None):
        """Add a message to the conversation."""
        msg = Message(
            role=role,
            content=content,
            trace_id=trace_id,
            citations=citations or [],
        )
        self.messages.append(msg)
        self.last_active = datetime.utcnow()

    def get_context(self, max_turns: int = 5) -> List[Dict[str, str]]:
        """
        Get recent conversation context for LLM prompt.

        Args:
            max_turns: Maximum number of turns to include

        Returns:
            List of message dicts with role and content
        """
        recent = self.messages[-(max_turns * 2):]  # Each turn = user + assistant
        return [{"role": msg.role, "content": msg.content} for msg in recent]

    def get_last_user_query(self) -> Optional[str]:
        """Get the most recent user query."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None


class SessionManager:
    """
    In-memory session manager for conversations.

    In production, this should be backed by Redis or a database.
    """

    def __init__(self):
        self.sessions: Dict[UUID, ConversationSession] = {}
        logger.info("SessionManager initialized")

    def create_session(self, metadata: Optional[Dict] = None) -> ConversationSession:
        """Create a new conversation session."""
        session = ConversationSession(metadata=metadata or {})
        self.sessions[session.session_id] = session
        logger.info(f"Created session {session.session_id}")
        return session

    def get_session(self, session_id: UUID) -> Optional[ConversationSession]:
        """Retrieve a session by ID."""
        return self.sessions.get(session_id)

    def get_or_create_session(self, session_id: Optional[UUID] = None) -> ConversationSession:
        """Get existing session or create new one."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        return self.create_session()

    def delete_session(self, session_id: UUID) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        return False

    def cleanup_expired(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours."""
        now = datetime.utcnow()
        expired = [
            sid
            for sid, session in self.sessions.items()
            if (now - session.last_active).total_seconds() > max_age_hours * 3600
        ]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
