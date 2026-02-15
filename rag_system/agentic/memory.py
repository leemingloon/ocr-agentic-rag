"""
Conversation Memory for Agentic RAG

Maintains context across multi-turn conversations
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    """Single conversation message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class ConversationMemory:
    """
    Manages conversation history and context
    
    Features:
    - Sliding window (keep last N messages)
    - Token-based truncation
    - Important message preservation
    """
    
    def __init__(
        self,
        max_messages: int = 10,
        max_tokens: int = 4000,
        preserve_system: bool = True
    ):
        """
        Initialize conversation memory
        
        Args:
            max_messages: Maximum messages to keep
            max_tokens: Maximum tokens in history
            preserve_system: Always keep system message
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.preserve_system = preserve_system
        
        self.messages: List[Message] = []
        self.system_message: Optional[Message] = None
        self.history = []
    
    def add_turn(self, query: str, response: str):
        """Add a conversation turn to memory"""
        self.history.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add message to history
        
        Args:
            role: "user", "assistant", or "system"
            content: Message content
            metadata: Optional metadata
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        if role == "system":
            self.system_message = message
        else:
            self.messages.append(message)
            self._truncate_if_needed()
    
    def get_history(self, include_system: bool = True) -> List[Dict]:
        """
        Get conversation history in format for LLM
        
        Args:
            include_system: Include system message
            
        Returns:
            List of message dicts
        """
        history = []
        
        # Add system message first
        if include_system and self.system_message:
            history.append({
                "role": "system",
                "content": self.system_message.content
            })
        
        # Add conversation messages
        for msg in self.messages:
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return history
    
    def get_context_string(self) -> str:
        """Get history as formatted string"""
        context_parts = []
        
        for msg in self.messages:
            context_parts.append(f"{msg.role}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def _truncate_if_needed(self):
        """Truncate history if it exceeds limits"""
        # Truncate by message count
        if len(self.messages) > self.max_messages:
            # Keep most recent messages
            self.messages = self.messages[-self.max_messages:]
        
        # Truncate by tokens (approximate)
        total_tokens = sum(len(msg.content.split()) for msg in self.messages)
        
        while total_tokens > self.max_tokens and len(self.messages) > 1:
            # Remove oldest message
            removed = self.messages.pop(0)
            total_tokens -= len(removed.content.split())
    
    def clear(self):
        """Clear all messages except system"""
        self.messages = []
    
    def get_last_user_message(self) -> Optional[str]:
        """Get most recent user message"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None
    
    def get_summary(self) -> Dict:
        """Get memory statistics"""
        return {
            "num_messages": len(self.messages),
            "has_system": self.system_message is not None,
            "total_tokens": sum(len(msg.content.split()) for msg in self.messages),
            "oldest_message": self.messages[0].timestamp if self.messages else None,
            "newest_message": self.messages[-1].timestamp if self.messages else None,
        }


# Example usage
if __name__ == "__main__":
    # Initialize memory
    memory = ConversationMemory(max_messages=5)
    
    # Add system message
    memory.add_message(
        "system",
        "You are a financial document analyst."
    )
    
    # Simulate conversation
    memory.add_message("user", "What was our Q3 revenue?")
    memory.add_message("assistant", "Q3 revenue was $2.5M.")
    memory.add_message("user", "How does that compare to Q2?")
    memory.add_message("assistant", "Q2 was $2.1M, so Q3 is up 19%.")
    
    # Get history
    history = memory.get_history()
    
    print("Conversation History:")
    for msg in history:
        print(f"{msg['role']}: {msg['content']}")
    
    print(f"\nMemory Summary: {memory.get_summary()}")