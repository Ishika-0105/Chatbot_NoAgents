# chat_memory.py
from typing import List, Dict, Any
from datetime import datetime

class ChatMemory:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        # These are instance-specific lists to hold chat history and conversation context
        self._chat_history = []
        self._conversation_context = []
    
    def add_message(self, role: str, content: str, sources: List[str] = None):
        """Add a message to chat history for this specific thread."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sources': sources or []
        }
        
        self._chat_history.append(message)
        
        # Maintain conversation context for better responses
        if role == 'user':
            self._conversation_context.append(f"Human: {content}")
        elif role == 'assistant':
            self._conversation_context.append(f"Assistant: {content}")
        
        # Keep only recent context to avoid token limits
        if len(self._conversation_context) > self.max_history * 2:
            self._conversation_context = self._conversation_context[-self.max_history * 2:]
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get full chat history for this specific thread."""
        return self._chat_history
    
    def get_conversation_context(self, include_last_n: int = 5) -> str:
        """Get recent conversation context for the model for this specific thread."""
        if not self._conversation_context:
            return ""
        
        recent_context = self._conversation_context[-include_last_n * 2:]
        return "\n".join(recent_context)
    
    def clear_history(self):
        """Clear chat history for this specific thread."""
        self._chat_history = []
        self._conversation_context = []
    
    def get_last_user_messages(self, n: int = 3) -> List[str]:
        """Get last n user messages for context from this specific thread."""
        user_messages = []
        for message in reversed(self._chat_history):
            if message['role'] == 'user':
                user_messages.append(message['content'])
                if len(user_messages) >= n:
                    break
        return list(reversed(user_messages))
