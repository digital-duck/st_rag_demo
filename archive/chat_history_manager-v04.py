# Location: models/chat_history_manager.py
# Update the save_chat_history method to include model information and other metadata

import os
import json
import time
import hashlib
from datetime import datetime

class ChatHistoryManager:
    """
    Manages the persistence of chat history to disk.
    Supports multiple chat sessions for different RAG types.
    """
    
    def __init__(self, history_dir="chat_history"):
        """
        Initialize the chat history manager.
        
        Args:
            history_dir (str): Directory to store chat history files
        """
        self.history_dir = history_dir
        
        # Create history directory if it doesn't exist
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            
        # Keep track of last saved message hash to avoid duplicate saves
        self.last_saved_hashes = {}
    
    def save_chat_history(self, messages, rag_type, session_id=None, metadata=None):
        """
        Save chat history to disk with enhanced metadata.
        
        Args:
            messages (list): List of message dictionaries
            rag_type (str): Type of RAG (pdf, csv, sqlite)
            session_id (str, optional): Unique session ID. If None, uses timestamp
            metadata (dict, optional): Additional metadata to store with the chat history
                This can include model_name, chunk_size, k_value, etc.
        
        Returns:
            str: Path to the saved history file
        """
        if not session_id:
            # Use timestamp as session ID if not provided
            session_id = f"{rag_type}_{int(time.time())}"
        
        # Calculate hash of messages to check for changes
        messages_str = json.dumps(messages, sort_keys=True)
        current_hash = hashlib.md5(messages_str.encode()).hexdigest()
        
        # Check if this exact message set was already saved
        if session_id in self.last_saved_hashes and self.last_saved_hashes[session_id] == current_hash:
            # No changes since last save, skip
            return None
        
        # Format filename
        filename = f"{session_id}.json"
        filepath = os.path.join(self.history_dir, filename)
        
        # Create metadata
        history_data = {
            "session_id": session_id,
            "rag_type": rag_type,
            "last_updated": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages
        }
        
        # Add additional metadata if provided
        if metadata:
            history_data["metadata"] = metadata
        
        # Save to disk
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        # Update hash
        self.last_saved_hashes[session_id] = current_hash
        
        return filepath
    
    def load_chat_history(self, session_id):
        """
        Load chat history from disk.
        
        Args:
            session_id (str): Session ID of the chat history to load
        
        Returns:
            tuple: (messages, metadata) or ([], None) if not found
        """
        filepath = os.path.join(self.history_dir, f"{session_id}.json")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                # Update the hash to prevent immediate re-save
                messages_str = json.dumps(history_data.get("messages", []), sort_keys=True)
                current_hash = hashlib.md5(messages_str.encode()).hexdigest()
                self.last_saved_hashes[session_id] = current_hash
                
                # Return both messages and metadata
                return history_data.get("messages", []), history_data.get("metadata", None)
            except Exception as e:
                print(f"Error loading chat history: {str(e)}")
                return [], None
        else:
            return [], None
    
    def list_sessions(self, rag_type=None):
        """
        List available chat sessions.
        
        Args:
            rag_type (str, optional): Filter by RAG type
        
        Returns:
            list: List of session information dictionaries
        """
        sessions = []
        
        for filename in os.listdir(self.history_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(self.history_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        history_data = json.load(f)
                    
                    # Extract basic info
                    session_info = {
                        "session_id": history_data.get("session_id"),
                        "rag_type": history_data.get("rag_type"),
                        "last_updated": history_data.get("last_updated"),
                        "message_count": history_data.get("message_count", 0),
                        "filename": filename
                    }
                    
                    # Filter by RAG type if specified
                    if rag_type is None or session_info["rag_type"] == rag_type:
                        sessions.append(session_info)
                except Exception as e:
                    print(f"Error loading session info from {filename}: {str(e)}")
        
        # Sort by last updated time (newest first)
        sessions.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
        
        return sessions
    
    def delete_session(self, session_id):
        """
        Delete a chat session.
        
        Args:
            session_id (str): Session ID to delete
        
        Returns:
            bool: True if successful, False otherwise
        """
        filepath = os.path.join(self.history_dir, f"{session_id}.json")
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                # Remove from hash cache
                if session_id in self.last_saved_hashes:
                    del self.last_saved_hashes[session_id]
                return True
            except Exception as e:
                print(f"Error deleting session {session_id}: {str(e)}")
                return False
        else:
            return False
    
    def get_latest_session(self, rag_type):
        """
        Get the most recent session for a specific RAG type.
        
        Args:
            rag_type (str): Type of RAG
        
        Returns:
            str: Session ID of the latest session or None if not found
        """
        sessions = self.list_sessions(rag_type)
        
        if sessions:
            return sessions[0]["session_id"]
        else:
            return None
    
    def save_all_sessions(self, session_state):
        """
        Save all active chat sessions found in Streamlit's session state.
        Useful for saving before page navigation or app closure.
        
        Args:
            session_state: Streamlit's session_state object
        
        Returns:
            dict: Summary of saved sessions
        """
        saved_sessions = {}
        
        # Look for common chat session patterns in session state
        for key in session_state:
            # Check for message lists with common naming patterns
            if key.endswith('_messages'):
                # Extract the RAG type from the key name
                rag_type = key.replace('_messages', '')
                
                # Check if there's a corresponding session ID
                session_id_key = f"{rag_type}_session_id"
                if session_id_key in session_state:
                    session_id = session_state[session_id_key]
                    messages = session_state[key]
                    
                    # Save this session
                    if messages:
                        filepath = self.save_chat_history(messages, rag_type, session_id)
                        if filepath:
                            saved_sessions[rag_type] = {
                                "session_id": session_id, 
                                "message_count": len(messages)
                            }
        
        return saved_sessions