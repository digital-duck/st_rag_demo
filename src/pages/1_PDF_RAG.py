# Location: pages/1_PDF_RAG.py
# Refactored implementation of PDF RAG using modular components
# With fixed example question buttons that properly trigger LLM calls

import os
import sys
import streamlit as st
from datetime import datetime

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.pdf_chatbot import PDFChatbot
from models.chat_history_manager import ChatHistoryManager

# Import shared utilities
from utils.ui_helpers import (
    initialize_session_state, display_chat_messages, 
    handle_chat_input, display_history_view,
    process_chat_message
)
from utils.sidebar import display_sidebar

# Page configuration
st.set_page_config(
    page_title="PDF RAG",
    page_icon="📄",
    layout="wide"
)

# Initialize application state
history_manager = ChatHistoryManager()
app_type, session_id_key, chatbot_key, messages_key = initialize_session_state(
    "pdf", PDFChatbot, history_manager
)

st.title("📄 PDF Document RAG")
st.subheader("Chat with your PDF documents")

# Handle history view if active
if st.session_state.viewing_history and st.session_state.history_session_id:
    display_history_view(history_manager)
else:
    # Display sidebar with controls
    display_sidebar(app_type, chatbot_key, messages_key, session_id_key, PDFChatbot, ['pdf'])
    
    # Display example questions with direct LLM triggering
    st.subheader("Example Questions")
    example_questions = [
        "What are the main topics in the document?",
        "Summarize the key points.",
        "What does the document say about X?"
    ]
    
    # Create columns for the example buttons
    col1, col2, col3 = st.columns(3)
    
    # Define a function to handle clicking example buttons
    def handle_example_click(question):
        # Add user message to chat history
        user_message = {
            "role": "user", 
            "content": question,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state[messages_key].append(user_message)
        
        # Process the message to get a response
        process_chat_message(app_type, chatbot_key, messages_key, session_id_key, question)
        
        # Force rerun to update the UI
        st.rerun()
    
    # Display buttons in columns with direct LLM call handling
    with col1:
        if st.button(example_questions[0]):
            handle_example_click(example_questions[0])
            
    with col2:
        if st.button(example_questions[1]):
            handle_example_click(example_questions[1])
            
    with col3:
        if st.button(example_questions[2]):
            handle_example_click(example_questions[2])
    
    # Main chat interface
    st.subheader("Chat with your PDFs")
    
    # Display chat messages
    display_chat_messages(messages_key)
    
    # Handle chat input
    handle_chat_input(app_type, chatbot_key, messages_key, session_id_key, 
                     "Ask a question about your PDFs")