# Location: pages/2_CSV_RAG.py
# Refactored implementation of CSV RAG using modular components

import os
import sys
import streamlit as st

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.csv_chatbot import CSVChatbot
from models.chat_history_manager import ChatHistoryManager

# Import shared utilities
from utils.ui_helpers import (
    initialize_session_state, display_example_questions,
    display_chat_messages, handle_chat_input, display_history_view
)
from utils.sidebar import display_sidebar

# Page configuration
st.set_page_config(
    page_title="CSV RAG",
    page_icon="📊",
    layout="wide"
)

# Initialize application state
history_manager = ChatHistoryManager()
app_type, session_id_key, chatbot_key, messages_key = initialize_session_state(
    "csv", CSVChatbot, history_manager
)

st.title("📊 CSV Document RAG")
st.subheader("Chat with your CSV data")

# Handle history view if active
if st.session_state.viewing_history and st.session_state.history_session_id:
    display_history_view(history_manager)
else:
    # Display sidebar with controls
    display_sidebar(app_type, chatbot_key, messages_key, session_id_key, CSVChatbot, ['csv'])
    
    # Display example questions
    example_questions = [
        "Summarize the CSV data.",
        "What trends do you see in the data?",
        "Calculate key statistics from the CSV."
    ]
    display_example_questions(app_type, messages_key, example_questions)
    
    # Main chat interface
    st.subheader("Chat with your CSV Data")
    
    # Display chat messages
    display_chat_messages(messages_key)
    
    # Handle chat input
    handle_chat_input(app_type, chatbot_key, messages_key, session_id_key,
                     "Ask a question about your CSV data")