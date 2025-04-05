# Location: pages/3_SQLite_RAG.py
# Refactored implementation of SQLite RAG using modular components

import os
import sys
import streamlit as st
import pandas as pd
import uuid
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.sqlite_chatbot import SQLiteChatbot
from models.chat_history_manager import ChatHistoryManager

# Import shared utilities
from utils.ui_helpers import (
    initialize_session_state, display_example_questions,
    display_chat_messages, handle_chat_input, display_history_view,
    add_user_message, add_assistant_message, save_chat_history, view_chat_history
)
from utils.sidebar import display_sidebar
from utils.visualization import generate_visualization, generate_visualization_code

# Page configuration
st.set_page_config(
    page_title="SQLite RAG",
    page_icon="🗃️",
    layout="wide"
)

# Initialize application state
history_manager = ChatHistoryManager()
app_type, session_id_key, chatbot_key, messages_key = initialize_session_state(
    "sqlite", SQLiteChatbot, history_manager
)

st.title("🗃️ SQLite Database RAG")
st.subheader("Chat with your SQLite Databases")

# Handle history view if active
if st.session_state.viewing_history and st.session_state.history_session_id:
    display_history_view(history_manager)
    
    # Display database info if available
    if hasattr(st.session_state[chatbot_key], 'db_connection') and st.session_state[chatbot_key].db_connection is not None:
        display_database_info()
else:
    # Display sidebar with controls
    display_sidebar(app_type, chatbot_key, messages_key, session_id_key, SQLiteChatbot, ['db', 'sqlite', 'sqlite3'])
    
    # Display database information if connected
    if hasattr(st.session_state[chatbot_key], 'table_info') and st.session_state[chatbot_key].table_info:
        st.subheader("Connected Database")
        
        # Create tabs for tables
        all_tables = list(st.session_state[chatbot_key].table_info.keys())
        if all_tables:
            tabs = st.tabs(all_tables)
            
            # Display info for each table in its own tab
            for i, table_name in enumerate(all_tables):
                info = st.session_state[chatbot_key].table_info[table_name]
                with tabs[i]:
                    st.write(f"**{table_name}** ({info['row_count']} rows)")
                    cols = ", ".join([f"{col[1]} ({col[2]})" for col in info['columns']])
                    st.write(f"Columns: {cols}")
                    
                    # Show sample data
                    st.write("Sample data:")
                    # Convert sample data to DataFrame for better display
                    columns = [col[1] for col in info['columns']]
                    sample_df = pd.DataFrame(info['sample_data'], columns=columns)
                    st.dataframe(sample_df)
    
    # Display example questions
    example_questions = [
        "Write a query to list all tables and their row counts",
        "How many records are in each table?",
        "Show me the schema for all tables"
    ]
    display_example_questions(app_type, messages_key, example_questions)
    
    # Display SQL query interface
    st.subheader("Direct SQL Query")
    if hasattr(st.session_state[chatbot_key], 'db_connection') and st.session_state[chatbot_key].db_connection is not None:
        # Create columns for SQL input and execution
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # SQL input text area
            sql_query = st.text_area("Enter SQL query", height=100, 
                                    placeholder="SELECT * FROM table_name LIMIT 5")
        
        with col2:
            # Execute button
            st.write("")  # Add some space for alignment
            st.write("")  # Add some space for alignment
            execute_sql = st.button("Execute Query")
            
            # Add visualization option
            viz_options = ["table", "bar", "line", "scatter", "pie"]
            selected_viz = st.selectbox(f"Visualization Type", viz_options)
        
        # Handle SQL execution
        if execute_sql and sql_query:
            process_direct_sql_query(sql_query, selected_viz)
    
    # Main chat interface
    st.subheader("Chat with your Database")
    
    # Display chat messages
    display_chat_messages(messages_key)
    
    # Handle chat input
    handle_chat_input(app_type, chatbot_key, messages_key, session_id_key,
                     "Ask a question about your database")

# Helper function for SQL-specific functionality
def process_direct_sql_query(sql_query, viz_type="table"):
    """Process a direct SQL query and display results"""
    with st.spinner("Executing query..."):
        try:
            # Execute the query
            result = st.session_state.sqlite_chatbot.execute_sql_query(sql_query)
            
            if isinstance(result, pd.DataFrame):
                # Display the results with tabular and visual representation
                st.subheader("Query Results")
                st.dataframe(result)
                
                # Create visualization if requested
                if viz_type != "table" and len(result) > 0:
                    with st.spinner("Generating visualization..."):
                        try:
                            fig = generate_visualization(result, chart_type=viz_type)
                            if isinstance(fig, go.Figure):
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(fig)  # Show error message
                        except Exception as viz_error:
                            st.error(f"Error creating visualization: {str(viz_error)}")
                
                # Add option to download results
                csv = result.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
                
                # Add successful query to chat history
                user_message = {
                    "role": "user", 
                    "content": f"Executed SQL query: ```sql\n{sql_query}\n```",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.sqlite_messages.append(user_message)
                
                # Format the result summary
                result_summary = f"Query executed successfully. Returned {len(result)} rows with {len(result.columns)} columns."
                
                # Add to chat history with specialized function for SQL results
                add_assistant_message("sqlite", "sqlite_messages", "sqlite_session_id", 
                                     result_summary, query=sql_query, data=result)
                
                # Store query results for visualization
                st.session_state.sqlite_chatbot.last_query_results = result
            else:
                # Non-SELECT query result (like UPDATE, INSERT, etc.)
                st.success(result)
                
                # Add to chat history
                user_message = {
                    "role": "user", 
                    "content": f"Executed SQL query: ```sql\n{sql_query}\n```",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.sqlite_messages.append(user_message)
                
                # Add the assistant message
                add_assistant_message("sqlite", "sqlite_messages", "sqlite_session_id", 
                                     result, query=sql_query)
                
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")

# Function to display database info (SQLite specific)
def display_database_info():
    """Display information about the connected database"""
    if hasattr(st.session_state.sqlite_chatbot, 'db_connection') and st.session_state.sqlite_chatbot.db_connection is not None:
        st.subheader("Connected Database")
        
        # Create tabs instead of nested expanders
        all_tables = list(st.session_state.sqlite_chatbot.table_info.keys())
        if all_tables:
            tabs = st.tabs(all_tables)
            
            # Display info for each table in its own tab
            for i, table_name in enumerate(all_tables):
                info = st.session_state.sqlite_chatbot.table_info[table_name]
                with tabs[i]:
                    st.write(f"**{table_name}** ({info['row_count']} rows)")
                    cols = ", ".join([f"{col[1]} ({col[2]})" for col in info['columns']])
                    st.write(f"Columns: {cols}")
                    
                    # Show sample data directly (no nested expander)
                    st.write("Sample data:")
                    # Convert sample data to DataFrame for better display
                    columns = [col[1] for col in info['columns']]
                    sample_df = pd.DataFrame(info['sample_data'], columns=columns)
                    st.dataframe(sample_df)