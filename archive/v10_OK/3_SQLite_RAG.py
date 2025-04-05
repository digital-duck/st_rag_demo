"""
find top 5 customers by sales
"""

import os
import sys
import streamlit as st
import pandas as pd
import uuid

from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.sqlite_chatbot import SQLiteChatbot
from models.chat_history_manager import ChatHistoryManager
from models.auto_save import AutoSave

# Page configuration
st.set_page_config(
    page_title="SQLite RAG",
    page_icon="🗃️",
    layout="wide"
)

#------------------------------------------------------------------------
# Helper Functions for UI Components
#------------------------------------------------------------------------

def initialize_session_state():
    """Initialize all session state variables"""
    # Initialize the chat history manager
    if "history_manager" not in st.session_state:
        st.session_state.history_manager = ChatHistoryManager()
        
    # Initialize auto-save utility
    if "auto_save" not in st.session_state:
        st.session_state.auto_save = AutoSave(st.session_state.history_manager)
        st.session_state.auto_save.setup_auto_save()

    # Initialize session state for current session ID
    if "sqlite_session_id" not in st.session_state:
        # Generate a unique session ID for this chat session
        st.session_state.sqlite_session_id = f"sqlite_{uuid.uuid4().hex[:8]}"

    # Initialize session state for chatbot and messages
    if "sqlite_chatbot" not in st.session_state:
        st.session_state.sqlite_chatbot = SQLiteChatbot()
        
    if "sqlite_messages" not in st.session_state:
        # Try to load previous chat history for this session
        previous_messages, session_metadata = st.session_state.history_manager.load_chat_history(st.session_state.sqlite_session_id)
        
        if previous_messages:
            st.session_state.sqlite_messages = previous_messages
            # Restore settings from metadata if available
            _restore_settings_from_metadata(session_metadata)
        else:
            st.session_state.sqlite_messages = []

    # Initialize RAG parameters in session state
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000
        
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 100
        
    if "k_value" not in st.session_state:
        st.session_state.k_value = 5

    # Initialize session state for viewing chat history
    if "viewing_history" not in st.session_state:
        st.session_state.viewing_history = False
    if "history_session_id" not in st.session_state:
        st.session_state.history_session_id = None

def _restore_settings_from_metadata(metadata):
    """Restore settings from session metadata"""
    if not metadata:
        return
        
    # Restore model if available
    if "model_name" in metadata:
        model_name = metadata["model_name"]
        if model_name != st.session_state.get("model_name", "gpt-3.5-turbo"):
            st.session_state.model_name = model_name
    
    # Restore RAG parameters if available
    if "k_value" in metadata:
        st.session_state.k_value = metadata["k_value"]
    if "chunk_size" in metadata:
        st.session_state.chunk_size = metadata["chunk_size"]
    if "chunk_overlap" in metadata:
        st.session_state.chunk_overlap = metadata["chunk_overlap"]

def save_current_chat_history():
    """Save the current chat history with metadata"""
    if st.session_state.sqlite_messages:
        current_metadata = {
            "model_name": st.session_state.get("model_name", "gpt-3.5-turbo"),
            "k_value": st.session_state.k_value,
            "chunk_size": st.session_state.chunk_size,
            "chunk_overlap": st.session_state.chunk_overlap,
            "files_processed": list(st.session_state.sqlite_chatbot.file_metadata.keys()) if hasattr(st.session_state.sqlite_chatbot, 'file_metadata') else []
        }
        st.session_state.history_manager.save_chat_history(
            st.session_state.sqlite_messages,
            "sqlite",
            st.session_state.sqlite_session_id,
            metadata=current_metadata
        )

def generate_visualization(data, chart_type="auto"):
    """Generate a Plotly visualization based on data and chart type"""
    if isinstance(data, str):
        return data  # Return error message if data is a string
    
    try:
        # Determine the best chart type if auto
        if chart_type == "auto":
            num_columns = sum(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns)
            categorical_columns = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
            
            if len(data) > 20 and num_columns >= 2:
                chart_type = "scatter"
            elif len(categorical_columns) >= 1 and num_columns >= 1:
                chart_type = "bar"
            elif num_columns >= 1:
                chart_type = "line"
            else:
                chart_type = "table"
        
        # Create visualization based on chart type
        if chart_type == "scatter" and len(data.columns) >= 2:
            numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            if len(numeric_cols) >= 2:
                fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1])
                return fig
        
        elif chart_type == "bar" and len(data.columns) >= 2:
            # Find a categorical column and a numeric column
            categorical_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
            numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            
            if categorical_cols and numeric_cols:
                fig = px.bar(data, x=categorical_cols[0], y=numeric_cols[0])
                return fig
        
        elif chart_type == "line" and len(data.columns) >= 2:
            numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            if len(numeric_cols) >= 1:
                # Use first column as x if it's datetime or the index
                if pd.api.types.is_datetime64_any_dtype(data.index):
                    fig = px.line(data, y=numeric_cols[0])
                else:
                    fig = px.line(data, y=numeric_cols[0], x=data.columns[0])
                return fig
        
        elif chart_type == "pie" and len(data.columns) >= 2:
            # Find a categorical column and a numeric column
            categorical_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
            numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            
            if categorical_cols and numeric_cols:
                fig = px.pie(data, names=categorical_cols[0], values=numeric_cols[0])
                return fig
        
        # Default to table view
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(data.columns)),
            cells=dict(values=[data[col] for col in data.columns])
        )])
        return fig
        
    except Exception as e:
        return f"Error generating visualization: {str(e)}"

def view_chat_history(session_id):
    """Set the state to view a specific chat history session"""
    st.session_state.viewing_history = True
    st.session_state.history_session_id = session_id
    st.rerun()

def display_history_view():
    """Display the chat history view"""
    session_id = st.session_state.history_session_id
    # Load the messages from this session
    loaded_messages, session_metadata = st.session_state.history_manager.load_chat_history(session_id)
    
    if loaded_messages:
        # Create a container for the history view
        st.subheader(f"Chat History for {session_id} ({len(loaded_messages)} messages)")
        
        # Display session details if metadata is available
        if session_metadata:
            with st.expander("Session Details", expanded=False):
                st.write(f"**Model:** {session_metadata.get('model_name', 'Unknown')}")
                st.write(f"**RAG Parameters:**")
                st.write(f"- k_value: {session_metadata.get('k_value', 'N/A')}")
                st.write(f"- chunk_size: {session_metadata.get('chunk_size', 'N/A')}")
                st.write(f"- chunk_overlap: {session_metadata.get('chunk_overlap', 'N/A')}")
                
                # Show files that were processed
                if "files_processed" in session_metadata and session_metadata["files_processed"]:
                    st.write("**Files:**")
                    for file in session_metadata["files_processed"]:
                        st.write(f"- {file}")
        
        # Chat container for messages
        chat_container = st.container()
        with chat_container:
            for msg in loaded_messages:
                with st.chat_message(msg.get("role", "unknown")):
                    st.markdown(msg.get("content", ""))
                    
                    # Show message details for assistant messages
                    if msg.get("role") == "assistant" and "model" in msg:
                        with st.expander("Message Details", expanded=False):
                            st.caption(f"**Model**: {msg.get('model', 'Unknown')}")
                            if "timestamp" in msg:
                                st.caption(f"**Time**: {msg.get('timestamp')}")
        
        # Button to exit history view
        if st.button("← Back to Current Session", key="exit_history"):
            st.session_state.viewing_history = False
            st.session_state.history_session_id = None
            st.rerun()
    else:
        st.error("Failed to load chat history")
        if st.button("← Back to Current Session", key="exit_history_error"):
            st.session_state.viewing_history = False
            st.session_state.history_session_id = None
            st.rerun()

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

def display_sidebar():
    """Display the sidebar with all options and controls"""
    with st.sidebar:
        st.header("Upload SQLite Database")
        handle_file_upload()
        
        st.header("Model Selection")
        handle_model_selection()
        
        st.header("RAG Configuration")
        handle_rag_config()
        
        st.subheader("Visualization Options")
        handle_viz_options()
        
        st.header("Chat History")
        handle_chat_history()
        
        handle_action_buttons()
        
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("Please set your OpenAI API key in a .env file or as an environment variable.")

        # Display current session info
        st.caption(f"Current Session ID: {st.session_state.sqlite_session_id}")

def handle_file_upload():
    """Handle file upload and database selection in the sidebar"""
    # First check if there are any previously processed databases
    available_dbs = st.session_state.sqlite_chatbot.get_available_databases()
    
    if available_dbs:
        st.write("Previously processed databases:")
        
        # Create a radio button for each available database
        db_options = ["Upload new database"] + [db["name"] for db in available_dbs]
        selected_option = st.radio("Select a database", db_options)
        
        if selected_option != "Upload new database":
            # User selected a previously processed database
            selected_db = next((db for db in available_dbs if db["name"] == selected_option), None)
            
            if selected_db and st.button(f"Load {selected_db['name']}"):
                with st.spinner(f"Loading {selected_db['name']}..."):
                    table_count = st.session_state.sqlite_chatbot.load_database(
                        selected_db["path"], 
                        selected_db["name"]
                    )
                    if table_count > 0:
                        st.success(f"Successfully loaded database with {table_count} tables!")
                    else:
                        st.error("Error loading database.")
            
            # Early return if user selected an existing database
            return
    
    # Regular file upload UI
    uploaded_files = st.file_uploader(
        "Upload SQLite database files", 
        type=["db", "sqlite", "sqlite3"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        process_clicked = st.button("Process Database Files")
        
        if process_clicked:
            with st.spinner("Processing files..."):
                # Clear existing data before processing new files
                st.session_state.sqlite_chatbot.clear()
                
                table_count = 0
                
                for file in uploaded_files:
                    count = st.session_state.sqlite_chatbot.process_sqlite(file)
                    st.write(f"Processed {file.name}: {count} tables")
                    table_count += count
                
                if table_count > 0:
                    st.success(f"Successfully processed {table_count} tables into document chunks!")
                else:
                    st.warning("No database tables were processed.")

def handle_model_selection():
    """Handle model selection in the sidebar"""
    model_name = st.selectbox(
        "Select OpenAI Model",
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"]
    )
    
    if "model_name" not in st.session_state or st.session_state.model_name != model_name:
        st.session_state.model_name = model_name
        # Create a new chatbot instance with the selected model
        temp_chatbot = SQLiteChatbot(model_name=model_name)
        # Transfer documents and vectorstore if they exist
        if hasattr(st.session_state.sqlite_chatbot, 'documents') and st.session_state.sqlite_chatbot.documents:
            temp_chatbot.documents = st.session_state.sqlite_chatbot.documents
            temp_chatbot.file_metadata = st.session_state.sqlite_chatbot.file_metadata
            temp_chatbot.db_connection = st.session_state.sqlite_chatbot.db_connection
            temp_chatbot.db_path = st.session_state.sqlite_chatbot.db_path
            temp_chatbot.sql_database = st.session_state.sqlite_chatbot.sql_database
            temp_chatbot.table_info = st.session_state.sqlite_chatbot.table_info
            # Rebuild the vectorstore with the new embeddings model
            temp_chatbot.build_vectorstore()
        
        st.session_state.sqlite_chatbot = temp_chatbot
        st.info(f"Model updated to {model_name}. This will apply to future questions.")

def handle_rag_config():
    """Handle RAG configuration in the sidebar"""
    # Add a toggle for debug mode
    debug_mode = st.checkbox("Debug Mode (Show RAG Context)", value=False)
    st.session_state.debug_mode = debug_mode
    
    # Add RAG controls when in debug mode
    if debug_mode:
        st.subheader("Retrieval Parameters")
        
        # Number of chunks to retrieve
        new_k = st.slider("Number of chunks (k)", 1, 20, st.session_state.k_value)
        if new_k != st.session_state.k_value:
            st.session_state.k_value = new_k
            st.info("K value updated. Changes will apply to the next query.")
        
        # Chunk size for text splitter
        new_chunk_size = st.slider("Chunk size", 200, 2000, st.session_state.chunk_size)
        if new_chunk_size != st.session_state.chunk_size:
            st.session_state.chunk_size = new_chunk_size
            st.warning("Chunk size updated. You'll need to reprocess files for this to take effect.")
        
        # Chunk overlap for text splitter
        new_chunk_overlap = st.slider("Chunk overlap", 0, 500, st.session_state.chunk_overlap)
        if new_chunk_overlap != st.session_state.chunk_overlap:
            st.session_state.chunk_overlap = new_chunk_overlap
            st.warning("Chunk overlap updated. You'll need to reprocess files for this to take effect.")

def handle_viz_options():
    """Handle visualization options in the sidebar"""
    viz_type = st.selectbox(
        "Default Chart Type",
        ["auto", "bar", "line", "scatter", "pie", "table"]
    )
    if "viz_type" not in st.session_state or st.session_state.viz_type != viz_type:
        st.session_state.viz_type = viz_type

def handle_chat_history():
    """Handle chat history management in the sidebar"""
    # Show available chat sessions with enhanced metadata
    sqlite_sessions = st.session_state.history_manager.list_sessions("sqlite")
    
    if sqlite_sessions:
        # Display count and create dropdown
        st.write(f"You have {len(sqlite_sessions)} saved chat sessions")
        
        # Create a new session option
        session_options = ["Current Session"] + [
            f"{s['session_id']} ({s['message_count']} msgs, {s.get('metadata', {}).get('model_name', 'unknown model')})" 
            for s in sqlite_sessions
        ]
        
        selected_session = st.selectbox(
            "Select a session to load",
            session_options
        )
        
        if selected_session != "Current Session" and "(" in selected_session:
            # Extract session ID from the selection
            session_id = selected_session.split(" ")[0]
            
            # Get session details for display
            session_info = next((s for s in sqlite_sessions if s["session_id"] == session_id), None)
            
            if session_info and "metadata" in session_info:
                # Display session metadata
                with st.expander("Session Details"):
                    metadata = session_info["metadata"]
                    st.write(f"**Model:** {metadata.get('model_name', 'Unknown')}")
                    st.write(f"**Last Updated:** {session_info.get('last_updated', 'Unknown')}")
                    st.write(f"**Messages:** {session_info.get('message_count', 0)}")
                    
                    # Show files that were processed
                    if "files_processed" in metadata and metadata["files_processed"]:
                        st.write("**Files:**")
                        for file in metadata["files_processed"]:
                            st.write(f"- {file}")
                    
                    # Show RAG parameters
                    st.write("**RAG Parameters:**")
                    st.write(f"- k_value: {metadata.get('k_value', 'N/A')}")
                    st.write(f"- chunk_size: {metadata.get('chunk_size', 'N/A')}")
                    st.write(f"- chunk_overlap: {metadata.get('chunk_overlap', 'N/A')}")
            
            # Button to load the selected session
            if st.button("Load Selected Session"):
                # Load the messages from this session
                loaded_messages, session_metadata = st.session_state.history_manager.load_chat_history(session_id)
                if loaded_messages:
                    # Update the current session ID and messages
                    st.session_state.sqlite_session_id = session_id
                    st.session_state.sqlite_messages = loaded_messages
                    
                    # Update RAG parameters from metadata if available
                    _restore_settings_from_metadata(session_metadata)
                    
                    st.success(f"Loaded chat session with {len(loaded_messages)} messages")
                    st.rerun()
                else:
                    st.error("Failed to load chat session")
                    
            # Button to view the selected chat history
            if st.button("👁️ View Chat History"):
                view_chat_history(session_id)

def handle_action_buttons():
    """Display and handle action buttons in the sidebar"""
    # Status information - check vector database
    if hasattr(st.session_state.sqlite_chatbot, 'vectorstore') and st.session_state.sqlite_chatbot.vectorstore:
        st.success("✅ Vector database is ready")
    else:
        st.warning("⚠️ No vector database available. Please upload and process files.")
    
    # Database connection status - check db_path instead of db_connection
    if hasattr(st.session_state.sqlite_chatbot, 'db_path') and st.session_state.sqlite_chatbot.db_path:
        st.success("✅ SQLite database is connected")
    else:
        st.warning("⚠️ No SQLite database connected. Please upload and process files.")
    
    # Add explicit save button
    if st.button("💾 Save Current Chat History"):
        if st.session_state.sqlite_messages:
            # Collect current metadata
            current_metadata = {
                "model_name": st.session_state.get("model_name", "gpt-3.5-turbo"),
                "k_value": st.session_state.k_value,
                "chunk_size": st.session_state.chunk_size,
                "chunk_overlap": st.session_state.chunk_overlap,
                "files_processed": list(st.session_state.sqlite_chatbot.file_metadata.keys()) if hasattr(st.session_state.sqlite_chatbot, 'file_metadata') else []
            }
            
            saved_path = st.session_state.history_manager.save_chat_history(
                st.session_state.sqlite_messages,
                "sqlite",
                st.session_state.sqlite_session_id,
                metadata=current_metadata
            )
            if saved_path:
                st.success(f"Chat history saved successfully! Session ID: {st.session_state.sqlite_session_id}")
            else:
                st.info("No changes to save.")
        else:
            st.info("No messages to save.")
    
    # Allow creating a new session
    if st.button("Start New Chat Session"):
        # Generate a new session ID
        st.session_state.sqlite_session_id = f"sqlite_{uuid.uuid4().hex[:8]}"
        # Clear messages
        st.session_state.sqlite_messages = []
        st.success("Started a new chat session")
        st.rerun()
    
    # Modify the Clear Chat button to preserve the saved history
    if st.button("Clear Chat Display"):
        save_current_chat_history()
        # Clear the display only
        st.session_state.sqlite_messages = []
        st.success("Chat display cleared! Your history has been saved and can be loaded again.")
    
    # Modify the Clear All Data button
    if st.button("Clear All Data"):
        save_current_chat_history()
        # Clear the chatbot data
        st.session_state.sqlite_chatbot.clear()
        # Clear the display
        st.session_state.sqlite_messages = []
        st.success("All data cleared! Your chat history has been saved and can be loaded again.")

def display_example_questions():
    """Display example question buttons"""
    st.subheader("Example Questions")
    example_qs = [
        "Write a query to list all tables and their row counts",
        "How many records are in each table?",
        "Show me the schema for all tables"
    ]

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(example_qs[0]):
            add_user_message(example_qs[0])
            
    with col2:
        if st.button(example_qs[1]):
            add_user_message(example_qs[1])
            
    with col3:
        if st.button(example_qs[2]):
            add_user_message(example_qs[2])

def add_user_message(prompt):
    """Add a user message to the chat history"""
    user_message = {
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.sqlite_messages.append(user_message)
    # Force a rerun to show the message immediately
    st.rerun()

def display_direct_sql_interface():
    """Display the direct SQL query interface section"""
    if hasattr(st.session_state.sqlite_chatbot, 'db_connection') and st.session_state.sqlite_chatbot.db_connection is not None:
        st.subheader("Direct SQL Query")
        
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
                
                assistant_message = {
                    "role": "assistant", 
                    "content": result_summary,
                    "model": st.session_state.get("model_name", "gpt-3.5-turbo"),
                    "timestamp": datetime.now().isoformat(),
                    "query": sql_query,
                    "has_data": True
                }
                st.session_state.sqlite_messages.append(assistant_message)
                
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
                
                assistant_message = {
                    "role": "assistant", 
                    "content": result,
                    "model": st.session_state.get("model_name", "gpt-3.5-turbo"),
                    "timestamp": datetime.now().isoformat(),
                    "query": sql_query,
                    "has_data": False
                }
                st.session_state.sqlite_messages.append(assistant_message)
                
                # Save the chat history
                save_current_chat_history()
                
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")


def add_assistant_response_with_context(response):
    """Add an assistant response with full context to chat history"""
    # Check if response was a dict with query information
    if isinstance(response, dict):
        assistant_message = {
            "role": "assistant", 
            "content": response.get("answer", str(response)),
            "model": st.session_state.get("model_name", "gpt-3.5-turbo"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add query information if available
        if "query" in response:
            assistant_message["query"] = response["query"]
        
        # Add data flag if available
        if "data" in response and isinstance(response["data"], pd.DataFrame):
            assistant_message["has_data"] = True
            
            # Use the serializable version of the data if available
            if "data_dict" in response:
                assistant_message["data_dict"] = response["data_dict"]
            else:
                # Create a serializable version of the DataFrame
                assistant_message["data_dict"] = {
                    'columns': response["data"].columns.tolist(),
                    'data': response["data"].values.tolist(),
                    'index': response["data"].index.tolist()
                }
            
            # Store this in the chatbot for later visualization (don't store in chat history)
            st.session_state.sqlite_chatbot.last_query_results = response["data"]
        
        # Add RAG parameters
        assistant_message["rag_params"] = {
            "k_value": st.session_state.k_value,
            "chunk_size": st.session_state.chunk_size,
            "chunk_overlap": st.session_state.chunk_overlap
        }
        
        # Add debug info if available
        if "debug_info" in response:
            assistant_message["debug_info"] = response["debug_info"]
    else:
        # Regular message
        assistant_message = {
            "role": "assistant", 
            "content": str(response),
            "model": st.session_state.get("model_name", "gpt-3.5-turbo"),
            "timestamp": datetime.now().isoformat(),
            "rag_params": {
                "k_value": st.session_state.k_value,
                "chunk_size": st.session_state.chunk_size,
                "chunk_overlap": st.session_state.chunk_overlap
            }
        }
    
    st.session_state.sqlite_messages.append(assistant_message)
    save_current_chat_history()


def generate_visualization_code(df, chart_type):
    """
    Generate Python code that would create the visualization
    
    Args:
        df: pandas DataFrame with the data
        chart_type: Type of chart to generate (bar, line, scatter, pie)
        
    Returns:
        str: Python code snippet
    """
    # Get column information
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Create sample code snippet
    code = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""
    
    # Add pandas DataFrame creation code with sample data
    code += "\n# Create DataFrame with your data\ndf = pd.DataFrame({\n"
    for col in columns:
        sample_values = str(df[col].head(3).tolist()).replace('[', '').replace(']', '')
        if pd.api.types.is_numeric_dtype(df[col]):
            code += f"    '{col}': [{sample_values}, ...],\n"
        else:
            code += f"    '{col}': ['{sample_values}', ...],\n"
    code += "})\n\n"
    
    # Add visualization code based on chart type
    if chart_type == "bar" and len(categorical_cols) > 0 and len(numeric_cols) > 0:
        x_col = categorical_cols[0]
        y_col = numeric_cols[0]
        
        code += f"""# Create bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='{x_col}', y='{y_col}', data=df)
plt.title('{y_col} by {x_col}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""

    elif chart_type == "line" and len(numeric_cols) > 0:
        y_col = numeric_cols[0]
        x_col = columns[0] if columns[0] != y_col else columns[1] if len(columns) > 1 else "index"
        
        if x_col == "index":
            code += f"""# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(df['{y_col}'])
plt.title('{y_col} over Index')
plt.ylabel('{y_col}')
plt.xlabel('Index')
plt.grid(True)
plt.tight_layout()
plt.show()"""
        else:
            code += f"""# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(df['{x_col}'], df['{y_col}'])
plt.title('{y_col} vs {x_col}')
plt.ylabel('{y_col}')
plt.xlabel('{x_col}')
plt.grid(True)
plt.tight_layout()
plt.show()"""

    elif chart_type == "scatter" and len(numeric_cols) >= 2:
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        
        code += f"""# Create scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='{x_col}', y='{y_col}', data=df)
plt.title('{y_col} vs {x_col}')
plt.tight_layout()
plt.show()"""

    elif chart_type == "pie" and len(categorical_cols) > 0 and len(numeric_cols) > 0:
        cat_col = categorical_cols[0]
        value_col = numeric_cols[0]
        
        code += f"""# Create pie chart
plt.figure(figsize=(10, 6))
df_grouped = df.groupby('{cat_col}')['{value_col}'].sum()
plt.pie(df_grouped, labels=df_grouped.index, autopct='%1.1f%%')
plt.title('{value_col} Distribution by {cat_col}')
plt.axis('equal')
plt.tight_layout()
plt.show()"""
        
    else:
        # Default to a generic histogram of the first numeric column
        if numeric_cols:
            col = numeric_cols[0]
            code += f"""# Create histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['{col}'], kde=True)
plt.title('Distribution of {col}')
plt.tight_layout()
plt.show()"""
        else:
            # Fallback if no numeric columns
            code += """# No numeric columns found for visualization
# Display the data instead
print(df)"""
    
    return code


def process_chat_message(prompt):
    """Process a chat message and generate a response"""
    with st.chat_message("assistant"):
        # Initialize debug info dictionary
        debug_info = {
            "prompt": prompt,
            "stage": "starting",
            "timestamp": datetime.now().isoformat()
        }
        
        with st.spinner("Thinking..."):
            try:
                # Check if database is connected
                if not hasattr(st.session_state.sqlite_chatbot, 'db_path') or not st.session_state.sqlite_chatbot.db_path:
                    answer_text = "Please upload and process a SQLite database file first."
                    st.warning(answer_text)
                    add_assistant_message(answer_text)
                    return
                
                # Get current model information
                current_model = st.session_state.get("model_name", "gpt-3.5-turbo")
                debug_info["model"] = current_model
                
                # Detect if this is a potential SQL or visualization request
                sql_keywords = ["sql", "query", "table", "database", "select", "count", "find", "show me", "list"]
                viz_keywords = ["chart", "plot", "graph", "visualize", "visualization", "show", "display"]
                
                is_sql_query = any(keyword in prompt.lower() for keyword in sql_keywords)
                is_viz_query = any(keyword in prompt.lower() for keyword in viz_keywords)
                
                debug_info["is_sql_query"] = is_sql_query
                debug_info["is_viz_query"] = is_viz_query
                
                # First approach: Always try SQL generation for database-related questions
                if is_sql_query or is_viz_query or "how many" in prompt.lower():
                    debug_info["stage"] = "attempting_sql_generation"
                    
                    try:
                        # Generate SQL
                        generated_query = st.session_state.sqlite_chatbot.generate_sql_query(prompt)
                        debug_info["generated_query"] = generated_query
                        
                        # Check if generation succeeded
                        if not generated_query.startswith("Error"):
                            debug_info["stage"] = "sql_generation_succeeded"
                            
                            # Execute the query
                            query_result = st.session_state.sqlite_chatbot.execute_sql_query(generated_query)
                            debug_info["query_result_type"] = str(type(query_result))
                            
                            if isinstance(query_result, pd.DataFrame):
                                debug_info["stage"] = "query_returned_dataframe"
                                debug_info["row_count"] = len(query_result)
                                debug_info["column_count"] = len(query_result.columns)
                                
                                # Format the answer WITHOUT including the table in the text
                                num_rows = len(query_result)
                                answer = f"I've executed the following SQL query based on your question:\n\n```sql\n{generated_query}\n```\n\nThe query returned {num_rows} rows."
                                
                                # Display the answer
                                st.markdown(answer)
                                
                                # ONLY display the dataframe here - not in the text
                                st.write("Here's the result:")
                                st.dataframe(query_result)
                                
                                # Create visualization for data if requested or data looks suitable for visualization
                                # Check if data is suitable for visualization (has numeric columns)
                                has_numeric_cols = any(pd.api.types.is_numeric_dtype(query_result[col]) for col in query_result.columns)
                                
                                # Always offer visualization if requested or if data is suitable
                                if is_viz_query or has_numeric_cols:
                                    # Create visualization section with tabs
                                    st.subheader("Visualization")
                                    
                                    # Create tabs for visualization and code
                                    viz_tab, code_tab = st.tabs(["Visualization", "Python Code"])
                                    
                                    # Get the selected visualization type
                                    viz_type = st.session_state.viz_type if hasattr(st.session_state, 'viz_type') else 'auto'
                                    
                                    # Create a unique ID for this visualization
                                    viz_id = f"viz_{datetime.now().timestamp()}"
                                    
                                    # Show visualization in the viz tab
                                    with viz_tab:
                                        fig = generate_visualization(query_result, chart_type=viz_type)
                                        if isinstance(fig, go.Figure):
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Generate and store Python code
                                            python_code = generate_visualization_code(query_result, viz_type)
                                            st.session_state[f"code_{viz_id}"] = python_code
                                        else:
                                            st.warning(f"Couldn't generate visualization: {fig}")
                                    
                                    # Show code in the code tab
                                    with code_tab:
                                        if f"code_{viz_id}" in st.session_state:
                                            st.code(st.session_state[f"code_{viz_id}"], language="python")
                                        else:
                                            st.info("Select the Visualization tab to see the visualization first.")
                                
                                # Add download button
                                csv = query_result.to_csv(index=False)
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=csv,
                                    file_name="query_results.csv",
                                    mime="text/csv"
                                )
                                
                                # Create serializable version of the DataFrame for storage
                                data_dict = {
                                    'columns': query_result.columns.tolist(),
                                    'data': query_result.values.tolist(),
                                    'index': query_result.index.tolist()
                                }
                                
                                # Add the assistant response to chat history
                                response_with_data = {
                                    "answer": answer,
                                    "query": generated_query,
                                    "data": query_result,
                                    "data_dict": data_dict,
                                    "query_type": "visualization" if is_viz_query else "sql_execution",
                                    "debug_info": debug_info
                                }
                                add_assistant_response_with_context(response_with_data)
                                return
                            else:
                                debug_info["stage"] = "query_returned_non_dataframe"
                                debug_info["result"] = str(query_result)
                                
                                # Non-SELECT query result
                                answer = f"I've executed the following SQL query based on your question:\n\n```sql\n{generated_query}\n```\n\nResult: {query_result}"
                                st.markdown(answer)
                                
                                # Add the assistant response to chat history
                                response_without_data = {
                                    "answer": answer,
                                    "query": generated_query,
                                    "query_type": "sql_execution",
                                    "debug_info": debug_info
                                }
                                add_assistant_response_with_context(response_without_data)
                                return
                        else:
                            # SQL generation had an error, log and continue to RAG
                            debug_info["stage"] = "sql_generation_failed"
                            debug_info["sql_error"] = generated_query
                            
                            # Only show warning in debug mode
                            if st.session_state.debug_mode:
                                st.warning(f"SQL generation failed, falling back to RAG: {generated_query}")
                    
                    except Exception as sql_error:
                        import traceback
                        error_traceback = traceback.format_exc()
                        
                        # Log the error but continue to RAG as fallback
                        debug_info["stage"] = "sql_execution_error"
                        debug_info["sql_error"] = str(sql_error)
                        debug_info["traceback"] = error_traceback
                        
                        # Only show error in debug mode
                        if st.session_state.debug_mode:
                            st.error(f"SQL execution error, falling back to RAG: {str(sql_error)}")
                            st.code(error_traceback)  # Avoid nesting in expander
                
                # If we get here, we're handling a general question with RAG
                debug_info["stage"] = "processing_with_rag"
                
                # Get response with RAG
                response = st.session_state.sqlite_chatbot.ask(prompt, return_context=True)
                debug_info["rag_response_received"] = True
                
                # Handle the response display
                if isinstance(response, dict) and "answer" in response:
                    st.markdown(response["answer"])
                    
                    # If there's query data, show it
                    if "data" in response and isinstance(response["data"], pd.DataFrame):
                        # Display the dataframe separately
                        st.write("Here's the result:")
                        st.dataframe(response["data"])
                        
                        # Check if data is suitable for visualization (has numeric columns)
                        has_numeric_cols = any(pd.api.types.is_numeric_dtype(response["data"][col]) for col in response["data"].columns)
                        
                        # Add visualization if requested or data is suitable
                        if is_viz_query or has_numeric_cols or ("query_type" in response and response["query_type"] == "visualization"):
                            # Create visualization section
                            st.subheader("Visualization")
                            
                            # Create tabs for visualization and code
                            viz_tab, code_tab = st.tabs(["Visualization", "Python Code"])
                            
                            # Get visualization type
                            viz_type = st.session_state.viz_type if hasattr(st.session_state, 'viz_type') else 'auto'
                            
                            # Create a unique ID for this visualization
                            viz_id = f"viz_{datetime.now().timestamp()}"
                            
                            # Show visualization in the viz tab
                            with viz_tab:
                                fig = generate_visualization(response["data"], chart_type=viz_type)
                                if isinstance(fig, go.Figure):
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Generate and store Python code
                                    python_code = generate_visualization_code(response["data"], viz_type)
                                    st.session_state[f"code_{viz_id}"] = python_code
                                else:
                                    st.warning(f"Couldn't generate visualization: {fig}")
                            
                            # Show code in the code tab
                            with code_tab:
                                if f"code_{viz_id}" in st.session_state:
                                    st.code(st.session_state[f"code_{viz_id}"], language="python")
                                else:
                                    st.info("Select the Visualization tab to see the visualization first.")
                            
                            # Add download button
                            csv = response["data"].to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv"
                            )
                    
                    # Add response to chat history with debug info
                    response["debug_info"] = debug_info
                    add_assistant_response_with_context(response)
                else:
                    st.markdown(str(response))
                    
                    # Add to chat history
                    add_assistant_message(str(response))
                
            except Exception as e:
                # Capture the full error for debugging
                import traceback
                error_traceback = traceback.format_exc()
                error_msg = f"An error occurred: {str(e)}"
                
                st.error(error_msg)
                
                # Show detailed error in debug mode without nesting expanders
                if st.session_state.debug_mode:
                    st.subheader("Error Details")
                    st.code(error_traceback)
                
                # Add error to chat history
                add_assistant_message(f"{error_msg}\n\nPlease try again with a different question.")

def display_chat_messages():
    """Display the chat message history with improved data handling"""
    for message in st.session_state.sqlite_messages:
        with st.chat_message(message["role"]):
            # Display the message content
            st.markdown(message["content"])
            
            # Check if this message has query data to visualize
            if message["role"] == "assistant" and "query" in message:
                # If we have data directly in the message
                if "data_dict" in message:
                    # Convert the serialized data back to a DataFrame
                    try:
                        data_dict = message["data_dict"]
                        df = pd.DataFrame(data=data_dict['data'], columns=data_dict['columns'])
                        
                        # Display the DataFrame
                        st.write("Here's the result:")
                        st.dataframe(df)
                        
                        # Check if data is suitable for visualization (has numeric columns)
                        has_numeric_cols = any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
                        
                        # Create visualization section if suitable
                        if has_numeric_cols:
                            # Create a unique ID for this message
                            msg_key = str(hash(message.get('timestamp', '')))
                            
                            # Create tabs for visualization and code
                            viz_tab, code_tab = st.tabs(["Visualization", "Python Code"])
                            
                            with viz_tab:
                                # Select visualization type
                                viz_options = ["bar", "line", "scatter", "pie", "table"]
                                selected_viz = st.selectbox(
                                    f"Visualization Type", 
                                    viz_options, 
                                    key=f"viz_type_{msg_key}"
                                )
                                
                                # Generate Visualization button
                                if st.button("Generate Visualization", key=f"gen_viz_{msg_key}"):
                                    fig = generate_visualization(df, chart_type=selected_viz)
                                    if isinstance(fig, go.Figure):
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Generate and store Python code
                                        python_code = generate_visualization_code(df, selected_viz)
                                        st.session_state[f"code_{msg_key}"] = python_code
                                    else:
                                        st.warning(f"Couldn't generate visualization: {fig}")
                            
                            # Show code in the code tab
                            with code_tab:
                                if f"code_{msg_key}" in st.session_state:
                                    st.code(st.session_state[f"code_{msg_key}"], language="python")
                                else:
                                    st.info("Generate a visualization first to see the corresponding Python code.")
                                    
                        # Add download option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv",
                            key=f"download_{hash(message.get('timestamp', ''))}"
                        )
                    except Exception as e:
                        st.error(f"Error displaying data: {str(e)}")
                
                # If no data_dict, check for last_query_results
                elif hasattr(st.session_state.sqlite_chatbot, 'last_query_results') and \
                     st.session_state.sqlite_chatbot.last_query_results is not None and \
                     message.get("has_data", False):
                    
                    df = st.session_state.sqlite_chatbot.last_query_results
                    
                    # Display the results
                    st.write("Query Results:")
                    st.dataframe(df)
                    
                    # Check if data is suitable for visualization (has numeric columns)
                    has_numeric_cols = any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
                    
                    # Create visualization section if suitable
                    if has_numeric_cols:
                        # Create a unique ID for this message
                        msg_key = str(hash(message.get('timestamp', '')))
                        
                        # Create tabs for visualization and code
                        viz_tab, code_tab = st.tabs(["Visualization", "Python Code"])
                        
                        with viz_tab:
                            # Select visualization type
                            viz_options = ["bar", "line", "scatter", "pie", "table"]
                            selected_viz = st.selectbox(
                                f"Visualization Type", 
                                viz_options, 
                                key=f"viz_type_{msg_key}"
                            )
                            
                            # Generate Visualization button
                            if st.button("Generate Visualization", key=f"gen_viz_{msg_key}"):
                                fig = generate_visualization(df, chart_type=selected_viz)
                                if isinstance(fig, go.Figure):
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Generate and store Python code
                                    python_code = generate_visualization_code(df, selected_viz)
                                    st.session_state[f"code_{msg_key}"] = python_code
                                else:
                                    st.warning(f"Couldn't generate visualization: {fig}")
                        
                        # Show code in the code tab
                        with code_tab:
                            if f"code_{msg_key}" in st.session_state:
                                st.code(st.session_state[f"code_{msg_key}"], language="python")
                            else:
                                st.info("Generate a visualization first to see the corresponding Python code.")
            
            # Display metadata for assistant messages
            if message["role"] == "assistant" and "model" in message:
                with st.expander("Message Details", expanded=False):
                    # Show model used
                    st.caption(f"**Model**: {message.get('model', 'Unknown')}")
                    
                    # Show timestamp if available
                    if "timestamp" in message:
                        st.caption(f"**Time**: {message['timestamp']}")
                    
                    # Show SQL query if available
                    if "query" in message:
                        st.caption("**SQL Query**:")
                        st.code(message["query"], language="sql")
                    
                    # Show RAG parameters if available
                    if "rag_params" in message:
                        st.caption("**RAG Parameters**:")
                        params = message["rag_params"]
                        st.caption(f"- k_value: {params.get('k_value', 'N/A')}")
                        st.caption(f"- chunk_size: {params.get('chunk_size', 'N/A')}")
                        st.caption(f"- chunk_overlap: {params.get('chunk_overlap', 'N/A')}")                        

def handle_chat_input():
    """Handle the chat input and generate responses"""
    if prompt := st.chat_input("Ask a question about your SQLite data"):
        # Add user message to chat history
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.sqlite_messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        process_chat_message(prompt)
        
        # No rerun here as it can interrupt the processing
        # The save_current_chat_history function will be called by add_assistant_message



def process_debug_mode_response(prompt, is_viz_query):
    """Process a response in debug mode with detailed information"""
    response = st.session_state.sqlite_chatbot.ask(prompt, return_context=True)
    
    # Create tabs for answer and debugging info
    answer_tab, context_tab, metrics_tab, sql_tab = st.tabs([
        "Answer", "Retrieved Context", "RAG Metrics", "SQL Debug"
    ])
    
    with answer_tab:
        if isinstance(response, dict) and "answer" in response:
            st.markdown(response["answer"])
            
            # If this is a visualization query that generated SQL, show a chart
            if is_viz_query and "data" in response and isinstance(response["data"], pd.DataFrame):
                # Create visualization
                viz_type = st.session_state.viz_type
                fig = generate_visualization(response["data"], chart_type=viz_type)
                if isinstance(fig, go.Figure):
                    st.plotly_chart(fig, use_container_width=True)
            
            # Store the answer for chat history
            answer_text = response["answer"]
        else:
            st.markdown(str(response))
            
            # Store the response for chat history
            answer_text = str(response)
    
    with context_tab:
        if isinstance(response, dict) and "retrieved_context" in response:
            st.text_area("Retrieved Documents", 
                        response["retrieved_context"], 
                        height=400)
        else:
            st.text("No context retrieved for this query.")
    
    with metrics_tab:
        # Show RAG metrics and stats
        st.subheader("RAG Retrieval Metrics")
        
        # Current configuration
        st.write("### Current Configuration")
        st.json({
            "k_value": st.session_state.k_value,
            "chunk_size": st.session_state.chunk_size,
            "chunk_overlap": st.session_state.chunk_overlap,
            "model": st.session_state.model_name
        })
        
        # Retrieved chunks stats
        if isinstance(response, dict) and "retrieved_context" in response and response["retrieved_context"]:
            st.write("### Retrieved Content Stats")
            chunks = response["retrieved_context"].split("\n\n---\n\n")
            chunk_lengths = [len(chunk) for chunk in chunks]
            
            st.json({
                "num_chunks_retrieved": len(chunks),
                "avg_chunk_length": sum(chunk_lengths) / max(1, len(chunk_lengths)),
                "min_chunk_length": min(chunk_lengths, default=0),
                "max_chunk_length": max(chunk_lengths, default=0),
                "total_context_length": len(response["retrieved_context"])
            })
        else:
            st.write("No retrieval metrics available for this query.")
    
    with sql_tab:
        st.subheader("SQL Query Information")
        
        if isinstance(response, dict) and "query" in response:
            st.write("### Generated SQL Query")
            st.code(response["query"], language="sql")
            
            if "data" in response and isinstance(response["data"], pd.DataFrame):
                st.write("### Query Results")
                st.dataframe(response["data"])
        else:
            st.write("No SQL query was generated for this question.")
    
    # Add the assistant response to chat history
    add_assistant_response_with_context(response)

def process_regular_response(prompt, is_viz_query):
    """Process a response in normal mode"""
    response = st.session_state.sqlite_chatbot.ask(prompt, return_context=True)
    
    if isinstance(response, dict):
        answer_text = response["answer"]
        
        # Check if this is a visualization query that generated SQL
        if is_viz_query and "data" in response and isinstance(response["data"], pd.DataFrame):
            st.markdown(answer_text)
            
            # Create visualization
            viz_type = st.session_state.viz_type
            st.subheader("Visualization")
            fig = generate_visualization(response["data"], chart_type=viz_type)
            if isinstance(fig, go.Figure):
                st.plotly_chart(fig, use_container_width=True)
            
            # Add download button for the data
            csv = response["data"].to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv"
            )
        elif "query" in response and "data" in response and isinstance(response["data"], pd.DataFrame):
            # Show the answer with SQL results
            st.markdown(answer_text)
            
            # Show the data in a collapsible section
            with st.expander("Query Results", expanded=False):
                st.dataframe(response["data"])
                
                # Add download button for the data
                csv = response["data"].to_csv(index=False)
                st.download_button(
                    label="Download Data as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
        else:
            # Regular text answer
            st.markdown(answer_text)
    else:
        # Fallback in case we get a string
        answer_text = response
        st.markdown(answer_text)
    
    # Add the assistant response to chat history
    add_assistant_response_with_context(response)

def add_assistant_message(answer_text):
    """Add a simple assistant message to chat history"""
    assistant_message = {
        "role": "assistant", 
        "content": answer_text,
        "model": st.session_state.get("model_name", "gpt-3.5-turbo"),
        "timestamp": datetime.now().isoformat(),
        "rag_params": {
            "k_value": st.session_state.k_value,
            "chunk_size": st.session_state.chunk_size,
            "chunk_overlap": st.session_state.chunk_overlap
        }
    }
    st.session_state.sqlite_messages.append(assistant_message)
    save_current_chat_history()



def direct_sql_generation(question):
    """Generate SQL directly using the LLM without going through complex chains"""
    if not hasattr(st.session_state.sqlite_chatbot, 'sql_database'):
        return "No database connected. Please upload a SQLite database file first."
        
    try:
        # Get schema information
        schema_info = st.session_state.sqlite_chatbot.sql_database.get_table_info()
        
        # Create a simple, direct prompt
        prompt = f"""You are a SQL expert. Generate a SQL query for SQLite that answers this question:
"{question}"

Here's the database schema:
{schema_info}

Your SQL query (just return the query, no explanation):
"""
        
        # Get the current model
        model_name = st.session_state.get("model_name", "gpt-3.5-turbo")
        
        # Create messages list
        messages = [
            {"role": "system", "content": "You are an expert SQL query generator. Output only the SQL query with no explanation or markdown formatting."},
            {"role": "user", "content": prompt}
        ]
        
        # Create a ChatOpenAI instance
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.1
        )
        
        # Call the LLM directly
        with st.spinner("Generating SQL query..."):
            response = llm.invoke(messages)
            
        # Extract just the SQL query
        sql_query = response.content.strip()
        
        # Clean up any markdown formatting or explanations
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
            
        # Execute the query if it looks valid
        if sql_query.lower().startswith(("select", "with")):
            with st.spinner("Executing query..."):
                result = st.session_state.sqlite_chatbot.execute_sql_query(sql_query)
                
            return {
                "query": sql_query,
                "result": result
            }
        
        return {
            "query": sql_query,
            "result": None
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def debug_sql_generation():
    """Add a section for direct SQL generation debugging"""
    st.subheader("Debug SQL Generation")
    
    st.write("This section lets you test SQL generation directly, bypassing the RAG system.")
    
    question = st.text_input("Enter a question for direct SQL generation")
    
    if st.button("Generate SQL Directly"):
        result = direct_sql_generation(question)
        
        if isinstance(result, dict) and "query" in result:
            st.write("### Generated SQL")
            st.code(result["query"], language="sql")
            
            if "result" in result and isinstance(result["result"], pd.DataFrame):
                st.write("### Query Result")
                st.dataframe(result["result"])
                
                st.download_button(
                    "Download Results as CSV",
                    result["result"].to_csv(index=False).encode('utf-8'),
                    "sql_results.csv",
                    "text/csv",
                    key="download-direct-sql"
                )
        elif isinstance(result, dict) and "error" in result:
            st.error(f"Error: {result['error']}")
            with st.expander("Traceback"):
                st.code(result["traceback"])
        else:
            st.write(result)
#------------------------------------------------------------------------
# Main Application
#------------------------------------------------------------------------

def main():
    """Main application entry point"""
    st.title("🗃️ SQLite Database RAG")
    st.subheader("Chat with your SQLite Databases")
    
    # Initialize session state
    initialize_session_state()
    
    # If viewing history, show it in the main content area
    if st.session_state.viewing_history and st.session_state.history_session_id:
        display_history_view()
        display_database_info()
        return
        
    # Display database information if connected
    display_database_info()
        
    # Display sidebar with controls
    display_sidebar()
    
    # Display example questions
    display_example_questions()
    
    # Display SQL query interface
    display_direct_sql_interface()
    
    # Add debug SQL generation section when in debug mode
    if st.session_state.debug_mode:
        debug_sql_generation()


    # Main chat interface
    st.subheader("Chat with your Database")
    
    # Display chat messages
    display_chat_messages()
    
    # Handle chat input
    handle_chat_input()

if __name__ == "__main__":
    main()