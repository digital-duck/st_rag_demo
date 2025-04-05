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
        with st.expander("Database Tables", expanded=False):
            for table_name, info in st.session_state.sqlite_chatbot.table_info.items():
                st.write(f"**{table_name}** ({info['row_count']} rows)")
                cols = ", ".join([f"{col[1]} ({col[2]})" for col in info['columns']])
                st.write(f"Columns: {cols}")
                
                # Add option to show sample data
                with st.expander(f"Sample data from {table_name}", expanded=False):
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
    """Handle file upload in the sidebar"""
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
                    with st.spinner("Building vector database..."):
                        if st.session_state.sqlite_chatbot.build_vectorstore():
                            st.success(f"Successfully processed {table_count} tables into document chunks!")
                        else:
                            st.error("Error building the vector database.")
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
    # Status information
    if hasattr(st.session_state.sqlite_chatbot, 'vectorstore') and st.session_state.sqlite_chatbot.vectorstore:
        st.success("✅ Vector database is ready")
    else:
        st.warning("⚠️ No vector database available. Please upload and process files.")
    
    # Database connection status
    if hasattr(st.session_state.sqlite_chatbot, 'db_connection') and st.session_state.sqlite_chatbot.db_connection is not None:
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