# Location: utils/sidebar.py
# Sidebar utilities for RAG applications

import os
import uuid
import pandas as pd
import streamlit as st
from utils.ui_helpers import restore_settings_from_metadata, save_chat_history, view_chat_history

def display_sidebar(app_type, chatbot_key, messages_key, session_id_key, chatbot_class, file_types=None):
    """
    Display the sidebar with all options and controls
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        chatbot_key (str): Key for chatbot in session state
        messages_key (str): Key for messages in session state
        session_id_key (str): Key for session ID in session state
        chatbot_class: The chatbot class to instantiate when model changes
        file_types (list): List of file extensions to accept in uploader
    """
    with st.sidebar:
        st.header(f"Upload {app_type.upper()} Files")
        handle_file_upload(app_type, chatbot_key, file_types)
        
        st.header("Model Selection")
        handle_model_selection(app_type, chatbot_key, chatbot_class)
        
        st.header("RAG Configuration")
        handle_rag_config()
        
        if app_type == 'sqlite' or app_type == 'csv':
            st.subheader("Visualization Options")
            handle_viz_options()
        
        st.header("Chat History")
        handle_chat_history(app_type, session_id_key, messages_key, chatbot_key)
        
        handle_action_buttons(app_type, chatbot_key, messages_key, session_id_key)
        
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("Please set your OpenAI API key in a .env file or as an environment variable.")

        # Display current session info
        st.caption(f"Current Session ID: {st.session_state[session_id_key]}")

def handle_file_upload(app_type, chatbot_key, file_types=None):
    """
    Handle file upload in the sidebar
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        chatbot_key (str): Key for chatbot in session state
        file_types (list): List of file extensions to accept in uploader
    """
    # Set default file types based on app type if not specified
    if file_types is None:
        if app_type == 'pdf':
            file_types = ['pdf']
        elif app_type == 'csv':
            file_types = ['csv']
        elif app_type == 'sqlite':
            file_types = ['db', 'sqlite', 'sqlite3']
    
    # SQLite specific: check for previously processed databases
    if app_type == 'sqlite' and hasattr(st.session_state[chatbot_key], 'get_available_databases'):
        available_dbs = st.session_state[chatbot_key].get_available_databases()
        
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
                        table_count = st.session_state[chatbot_key].load_database(
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
        f"Upload {app_type.upper()} files", 
        type=file_types, 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        process_clicked = st.button(f"Process {app_type.upper()} Files")
        
        if process_clicked:
            with st.spinner("Processing files..."):
                # Clear existing data before processing new files
                st.session_state[chatbot_key].clear()
                
                if app_type == 'pdf':
                    # Process PDF files
                    doc_count = 0
                    for file in uploaded_files:
                        count = st.session_state[chatbot_key].process_pdf(file)
                        st.write(f"Processed {file.name}: {count} pages")
                        doc_count += count
                    
                    if doc_count > 0:
                        with st.spinner("Building vector database..."):
                            if st.session_state[chatbot_key].build_vectorstore():
                                st.success(f"Successfully processed {doc_count} pages into document chunks!")
                            else:
                                st.error("Error building the vector database.")
                    else:
                        st.warning("No documents were processed.")
                        
                elif app_type == 'csv':
                    # Process CSV files
                    file_count = 0
                    for file in uploaded_files:
                        try:
                            # Preview the CSV data
                            df = pd.read_csv(file)
                            st.write(f"Processing {file.name}: {len(df)} rows × {len(df.columns)} columns")
                            
                            # Display sample of the data
                            with st.expander(f"Preview of {file.name}"):
                                st.dataframe(df.head())
                            
                            # Process the file
                            count = st.session_state[chatbot_key].process_csv(file)
                            st.write(f"✅ Processed {file.name}: {count} records")
                            file_count += 1
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                    
                    if file_count > 0:
                        with st.spinner("Building vector database..."):
                            if st.session_state[chatbot_key].build_vectorstore():
                                st.success(f"Successfully processed {file_count} CSV files!")
                            else:
                                st.error("Error building the vector database.")
                    else:
                        st.warning("No CSV files were processed.")
                        
                elif app_type == 'sqlite':
                    # Process SQLite files
                    table_count = 0
                    for file in uploaded_files:
                        count = st.session_state[chatbot_key].process_sqlite(file)
                        st.write(f"Processed {file.name}: {count} tables")
                        table_count += count
                    
                    if table_count > 0:
                        st.success(f"Successfully processed {table_count} tables into document chunks!")
                    else:
                        st.warning("No database tables were processed.")

def handle_model_selection(app_type, chatbot_key, chatbot_class):
    """
    Handle model selection in the sidebar
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        chatbot_key (str): Key for chatbot in session state
        chatbot_class: The chatbot class to instantiate when model changes
    """
    model_name = st.selectbox(
        "Select OpenAI Model",
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"]
    )
    
    if "model_name" not in st.session_state or st.session_state.model_name != model_name:
        st.session_state.model_name = model_name
        # Create a new chatbot instance with the selected model
        temp_chatbot = chatbot_class(model_name=model_name)
        
        # Transfer documents and vectorstore if they exist
        if hasattr(st.session_state[chatbot_key], 'documents') and st.session_state[chatbot_key].documents:
            temp_chatbot.documents = st.session_state[chatbot_key].documents
            temp_chatbot.file_metadata = st.session_state[chatbot_key].file_metadata
            
            # Transfer app-specific attributes
            if app_type == 'csv' and hasattr(st.session_state[chatbot_key], 'data_frames'):
                temp_chatbot.data_frames = st.session_state[chatbot_key].data_frames
            
            if app_type == 'sqlite':
                if hasattr(st.session_state[chatbot_key], 'db_connection'):
                    temp_chatbot.db_connection = st.session_state[chatbot_key].db_connection
                if hasattr(st.session_state[chatbot_key], 'db_path'):
                    temp_chatbot.db_path = st.session_state[chatbot_key].db_path
                if hasattr(st.session_state[chatbot_key], 'sql_database'):
                    temp_chatbot.sql_database = st.session_state[chatbot_key].sql_database
                if hasattr(st.session_state[chatbot_key], 'table_info'):
                    temp_chatbot.table_info = st.session_state[chatbot_key].table_info
            
            # Rebuild the vectorstore with the new embeddings model
            temp_chatbot.build_vectorstore()
        
        st.session_state[chatbot_key] = temp_chatbot
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

def handle_chat_history(app_type, session_id_key, messages_key, chatbot_key):
    """
    Handle chat history management in the sidebar
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        session_id_key (str): Key for session ID in session state
        messages_key (str): Key for messages in session state
        chatbot_key (str): Key for chatbot in session state
    """
    # Show available chat sessions with enhanced metadata
    sessions = st.session_state.history_manager.list_sessions(app_type)
    
    if sessions:
        # Display count and create dropdown
        st.write(f"You have {len(sessions)} saved chat sessions")
        
        # Create a new session option
        session_options = ["Current Session"] + [
            f"{s['session_id']} ({s['message_count']} msgs, {s.get('metadata', {}).get('model_name', 'unknown model')})" 
            for s in sessions
        ]
        
        selected_session = st.selectbox(
            "Select a session to load",
            session_options
        )
        
        if selected_session != "Current Session" and "(" in selected_session:
            # Extract session ID from the selection
            session_id = selected_session.split(" ")[0]
            
            # Get session details for display
            session_info = next((s for s in sessions if s["session_id"] == session_id), None)
            
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
                    st.session_state[session_id_key] = session_id
                    st.session_state[messages_key] = loaded_messages
                    
                    # Update RAG parameters from metadata if available
                    restore_settings_from_metadata(session_metadata)
                    
                    st.success(f"Loaded chat session with {len(loaded_messages)} messages")
                    st.rerun()
                else:
                    st.error("Failed to load chat session")
                    
            # Button to view the selected chat history
            if st.button("👁️ View Chat History"):
                view_chat_history(session_id)

def handle_action_buttons(app_type, chatbot_key, messages_key, session_id_key):
    """
    Display and handle action buttons in the sidebar
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        chatbot_key (str): Key for chatbot in session state
        messages_key (str): Key for messages in session state
        session_id_key (str): Key for session ID in session state
    """
    # Status information - check vector database
    if hasattr(st.session_state[chatbot_key], 'vectorstore') and st.session_state[chatbot_key].vectorstore:
        st.success("✅ Vector database is ready")
    else:
        st.warning("⚠️ No vector database available. Please upload and process files.")
    
    # Check additional status for SQLite
    if app_type == 'sqlite':
        # Check database connection status
        if hasattr(st.session_state[chatbot_key], 'db_path') and st.session_state[chatbot_key].db_path:
            st.success("✅ SQLite database is connected")
        else:
            st.warning("⚠️ No SQLite database connected. Please upload and process files.")
    
    # Add explicit save button
    if st.button("💾 Save Current Chat History"):
        if st.session_state[messages_key]:
            saved_path = save_chat_history(app_type, session_id_key, messages_key, chatbot_key)
            if saved_path:
                st.success(f"Chat history saved successfully! Session ID: {st.session_state[session_id_key]}")
            else:
                st.info("No changes to save.")
        else:
            st.info("No messages to save.")
    
    # Allow creating a new session
    if st.button("Start New Chat Session"):
        # Generate a new session ID
        st.session_state[session_id_key] = f"{app_type}_{uuid.uuid4().hex[:8]}"
        # Clear messages
        st.session_state[messages_key] = []
        st.success("Started a new chat session")
        st.rerun()
    
    # Modify the Clear Chat button to preserve the saved history
    if st.button("Clear Chat Display"):
        # Keep a backup of messages before clearing if they should be auto-saved
        if st.session_state[messages_key]:
            save_chat_history(app_type, session_id_key, messages_key, chatbot_key)
        
        # Clear the display only
        st.session_state[messages_key] = []
        st.success("Chat display cleared! Your history has been saved and can be loaded again.")
    
    # Modify the Clear All Data button
    if st.button("Clear All Data"):
        # Auto-save before clearing if there's anything to save
        if st.session_state[messages_key]:
            save_chat_history(app_type, session_id_key, messages_key, chatbot_key)
        
        # Clear the chatbot data
        st.session_state[chatbot_key].clear()
        # Clear the display
        st.session_state[messages_key] = []
        st.success("All data cleared! Your chat history has been saved and can be loaded again.")