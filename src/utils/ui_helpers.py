# Location: utils/ui_helpers.py
# Common UI helper functions for RAG applications
# Updated with fixed example questions implementation

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def initialize_session_state(app_type, chatbot_class, history_manager):
    """
    Initialize all session state variables for a RAG app
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        chatbot_class: The chatbot class to instantiate
        history_manager: ChatHistoryManager instance
        
    Returns:
        tuple: (app_type, session_id_key, chatbot_key, messages_key)
    """
    import uuid
    from models.auto_save import AutoSave
    
    # Initialize session state for viewing history
    if "viewing_history" not in st.session_state:
        st.session_state.viewing_history = False
    if "history_session_id" not in st.session_state:
        st.session_state.history_session_id = None
    
    # Initialize the chat history manager
    if "history_manager" not in st.session_state:
        st.session_state.history_manager = history_manager
        
    # Initialize auto-save utility
    if "auto_save" not in st.session_state:
        st.session_state.auto_save = AutoSave(st.session_state.history_manager)
        st.session_state.auto_save.setup_auto_save()

    # Initialize session state for current session ID
    session_id_key = f"{app_type}_session_id"
    if session_id_key not in st.session_state:
        # Generate a unique session ID for this chat session
        st.session_state[session_id_key] = f"{app_type}_{uuid.uuid4().hex[:8]}"

    # Initialize session state for chatbot and messages
    chatbot_key = f"{app_type}_chatbot"
    messages_key = f"{app_type}_messages"
    
    if chatbot_key not in st.session_state:
        st.session_state[chatbot_key] = chatbot_class()
        
    if messages_key not in st.session_state:
        # Try to load previous chat history for this session
        previous_messages, session_metadata = st.session_state.history_manager.load_chat_history(
            st.session_state[session_id_key]
        )
        
        if previous_messages:
            st.session_state[messages_key] = previous_messages
            # Restore settings from metadata if available
            restore_settings_from_metadata(session_metadata)
        else:
            st.session_state[messages_key] = []

    # Initialize RAG parameters in session state
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000
        
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 100
        
    if "k_value" not in st.session_state:
        st.session_state.k_value = 5
    
    # Initialize debug mode
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
        
    return app_type, session_id_key, chatbot_key, messages_key

def restore_settings_from_metadata(metadata):
    """
    Restore settings from session metadata
    
    Args:
        metadata (dict): Session metadata with settings
    """
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

def save_chat_history(app_type, session_id_key, messages_key, chatbot_key):
    """
    Save the current chat history with metadata
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        session_id_key (str): Key for session ID in session state
        messages_key (str): Key for messages in session state
        chatbot_key (str): Key for chatbot in session state
    
    Returns:
        str or None: Path to saved file or None if no changes
    """
    if not st.session_state[messages_key]:
        return None
        
    # Collect current metadata
    current_metadata = {
        "model_name": st.session_state.get("model_name", "gpt-3.5-turbo"),
        "k_value": st.session_state.k_value,
        "chunk_size": st.session_state.chunk_size,
        "chunk_overlap": st.session_state.chunk_overlap,
        "files_processed": list(st.session_state[chatbot_key].file_metadata.keys()) 
                          if hasattr(st.session_state[chatbot_key], 'file_metadata') else []
    }
    
    # Save the chat history
    return st.session_state.history_manager.save_chat_history(
        st.session_state[messages_key],
        app_type,
        st.session_state[session_id_key],
        metadata=current_metadata
    )

def display_example_questions(app_type, messages_key, examples):
    """
    Display example question buttons that will properly trigger the LLM
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        messages_key (str): Key for messages in session state
        examples (list): List of example questions to display
    """
    st.subheader("Example Questions")
    
    # Create columns based on the number of examples (max 3)
    cols = st.columns(min(len(examples), 3))
    
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
        chatbot_key = f"{app_type}_chatbot"
        session_id_key = f"{app_type}_session_id"
        process_chat_message(app_type, chatbot_key, messages_key, session_id_key, question)
        
        # Force rerun to update the UI
        st.rerun()
    
    # Display buttons in columns with direct LLM call handling
    for i, example in enumerate(examples[:3]):  # Limit to first 3 examples
        with cols[i]:
            if st.button(example):
                handle_example_click(example)

def add_user_message(app_type, messages_key, prompt):
    """
    Add a user message to the chat history and force a rerun
    Note: This method is kept for backward compatibility but no longer needed
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        messages_key (str): Key for messages in session state
        prompt (str): The user's prompt to add
    """
    user_message = {
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state[messages_key].append(user_message)
    
    # Force a rerun to show the message immediately and trigger processing
    st.rerun()

def display_chat_messages(messages_key):
    """
    Display the chat message history with metadata
    
    Args:
        messages_key (str): Key for messages in session state
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    
    for message in st.session_state[messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display SQL data and visualization if present
            if message["role"] == "assistant" and message.get("has_data", False):
                # Get app type from messages_key
                app_type = messages_key.split("_")[0]
                
                # Try to get data from chatbot
                data = None
                chatbot_key = f"{app_type}_chatbot"
                
                if hasattr(st.session_state, chatbot_key) and hasattr(st.session_state[chatbot_key], 'last_query_results'):
                    data = st.session_state[chatbot_key].last_query_results
                
                if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                    # Display the dataframe
                    st.dataframe(data)
                    
                    # Create visualization select box
                    viz_options = ["table", "bar", "line", "scatter", "pie"]
                    # Create a unique key for this message
                    msg_id = f"viz_{hash(str(message.get('timestamp', '')))}"
                    selected_viz = st.selectbox("Visualization Type", viz_options, key=f"select_{msg_id}")
                    
                    # Generate visualization button
                    if st.button("Generate Visualization", key=f"button_{msg_id}"):
                        if selected_viz != "table":
                            try:
                                # Direct visualization function
                                def generate_visualization(data, chart_type="auto"):
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
                                
                                # Create the visualization
                                fig = generate_visualization(data, chart_type=selected_viz)
                                if isinstance(fig, go.Figure):
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Add download button
                                    csv = data.to_csv(index=False)
                                    st.download_button(
                                        label="Download Results as CSV",
                                        data=csv,
                                        file_name="query_results.csv",
                                        mime="text/csv",
                                        key=f"download_{msg_id}"
                                    )
                                else:
                                    st.warning(fig)  # Show error message
                            except Exception as viz_error:
                                st.error(f"Error creating visualization: {str(viz_error)}")
            
            # Display metadata for assistant messages
            if message["role"] == "assistant" and "model" in message:
                with st.expander("Message Details", expanded=False):
                    # Show model used
                    st.caption(f"**Model**: {message.get('model', 'Unknown')}")
                    
                    # Show timestamp if available
                    if "timestamp" in message:
                        st.caption(f"**Time**: {message.get('timestamp')}")
                    
                    # Show RAG parameters if available
                    if "rag_params" in message:
                        st.caption("**RAG Parameters**:")
                        params = message["rag_params"]
                        st.caption(f"- k_value: {params.get('k_value', 'N/A')}")
                        st.caption(f"- chunk_size: {params.get('chunk_size', 'N/A')}")
                        st.caption(f"- chunk_overlap: {params.get('chunk_overlap', 'N/A')}")
                    
                    # Show query if available (for SQLite)
                    if "query" in message:
                        st.caption("**SQL Query**:")
                        st.code(message["query"], language="sql")

def handle_chat_input(app_type, chatbot_key, messages_key, session_id_key, input_placeholder="Ask a question about your documents"):
    """
    Handle the chat input and generate responses
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        chatbot_key (str): Key for chatbot in session state
        messages_key (str): Key for messages in session state
        session_id_key (str): Key for session ID in session state
        input_placeholder (str): Placeholder text for the chat input
    """
    if prompt := st.chat_input(input_placeholder):
        # Add user message to chat history
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state[messages_key].append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        process_chat_message(app_type, chatbot_key, messages_key, session_id_key, prompt)
        
        # Force a rerun to update the UI
        st.rerun()

def process_chat_message(app_type, chatbot_key, messages_key, session_id_key, prompt):
    """
    Process a chat message and generate a response
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        chatbot_key (str): Key for chatbot in session state
        messages_key (str): Key for messages in session state
        session_id_key (str): Key for session ID in session state
        prompt (str): The user's prompt to process
    """
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check if vectorstore exists
                if not hasattr(st.session_state[chatbot_key], 'vectorstore') or not st.session_state[chatbot_key].vectorstore:
                    answer_text = f"Please upload and process {'files' if app_type == 'sqlite' else app_type.upper() + ' files'} first to create a knowledge base."
                    st.warning(answer_text)
                    add_assistant_message(app_type, messages_key, session_id_key, answer_text)
                    return
                
                # Get current model information
                current_model = st.session_state.get("model_name", "gpt-3.5-turbo")
                
                # If debug mode is enabled, show additional information
                if st.session_state.debug_mode:
                    response = st.session_state[chatbot_key].ask(prompt, return_context=True)
                    
                    # Create tabs for answer and debugging info
                    answer_tab, context_tab, metrics_tab = st.tabs([
                        "Answer", "Retrieved Context", "RAG Metrics"
                    ])
                    
                    with answer_tab:
                        if isinstance(response, dict) and "answer" in response:
                            st.markdown(response["answer"])
                            
                            # Display dataframe if present for SQLite queries
                            if app_type == 'sqlite' and "data" in response and isinstance(response["data"], pd.DataFrame):
                                st.write("Here's the result:")
                                st.dataframe(response["data"])
                                
                                # Try visualization
                                if "query" in response:
                                    with st.expander("Query Visualization", expanded=False):
                                        viz_options = ["bar", "line", "scatter", "pie"]
                                        selected_viz = st.selectbox("Select Visualization Type", viz_options)
                                        
                                        if st.button("Generate Visualization"):
                                            try:
                                                # Import from utils to ensure visual consistency
                                                from utils.visualization import generate_visualization
                                                fig = generate_visualization(response["data"], chart_type=selected_viz)
                                                if isinstance(fig, go.Figure):
                                                    st.plotly_chart(fig, use_container_width=True)
                                                else:
                                                    st.warning(fig)
                                            except Exception as viz_error:
                                                st.error(f"Error creating visualization: {str(viz_error)}")
                            
                            answer_text = response["answer"]
                        else:
                            st.markdown(str(response))
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
                            
                else:
                    # Regular mode, just show the answer
                    if app_type == 'sqlite':
                        # SQLite has special handling for SQL generation
                        response = st.session_state[chatbot_key].ask(prompt, return_context=True)
                        
                        if isinstance(response, dict) and "answer" in response:
                            st.markdown(response["answer"])
                            
                            # Display dataframe if present
                            if "data" in response and isinstance(response["data"], pd.DataFrame):
                                st.write("Here's the result:")
                                st.dataframe(response["data"])
                                
                                # Add support for visualization
                                viz_options = ["bar", "line", "scatter", "pie"]
                                selected_viz = st.selectbox("Visualization Type", viz_options)
                                
                                if st.button("Generate Visualization"):
                                    try:
                                        # Direct visualization function
                                        def generate_visualization(data, chart_type="auto"):
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
                                        
                                        fig = generate_visualization(response["data"], chart_type=selected_viz)
                                        if isinstance(fig, go.Figure):
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.warning(fig)  # Show error message
                                    except Exception as viz_error:
                                        st.error(f"Error creating visualization: {str(viz_error)}")
                                
                                # Add download button
                                csv = response["data"].to_csv(index=False)
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=csv,
                                    file_name="query_results.csv",
                                    mime="text/csv"
                                )
                            
                            answer_text = response["answer"]
                        else:
                            st.markdown(str(response))
                            answer_text = str(response)
                    else:
                        # PDF and CSV can use the regular approach
                        answer_text = st.session_state[chatbot_key].ask(prompt)
                        st.markdown(answer_text)
            except Exception as e:
                answer_text = f"An error occurred: {str(e)}"
                st.error(answer_text)
    
    # Add assistant response to chat history with metadata
    add_assistant_message(app_type, messages_key, session_id_key, answer_text)
    
    # Save chat history
    save_chat_history(app_type, session_id_key, messages_key, chatbot_key)

def add_assistant_message(app_type, messages_key, session_id_key, answer_text, query=None, data=None):
    """
    Add an assistant message to the chat history with metadata
    
    Args:
        app_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        messages_key (str): Key for messages in session state
        session_id_key (str): Key for session ID in session state
        answer_text (str): The assistant's response
        query (str, optional): SQL query for SQLite responses
        data (pd.DataFrame, optional): Query result data for visualization
    """
    # Create assistant message with metadata
    assistant_message = {
        "role": "assistant",
        "content": answer_text,
        "timestamp": datetime.now().isoformat(),
        "model": st.session_state.get("model_name", "gpt-3.5-turbo"),
        "rag_params": {
            "k_value": st.session_state.k_value,
            "chunk_size": st.session_state.chunk_size,
            "chunk_overlap": st.session_state.chunk_overlap
        }
    }
    
    # Add SQL query if provided
    if query:
        assistant_message["query"] = query
    
    # Add data flag if DataFrame is provided
    if data is not None:
        assistant_message["has_data"] = True
        # Store data reference - the actual DataFrame will be accessed via the chatbot object
        if hasattr(st.session_state, f"{app_type}_chatbot"):
            st.session_state[f"{app_type}_chatbot"].last_query_results = data
    
    # Add the message to session state
    st.session_state[messages_key].append(assistant_message)
    
    # Save the chat history
    return save_chat_history(app_type, session_id_key, messages_key, f"{app_type}_chatbot")

def display_history_view(history_manager):
    """
    Display the chat history view for reviewing past conversations
    
    Args:
        history_manager: ChatHistoryManager instance
    """
    st.subheader("Chat History View")
    
    # Create a button to exit history view mode
    if st.button("Return to Chat"):
        st.session_state.viewing_history = False
        st.session_state.history_session_id = None
        st.rerun()
    
    # Load and display the selected chat history
    if st.session_state.history_session_id:
        try:
            history, metadata = history_manager.load_chat_history(st.session_state.history_session_id)
            
            if history:
                # Display metadata if available
                if metadata:
                    st.subheader("Session Information")
                    
                    # Create two columns for metadata display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Session ID and type
                        st.write(f"**Session ID:** {st.session_state.history_session_id}")
                        st.write(f"**Type:** {metadata.get('app_type', 'Unknown')}")
                        
                        # Show model used
                        st.write(f"**Model:** {metadata.get('model_name', 'Unknown')}")
                    
                    with col2:
                        # RAG parameters
                        st.write("**RAG Parameters:**")
                        st.write(f"- k_value: {metadata.get('k_value', 'N/A')}")
                        st.write(f"- chunk_size: {metadata.get('chunk_size', 'N/A')}")
                        st.write(f"- chunk_overlap: {metadata.get('chunk_overlap', 'N/A')}")
                
                # Display the conversation
                st.subheader("Conversation")
                
                for message in history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        
                        # Show timestamp if available
                        if "timestamp" in message:
                            st.caption(f"Time: {message.get('timestamp')}")
            else:
                st.warning(f"No chat history found for session ID: {st.session_state.history_session_id}")
                
        except Exception as e:
            st.error(f"Error loading chat history: {str(e)}")
    else:
        st.warning("No history session selected.")


def view_chat_history(session_id):
    """
    Set the state to view a specific chat history session
    
    Args:
        session_id (str): The session ID to view
    """
    st.session_state.viewing_history = True
    st.session_state.history_session_id = session_id
    st.rerun()
