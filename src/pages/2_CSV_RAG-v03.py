import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import uuid

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.csv_chatbot import CSVChatbot
from models.chat_history_manager import ChatHistoryManager

# Page configuration
st.set_page_config(
    page_title="CSV RAG",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CSV Document RAG")
st.subheader("Chat with your CSV Data")

# Initialize the chat history manager
if "history_manager" not in st.session_state:
    st.session_state.history_manager = ChatHistoryManager()

# Initialize session state for current session ID
if "csv_session_id" not in st.session_state:
    # Generate a unique session ID for this chat session
    st.session_state.csv_session_id = f"csv_{uuid.uuid4().hex[:8]}"

# Initialize session state for chatbot and messages
if "csv_chatbot" not in st.session_state:
    st.session_state.csv_chatbot = CSVChatbot()
    
if "csv_messages" not in st.session_state:
    # Try to load previous chat history for this session
    previous_messages = st.session_state.history_manager.load_chat_history(st.session_state.csv_session_id)
    if previous_messages:
        st.session_state.csv_messages = previous_messages
    else:
        st.session_state.csv_messages = []

# Initialize RAG parameters in session state
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
    
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 100
    
if "k_value" not in st.session_state:
    st.session_state.k_value = 5

# Sidebar for file upload and RAG configuration
with st.sidebar:
    st.header("Upload CSV Files")
    
    uploaded_files = st.file_uploader(
        "Upload CSV files", 
        type=["csv"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        process_clicked = st.button("Process CSV Files")
        
        if process_clicked:
            with st.spinner("Processing files..."):
                # Clear existing data before processing new files
                st.session_state.csv_chatbot.clear()
                
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
                        count = st.session_state.csv_chatbot.process_csv(file)
                        st.write(f"✅ Processed {file.name}: {count} records")
                        file_count += 1
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                
                if file_count > 0:
                    with st.spinner("Building vector database..."):
                        if st.session_state.csv_chatbot.build_vectorstore():
                            st.success(f"Successfully processed {file_count} CSV files!")
                        else:
                            st.error("Error building the vector database.")
                else:
                    st.warning("No CSV files were processed.")
    
    # Model selection
    st.header("Model Selection")
    model_name = st.selectbox(
        "Select OpenAI Model",
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"]
    )
    
    if "model_name" not in st.session_state or st.session_state.model_name != model_name:
        st.session_state.model_name = model_name
        # Create a new chatbot instance with the selected model
        temp_chatbot = CSVChatbot(model_name=model_name)
        # Transfer documents and vectorstore if they exist
        if hasattr(st.session_state.csv_chatbot, 'documents') and st.session_state.csv_chatbot.documents:
            temp_chatbot.documents = st.session_state.csv_chatbot.documents
            temp_chatbot.file_metadata = st.session_state.csv_chatbot.file_metadata
            temp_chatbot.data_frames = st.session_state.csv_chatbot.data_frames
            # Rebuild the vectorstore with the new embeddings model
            temp_chatbot.build_vectorstore()
        
        st.session_state.csv_chatbot = temp_chatbot
        st.info(f"Model updated to {model_name}. This will apply to future questions.")
    
    # RAG Configuration
    st.header("RAG Configuration")
    
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
        
        # Add toggle for data visualization
        st.subheader("Visualization Settings")
        viz_enabled = st.checkbox("Enable Auto-Visualization", value=True)
        st.session_state.viz_enabled = viz_enabled
        
        if viz_enabled:
            viz_type = st.selectbox(
                "Default Chart Type",
                ["auto", "bar", "line", "scatter", "pie", "table"]
            )
            st.session_state.viz_type = viz_type
    
    # Chat History Management
    st.header("Chat History")
    
    # Show available chat sessions
    csv_sessions = st.session_state.history_manager.list_sessions("csv")
    
    if csv_sessions:
        st.write(f"You have {len(csv_sessions)} saved chat sessions")
        
        # Create a new session option
        session_options = ["Current Session"] + [
            f"{s['session_id']} ({s['message_count']} messages)" for s in csv_sessions
        ]
        
        selected_session = st.selectbox(
            "Select a session to load",
            session_options
        )
        
        if selected_session != "Current Session" and "(" in selected_session:
            # Extract session ID from the selection
            session_id = selected_session.split(" ")[0]
            
            # Button to load the selected session
            if st.button("Load Selected Session"):
                # Load the messages from this session
                loaded_messages = st.session_state.history_manager.load_chat_history(session_id)
                if loaded_messages:
                    # Update the current session ID and messages
                    st.session_state.csv_session_id = session_id
                    st.session_state.csv_messages = loaded_messages
                    st.success(f"Loaded chat session with {len(loaded_messages)} messages")
                    st.experimental_rerun()
                else:
                    st.error("Failed to load chat session")
    
    # Allow creating a new session
    if st.button("Start New Chat Session"):
        # Generate a new session ID
        st.session_state.csv_session_id = f"csv_{uuid.uuid4().hex[:8]}"
        # Clear messages
        st.session_state.csv_messages = []
        st.success("Started a new chat session")
        st.experimental_rerun()
    
    # Status information
    if hasattr(st.session_state.csv_chatbot, 'vectorstore') and st.session_state.csv_chatbot.vectorstore:
        st.success("✅ Vector database is ready")
    else:
        st.warning("⚠️ No vector database available. Please upload and process files.")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.csv_messages = []
        # Save empty history to show we've cleared it
        st.session_state.history_manager.save_chat_history(
            [], "csv", st.session_state.csv_session_id
        )
        st.success("Chat history cleared!")
    
    # Clear all data button
    if st.button("Clear All Data"):
        st.session_state.csv_chatbot.clear()
        st.session_state.csv_messages = []
        # Save empty history
        st.session_state.history_manager.save_chat_history(
            [], "csv", st.session_state.csv_session_id
        )
        st.success("All data and chat history cleared! Please upload and process files again.")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Please set your OpenAI API key in a .env file or as an environment variable.")

# Display current session info
st.caption(f"Current Session ID: {st.session_state.csv_session_id}")

# Display example prompts
st.subheader("Example Questions")
example_qs = [
    "Summarize the CSV data.",
    "What trends do you see in the data?",
    "Calculate key statistics from the CSV."
]

col1, col2, col3 = st.columns(3)
with col1:
    if st.button(example_qs[0]):
        prompt = example_qs[0]
        st.session_state.csv_messages.append({"role": "user", "content": prompt})
        
with col2:
    if st.button(example_qs[1]):
        prompt = example_qs[1]
        st.session_state.csv_messages.append({"role": "user", "content": prompt})
        
with col3:
    if st.button(example_qs[2]):
        prompt = example_qs[2]
        st.session_state.csv_messages.append({"role": "user", "content": prompt})

# Main chat interface
st.subheader("Chat with your CSV Data")

# Display chat messages
for message in st.session_state.csv_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your CSV data"):
    # Add user message to chat history
    st.session_state.csv_messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check if vectorstore exists
                if not hasattr(st.session_state.csv_chatbot, 'vectorstore') or not st.session_state.csv_chatbot.vectorstore:
                    answer_text = "Please upload and process CSV files first to create a knowledge base."
                    st.warning(answer_text)
                else:
                    if st.session_state.debug_mode:
                        response = st.session_state.csv_chatbot.ask(prompt, return_context=True)
                        
                        # Check if visualization is requested
                        viz_keywords = ["chart", "plot", "graph", "visualize", "visualization", "show", "display"]
                        is_viz_query = any(keyword in prompt.lower() for keyword in viz_keywords)
                        viz_enabled = st.session_state.get("viz_enabled", False)
                        
                        # Create tabs for answer and debugging info
                        if is_viz_query and viz_enabled:
                            answer_tab, viz_tab, context_tab, metrics_tab = st.tabs([
                                "Answer", "Visualization", "Retrieved Context", "RAG Metrics"
                            ])
                        else:
                            answer_tab, context_tab, metrics_tab = st.tabs([
                                "Answer", "Retrieved Context", "RAG Metrics"
                            ])
                        
                        with answer_tab:
                            if isinstance(response, dict) and "answer" in response:
                                st.markdown(response["answer"])
                                
                                # Store the answer for chat history
                                answer_text = response["answer"]
                            else:
                                st.markdown(str(response))
                                
                                # Store the response for chat history
                                answer_text = str(response)
                        
                        # If visualization is enabled and requested
                        if is_viz_query and viz_enabled and isinstance(response, dict) and "answer" in response:
                            # Try to extract data from the chatbot
                            if hasattr(st.session_state.csv_chatbot, 'data_frames') and st.session_state.csv_chatbot.data_frames:
                                with viz_tab:
                                    # Use the first dataframe for now
                                    df_key = list(st.session_state.csv_chatbot.data_frames.keys())[0]
                                    df = st.session_state.csv_chatbot.data_frames[df_key]
                                    
                                    # Try to create an automatic visualization
                                    viz_type = st.session_state.get("viz_type", "auto")
                                    
                                    if viz_type == "auto":
                                        # Auto-detect the best chart type
                                        numeric_cols = df.select_dtypes(include=['number']).columns
                                        categorical_cols = df.select_dtypes(include=['object']).columns
                                        
                                        if len(numeric_cols) >= 2:
                                            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
                                            st.plotly_chart(fig, use_container_width=True)
                                        elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                                            fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], title=f"Bar Chart: {numeric_cols[0]} by {categorical_cols[0]}")
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.write("Could not automatically determine the best visualization. Here's the data:")
                                            st.dataframe(df.head(10))
                                    elif viz_type == "bar" and len(df.columns) >= 2:
                                        # Find a categorical and numeric column
                                        categorical_col = df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) > 0 else df.columns[0]
                                        numeric_col = df.select_dtypes(include=['number']).columns[0] if len(df.select_dtypes(include=['number']).columns) > 0 else df.columns[1]
                                        fig = px.bar(df, x=categorical_col, y=numeric_col, title=f"Bar Chart: {numeric_col} by {categorical_col}")
                                        st.plotly_chart(fig, use_container_width=True)
                                    elif viz_type == "line" and len(df.columns) >= 2:
                                        # Assume first column is x-axis
                                        fig = px.line(df, x=df.columns[0], y=df.columns[1], title=f"Line Chart: {df.columns[1]} over {df.columns[0]}")
                                        st.plotly_chart(fig, use_container_width=True)
                                    elif viz_type == "scatter" and len(df.columns) >= 2:
                                        numeric_cols = df.select_dtypes(include=['number']).columns
                                        if len(numeric_cols) >= 2:
                                            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.write("Not enough numeric columns for scatter plot. Here's the data:")
                                            st.dataframe(df.head(10))
                                    elif viz_type == "pie" and len(df.columns) >= 2:
                                        # Find a categorical and numeric column
                                        categorical_col = df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) > 0 else df.columns[0]
                                        numeric_col = df.select_dtypes(include=['number']).columns[0] if len(df.select_dtypes(include=['number']).columns) > 0 else df.columns[1]
                                        fig = px.pie(df, names=categorical_col, values=numeric_col, title=f"Pie Chart: {numeric_col} by {categorical_col}")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        # Default to table view
                                        st.write("Showing data table:")
                                        st.dataframe(df.head(10))
                        
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
                        answer_text = st.session_state.csv_chatbot.ask(prompt)
                        
                        # Check if this is a visualization request
                        viz_keywords = ["chart", "plot", "graph", "visualize", "visualization", "show", "display"]
                        is_viz_query = any(keyword in prompt.lower() for keyword in viz_keywords)
                        
                        if is_viz_query and hasattr(st.session_state.csv_chatbot, 'data_frames') and st.session_state.csv_chatbot.data_frames:
                            # Display the answer
                            st.markdown(answer_text)
                            
                            # Try to create a basic visualization
                            st.subheader("Visualization")
                            df_key = list(st.session_state.csv_chatbot.data_frames.keys())[0]
                            df = st.session_state.csv_chatbot.data_frames[df_key]
                            
                            # Choose a simple visualization type based on data
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) >= 2:
                                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
                                st.plotly_chart(fig, use_container_width=True)
                            elif len(df.columns) >= 2:
                                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=f"Bar Chart: {df.columns[1]} by {df.columns[0]}")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.dataframe(df.head(10))
                        else:
                            st.markdown(answer_text)
            except Exception as e:
                answer_text = f"An error occurred: {str(e)}"
                st.error(answer_text)
    
    # Add assistant response to chat history
    st.session_state.csv_messages.append({"role": "assistant", "content": answer_text})
    
    # Save chat history to disk
    st.session_state.history_manager.save_chat_history(
        st.session_state.csv_messages,
        "csv",
        st.session_state.csv_session_id
    )
    
    # Force a rerun to update the chat history display
    st.rerun()