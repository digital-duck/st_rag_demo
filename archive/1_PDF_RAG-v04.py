# Location: pages/1_PDF_RAG.py
# Complete implementation of PDF RAG with chat history persistence and model tracking

import os
import sys
import streamlit as st
import uuid
from datetime import datetime

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.pdf_chatbot import PDFChatbot
from models.chat_history_manager import ChatHistoryManager
from models.auto_save import AutoSave

# Page configuration
st.set_page_config(
    page_title="PDF RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Document RAG")

# Initialize the chat history manager
if "history_manager" not in st.session_state:
    st.session_state.history_manager = ChatHistoryManager()
    
# Initialize auto-save utility
if "auto_save" not in st.session_state:
    st.session_state.auto_save = AutoSave(st.session_state.history_manager)

# Setup auto-save for page navigation
st.session_state.auto_save.setup_auto_save()

# Initialize session state for current session ID
if "pdf_session_id" not in st.session_state:
    # Generate a unique session ID for this chat session
    st.session_state.pdf_session_id = f"pdf_{uuid.uuid4().hex[:8]}"

# Initialize session state for chatbot and messages
if "pdf_chatbot" not in st.session_state:
    st.session_state.pdf_chatbot = PDFChatbot()
    
if "pdf_messages" not in st.session_state:
    # Try to load previous chat history for this session
    previous_messages, session_metadata = st.session_state.history_manager.load_chat_history(st.session_state.pdf_session_id)
    
    if previous_messages:
        st.session_state.pdf_messages = previous_messages
        
        # If we have metadata, restore relevant settings
        if session_metadata:
            # Restore model if available
            if "model_name" in session_metadata:
                model_name = session_metadata["model_name"]
                if model_name != st.session_state.get("model_name", "gpt-3.5-turbo"):
                    st.session_state.model_name = model_name
                    # Will need to recreate chatbot with the right model later
            
            # Restore RAG parameters if available
            if "k_value" in session_metadata:
                st.session_state.k_value = session_metadata["k_value"]
            if "chunk_size" in session_metadata:
                st.session_state.chunk_size = session_metadata["chunk_size"]
            if "chunk_overlap" in session_metadata:
                st.session_state.chunk_overlap = session_metadata["chunk_overlap"]
    else:
        st.session_state.pdf_messages = []

# Initialize RAG parameters in session state
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
    
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 100
    
if "k_value" not in st.session_state:
    st.session_state.k_value = 5

# Sidebar for file upload and RAG configuration
with st.sidebar:
    st.header("Upload PDF Files")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        process_clicked = st.button("Process PDF Files")
        
        if process_clicked:
            with st.spinner("Processing files..."):
                # Clear existing data before processing new files
                st.session_state.pdf_chatbot.clear()
                
                doc_count = 0
                
                for file in uploaded_files:
                    count = st.session_state.pdf_chatbot.process_pdf(file)
                    st.write(f"Processed {file.name}: {count} pages")
                    doc_count += count
                
                if doc_count > 0:
                    with st.spinner("Building vector database..."):
                        if st.session_state.pdf_chatbot.build_vectorstore():
                            st.success(f"Successfully processed {doc_count} pages into document chunks!")
                        else:
                            st.error("Error building the vector database.")
                else:
                    st.warning("No documents were processed.")
    
    # Model selection
    st.header("Model Selection")
    model_name = st.selectbox(
        "Select OpenAI Model",
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"]
    )
    
    if "model_name" not in st.session_state or st.session_state.model_name != model_name:
        st.session_state.model_name = model_name
        # Create a new chatbot instance with the selected model
        temp_chatbot = PDFChatbot(model_name=model_name)
        # Transfer documents and vectorstore if they exist
        if hasattr(st.session_state.pdf_chatbot, 'documents') and st.session_state.pdf_chatbot.documents:
            temp_chatbot.documents = st.session_state.pdf_chatbot.documents
            temp_chatbot.file_metadata = st.session_state.pdf_chatbot.file_metadata
            # Rebuild the vectorstore with the new embeddings model
            temp_chatbot.build_vectorstore()
        
        st.session_state.pdf_chatbot = temp_chatbot
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
    
    # Chat History Management
    st.header("Chat History")
    
    # Show available chat sessions with enhanced metadata
    pdf_sessions = st.session_state.history_manager.list_sessions("pdf")
    
    if pdf_sessions:
        # Display count and create dropdown
        st.write(f"You have {len(pdf_sessions)} saved chat sessions")
        
        # Create a new session option
        session_options = ["Current Session"] + [
            f"{s['session_id']} ({s['message_count']} msgs, {s.get('metadata', {}).get('model_name', 'unknown model')})" 
            for s in pdf_sessions
        ]
        
        selected_session = st.selectbox(
            "Select a session to load",
            session_options
        )
        
        if selected_session != "Current Session" and "(" in selected_session:
            # Extract session ID from the selection
            session_id = selected_session.split(" ")[0]
            
            # Get session details for display
            session_info = next((s for s in pdf_sessions if s["session_id"] == session_id), None)
            
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
                    st.session_state.pdf_session_id = session_id
                    st.session_state.pdf_messages = loaded_messages
                    
                    # Update RAG parameters from metadata if available
                    if session_metadata:
                        if "model_name" in session_metadata:
                            st.session_state.model_name = session_metadata["model_name"]
                        if "k_value" in session_metadata:
                            st.session_state.k_value = session_metadata["k_value"]
                        if "chunk_size" in session_metadata:
                            st.session_state.chunk_size = session_metadata["chunk_size"]
                        if "chunk_overlap" in session_metadata:
                            st.session_state.chunk_overlap = session_metadata["chunk_overlap"]
                    
                    st.success(f"Loaded chat session with {len(loaded_messages)} messages")
                    st.rerun()
                else:
                    st.error("Failed to load chat session")
    
    # Add explicit save button
    if st.button("💾 Save Current Chat History"):
        if st.session_state.pdf_messages:
            # Collect current metadata
            current_metadata = {
                "model_name": st.session_state.get("model_name", "gpt-3.5-turbo"),
                "k_value": st.session_state.k_value,
                "chunk_size": st.session_state.chunk_size,
                "chunk_overlap": st.session_state.chunk_overlap,
                "files_processed": list(st.session_state.pdf_chatbot.file_metadata.keys()) if hasattr(st.session_state.pdf_chatbot, 'file_metadata') else []
            }
            
            saved_path = st.session_state.history_manager.save_chat_history(
                st.session_state.pdf_messages,
                "pdf",
                st.session_state.pdf_session_id,
                metadata=current_metadata
            )
            if saved_path:
                st.success(f"Chat history saved successfully! Session ID: {st.session_state.pdf_session_id}")
            else:
                st.info("No changes to save.")
        else:
            st.info("No messages to save.")
    
    # Allow creating a new session
    if st.button("Start New Chat Session"):
        # Generate a new session ID
        st.session_state.pdf_session_id = f"pdf_{uuid.uuid4().hex[:8]}"
        # Clear messages
        st.session_state.pdf_messages = []
        st.success("Started a new chat session")
        st.rerun()
    
    # Status information
    if hasattr(st.session_state.pdf_chatbot, 'vectorstore') and st.session_state.pdf_chatbot.vectorstore:
        st.success("✅ Vector database is ready")
    else:
        st.warning("⚠️ No vector database available. Please upload and process files.")
    
    # Modify the Clear Chat button to preserve the saved history
    if st.button("Clear Chat Display"):
        # Keep a backup of messages before clearing if they should be auto-saved
        if st.session_state.pdf_messages:
            # Auto-save before clearing
            current_metadata = {
                "model_name": st.session_state.get("model_name", "gpt-3.5-turbo"),
                "k_value": st.session_state.k_value,
                "chunk_size": st.session_state.chunk_size,
                "chunk_overlap": st.session_state.chunk_overlap,
                "files_processed": list(st.session_state.pdf_chatbot.file_metadata.keys()) if hasattr(st.session_state.pdf_chatbot, 'file_metadata') else []
            }
            st.session_state.history_manager.save_chat_history(
                st.session_state.pdf_messages,
                "pdf",
                st.session_state.pdf_session_id,
                metadata=current_metadata
            )
        
        # Clear the display only
        st.session_state.pdf_messages = []
        st.success("Chat display cleared! Your history has been saved and can be loaded again.")
    
    # Modify the Clear All Data button
    if st.button("Clear All Data"):
        # Auto-save before clearing if there's anything to save
        if st.session_state.pdf_messages:
            current_metadata = {
                "model_name": st.session_state.get("model_name", "gpt-3.5-turbo"),
                "k_value": st.session_state.k_value,
                "chunk_size": st.session_state.chunk_size,
                "chunk_overlap": st.session_state.chunk_overlap,
                "files_processed": list(st.session_state.pdf_chatbot.file_metadata.keys()) if hasattr(st.session_state.pdf_chatbot, 'file_metadata') else []
            }
            st.session_state.history_manager.save_chat_history(
                st.session_state.pdf_messages,
                "pdf",
                st.session_state.pdf_session_id,
                metadata=current_metadata
            )
        
        # Clear the chatbot data
        st.session_state.pdf_chatbot.clear()
        # Clear the display
        st.session_state.pdf_messages = []
        st.success("All data cleared! Your chat history has been saved and can be loaded again.")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Please set your OpenAI API key in a .env file or as an environment variable.")

    # Display current session info
    st.caption(f"Current Session ID: {st.session_state.pdf_session_id}")

# Display example prompts
st.subheader("Example Questions")
example_qs = [
    "What are the main topics in the document?",
    "Summarize the key points.",
    "What does the document say about X?"
]

col1, col2, col3 = st.columns(3)
with col1:
    if st.button(example_qs[0]):
        prompt = example_qs[0]
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.pdf_messages.append(user_message)
        
with col2:
    if st.button(example_qs[1]):
        prompt = example_qs[1]
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.pdf_messages.append(user_message)
        
with col3:
    if st.button(example_qs[2]):
        prompt = example_qs[2]
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.pdf_messages.append(user_message)

# Main chat interface
st.subheader("Chat with your PDFs")

# Display chat messages with enhanced metadata
for message in st.session_state.pdf_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display metadata for assistant messages
        if message["role"] == "assistant" and "model" in message:
            with st.expander("Message Details"):
                # Show model used
                st.caption(f"**Model**: {message.get('model', 'Unknown')}")
                
                # Show timestamp if available
                if "timestamp" in message:
                    st.caption(f"**Time**: {message['timestamp']}")
                
                # Show RAG parameters if available
                if "rag_params" in message:
                    st.caption("**RAG Parameters**:")
                    params = message["rag_params"]
                    st.caption(f"- k_value: {params.get('k_value', 'N/A')}")
                    st.caption(f"- chunk_size: {params.get('chunk_size', 'N/A')}")
                    st.caption(f"- chunk_overlap: {params.get('chunk_overlap', 'N/A')}")

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs"):
    # Add user message to chat history with current timestamp
    user_message = {
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.pdf_messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check if vectorstore exists
                if not hasattr(st.session_state.pdf_chatbot, 'vectorstore') or not st.session_state.pdf_chatbot.vectorstore:
                    answer_text = "Please upload and process PDF files first to create a knowledge base."
                    st.warning(answer_text)
                else:
                    # Get current model information
                    current_model = st.session_state.get("model_name", "gpt-3.5-turbo")
                    
                    if st.session_state.debug_mode:
                        response = st.session_state.pdf_chatbot.ask(prompt, return_context=True)
                        
                        # Create tabs for answer and debugging info
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
                        answer_text = st.session_state.pdf_chatbot.ask(prompt)
                        st.markdown(answer_text)
            except Exception as e:
                answer_text = f"An error occurred: {str(e)}"
                st.error(answer_text)
    
    # Add assistant response to chat history with model information and timestamp
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
    st.session_state.pdf_messages.append(assistant_message)
    
    # Save chat history to disk with metadata
    session_metadata = {
        "model_name": st.session_state.get("model_name", "gpt-3.5-turbo"),
        "k_value": st.session_state.k_value,
        "chunk_size": st.session_state.chunk_size,
        "chunk_overlap": st.session_state.chunk_overlap,
        "files_processed": list(st.session_state.pdf_chatbot.file_metadata.keys()) if hasattr(st.session_state.pdf_chatbot, 'file_metadata') else []
    }
    
    st.session_state.history_manager.save_chat_history(
        st.session_state.pdf_messages,
        "pdf",
        st.session_state.pdf_session_id,
        metadata=session_metadata
    )
    
    # Force a rerun to update the chat history display
    st.rerun()