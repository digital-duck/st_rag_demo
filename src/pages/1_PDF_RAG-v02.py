import os
import sys
import streamlit as st

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.pdf_chatbot import PDFChatbot

# Page configuration
st.set_page_config(
    page_title="PDF RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Document RAG")
st.subheader("Chat with your PDF Documents")

# Initialize session state for chatbot and messages
if "pdf_chatbot" not in st.session_state:
    st.session_state.pdf_chatbot = PDFChatbot()
    
if "pdf_messages" not in st.session_state:
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
    
    # Status information
    if hasattr(st.session_state.pdf_chatbot, 'vectorstore') and st.session_state.pdf_chatbot.vectorstore:
        st.success("✅ Vector database is ready")
    else:
        st.warning("⚠️ No vector database available. Please upload and process files.")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.pdf_messages = []
        st.success("Chat history cleared!")
    
    # Clear all data button
    if st.button("Clear All Data"):
        st.session_state.pdf_chatbot.clear()
        st.session_state.pdf_messages = []
        st.success("All data and chat history cleared! Please upload and process files again.")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Please set your OpenAI API key in a .env file or as an environment variable.")

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
        st.session_state.pdf_messages.append({"role": "user", "content": prompt})
        
with col2:
    if st.button(example_qs[1]):
        prompt = example_qs[1]
        st.session_state.pdf_messages.append({"role": "user", "content": prompt})
        
with col3:
    if st.button(example_qs[2]):
        prompt = example_qs[2]
        st.session_state.pdf_messages.append({"role": "user", "content": prompt})

# Main chat interface
st.subheader("Chat with your PDFs")

# Display chat messages
for message in st.session_state.pdf_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs"):
    # Add user message to chat history
    st.session_state.pdf_messages.append({"role": "user", "content": prompt})
    
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
    
    # Add assistant response to chat history
    st.session_state.pdf_messages.append({"role": "assistant", "content": answer_text})
    
    # Force a rerun to update the chat history display
    st.rerun()