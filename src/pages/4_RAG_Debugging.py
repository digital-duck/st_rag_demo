# Location: pages/4_RAG_Debugging.py
# Implementation of the RAG Debugging page

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.pdf_chatbot import PDFChatbot
from models.csv_chatbot import CSVChatbot
from models.sqlite_chatbot import SQLiteChatbot
from models.chat_history_manager import ChatHistoryManager

# Import debugging utilities
from utils.debugging import (
    display_debug_page_header, compare_retrieval_parameters,
    display_retrieval_comparison, analyze_document_chunks,
    run_rag_evaluation, display_evaluation_results,
    generate_test_questions, export_debug_report
)

# Page configuration
st.set_page_config(
    page_title="RAG Debugging",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state for RAG debugging
if "debug_rag_type" not in st.session_state:
    st.session_state.debug_rag_type = "pdf"
if "debug_question" not in st.session_state:
    st.session_state.debug_question = "What are the main topics in these documents?"
if "retrieval_results" not in st.session_state:
    st.session_state.retrieval_results = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "show_doc_analysis" not in st.session_state:
    st.session_state.show_doc_analysis = False

# Initialize RAG parameters in session state if not already present
if "k_value" not in st.session_state:
    st.session_state.k_value = 5
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 100
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-3.5-turbo"

# Initialize chatbots if not already done
if "pdf_chatbot" not in st.session_state:
    st.session_state.pdf_chatbot = PDFChatbot()
if "csv_chatbot" not in st.session_state:
    st.session_state.csv_chatbot = CSVChatbot()
if "sqlite_chatbot" not in st.session_state:
    st.session_state.sqlite_chatbot = SQLiteChatbot()

# Display page header
display_debug_page_header()

# Sidebar for selecting RAG type and parameters
with st.sidebar:
    st.header("RAG Debugging Settings")
    
    # Select RAG type to debug
    rag_type = st.selectbox(
        "Select RAG Type to Debug",
        ["pdf", "csv", "sqlite"],
        index=["pdf", "csv", "sqlite"].index(st.session_state.debug_rag_type)
    )
    st.session_state.debug_rag_type = rag_type
    
    # Get the appropriate chatbot
    if rag_type == "pdf":
        chatbot = st.session_state.pdf_chatbot
    elif rag_type == "csv":
        chatbot = st.session_state.csv_chatbot
    else:  # sqlite
        chatbot = st.session_state.sqlite_chatbot
    
    # Check if documents are loaded
    has_docs = (
        hasattr(chatbot, 'vectorstore') and 
        chatbot.vectorstore is not None
    )
    
    if has_docs:
        st.success(f"✅ Vector database is ready for {rag_type.upper()}")
    else:
        st.warning(f"⚠️ No vector database available for {rag_type.upper()}. Please upload and process files on the {rag_type.upper()} page first.")
    
    # RAG parameters
    st.subheader("RAG Parameters")
    
    # Number of chunks to retrieve
    k_value = st.slider("Number of chunks (k)", 1, 20, st.session_state.k_value)
    if k_value != st.session_state.k_value:
        st.session_state.k_value = k_value
    
    # Chunk size for text splitter
    chunk_size = st.slider("Chunk size", 200, 2000, st.session_state.chunk_size)
    if chunk_size != st.session_state.chunk_size:
        st.session_state.chunk_size = chunk_size
    
    # Chunk overlap for text splitter
    chunk_overlap = st.slider("Chunk overlap", 0, 500, st.session_state.chunk_overlap)
    if chunk_overlap != st.session_state.chunk_overlap:
        st.session_state.chunk_overlap = chunk_overlap
    
    # Select model for testing
    model_name = st.selectbox(
        "Select Model",
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"],
        index=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"].index(st.session_state.model_name)
    )
    if model_name != st.session_state.model_name:
        st.session_state.model_name = model_name
        # Create a new chatbot instance with the selected model
        if rag_type == "pdf":
            st.session_state.pdf_chatbot = PDFChatbot(model_name=model_name)
        elif rag_type == "csv":
            st.session_state.csv_chatbot = CSVChatbot(model_name=model_name)
        else:  # sqlite
            st.session_state.sqlite_chatbot = SQLiteChatbot(model_name=model_name)
    
    # Debugging tabs in sidebar
    st.subheader("Testing Options")
    
    # Test question for parameter comparison
    st.text_area(
        "Test Question", 
        st.session_state.debug_question,
        key="debug_question_input",
        on_change=lambda: setattr(st.session_state, "debug_question", st.session_state.debug_question_input)
    )
    
    # Number of K values to compare
    num_k_values = st.number_input("Number of k values to compare", 2, 10, 4)
    
    # Number of test questions for evaluation
    num_test_questions = st.number_input("Number of test questions for evaluation", 2, 10, 5)
    
    # Buttons for different tests
    st.subheader("Run Tests")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Compare Parameters"):
            if has_docs:
                with st.spinner("Comparing retrieval parameters..."):
                    # Generate k values from 1 to max k
                    k_values = list(range(1, max(num_k_values + 1, 2)))
                    # Run comparison
                    results = compare_retrieval_parameters(
                        chatbot, st.session_state.debug_question, k_values=k_values
                    )
                    # Store results
                    st.session_state.retrieval_results = results
                    st.rerun()
            else:
                st.error("Please load documents first.")

# Main content area
# Create tabs for different debugging views
if has_docs:
    debug_tabs = st.tabs([
        "Parameter Comparison", 
        "Document Analysis", 
        "RAG Evaluation",
        "Single Query Test"
    ])
    
    # Parameter Comparison tab
    with debug_tabs[0]:
        st.subheader("Parameter Comparison")
        st.write("""
        Compare retrieval results with different parameter settings to find the optimal configuration.
        Use the "Compare Parameters" button in the sidebar to run a new comparison.
        """)
        
        if st.session_state.retrieval_results:
            display_retrieval_comparison(st.session_state.retrieval_results)
        else:
            st.info("No parameter comparison results yet. Click 'Compare Parameters' in the sidebar to start.")
    
    # Document Analysis tab
    with debug_tabs[1]:
        st.subheader("Document Analysis")
        st.write("""
        Analyze the document chunks in your knowledge base to understand their distribution and content.
        Use this to identify potential issues with chunking or embedding.
        """)
        
        if st.session_state.show_doc_analysis:
            analyze_document_chunks(chatbot)
        else:
            st.info("Click 'Analyze Documents' in the sidebar to start document analysis.")
    
    # RAG Evaluation tab
    with debug_tabs[2]:
        st.subheader("RAG Evaluation")
        st.write("""
        Evaluate your RAG system with a set of test questions to measure performance.
        Use the "Run Evaluation" button in the sidebar to start a new evaluation.
        """)
        
        if st.session_state.evaluation_results:
            display_evaluation_results(st.session_state.evaluation_results)
        else:
            st.info("No evaluation results yet. Click 'Run Evaluation' in the sidebar to start.")
    
    # Single Query Test tab
    with debug_tabs[3]:
        st.subheader("Single Query Test")
        st.write("""
        Test a single query and inspect the retrieved context, prompt, and response.
        This helps you understand how your RAG system processes a specific question.
        """)
        
        test_query = st.text_area("Enter a test query", st.session_state.debug_question)
        
        if st.button("Run Test Query"):
            with st.spinner("Processing query..."):
                # Run the query with debug info
                response = chatbot.ask(test_query, return_context=True)
                
                # Display results in subtabs
                query_tabs = st.tabs([
                    "Answer", "Retrieved Context", "Prompt", "Metrics"
                ])
                
                with query_tabs[0]:
                    st.subheader("Answer")
                    if isinstance(response, dict) and "answer" in response:
                        st.write(response["answer"])
                    else:
                        st.write(str(response))
                
                with query_tabs[1]:
                    st.subheader("Retrieved Context")
                    if isinstance(response, dict) and "retrieved_context" in response:
                        st.text_area("Retrieved Documents", response["retrieved_context"], height=400)
                    else:
                        st.info("No context retrieved or not available.")
                
                with query_tabs[2]:
                    st.subheader("Prompt")
                    if isinstance(response, dict) and "formatted_prompt" in response:
                        st.text_area("Formatted Prompt", response["formatted_prompt"], height=400)
                    else:
                        st.info("Formatted prompt not available.")
                
                with query_tabs[3]:
                    st.subheader("Metrics")
                    metrics_dict = {}
                    
                    if isinstance(response, dict) and "retrieved_context" in response:
                        chunks = response["retrieved_context"].split("\n\n---\n\n")
                        chunk_lengths = [len(chunk) for chunk in chunks]
                        
                        metrics_dict.update({
                            "num_chunks_retrieved": len(chunks),
                            "avg_chunk_length": sum(chunk_lengths) / max(1, len(chunk_lengths)),
                            "min_chunk_length": min(chunk_lengths, default=0),
                            "max_chunk_length": max(chunk_lengths, default=0),
                            "total_context_length": len(response["retrieved_context"])
                        })
                    
                    metrics_dict.update({
                        "model": st.session_state.model_name,
                        "k_value": st.session_state.k_value,
                        "chunk_size": st.session_state.chunk_size,
                        "chunk_overlap": st.session_state.chunk_overlap
                    })
                    
                    st.json(metrics_dict)
else:
    st.warning("""
    No documents loaded. Please go to the appropriate RAG page to upload and process documents first.
    
    1. Navigate to the PDF, CSV, or SQLite RAG page
    2. Upload and process your documents
    3. Return to this page to debug your RAG system
    """)
    
    # Show placeholder buttons that direct to the right page
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Go to PDF RAG"):
            st.switch_page("pages/1_PDF_RAG.py")
    with col2:
        if st.button("Go to CSV RAG"):
            st.switch_page("pages/2_CSV_RAG.py")
    with col3:
        if st.button("Go to SQLite RAG"):
            st.switch_page("pages/3_SQLite_RAG.py")
    
    with col2:
        if st.button("Run Evaluation"):
            if has_docs:
                with st.spinner("Generating test questions..."):
                    # Generate test questions
                    test_questions = generate_test_questions(num_test_questions)
                    # Run evaluation
                    results = run_rag_evaluation(chatbot, test_questions)
                    # Store results
                    st.session_state.evaluation_results = results
                    st.rerun()
            else:
                st.error("Please load documents first.")
    
    # Document analysis button
    if st.button("Analyze Documents"):
        if has_docs:
            st.session_state.show_doc_analysis = True
            st.rerun()
        else:
            st.error("Please load documents first.")
    
    # Export debug report
    if st.button("Export Debug Report"):
        if has_docs:
            report_path = export_debug_report(
                rag_type, 
                chatbot, 
                st.session_state.retrieval_results,
                st.session_state.evaluation_results
            )
            
            with open(report_path, "r") as f:
                report_content = f.read()
            
            st.download_button(
                "Download Debug Report",
                report_content,
                file_name=f"rag_debug_report_{rag_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.error("Please load documents first.")