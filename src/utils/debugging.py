# Location: utils/debugging.py
# Utility functions for RAG debugging and evaluation

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import json
import numpy as np
from datetime import datetime

def display_debug_page_header():
    """
    Display the header for the RAG debugging page
    """
    st.title("🔍 RAG Debugging Tools")
    st.header("Advanced Tools for RAG Development and Debugging")
    st.write("This page provides tools to analyze, debug, and optimize your RAG implementation. "
             "You can compare different parameter settings, analyze retrieved chunks, and evaluate retrieval quality.")
    
    # Initialize k_value in session state if not exists
    if "k_value" not in st.session_state:
        st.session_state.k_value = 5

def compare_retrieval_parameters(chatbot, question, k_values=None, chunk_sizes=None, chunk_overlaps=None):
    """
    Compare retrieval with different parameter settings
    
    Args:
        chatbot: The chatbot instance to use
        question (str): The question to use for comparison
        k_values (list): List of k values to compare
        chunk_sizes (list): List of chunk sizes to compare
        chunk_overlaps (list): List of chunk overlaps to compare
    
    Returns:
        dict: Dictionary of comparison results
    """
    results = {
        "question": question,
        "comparisons": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Compare k values
    if k_values:
        current_k = st.session_state.k_value
        comparison = {"type": "k_value", "values": [], "contexts": [], "answers": []}
        
        for k in k_values:
            st.session_state.k_value = k
            response = chatbot.ask(question, return_context=True)
            
            if isinstance(response, dict):
                comparison["values"].append(k)
                comparison["contexts"].append(response.get("retrieved_context", ""))
                comparison["answers"].append(response.get("answer", str(response)))
            else:
                comparison["values"].append(k)
                comparison["contexts"].append("")
                comparison["answers"].append(str(response))
        
        # Reset k value
        st.session_state.k_value = current_k
        results["comparisons"].append(comparison)
    
    # Compare chunk sizes (would require reprocessing documents)
    # This is a placeholder for future implementation
    if chunk_sizes:
        results["comparisons"].append({"type": "chunk_size", "message": "Chunk size comparison requires reprocessing documents"})
    
    # Compare chunk overlaps (would require reprocessing documents)
    # This is a placeholder for future implementation
    if chunk_overlaps:
        results["comparisons"].append({"type": "chunk_overlap", "message": "Chunk overlap comparison requires reprocessing documents"})
    
    return results

def display_retrieval_comparison(results):
    """
    Display retrieval comparison results
    
    Args:
        results (dict): Dictionary of comparison results from compare_retrieval_parameters
    """
    st.subheader(f"Comparison Results for: '{results['question']}'")
    
    for comparison in results["comparisons"]:
        if comparison["type"] == "k_value" and "values" in comparison:
            # Create a tab for each k value
            if len(comparison["values"]) > 0:
                k_tabs = st.tabs([f"k = {k}" for k in comparison["values"]])
                
                for i, k in enumerate(comparison["values"]):
                    with k_tabs[i]:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("Retrieved Context")
                            st.text_area(f"Context (k={k})", comparison["contexts"][i], height=400)
                            
                            # Show context length
                            if comparison["contexts"][i]:
                                context_length = len(comparison["contexts"][i])
                                st.caption(f"Context length: {context_length} characters")
                                
                                # Count chunks
                                chunks = comparison["contexts"][i].split("\n\n---\n\n")
                                st.caption(f"Chunks retrieved: {len(chunks)}")
                        
                        with col2:
                            st.subheader("Answer")
                            st.markdown(comparison["answers"][i])
                
                # Create a comparison view
                st.subheader("Context Length Comparison")
                
                # Extract context lengths
                context_lengths = [len(ctx) for ctx in comparison["contexts"]]
                
                # Create a bar chart of context lengths
                fig = px.bar(
                    x=[f"k = {k}" for k in comparison["values"]],
                    y=context_lengths,
                    labels={"x": "k value", "y": "Context Length (chars)"},
                    title="Context Length by k Value"
                )
                st.plotly_chart(fig)
                
                # Show metrics
                metrics_df = pd.DataFrame({
                    "k Value": comparison["values"],
                    "Context Length": context_lengths,
                    "Chunks Retrieved": [len(ctx.split("\n\n---\n\n")) if ctx else 0 for ctx in comparison["contexts"]],
                    "Avg. Chunk Length": [
                        sum([len(chunk) for chunk in ctx.split("\n\n---\n\n")]) / max(1, len(ctx.split("\n\n---\n\n"))) 
                        if ctx else 0 
                        for ctx in comparison["contexts"]
                    ]
                })
                st.dataframe(metrics_df)
        else:
            st.info(comparison.get("message", "No comparison data available"))

def analyze_document_chunks(chatbot):
    """
    Analyze document chunks in the vectorstore
    
    Args:
        chatbot: The chatbot instance to use
    """
    if not hasattr(chatbot, 'vectorstore') or not chatbot.vectorstore:
        st.warning("No vectorstore available. Please process documents first.")
        return
    
    # Get document chunks from vectorstore
    # This implementation may vary based on your vectorstore implementation
    try:
        # Assuming FAISS or Chroma vectorstore with get_documents method
        if hasattr(chatbot.vectorstore, 'get'):
            documents = chatbot.vectorstore.get()
            if not documents:
                st.warning("No documents found in vectorstore.")
                return
        # If vectorstore is accessible through a different method, adjust accordingly
        elif hasattr(chatbot.vectorstore, '_collection'):
            documents = chatbot.vectorstore._collection.get()
        else:
            st.warning("Vectorstore type not supported for detailed analysis.")
            return
            
        # Extract document contents and metadata
        doc_contents = []
        doc_metadata = []
        
        # Handle different vectorstore types
        if isinstance(documents, dict) and 'documents' in documents:
            doc_contents = documents['documents']
            if 'metadatas' in documents:
                doc_metadata = documents['metadatas']
        elif isinstance(documents, list):
            # Assume list of document objects with page_content and metadata
            doc_contents = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
            doc_metadata = [doc.metadata if hasattr(doc, 'metadata') else {} for doc in documents]
        
        # Create analysis
        st.subheader("Document Chunk Analysis")
        
        # Create dataframe for analysis
        chunk_data = []
        for i, (content, metadata) in enumerate(zip(doc_contents, doc_metadata if doc_metadata else [{}] * len(doc_contents))):
            chunk_data.append({
                "Chunk ID": i,
                "Content Length": len(content),
                "Content Preview": content[:100] + "..." if len(content) > 100 else content,
                "Source": metadata.get("source", "Unknown"),
                "Page": metadata.get("page", "N/A")
            })
        
        chunk_df = pd.DataFrame(chunk_data)
        
        # Display chunk statistics
        st.write("### Chunk Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Chunks", len(chunk_data))
        
        with col2:
            avg_length = chunk_df["Content Length"].mean()
            st.metric("Avg. Chunk Length", f"{avg_length:.1f}")
        
        with col3:
            min_length = chunk_df["Content Length"].min()
            st.metric("Min Chunk Length", min_length)
        
        with col4:
            max_length = chunk_df["Content Length"].max()
            st.metric("Max Chunk Length", max_length)
        
        # Create histogram of chunk lengths
        st.write("### Chunk Length Distribution")
        fig = px.histogram(
            chunk_df, 
            x="Content Length",
            title="Distribution of Chunk Lengths",
            nbins=20
        )
        st.plotly_chart(fig)
        
        # Display chunk data table
        st.write("### Document Chunks")
        st.dataframe(chunk_df)
        
        # Source document distribution
        if "Source" in chunk_df.columns:
            st.write("### Source Document Distribution")
            source_counts = chunk_df["Source"].value_counts().reset_index()
            source_counts.columns = ["Source", "Chunk Count"]
            
            fig = px.bar(
                source_counts,
                x="Source",
                y="Chunk Count",
                title="Chunks per Source Document"
            )
            st.plotly_chart(fig)
            
            # Display source statistics
            st.dataframe(source_counts)
    
    except Exception as e:
        st.error(f"Error analyzing document chunks: {str(e)}")

def generate_test_questions(num_questions=5):
    """
    Generate test questions for RAG evaluation
    
    Args:
        num_questions (int): Number of test questions to generate
    
    Returns:
        list: List of test questions
    """
    # In a real implementation, this would generate questions based on document content
    # Here we provide some generic sample questions
    sample_questions = [
        "What are the main topics covered in the documents?",
        "Summarize the key points of the documents.",
        "What is the relationship between X and Y in the documents?",
        "How does the document explain the process of Z?",
        "What evidence supports the main argument in the documents?",
        "What are the limitations mentioned in the documents?",
        "How do the documents compare different approaches to the problem?",
        "What are the recommendations provided in the documents?",
        "What data or metrics are presented in the documents?",
        "What is the historical context provided in the documents?"
    ]
    
    # Return requested number of questions
    return sample_questions[:min(num_questions, len(sample_questions))]

def run_rag_evaluation(chatbot, test_questions):
    """
    Run RAG evaluation with test questions
    
    Args:
        chatbot: The chatbot instance to use
        test_questions (list): List of test questions to evaluate
    
    Returns:
        dict: Evaluation results
    """
    results = {
        "questions": test_questions,
        "answers": [],
        "contexts": [],
        "metrics": [],
        "timestamp": datetime.now().isoformat()
    }
    
    for question in test_questions:
        with st.spinner(f"Evaluating question: {question}"):
            response = chatbot.ask(question, return_context=True)
            
            if isinstance(response, dict):
                results["answers"].append(response.get("answer", str(response)))
                context = response.get("retrieved_context", "")
                results["contexts"].append(context)
                
                # Calculate metrics
                chunks = context.split("\n\n---\n\n") if context else []
                chunk_lengths = [len(chunk) for chunk in chunks]
                
                metrics = {
                    "num_chunks": len(chunks),
                    "avg_chunk_length": sum(chunk_lengths) / max(1, len(chunk_lengths)) if chunk_lengths else 0,
                    "total_context_length": len(context),
                    "retrieval_time": response.get("retrieval_time", 0),
                    "generation_time": response.get("generation_time", 0)
                }
                results["metrics"].append(metrics)
            else:
                results["answers"].append(str(response))
                results["contexts"].append("")
                results["metrics"].append({})
    
    return results

def display_evaluation_results(results):
    """
    Display RAG evaluation results
    
    Args:
        results (dict): Evaluation results from run_rag_evaluation
    """
    st.subheader("RAG Evaluation Results")
    
    # Create tabs for each question
    if results["questions"]:
        q_tabs = st.tabs([f"Q{i+1}" for i in range(len(results["questions"]))])
        
        for i, question in enumerate(results["questions"]):
            with q_tabs[i]:
                st.write(f"**Question:** {question}")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Retrieved Context")
                    st.text_area(f"Context", results["contexts"][i], height=300)
                    
                    # Context metrics
                    if results["contexts"][i]:
                        metrics = results["metrics"][i]
                        st.caption(f"Context length: {metrics.get('total_context_length', 0)} characters")
                        st.caption(f"Chunks retrieved: {metrics.get('num_chunks', 0)}")
                
                with col2:
                    st.subheader("Answer")
                    st.markdown(results["answers"][i])
                    
                    # Time metrics if available
                    metrics = results["metrics"][i]
                    if "retrieval_time" in metrics:
                        st.caption(f"Retrieval time: {metrics['retrieval_time']:.2f}s")
                    if "generation_time" in metrics:
                        st.caption(f"Generation time: {metrics['generation_time']:.2f}s")
        
        # Create summary metrics
        st.subheader("Evaluation Summary")
        
        # Calculate average metrics
        avg_metrics = {
            "avg_chunks": sum(m.get("num_chunks", 0) for m in results["metrics"]) / max(1, len(results["metrics"])),
            "avg_context_length": sum(m.get("total_context_length", 0) for m in results["metrics"]) / max(1, len(results["metrics"])),
            "avg_retrieval_time": sum(m.get("retrieval_time", 0) for m in results["metrics"]) / max(1, len(results["metrics"])),
            "avg_generation_time": sum(m.get("generation_time", 0) for m in results["metrics"]) / max(1, len(results["metrics"]))
        }
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg. Chunks", f"{avg_metrics['avg_chunks']:.1f}")
        
        with col2:
            st.metric("Avg. Context Length", f"{avg_metrics['avg_context_length']:.1f}")
        
        with col3:
            st.metric("Avg. Retrieval Time", f"{avg_metrics['avg_retrieval_time']:.2f}s")
        
        with col4:
            st.metric("Avg. Generation Time", f"{avg_metrics['avg_generation_time']:.2f}s")
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame([
            {
                "Question": f"Q{i+1}",
                "Chunks": m.get("num_chunks", 0),
                "Context Length": m.get("total_context_length", 0),
                "Retrieval Time (s)": m.get("retrieval_time", 0),
                "Generation Time (s)": m.get("generation_time", 0)
            }
            for i, m in enumerate(results["metrics"])
        ])
        
        st.dataframe(metrics_df)
    else:
        st.info("No evaluation results available.")

def export_debug_report(rag_type, chatbot, retrieval_results, evaluation_results):
    """
    Export a debug report with all analysis results
    
    Args:
        rag_type (str): Type of RAG app ('pdf', 'csv', 'sqlite')
        chatbot: The chatbot instance
        retrieval_results (dict): Retrieval parameter comparison results
        evaluation_results (dict): Evaluation results
    
    Returns:
        str: Path to the saved report file
    """
    # Create report directory if it doesn't exist
    report_dir = "debug_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"rag_debug_report_{rag_type}_{timestamp}.md")
    
    # Generate report content
    report_content = f"""# RAG Debugging Report
    
## Overview
- **RAG Type:** {rag_type.upper()}
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Model:** {st.session_state.get("model_name", "Unknown")}

## Configuration
- **k value:** {st.session_state.k_value}
- **Chunk size:** {st.session_state.chunk_size}
- **Chunk overlap:** {st.session_state.chunk_overlap}

"""
    
    # Add document stats if available
    try:
        if hasattr(chatbot, 'vectorstore') and chatbot.vectorstore:
            # Get document count if available
            doc_count = 0
            if hasattr(chatbot.vectorstore, '_collection'):
                doc_stats = chatbot.vectorstore._collection.count()
                doc_count = doc_stats
            
            report_content += f"""## Document Statistics
- **Documents in vectorstore:** {doc_count}
"""
            
            # Add file metadata if available
            if hasattr(chatbot, 'file_metadata') and chatbot.file_metadata:
                report_content += "- **Processed files:**\n"
                for file_name, metadata in chatbot.file_metadata.items():
                    report_content += f"  - {file_name}\n"
    except Exception as e:
        report_content += f"Error getting document statistics: {str(e)}\n\n"
    
    # Add retrieval comparison results if available
    if retrieval_results:
        report_content += f"""
## Parameter Comparison
- **Test question:** {retrieval_results.get("question", "N/A")}
- **Timestamp:** {retrieval_results.get("timestamp", "N/A")}

"""
        
        for comparison in retrieval_results.get("comparisons", []):
            if comparison["type"] == "k_value" and "values" in comparison:
                report_content += "### k Value Comparison\n\n"
                
                # Add metrics table
                report_content += "| k Value | Chunks Retrieved | Context Length | Avg. Chunk Length |\n"
                report_content += "|---------|-----------------|----------------|------------------|\n"
                
                for i, k in enumerate(comparison["values"]):
                    context = comparison["contexts"][i]
                    chunks = context.split("\n\n---\n\n") if context else []
                    chunk_lengths = [len(chunk) for chunk in chunks]
                    avg_chunk_length = sum(chunk_lengths) / max(1, len(chunk_lengths)) if chunk_lengths else 0
                    
                    report_content += f"| {k} | {len(chunks)} | {len(context)} | {avg_chunk_length:.1f} |\n"
                
                report_content += "\n"
    
    # Add evaluation results if available
    if evaluation_results:
        report_content += f"""
## RAG Evaluation
- **Timestamp:** {evaluation_results.get("timestamp", "N/A")}
- **Number of test questions:** {len(evaluation_results.get("questions", []))}

"""
        
        if "questions" in evaluation_results and evaluation_results["questions"]:
            # Add questions and metrics
            report_content += "### Test Questions and Metrics\n\n"
            report_content += "| Question | Chunks | Context Length | Retrieval Time | Generation Time |\n"
            report_content += "|----------|--------|----------------|---------------|----------------|\n"
            
            for i, question in enumerate(evaluation_results["questions"]):
                metrics = evaluation_results["metrics"][i]
                report_content += f"| Q{i+1}: {question[:30]}... | {metrics.get('num_chunks', 0)} | {metrics.get('total_context_length', 0)} | {metrics.get('retrieval_time', 0):.2f}s | {metrics.get('generation_time', 0):.2f}s |\n"
            
            report_content += "\n"
            
            # Calculate average metrics
            avg_metrics = {
                "avg_chunks": sum(m.get("num_chunks", 0) for m in evaluation_results["metrics"]) / max(1, len(evaluation_results["metrics"])),
                "avg_context_length": sum(m.get("total_context_length", 0) for m in evaluation_results["metrics"]) / max(1, len(evaluation_results["metrics"])),
                "avg_retrieval_time": sum(m.get("retrieval_time", 0) for m in evaluation_results["metrics"]) / max(1, len(evaluation_results["metrics"])),
                "avg_generation_time": sum(m.get("generation_time", 0) for m in evaluation_results["metrics"]) / max(1, len(evaluation_results["metrics"]))
            }
            
            report_content += "### Summary Metrics\n\n"
            report_content += f"- **Average chunks retrieved:** {avg_metrics['avg_chunks']:.1f}\n"
            report_content += f"- **Average context length:** {avg_metrics['avg_context_length']:.1f}\n"
            report_content += f"- **Average retrieval time:** {avg_metrics['avg_retrieval_time']:.2f}s\n"
            report_content += f"- **Average generation time:** {avg_metrics['avg_generation_time']:.2f}s\n"
    
    # Add timestamp and footer
    report_content += f"""
## Conclusion
This report was generated automatically by the RAG Debugging Tools.

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    # Write report to file
    with open(report_file, "w") as f:
        f.write(report_content)
    
    return report_file