import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st

def process_pdf_file(file):
    """
    Process a PDF file and return the extracted document pages
    
    Args:
        file: A streamlit uploaded file object
        
    Returns:
        tuple: (pdf_docs, tmp_path) - A tuple of the extracted documents and the temp file path
    """
    try:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        # Load PDF file
        loader = PyPDFLoader(tmp_path)
        pdf_docs = loader.load()
        
        return pdf_docs, tmp_path
    except Exception as e:
        st.error(f"Error processing PDF file: {str(e)}")
        return [], None

def extract_pdf_metadata(file, pdf_docs):
    """
    Extract metadata from a PDF file
    
    Args:
        file: A streamlit uploaded file object
        pdf_docs: The extracted document pages
        
    Returns:
        str: A string containing the PDF metadata
    """
    return f"""
    PDF File: {file.name}
    Total Pages: {len(pdf_docs)}
    """

def cleanup_temp_file(tmp_path):
    """
    Clean up a temporary file
    
    Args:
        tmp_path: The path to the temporary file
    """
    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)