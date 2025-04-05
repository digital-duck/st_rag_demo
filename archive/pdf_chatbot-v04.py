import tempfile
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from .document_chatbot import DocumentChatbot

class PDFChatbot(DocumentChatbot):
    """Chatbot specialized for PDF document processing and Q&A"""
    
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        """Initialize PDF chatbot with the base document chatbot functionality"""
        super().__init__(api_key=api_key, model_name=model_name)
    
    def process_pdf(self, file):
        """Process a PDF file and add to documents"""
        try:
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            # Load PDF file
            loader = PyPDFLoader(tmp_path)
            pdf_docs = loader.load()
            
            # Create metadata for the PDF
            pdf_metadata = f"""
            PDF File: {file.name}
            Total Pages: {len(pdf_docs)}
            """
            
            # Store metadata
            self.file_metadata[file.name] = pdf_metadata
            
            # Add metadata to documents
            for doc in pdf_docs:
                doc.metadata["file_type"] = "pdf"
                doc.metadata["file_name"] = file.name
                doc.metadata["file_info"] = pdf_metadata
            
            # Add to documents collection
            self.documents.extend(pdf_docs)
            
            # Clean up
            os.unlink(tmp_path)
            
            return len(pdf_docs)
        except Exception as e:
            st.error(f"Error processing PDF file: {str(e)}")
            return 0
            
    def ask(self, question, return_context=False):
        """Ask a question about the PDF documents"""
        try:
            # Check if vectorstore exists
            if not self.vectorstore:
                error_msg = "Please process PDF files first to create a knowledge base."
                if return_context:
                    return {
                        "answer": error_msg,
                        "retrieved_context": "No vector database available.",
                        "formatted_prompt": "Error: No vector database"
                    }
                return error_msg
                
            # Get the response from the base class
            response = super().ask(question, return_context=True)
            
            # If return_context is true, we need to return the full context
            if return_context and isinstance(response, dict):
                return response
            elif isinstance(response, dict) and "answer" in response:
                # Just return the answer
                return response["answer"]
            else:
                # Return the response as is
                return response
                
        except Exception as e:
            error_msg = f"Error querying PDF knowledge base: {str(e)}"
            if return_context:
                return {
                    "answer": error_msg,
                    "retrieved_context": "Error occurred during retrieval",
                    "formatted_prompt": "Error occurred"
                }
            return error_msg
    
    def clear(self):
        """Clear all documents and reset the chatbot"""
        # Call the parent clear method
        super().clear()