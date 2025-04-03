import tempfile
import os
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
from .document_chatbot import DocumentChatbot

class CSVChatbot(DocumentChatbot):
    """Chatbot specialized for CSV data processing and Q&A"""
    
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        """Initialize CSV chatbot with the base document chatbot functionality"""
        super().__init__(api_key=api_key, model_name=model_name)
        # Store the actual DataFrame objects for visualization
        self.data_frames = {}
    
    def process_csv(self, file):
        """Process a CSV file and add to documents"""
        try:
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            # Load CSV file into a pandas DataFrame
            df = pd.read_csv(tmp_path)
            
            # Store the DataFrame for later use in visualizations
            self.data_frames[file.name] = df
            
            # Get DataFrame info as a string
            buffer = pd.io.StringIO()
            df.info(buf=buffer)
            df_info = buffer.getvalue()
            
            # Get DataFrame statistics
            df_stats = df.describe().to_string()
            
            # Sample rows from the DataFrame
            df_sample = df.head(5).to_string()
            
            # Create metadata for the DataFrame
            csv_metadata = f"""
            CSV File: {file.name}
            
            DataFrame Information:
            {df_info}
            
            DataFrame Statistics:
            {df_stats}
            
            Sample Data (first 5 rows):
            {df_sample}
            """
            
            # Store metadata
            self.file_metadata[file.name] = csv_metadata
            
            # Convert DataFrame to documents
            # Use the first column as the page content, or a concatenation of all columns
            if len(df.columns) > 0:
                # Create multiple documents, one for each row with all columns
                documents = []
                for idx, row in df.iterrows():
                    # Convert row to string with column names
                    row_content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                    
                    # Create document
                    doc = Document(
                        page_content=row_content,
                        metadata={
                            "file_type": "csv",
                            "file_name": file.name,
                            "row_index": idx,
                            "file_info": csv_metadata
                        }
                    )
                    documents.append(doc)
                
                # Add to documents collection
                self.documents.extend(documents)
            else:
                # Fallback method using DataFrameLoader
                loader = DataFrameLoader(df)
                csv_docs = loader.load()
                
                # Add metadata to documents
                for doc in csv_docs:
                    doc.metadata["file_type"] = "csv"
                    doc.metadata["file_name"] = file.name
                    doc.metadata["file_info"] = csv_metadata
                
                # Add to documents collection
                self.documents.extend(csv_docs)
            
            # Clean up
            os.unlink(tmp_path)
            
            return len(df)
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
            return 0
            
    def ask(self, question, return_context=False):
        """Ask a question about the CSV data"""
        try:
            # Check if vectorstore exists
            if not self.vectorstore:
                error_msg = "Please process CSV files first to create a knowledge base."
                if return_context:
                    return {
                        "answer": error_msg,
                        "retrieved_context": "No vector database available.",
                        "formatted_prompt": "Error: No vector database"
                    }
                return error_msg
                
            # Check if this is a visualization-related question
            viz_keywords = ["chart", "plot", "graph", "visualize", "visualization", "show", "display"]
            is_viz_query = any(keyword in question.lower() for keyword in viz_keywords)
            
            # Enhance the prompt for visualization queries
            if is_viz_query:
                # Get the response from the base class with enhanced prompt
                viz_question = f"""
                {question}
                
                If this is asking for a visualization, please:
                1. Describe what the visualization would look like
                2. Recommend the best chart type (bar, line, scatter, pie)
                3. Specify which columns would be most appropriate for the x and y axes
                4. Include any relevant statistics or trends that would be shown in the visualization
                """
                
                response = super().ask(viz_question, return_context=True)
            else:
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
            error_msg = f"Error querying CSV knowledge base: {str(e)}"
            if return_context:
                return {
                    "answer": error_msg,
                    "retrieved_context": "Error occurred during retrieval",
                    "formatted_prompt": "Error occurred"
                }
            return error_msg
    
    def clear(self):
        """Clear all documents and reset the chatbot"""
        self.documents = []
        self.file_metadata = {}
        self.vectorstore = None
        self.chat_history = []
        self.data_frames = {}
        
        # Create a new temporary directory for Chroma DB
        self.persist_directory = tempfile.mkdtemp()