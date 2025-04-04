import tempfile
import os
import pandas as pd
import io  # Import the standard io module
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
            
            # Get DataFrame info as a string - using standard io.StringIO instead of pandas.io.StringIO
            buffer = io.StringIO()
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
            try:
                # Simpler approach - create a single document with full DataFrame summary
                full_csv_content = f"""
                CSV File: {file.name}
                Shape: {df.shape[0]} rows × {df.shape[1]} columns
                Columns: {', '.join(df.columns.tolist())}
                
                DataFrame Summary:
                {df_info}
                
                Statistics:
                {df_stats}
                
                Sample Data:
                {df_sample}
                """
                
                # Create a single document with the full content
                doc = Document(
                    page_content=full_csv_content,
                    metadata={
                        "file_type": "csv",
                        "file_name": file.name,
                        "row_count": df.shape[0],
                        "column_count": df.shape[1],
                        "file_info": csv_metadata
                    }
                )
                
                # Add to documents collection
                self.documents.append(doc)
                
                # Also add documents for each column
                for col in df.columns:
                    col_stats = df[col].describe().to_string() if pd.api.types.is_numeric_dtype(df[col]) else f"Unique values: {df[col].nunique()}"
                    col_content = f"""
                    Column: {col}
                    Data Type: {df[col].dtype}
                    
                    Statistics:
                    {col_stats}
                    
                    Sample Values:
                    {df[col].head(10).to_string()}
                    """
                    
                    col_doc = Document(
                        page_content=col_content,
                        metadata={
                            "file_type": "csv_column",
                            "file_name": file.name,
                            "column_name": col,
                            "file_info": csv_metadata
                        }
                    )
                    
                    self.documents.append(col_doc)
                
            except Exception as column_error:
                st.warning(f"Warning creating column-specific documents: {str(column_error)}")
                # Fall back to a simpler approach
                doc = Document(
                    page_content=csv_metadata,
                    metadata={
                        "file_type": "csv",
                        "file_name": file.name
                    }
                )
                self.documents.append(doc)
            
            # Clean up
            os.unlink(tmp_path)
            
            return df.shape[0]  # Return number of rows processed
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
        self.chat_history = []
        self.data_frames = {}
        
        # Note: We don't reset the vectorstore to preserve persistence across sessions
        # If you want to truly clear everything, uncomment below:
        # import shutil
        # if os.path.exists(self.persist_directory):
        #     shutil.rmtree(self.persist_directory)
        # self.vectorstore = None