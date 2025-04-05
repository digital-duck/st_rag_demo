import tempfile
import os
import pandas as pd
import sqlite3
import streamlit as st
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase
from .document_chatbot import DocumentChatbot

class SQLiteChatbot(DocumentChatbot):
    """Chatbot specialized for SQLite database processing and Q&A"""
    
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        """Initialize SQLite chatbot with the base document chatbot functionality"""
        super().__init__(api_key=api_key, model_name=model_name)
        
        # Initialize SQLite related attributes
        self.db_connection = None
        self.db_path = None
        self.sql_database = None
        self.table_info = {}
        
        # Store query results for visualization
        self.last_query_results = None
    
    def process_sqlite(self, file):
        """
        Process a SQLite database file and set up SQL database connection.
        Only schema information is vectorized, not the actual data rows.
        """
        try:
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            # Store the path for later use
            self.db_path = tmp_path
            
            # Connect to the database
            self.db_connection = sqlite3.connect(tmp_path)
            
            # Get list of tables
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # Get schema for each table
            table_info = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                # Get sample data (first 5 rows)
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                sample_data = cursor.fetchall()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                
                # Store table info
                table_info[table_name] = {
                    "columns": columns,
                    "sample_data": sample_data,
                    "row_count": row_count
                }
            
            # Store table info for later use
            self.table_info = table_info
            
            # Create a LangChain SQLDatabase object
            self.sql_database = SQLDatabase.from_uri(f"sqlite:///{tmp_path}")
            
            # Create database schema metadata
            db_metadata = self._create_database_metadata(file.name, table_info)
            
            # Add to file metadata
            self.file_metadata[file.name] = db_metadata
            
            # Generate documents for vectorstore - ONLY schema information, not data
            self._generate_schema_documents(file.name, table_info, db_metadata)
            
            return len(tables)
            
        except Exception as e:
            st.error(f"Error processing SQLite file: {str(e)}")
            if self.db_connection:
                self.db_connection.close()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return 0
            
    def _create_database_metadata(self, filename, table_info):
        """Create a metadata string describing the database schema"""
        db_metadata = f"""
        SQLite Database: {filename}
        
        Tables:
        """
        
        for table_name, info in table_info.items():
            db_metadata += f"\n- {table_name} ({info['row_count']} rows)"
            db_metadata += "\n  Columns: "
            db_metadata += ", ".join([f"{col[1]} ({col[2]})" for col in info['columns']])
        
        return db_metadata
        
    def _generate_schema_documents(self, filename, table_info, db_metadata):
        """
        Generate documents for vectorstore containing only schema information.
        This explicitly avoids vectorizing the actual data rows.
        """
        # Create a document for overall database structure
        overall_schema = f"""
        Database: {filename}
        
        Schema Overview:
        """
        
        for table_name, info in table_info.items():
            overall_schema += f"\nTable: {table_name}"
            overall_schema += f"\nColumns: {', '.join([col[1] for col in info['columns']])}"
        
        # Add overall schema document
        doc = Document(
            page_content=overall_schema,
            metadata={
                "file_type": "sqlite_schema",
                "file_name": filename,
                "content_type": "database_overview",
                "file_info": db_metadata
            }
        )
        self.documents.append(doc)
        
        # Add individual table schema documents
        for table_name, info in table_info.items():
            # Create detailed schema information without actual data values
            columns_desc = "\n".join([f"- {col[1]} ({col[2]})" for col in info['columns']])
            foreign_keys = self._get_foreign_keys(table_name)
            
            content = f"""
            Table: {table_name}
            Row count: {info['row_count']}
            
            Columns:
            {columns_desc}
            
            {foreign_keys}
            """
            
            # Create a document with table schema only
            doc = Document(
                page_content=content,
                metadata={
                    "file_type": "sqlite_schema",
                    "file_name": filename,
                    "table_name": table_name,
                    "content_type": "table_schema",
                    "file_info": db_metadata
                }
            )
            
            # Add to documents collection
            self.documents.append(doc)
            
    def _get_foreign_keys(self, table_name):
        """Get foreign key relationships for a table"""
        if not self.db_connection:
            return ""
            
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            fk_data = cursor.fetchall()
            
            if not fk_data:
                return "Foreign Keys: None"
                
            fk_info = "Foreign Keys:\n"
            for fk in fk_data:
                fk_info += f"- {fk[3]} references {fk[2]}({fk[4]})\n"
                
            return fk_info
        except Exception:
            return "Foreign Keys: Could not retrieve"

    def execute_sql_query(self, query):
        """Execute a SQL query on the connected database"""
        if not self.db_connection:
            return "No database connected. Please upload a SQLite database file."
        
        try:
            # Execute the query
            cursor = self.db_connection.cursor()
            cursor.execute(query)
            
            # Check if it's a SELECT query
            if query.strip().upper().startswith("SELECT"):
                # Fetch all results
                results = cursor.fetchall()
                
                # Get column names
                column_names = [description[0] for description in cursor.description]
                
                # Return as DataFrame
                df = pd.DataFrame(results, columns=column_names)
                
                # Store for visualization
                self.last_query_results = df
                
                return df
            else:
                # For non-SELECT queries, commit and return affected row count
                self.db_connection.commit()
                return f"Query executed successfully. Rows affected: {cursor.rowcount}"
                
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def generate_sql_query(self, question):
        """Generate a SQL query based on a natural language question"""
        if not self.sql_database:
            return "No database connected. Please upload a SQLite database file."
        
        try:
            # We'll use LLM directly with an appropriate prompt
            # Get database schema information
            schema_info = self.sql_database.get_table_info()
            
            # Construct a prompt for the LLM
            prompt = f"""You are an expert in converting natural language questions into SQL queries.
            
            The database schema is as follows:
            {schema_info}
            
            Please convert the following natural language question to a valid SQL query:
            "{question}"
            
            Return only the SQL query without any explanations or markdown formatting.
            """
            
            # Create messages for LLM
            messages = [
                {"role": "system", "content": "You are an expert SQL assistant that converts natural language to SQL queries."},
                {"role": "user", "content": prompt}
            ]
            
            # Get response from LLM
            llm_response = self.llm.invoke(messages)
            generated_query = llm_response.content.strip()
            
            # Clean up the query - if it includes triple backticks, extract just the SQL
            if "```sql" in generated_query:
                # Extract content between ```sql and ```
                generated_query = generated_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in generated_query:
                # Extract content between ``` and ```
                generated_query = generated_query.split("```")[1].strip()
            
            # Add the generated query to the chat history
            self.chat_history.append(("human", question))
            self.chat_history.append(("ai", f"I've generated the following SQL query based on your question:\n\n```sql\n{generated_query}\n```\n\nLet me execute this query for you."))
            
            return generated_query
            
        except Exception as e:
            return f"Error generating SQL query: {str(e)}"
    
    def ask(self, question, return_context=False):
        """
        Ask a question about the database
        
        Args:
            question (str): The question to ask
            return_context (bool): Whether to return additional context
            
        Returns:
            str or dict: Response or dict with additional info
        """
        try:
            # Check if database is connected
            if not self.db_connection or not self.sql_database:
                error_msg = "Please upload and process a SQLite database file first."
                if return_context:
                    return {
                        "answer": error_msg,
                        "query_type": "error",
                        "retrieved_context": "",
                        "formatted_prompt": ""
                    }
                return error_msg
            
            # Detect if this is a SQL-related query or a visualization request
            sql_keywords = ["sql", "query", "database", "table", "select", "from", "where", "group by", "order by"]
            viz_keywords = ["plot", "chart", "graph", "visualize", "visualization", "show", "display"]
            
            is_sql_query = any(keyword in question.lower() for keyword in sql_keywords)
            is_viz_query = any(keyword in question.lower() for keyword in viz_keywords)
            
            # If it's a direct SQL query or visualization request
            if "write a query" in question.lower() or "generate a query" in question.lower() or is_sql_query:
                # Generate SQL query
                generated_query = self.generate_sql_query(question)
                
                if generated_query.startswith("Error"):
                    # Return the error
                    response = {
                        "answer": generated_query,
                        "query_type": "error",
                        "retrieved_context": "",
                        "formatted_prompt": ""
                    }
                else:
                    try:
                        # Execute the query
                        query_result = self.execute_sql_query(generated_query)
                        
                        if isinstance(query_result, pd.DataFrame):
                            # Format the answer
                            num_rows = len(query_result)
                            answer = f"I've executed the following SQL query based on your question:\n\n```sql\n{generated_query}\n```\n\nThe query returned {num_rows} rows."
                            
                            if num_rows > 0 and num_rows <= 10:
                                # Show the full result for small datasets
                                answer += f"\n\nHere's the result:\n\n{query_result.to_markdown()}"
                            elif num_rows > 10:
                                # Show just a sample for larger datasets
                                answer += f"\n\nHere's a sample of the results (first 5 rows):\n\n{query_result.head(5).to_markdown()}"
                            
                            # For visualization queries, add visualization recommendations
                            if is_viz_query:
                                numeric_cols = query_result.select_dtypes(include=['number']).columns.tolist()
                                categorical_cols = query_result.select_dtypes(exclude=['number']).columns.tolist()
                                
                                answer += "\n\n**Visualization Recommendations:**\n"
                                
                                if len(numeric_cols) >= 2:
                                    answer += f"\n- You could create a scatter plot using {numeric_cols[0]} and {numeric_cols[1]} for the axes."
                                
                                if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                                    answer += f"\n- A bar chart with {categorical_cols[0]} on the x-axis and {numeric_cols[0]} on the y-axis would be effective."
                                
                                if len(numeric_cols) >= 1:
                                    answer += f"\n- A histogram of {numeric_cols[0]} would show its distribution."
                            
                            response = {
                                "answer": answer,
                                "query_type": "visualization" if is_viz_query else "sql_execution",
                                "query": generated_query,
                                "data": query_result,
                                "retrieved_context": "",
                                "formatted_prompt": ""
                            }
                        else:
                            # Non-SELECT query result
                            response = {
                                "answer": f"I've executed the following SQL query based on your question:\n\n```sql\n{generated_query}\n```\n\nResult: {query_result}",
                                "query_type": "sql_execution",
                                "query": generated_query,
                                "retrieved_context": "",
                                "formatted_prompt": ""
                            }
                    except Exception as execution_error:
                        response = {
                            "answer": f"I generated the following SQL query:\n\n```sql\n{generated_query}\n```\n\nBut encountered an error when executing it: {str(execution_error)}",
                            "query_type": "error",
                            "query": generated_query,
                            "retrieved_context": "",
                            "formatted_prompt": ""
                        }
            else:
                # Use the RAG approach for general database questions
                response = super().ask(question, return_context=True)
            
            if return_context:
                return response
            elif isinstance(response, dict) and "answer" in response:
                return response["answer"]
            else:
                return response
                
        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
            if return_context:
                return {
                    "answer": error_msg,
                    "query_type": "error",
                    "retrieved_context": "",
                    "formatted_prompt": ""
                }
            return error_msg
    
    def clear(self):
        """Clear all documents and reset the chatbot"""
        self.documents = []
        self.file_metadata = {}
        self.chat_history = []
        self.last_query_results = None
        
        # Close the database connection if it exists
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None
        
        # Remove the database file if it exists
        if self.db_path and os.path.exists(self.db_path):
            os.unlink(self.db_path)
            self.db_path = None
        
        # Reset database-related attributes
        self.sql_database = None
        self.table_info = {}