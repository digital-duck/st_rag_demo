import os
import hashlib
import tempfile
import sqlite3
import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from .document_chatbot import DocumentChatbot

def format_query_response(query, query_result, include_table=False):
    """
    Format a SQL query response with optional table inclusion
    
    Args:
        query: The SQL query that was executed
        query_result: The results DataFrame
        include_table: Whether to include table markdown in the response
        
    Returns:
        str: Formatted response text
    """
    if isinstance(query_result, pd.DataFrame):
        num_rows = len(query_result)
        answer = f"I've executed the following SQL query based on your question:\n\n```sql\n{query}\n```\n\nThe query returned {num_rows} rows."
        
        # Only include the table in the text if explicitly requested
        if include_table:
            if num_rows > 0 and num_rows <= 10:
                # Show the full result for small datasets
                try:
                    # Try to use to_markdown if available
                    table_text = query_result.to_markdown()
                except:
                    # Fall back to string representation
                    table_text = query_result.to_string(index=False)
                    
                answer += f"\n\nHere's the result:\n\n```\n{table_text}\n```"
            elif num_rows > 10:
                # Show just a sample for larger datasets
                try:
                    # Try to use to_markdown if available
                    table_text = query_result.head(5).to_markdown()
                except:
                    # Fall back to string representation
                    table_text = query_result.head(5).to_string(index=False)
                    
                answer += f"\n\nHere's a sample of the results (first 5 rows):\n\n```\n{table_text}\n```"
        
        return answer
    else:
        # Non-DataFrame result
        return f"I've executed the following SQL query based on your question:\n\n```sql\n{query}\n```\n\nResult: {query_result}"



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
        
        # Create a persistent directory for this chatbot type
        self.sqlite_persist_dir = os.path.join(os.getcwd(), "data", "sqlite_data")
        os.makedirs(self.sqlite_persist_dir, exist_ok=True)

    def load_database(self, file_path, file_name):
        """Load a previously processed database by its path with thread safety"""
        try:
            # Store the path for later use in thread-safe connections
            self.db_path = file_path
            
            # Create a thread-local connection for initial setup
            thread_conn = sqlite3.connect(file_path)
            
            try:
                # Get list of tables
                cursor = thread_conn.cursor()
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
                # This will create its own connections when needed
                self.sql_database = SQLDatabase.from_uri(f"sqlite:///{file_path}")
                
                # Create database schema metadata
                db_metadata = self._create_database_metadata(file_name, table_info)
                
                # Add to file metadata
                self.file_metadata[file_name] = db_metadata
                
                # Get a unique identifier for this database to use in the persist directory
                db_id = hashlib.md5(file_name.encode()).hexdigest()[:10]
                vector_persist_dir = os.path.join(self.sqlite_persist_dir, f"vectors_{db_id}")
                
                # Load the persisted vectorstore if it exists
                if os.path.exists(vector_persist_dir) and os.path.isdir(vector_persist_dir):
                    self.vectorstore = Chroma(
                        persist_directory=vector_persist_dir,
                        embedding_function=self.embeddings
                    )
                    return len(tables)
                else:
                    # If vectorstore doesn't exist, recreate it
                    self._generate_schema_documents(file_name, table_info, db_metadata)
                    self._build_and_persist_vectorstore(vector_persist_dir)
                    return len(tables)
            finally:
                # Always close the thread-local connection
                if thread_conn:
                    thread_conn.close()
                
        except Exception as e:
            st.error(f"Error loading database: {str(e)}")
            return 0

    def process_sqlite(self, file):
        """
        Process a SQLite database file and set up SQL database connection.
        Only schema information is vectorized, not the actual data rows.
        Uses thread-safe connection handling.
        """
        try:
            # Generate a consistent filename based on the hash of the file content
            file_content = file.getvalue()
            file_hash = hashlib.md5(file_content).hexdigest()
            db_filename = f"{file_hash}_{file.name}"
            
            # Create a persistent path for this specific database
            persistent_db_path = os.path.join(self.sqlite_persist_dir, db_filename)
            
            # Check if we've already processed this exact file before
            if os.path.exists(persistent_db_path):
                # Use the existing database file
                self.db_path = persistent_db_path
            else:
                # Save the uploaded file to the persistent location
                with open(persistent_db_path, 'wb') as f:
                    f.write(file_content)
                self.db_path = persistent_db_path
            
            # Create a thread-local connection for setup
            thread_conn = sqlite3.connect(self.db_path)
            
            try:
                # Get list of tables
                cursor = thread_conn.cursor()
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
            finally:
                # Always close the thread-local connection
                if thread_conn:
                    thread_conn.close()
            
            # Store table info for later use
            self.table_info = table_info
            
            # Create a LangChain SQLDatabase object
            # This creates its own connections as needed
            self.sql_database = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            
            # Create database schema metadata
            db_metadata = self._create_database_metadata(file.name, table_info)
            
            # Add to file metadata
            self.file_metadata[file.name] = db_metadata
            
            # Generate documents for vectorstore - ONLY schema information, not data
            self._generate_schema_documents(file.name, table_info, db_metadata)
            
            # Get a unique identifier for this database to use in the persist directory
            db_id = hashlib.md5(file.name.encode()).hexdigest()[:10]
            vector_persist_dir = os.path.join(self.sqlite_persist_dir, f"vectors_{db_id}")
            os.makedirs(vector_persist_dir, exist_ok=True)
            
            # Build and persist the vectorstore
            self._build_and_persist_vectorstore(vector_persist_dir)
            
            return len(tables)
            
        except Exception as e:
            st.error(f"Error processing SQLite file: {str(e)}")
            return 0



    def _build_and_persist_vectorstore(self, persist_dir):
        """Build and persist the vectorstore to disk"""
        if not self.documents:
            return False
            
        try:
            # Update text splitter with current settings (if available)
            if hasattr(st, 'session_state') and 'chunk_size' in st.session_state and 'chunk_overlap' in st.session_state:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap
                )
                
            # Split documents into chunks
            splits = self.text_splitter.split_documents(self.documents)
            
            if not splits:
                return False
                
            # Create Chroma vectorstore with persistence
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            
            # Explicitly persist to disk
            self.vectorstore.persist()
            
            return True
        except Exception as e:
            st.error(f"Error building vector database: {str(e)}")
            return False


    def clear(self):
        """Clear all documents and reset the chatbot"""
        # Don't delete the database file or the vectorstore - we want persistence
        
        # Just close the connection
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None
        
        # Reset other state
        self.db_path = None
        self.sql_database = None
        self.table_info = {}
        self.documents = []
        self.file_metadata = {}
        self.chat_history = []
        self.last_query_results = None
        self.vectorstore = None

    def get_available_databases(self):
        """Get a list of previously processed databases"""
        try:
            databases = []
            if os.path.exists(self.sqlite_persist_dir):
                for filename in os.listdir(self.sqlite_persist_dir):
                    if filename.endswith(('.db', '.sqlite', '.sqlite3')) or '_' in filename:
                        # This looks like a saved database file
                        file_path = os.path.join(self.sqlite_persist_dir, filename)
                        if os.path.isfile(file_path):
                            # Extract the original filename (after the hash)
                            original_name = filename
                            if '_' in filename:
                                original_name = filename.split('_', 1)[1]
                            
                            # Check if vectorstore exists for this database
                            db_id = hashlib.md5(original_name.encode()).hexdigest()[:10]
                            vector_dir = os.path.join(self.sqlite_persist_dir, f"vectors_{db_id}")
                            has_vectors = os.path.exists(vector_dir) and os.path.isdir(vector_dir)
                            
                            databases.append({
                                "path": file_path,
                                "name": original_name,
                                "has_vectors": has_vectors
                            })
            return databases
        except Exception as e:
            st.error(f"Error listing databases: {str(e)}")
            return []
        
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
        from langchain_core.documents import Document
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
        """Execute a SQL query on the connected database with thread safety"""
        if not self.db_path:
            return "No database connected. Please upload a SQLite database file."
        
        try:
            # Create a new connection in the current thread
            # This is necessary because SQLite connections are thread-local
            # and cannot be shared between threads
            conn = sqlite3.connect(self.db_path)
            
            try:
                # Execute the query
                cursor = conn.cursor()
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
                    
                    # Close this thread's connection
                    conn.close()
                    
                    return df
                else:
                    # For non-SELECT queries, commit and return affected row count
                    conn.commit()
                    rowcount = cursor.rowcount
                    
                    # Close this thread's connection
                    conn.close()
                    
                    return f"Query executed successfully. Rows affected: {rowcount}"
            finally:
                # Ensure connection is closed even if an error occurs
                if conn:
                    conn.close()
                    
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    
    def generate_sql_query(self, question):
        """Generate a SQL query based on a natural language question"""
        if not self.sql_database:
            return "No database connected. Please upload a SQLite database file."
        
        try:
            # Get database schema information
            schema_info = self.sql_database.get_table_info()
            
            # Construct a prompt for the LLM that's more explicit about SQL generation
            prompt = f"""You are a SQL expert. Convert the following natural language question to a SQL query for a SQLite database.

    Database Schema:
    {schema_info}

    Natural Language Question: "{question}"

    Your task:
    1. Generate a valid SQL query that answers this question
    2. ONLY return the SQL query itself - no explanations, no markdown formatting
    3. Make sure to use proper table and column names from the schema
    4. If the question asks for a limited number of results, make sure to include LIMIT in your query
    5. If the question mentions ordering or ranking, use ORDER BY appropriately
    6. If the question is too vague or can't be answered with the schema, generate a simple exploratory query

    SQL Query:"""
            
            # Create messages for LLM with the enhanced prompt
            messages = [
                {"role": "system", "content": "You are an expert SQL query generator. You only respond with the SQL query, nothing else."},
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
            
            # If it contains explanations before or after the SQL, try to extract just the SQL query
            # Look for common SQL starting keywords
            sql_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
            for keyword in sql_keywords:
                if keyword in generated_query.upper():
                    # Find the start of the SQL query
                    start_idx = generated_query.upper().find(keyword)
                    if start_idx >= 0:
                        # Extract from the keyword to the end
                        query_part = generated_query[start_idx:]
                        # Look for potential end markers like explanations
                        end_markers = ["\n\n", "\nNote:", "\nExplanation:"]
                        for marker in end_markers:
                            if marker in query_part:
                                query_part = query_part.split(marker)[0]
                        generated_query = query_part.strip()
                        break
            
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
            # Check if database is connected using db_path instead of db_connection
            if not self.db_path or not self.sql_database:
                error_msg = "Please upload and process a SQLite database file first."
                if return_context:
                    return {
                        "answer": error_msg,
                        "query_type": "error",
                        "retrieved_context": "",
                        "formatted_prompt": ""
                    }
                return error_msg
            
            # Check if this looks like a request for SQL generation or data visualization
            sql_keywords = ["sql", "query", "table", "count", "find", "show", "list", "top", "customers", "orders", "sales"]
            viz_keywords = ["chart", "plot", "graph", "visualize", "visualization", "show", "display"]
            
            is_sql_query = any(keyword in question.lower() for keyword in sql_keywords)
            is_viz_query = any(keyword in question.lower() for keyword in viz_keywords)
            
            # SQL generation and execution logic
            if is_sql_query or "write a query" in question.lower() or "generate a query" in question.lower():
                try:
                    # Generate SQL query
                    generated_query = self.generate_sql_query(question)
                    
                    if generated_query.startswith("Error"):
                        # Return the error
                        return {
                            "answer": generated_query,
                            "query_type": "error",
                            "retrieved_context": "",
                            "formatted_prompt": ""
                        }
                    
                    # Try to execute the query
                    try:
                        query_result = self.execute_sql_query(generated_query)
                        if isinstance(query_result, pd.DataFrame):
                            # Format a helpful answer that includes the query but NOT the table output
                            num_rows = len(query_result)
                            answer = f"I've executed the following SQL query based on your question:\n\n```sql\n{generated_query}\n```\n\nThe query returned {num_rows} rows."
                            
                            # Don't add table text to the answer
                            
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
                            
                            return {
                                "answer": answer,
                                "query_type": "visualization" if is_viz_query else "sql_execution",
                                "query": generated_query,
                                "data": query_result,  # Include the raw data for display in the UI
                                "retrieved_context": "",
                                "formatted_prompt": ""
                            }
                        else:
                            # Non-SELECT query result
                            return {
                                "answer": f"I've executed the following SQL query based on your question:\n\n```sql\n{generated_query}\n```\n\nResult: {query_result}",
                                "query_type": "sql_execution",
                                "query": generated_query,
                                "retrieved_context": "",
                                "formatted_prompt": ""
                            }                        
                        
                    except Exception as execution_error:
                        return {
                            "answer": f"I generated the following SQL query:\n\n```sql\n{generated_query}\n```\n\nBut encountered an error when executing it: {str(execution_error)}",
                            "query_type": "error",
                            "query": generated_query,
                            "retrieved_context": "",
                            "formatted_prompt": ""
                        }
                except Exception as sql_error:
                    # If SQL generation fails, fall back to RAG
                    rag_response = super().ask(question, return_context=True)
                    if return_context:
                        return rag_response
                    return rag_response.get("answer", str(rag_response))
                    
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
    