# Location: pages/3_SQLite_RAG.py
# Refactored implementation of SQLite RAG using modular components
# With fixed example question buttons that properly trigger LLM calls

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.sqlite_chatbot import SQLiteChatbot
from models.chat_history_manager import ChatHistoryManager

# Import shared utilities
from utils.ui_helpers import (
    initialize_session_state, display_chat_messages,
    handle_chat_input, display_history_view,
    add_assistant_message, process_chat_message, save_chat_history
)
from utils.sidebar import display_sidebar

# Helper function for auto-visualizing data
def auto_visualize(data):
    """
    Automatically create an appropriate visualization based on the data
    
    Args:
        data (pd.DataFrame): DataFrame to visualize
        
    Returns:
        plotly.graph_objects.Figure or None: Plotly figure object or None if visualization failed
    """
    try:
        # Get column types
        cat_cols = data.select_dtypes(include=['object']).columns
        num_cols = data.select_dtypes(include=['number']).columns
        
        # Choose visualization based on data characteristics
        if len(num_cols) >= 2:
            # Scatter plot for numeric correlations
            return px.scatter(data, x=num_cols[0], y=num_cols[1], title=f"{num_cols[1]} vs {num_cols[0]}")
        elif len(cat_cols) > 0 and len(num_cols) > 0:
            # Bar chart for categorical vs numeric
            return px.bar(data, x=cat_cols[0], y=num_cols[0], title=f"{num_cols[0]} by {cat_cols[0]}")
        elif len(num_cols) > 0:
            # Line chart for single numeric column
            return px.line(data, y=num_cols[0], title=f"{num_cols[0]} over index")
        else:
            # Table view as fallback
            return go.Figure(data=[go.Table(
                header=dict(values=list(data.columns)),
                cells=dict(values=[data[col] for col in data.columns])
            )])
    except Exception as e:
        st.warning(f"Auto-visualization error: {str(e)}")
        return None

# Function to display database info (SQLite specific)
def display_database_info():
    """Display information about the connected database"""
    if hasattr(st.session_state.sqlite_chatbot, 'db_connection') and st.session_state.sqlite_chatbot.db_connection is not None:
        st.subheader("Connected Database")
        
        # Create tabs instead of nested expanders
        all_tables = list(st.session_state.sqlite_chatbot.table_info.keys())
        if all_tables:
            tabs = st.tabs(all_tables)
            
            # Display info for each table in its own tab
            for i, table_name in enumerate(all_tables):
                info = st.session_state.sqlite_chatbot.table_info[table_name]
                with tabs[i]:
                    st.write(f"**{table_name}** ({info['row_count']} rows)")
                    cols = ", ".join([f"{col[1]} ({col[2]})" for col in info['columns']])
                    st.write(f"Columns: {cols}")
                    
                    # Show sample data directly (no nested expander)
                    st.write("Sample data:")
                    # Convert sample data to DataFrame for better display
                    columns = [col[1] for col in info['columns']]
                    sample_df = pd.DataFrame(info['sample_data'], columns=columns)
                    st.dataframe(sample_df)

# Helper function to display SQL chat messages with visualizations
def display_sql_chat_messages(messages_key):
    """
    Display the chat message history with visualization support for SQL results
    
    Args:
        messages_key (str): Key for messages in session state
    """
    for message in st.session_state[messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Special handling for assistant messages with SQL results
            if message["role"] == "assistant" and "query" in message:
                # Check if we have results data to visualize
                if hasattr(st.session_state.sqlite_chatbot, 'last_query_results'):
                    data = st.session_state.sqlite_chatbot.last_query_results
                    
                    # Only display visualization options if data exists and is a DataFrame
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        # Display the dataframe
                        st.dataframe(data)
                        
                        # Create visualization selector with a unique key
                        viz_id = f"viz_{hash(message.get('timestamp', datetime.now().isoformat()))}"
                        
                        # Add visualization options
                        viz_options = ["table", "bar", "line", "scatter", "pie"]
                        selected_viz = st.selectbox(
                            "Visualization Type", 
                            viz_options, 
                            key=f"select_{viz_id}"
                        )
                        
                        # Generate visualization button
                        if st.button("Generate Visualization", key=f"button_{viz_id}"):
                            try:
                                # Try to create visualization based on selected type
                                fig = None
                                
                                # Choose appropriate columns for visualization
                                cat_cols = data.select_dtypes(include=['object']).columns
                                num_cols = data.select_dtypes(include=['number']).columns
                                
                                # Default columns to use
                                x_column = None
                                y_column = None
                                
                                # For bar/pie charts, use categorical column for x and numeric for y
                                if selected_viz in ["bar", "pie"] and len(cat_cols) > 0 and len(num_cols) > 0:
                                    x_column = cat_cols[0]
                                    y_column = num_cols[0]
                                # For line/scatter charts, prefer numeric columns for both axes
                                elif selected_viz in ["line", "scatter"]:
                                    if len(num_cols) >= 2:
                                        x_column = num_cols[0]
                                        y_column = num_cols[1]
                                    elif len(num_cols) == 1 and len(data.columns) > 1:
                                        # Use first non-numeric column for x if only one numeric column
                                        non_numeric_cols = [col for col in data.columns if col not in num_cols]
                                        if non_numeric_cols:
                                            x_column = non_numeric_cols[0]
                                            y_column = num_cols[0]
                                
                                # Create visualization based on type and columns
                                if selected_viz == "bar" and x_column and y_column:
                                    fig = px.bar(data, x=x_column, y=y_column, title=f"{y_column} by {x_column}")
                                elif selected_viz == "line" and x_column and y_column:
                                    fig = px.line(data, x=x_column, y=y_column, title=f"{y_column} over {x_column}")
                                elif selected_viz == "scatter" and x_column and y_column:
                                    fig = px.scatter(data, x=x_column, y=y_column, title=f"{y_column} vs {x_column}")
                                elif selected_viz == "pie" and x_column and y_column:
                                    fig = px.pie(data, names=x_column, values=y_column, title=f"Distribution of {y_column} by {x_column}")
                                
                                # Display the visualization
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Couldn't create visualization with the selected chart type and available columns.")
                                    
                                    # Try auto visualization as fallback
                                    st.write("Trying automatic visualization...")
                                    fig = auto_visualize(data)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating visualization: {str(e)}")
                        
                        # Add download button
                        csv = data.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv",
                            key=f"download_{viz_id}"
                        )
            
            # Display metadata for assistant messages
            if message["role"] == "assistant" and "model" in message:
                with st.expander("Message Details", expanded=False):
                    # Show model used
                    st.caption(f"**Model**: {message.get('model', 'Unknown')}")
                    
                    # Show timestamp if available
                    if "timestamp" in message:
                        st.caption(f"**Time**: {message.get('timestamp')}")
                    
                    # Show RAG parameters if available
                    if "rag_params" in message:
                        st.caption("**RAG Parameters**:")
                        params = message["rag_params"]
                        st.caption(f"- k_value: {params.get('k_value', 'N/A')}")
                        st.caption(f"- chunk_size: {params.get('chunk_size', 'N/A')}")
                        st.caption(f"- chunk_overlap: {params.get('chunk_overlap', 'N/A')}")
                    
                    # Show query if available
                    if "query" in message:
                        st.caption("**SQL Query**:")
                        st.code(message["query"], language="sql")

# Helper function for SQL-specific functionality
def process_direct_sql_query(sql_query, viz_type="table"):
    """Process a direct SQL query and display results"""
    with st.spinner("Executing query..."):
        try:
            # Execute the query
            result = st.session_state.sqlite_chatbot.execute_sql_query(sql_query)
            
            if isinstance(result, pd.DataFrame) and not result.empty:
                # Display the results with tabular and visual representation
                st.subheader("Query Results")
                st.dataframe(result)
                
                # Create visualization if requested
                if viz_type != "table" and len(result) > 0:
                    with st.spinner("Generating visualization..."):
                        try:
                            # Choose appropriate columns for visualization
                            x_column = None
                            y_column = None
                            
                            # Auto-select columns based on data types
                            cat_cols = result.select_dtypes(include=['object']).columns
                            num_cols = result.select_dtypes(include=['number']).columns
                            
                            # For bar/pie charts, use categorical column for x and numeric for y
                            if viz_type in ["bar", "pie"] and len(cat_cols) > 0 and len(num_cols) > 0:
                                x_column = cat_cols[0]
                                y_column = num_cols[0]
                            # For line/scatter charts, prefer numeric columns for both axes
                            elif viz_type in ["line", "scatter"]:
                                if len(num_cols) >= 2:
                                    x_column = num_cols[0]
                                    y_column = num_cols[1]
                                elif len(num_cols) == 1 and len(result.columns) > 1:
                                    # Use first non-numeric column for x if only one numeric column
                                    non_numeric_cols = [col for col in result.columns if col not in num_cols]
                                    if non_numeric_cols:
                                        x_column = non_numeric_cols[0]
                                        y_column = num_cols[0]
                            
                            # Create visualization based on type and columns
                            fig = None
                            
                            # For bar charts
                            if viz_type == "bar" and x_column and y_column:
                                fig = px.bar(result, x=x_column, y=y_column, title=f"{y_column} by {x_column}")
                            # For line charts
                            elif viz_type == "line" and x_column and y_column:
                                fig = px.line(result, x=x_column, y=y_column, title=f"{y_column} over {x_column}")
                            # For scatter plots
                            elif viz_type == "scatter" and x_column and y_column:
                                fig = px.scatter(result, x=x_column, y=y_column, title=f"{y_column} vs {x_column}")
                            # For pie charts
                            elif viz_type == "pie" and x_column and y_column:
                                fig = px.pie(result, names=x_column, values=y_column, title=f"Distribution of {y_column} by {x_column}")
                            
                            # Display the figure if successfully created
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Couldn't create visualization with the selected parameters.")
                                # Try auto visualization
                                st.write("Trying automatic visualization...")
                                fig = auto_visualize(result)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                        except Exception as viz_error:
                            st.error(f"Error creating visualization: {str(viz_error)}")
                
                # Add option to download results
                csv = result.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
                
                # Add successful query to chat history
                user_message = {
                    "role": "user", 
                    "content": f"Executed SQL query: ```sql\n{sql_query}\n```",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.sqlite_messages.append(user_message)
                
                # Format the result summary
                result_summary = f"Query executed successfully. Returned {len(result)} rows with {len(result.columns)} columns."
                
                # Add to chat history with specialized function for SQL results
                assistant_message = {
                    "role": "assistant",
                    "content": result_summary,
                    "model": st.session_state.get("model_name", "gpt-3.5-turbo"),
                    "timestamp": datetime.now().isoformat(),
                    "query": sql_query,
                    "rag_params": {
                        "k_value": st.session_state.k_value,
                        "chunk_size": st.session_state.chunk_size,
                        "chunk_overlap": st.session_state.chunk_overlap
                    }
                }
                
                # Add to session state
                st.session_state.sqlite_messages.append(assistant_message)
                
                # Store query results for visualization
                st.session_state.sqlite_chatbot.last_query_results = result
                
                # Save the chat history
                save_chat_history("sqlite", "sqlite_session_id", "sqlite_messages", "sqlite_chatbot")
            else:
                # Non-SELECT query result (like UPDATE, INSERT, etc.)
                if isinstance(result, pd.DataFrame) and result.empty:
                    st.info("Query executed successfully but returned no rows.")
                    result_summary = "Query executed successfully but returned no rows."
                else:
                    st.success(result)
                    result_summary = result
                
                # Add to chat history
                user_message = {
                    "role": "user", 
                    "content": f"Executed SQL query: ```sql\n{sql_query}\n```",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.sqlite_messages.append(user_message)
                
                # Add the assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": result_summary,
                    "model": st.session_state.get("model_name", "gpt-3.5-turbo"),
                    "timestamp": datetime.now().isoformat(),
                    "query": sql_query,
                    "rag_params": {
                        "k_value": st.session_state.k_value,
                        "chunk_size": st.session_state.chunk_size,
                        "chunk_overlap": st.session_state.chunk_overlap
                    }
                }
                st.session_state.sqlite_messages.append(assistant_message)
                
                # Save the chat history
                save_chat_history("sqlite", "sqlite_session_id", "sqlite_messages", "sqlite_chatbot")
                
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="SQLite RAG",
        page_icon="🗃️",
        layout="wide"
    )
    
    # Initialize application state
    history_manager = ChatHistoryManager()
    app_type, session_id_key, chatbot_key, messages_key = initialize_session_state(
        "sqlite", SQLiteChatbot, history_manager
    )
    
    st.title("🗃️ SQLite Database RAG")
    st.subheader("Chat with your SQLite Databases")
    
    # Handle history view if active
    if st.session_state.viewing_history and st.session_state.history_session_id:
        display_history_view(history_manager)
        
        # Display database info if available
        if hasattr(st.session_state[chatbot_key], 'db_connection') and st.session_state[chatbot_key].db_connection is not None:
            display_database_info()
    else:
        # Display sidebar with controls
        display_sidebar(app_type, chatbot_key, messages_key, session_id_key, SQLiteChatbot, ['db', 'sqlite', 'sqlite3'])
        
        # Display database information if connected
        if hasattr(st.session_state[chatbot_key], 'table_info') and st.session_state[chatbot_key].table_info:
            st.subheader("Connected Database")
            
            # Create tabs for tables
            all_tables = list(st.session_state[chatbot_key].table_info.keys())
            if all_tables:
                tabs = st.tabs(all_tables)
                
                # Display info for each table in its own tab
                for i, table_name in enumerate(all_tables):
                    info = st.session_state[chatbot_key].table_info[table_name]
                    with tabs[i]:
                        st.write(f"**{table_name}** ({info['row_count']} rows)")
                        cols = ", ".join([f"{col[1]} ({col[2]})" for col in info['columns']])
                        st.write(f"Columns: {cols}")
                        
                        # Show sample data
                        st.write("Sample data:")
                        # Convert sample data to DataFrame for better display
                        columns = [col[1] for col in info['columns']]
                        sample_df = pd.DataFrame(info['sample_data'], columns=columns)
                        st.dataframe(sample_df)
        
        # Display example questions with direct LLM triggering
        st.subheader("Example Questions")
        example_questions = [
            "Write a query to list all tables and their row counts",
            "How many records are in each table?",
            "Show me the schema for all tables"
        ]
        
        # Create columns for the example buttons
        col1, col2, col3 = st.columns(3)
        
        # Define a function to handle clicking example buttons
        def handle_example_click(question):
            # Add user message to chat history
            user_message = {
                "role": "user", 
                "content": question,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state[messages_key].append(user_message)
            
            # Process the message to get a response
            process_chat_message(app_type, chatbot_key, messages_key, session_id_key, question)
            
            # Force rerun to update the UI
            st.rerun()
        
        # Display buttons in columns with direct LLM call handling
        with col1:
            if st.button(example_questions[0]):
                handle_example_click(example_questions[0])
                
        with col2:
            if st.button(example_questions[1]):
                handle_example_click(example_questions[1])
                
        with col3:
            if st.button(example_questions[2]):
                handle_example_click(example_questions[2])
        
        # Display SQL query interface
        st.subheader("Direct SQL Query")
        if hasattr(st.session_state[chatbot_key], 'db_connection') and st.session_state[chatbot_key].db_connection is not None:
            # Create columns for SQL input and execution
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # SQL input text area
                sql_query = st.text_area("Enter SQL query", height=100, 
                                        placeholder="SELECT * FROM table_name LIMIT 5")
            
            with col2:
                # Execute button
                st.write("")  # Add some space for alignment
                st.write("")  # Add some space for alignment
                execute_sql = st.button("Execute Query")
                
                # Add visualization option
                viz_options = ["table", "bar", "line", "scatter", "pie"]
                selected_viz = st.selectbox(f"Visualization Type", viz_options)
            
            # Handle SQL execution
            if execute_sql and sql_query:
                process_direct_sql_query(sql_query, selected_viz)
        
        # Main chat interface
        st.subheader("Chat with your Database")
        
        # Display chat messages - customize display for SQL results
        display_sql_chat_messages(messages_key)
        
        # Handle chat input
        handle_chat_input(app_type, chatbot_key, messages_key, session_id_key,
                         "Ask a question about your database")

# Run the main application
if __name__ == "__main__":
    main()