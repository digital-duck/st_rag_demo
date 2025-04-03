import os
import sys
import streamlit as st

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Page configuration
st.set_page_config(
    page_title="SQLite RAG",
    page_icon="🗃️",
    layout="wide"
)

st.title("🗃️ SQLite Database RAG")
st.subheader("Chat with your SQLite Databases")

st.info("""
## Coming Soon!

This page will allow you to upload SQLite database files and interact with them using natural language.

Features will include:
- Text-to-SQL query generation
- RAG-based question answering on your database schema and content
- Data visualizations based on query results
- Insightful analytics on your database data

Please check back later for this functionality.
""")

# Sidebar content
with st.sidebar:
    st.header("SQLite RAG")
    st.write("This feature is under development. Please try the PDF RAG feature in the meantime.")