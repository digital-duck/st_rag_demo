import os
import sys
import streamlit as st

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Page configuration
st.set_page_config(
    page_title="CSV RAG",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CSV Data RAG")
st.subheader("Chat with your CSV Data")

st.info("""
## Coming Soon!

This page will allow you to upload CSV files and chat with your data.

Features will include:
- CSV data processing and analysis
- RAG-based question answering on your tabular data
- Data visualizations based on your questions
- Statistical insights from your data

Please check back later for this functionality.
""")

# Sidebar content
with st.sidebar:
    st.header("CSV RAG")
    st.write("This feature is under development. Please try the PDF RAG feature in the meantime.")