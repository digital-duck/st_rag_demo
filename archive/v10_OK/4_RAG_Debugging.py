import os
import sys
import streamlit as st

# Add the src directory to the path to allow importing from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Page configuration
st.set_page_config(
    page_title="RAG Debugging",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Debugging Tools")
st.subheader("Advanced Tools for RAG Development and Debugging")

st.info("""
## Coming Soon!

This page will provide advanced tools for debugging and optimizing your RAG systems.

Features will include:
- View retrieved document chunks
- Examine the exact prompts sent to the LLM
- Analyze metrics about retrieval performance
- Test different chunking parameters
- Compare different retrieval methods
- Visualize embeddings and vector spaces

Please check back later for this functionality, after the PDF RAG feature is fully implemented.
""")

# Sidebar content
with st.sidebar:
    st.header("RAG Debugging")
    st.write("This feature is under development. Please try the PDF RAG feature in the meantime.")