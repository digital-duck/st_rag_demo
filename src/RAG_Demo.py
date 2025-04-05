import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in your .env file or as an environment variable.")

# Set page configuration
st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("📄 Advanced RAG Chatbot")
st.subheader("Chat with your Documents, Databases and Visualize Data")

st.markdown("""
### Welcome to the Advanced RAG Chatbot!

This application demonstrates advanced Retrieval Augmented Generation (RAG) capabilities 
for different types of data sources. Navigate to the different pages to:

- **PDF RAG**: Process PDF documents and ask questions about their content
- **CSV RAG**: Analyze CSV data files and get insights through natural language
- **SQLite RAG**: Query SQLite databases using natural language and visualize results
- **RAG Debugging**: Advanced tools for debugging and optimizing your RAG system

#### Getting Started

1. Select a page from the sidebar
2. Upload your files
3. Process the files
4. Start asking questions!

#### About RAG

Retrieval Augmented Generation (RAG) combines the power of large language models with 
retrieval of relevant information from your data sources. This means the system can:

- Answer questions specific to your documents
- Perform data analysis on your CSV files
- Generate SQL queries from natural language questions
- Create visualizations based on your data
""")

# Display info about OpenAI API key
st.sidebar.header("API Configuration")
if openai_api_key:
    st.sidebar.success("OpenAI API key is configured.")
else:
    st.sidebar.warning("OpenAI API key is not configured. Please add it to your .env file.")
    st.sidebar.code("OPENAI_API_KEY=your_api_key_here")

# App information
st.sidebar.header("About")
st.sidebar.info("""
This app demonstrates advanced RAG capabilities built with:
- Streamlit
- LangChain
- OpenAI
- ChromaDB
- Plotly
""")