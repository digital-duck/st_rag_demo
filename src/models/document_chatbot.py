from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import os
import tempfile

class DocumentChatbot:
    """Base class for document chatbots"""
    
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        """Initialize the document chatbot with OpenAI and ChromaDB"""
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize the OpenAI model
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=self.api_key,
            temperature=0.2
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        
        # Initialize document splitter with default values
        # These can be updated via the UI in debug mode
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Store debugging information
        self.last_formatted_prompt = ""
        self.last_retrieved_context = ""
        self.retriever = None
        
        # Initialize storage for documents and metadata
        self.vectorstore = None
        self.documents = []
        self.file_metadata = {}
        self.chat_history = []
        
        # Create a temporary directory for Chroma DB
        self.persist_directory = tempfile.mkdtemp()
        
    def update_chunk_settings(self, chunk_size=None, chunk_overlap=None):
        """Update text splitter settings if provided"""
        if chunk_size:
            self.chunk_size = chunk_size
        if chunk_overlap:
            self.chunk_overlap = chunk_overlap
            
        # Recreate the text splitter with new settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
    def build_vectorstore(self):
        """Build the vector store from the processed documents"""
        if not self.documents:
            return False
        
        # Update text splitter with current settings (for debug mode)
        if hasattr(st, 'session_state') and 'chunk_size' in st.session_state and 'chunk_overlap' in st.session_state:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(self.documents)
        
        if not splits:
            return False
            
        # Log chunking stats for debugging
        # Don't try to access internal attributes - use the values from session state
        # or use defaults if they're not available
        chunk_size = st.session_state.get('chunk_size', 1000) if hasattr(st, 'session_state') else 1000
        chunk_overlap = st.session_state.get('chunk_overlap', 100) if hasattr(st, 'session_state') else 100
        
        self.chunk_stats = {
            "total_chunks": len(splits),
            "avg_chunk_size": sum(len(d.page_content) for d in splits) / max(1, len(splits)),
            "chunk_size_setting": chunk_size,
            "chunk_overlap_setting": chunk_overlap
        }
        
        try:
            # Create vector store with persistent storage
            self.vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Ensure the vector store is persisted
            if hasattr(self.vectorstore, '_persist'):
                self.vectorstore._persist()
                
            return True
        except Exception as e:
            st.error(f"Error building vector database: {str(e)}")
            return False
    
    def get_file_info(self):
        """Get all file information"""
        if not self.file_metadata:
            return "No files have been uploaded yet."
            
        file_info = "Uploaded Files:\n"
        for filename, metadata in self.file_metadata.items():
            file_info += f"\n{metadata}\n"
            
        return file_info
        
    def ask(self, question, return_context=False):
        """
        Ask a question about the documents.
        This is the base implementation that should be overridden by subclasses.
        
        Args:
            question (str): The question to ask
            return_context (bool): Whether to return the retrieval context along with the answer
            
        Returns:
            str or dict: Just the answer or a dict with answer and context
        """
        # Standard RAG query
        if not self.vectorstore:
            return "Please upload and process documents first."
            
        try:
            # Get retriever directly for debugging with current k value
            k_value = 5  # Default
            if hasattr(st, 'session_state') and 'k_value' in st.session_state:
                k_value = st.session_state.k_value
                
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_value}
            )
            
            # Get retrieved documents separately 
            retrieved_docs = retriever.invoke(question)
            
            # Format the retrieved context as a string for debugging
            retrieved_context = "\n\n---\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                                for i, doc in enumerate(retrieved_docs)])
            
            # Create a direct call to the LLM with the retrieved documents as context
            file_info = self.get_file_info()
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            system_prompt = f"""You are a helpful assistant that answers questions about document data.
            
Use the following context from the uploaded files to answer the user's question. 
If you don't know the answer, say that you don't know. 
Include relevant statistics and insights from the data where appropriate.

Context: {context_text}

Files metadata: {file_info}
"""
            
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add chat history if available
            for msg in self.chat_history:
                if isinstance(msg, tuple) and len(msg) == 2:
                    role, content = msg
                    messages.append({"role": role, "content": content})
            
            # Add user question
            messages.append({"role": "user", "content": question})
            
            # Call LLM directly
            llm_response = self.llm.invoke(messages)
            
            # Create formatted prompt text that would be sent to the LLM (for debugging)
            formatted_prompt = f"""System: You are a helpful assistant that answers questions about document data.
                
Use the following context from the uploaded files to answer the user's question. 
If you don't know the answer, say that you don't know. 
Include relevant statistics and insights from the data where appropriate.

Context: {retrieved_context}

Files metadata: {file_info}

"""
            # Store the formatted prompt for debugging
            self.last_formatted_prompt = formatted_prompt
            self.last_retrieved_context = retrieved_context
            
            # Get the answer from the LLM
            answer = llm_response.content
            
            # Update chat history
            self.chat_history.append(("human", question))
            self.chat_history.append(("ai", answer))
            
            # Return the appropriate response
            if return_context:
                return {
                    "answer": answer,
                    "retrieved_context": retrieved_context,
                    "formatted_prompt": formatted_prompt
                }
            
            return answer
            
        except Exception as e:
            error_message = f"Error while retrieving information: {str(e)}"
            if return_context:
                return {
                    "answer": error_message,
                    "retrieved_context": "Error occurred during retrieval",
                    "formatted_prompt": "Error occurred"
                }
            return error_message
    
    def clear(self):
        """Clear all documents and reset the chatbot"""
        self.documents = []
        self.file_metadata = {}
        self.vectorstore = None
        self.chat_history = []
        
        # Create a new temporary directory for Chroma DB
        self.persist_directory = tempfile.mkdtemp()