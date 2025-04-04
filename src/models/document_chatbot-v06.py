from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
# Updated import path for HuggingFaceEmbeddings as per warning
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import os
import tempfile

class DocumentChatbot:
    """Base class for document chatbots"""
    
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        """Initialize the document chatbot with LLM and vector store"""
        # Use provided API key or get from environment for the LLM
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required for the LLM")
        
        # Initialize the OpenAI model for LLM
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=self.api_key,
            temperature=0.2
        )
        
        # Initialize embeddings with an open-source model (no API key required)
        # all-MiniLM-L6-v2 is a good balance between speed and quality
        # Using the updated HuggingFaceEmbeddings from langchain_huggingface
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use CPU for compatibility
        )
        
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
        
        # Create a persistent directory for Chroma DB
        self.persist_directory = os.path.join(os.getcwd(), "vector_db", f"{self.__class__.__name__.lower()}_db")
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the vectorstore from the persistent directory if it exists
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """Load vectorstore from persistent directory if it exists"""
        try:
            if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                return True
            return False
        except Exception as e:
            print(f"Error loading vectorstore: {str(e)}")
            return False
            
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
        chunk_size = st.session_state.get('chunk_size', 1000) if hasattr(st, 'session_state') else 1000
        chunk_overlap = st.session_state.get('chunk_overlap', 100) if hasattr(st, 'session_state') else 100
        
        self.chunk_stats = {
            "total_chunks": len(splits),
            "avg_chunk_size": sum(len(d.page_content) for d in splits) / max(1, len(splits)),
            "chunk_size_setting": chunk_size,
            "chunk_overlap_setting": chunk_overlap
        }
        
        try:
            # Create or update vector store with persistent storage
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                # Add documents to existing vectorstore
                self.vectorstore.add_documents(splits)
            
            # NOTE: Chroma 0.4.x+ automatically persists documents, no need for explicit persist()
                
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
        self.chat_history = []
        
        # Note: We don't reset the vectorstore to preserve persistence
        # If you want to truly clear everything, uncomment below:
        # import shutil
        # if os.path.exists(self.persist_directory):
        #     shutil.rmtree(self.persist_directory)
        # self.vectorstore = None