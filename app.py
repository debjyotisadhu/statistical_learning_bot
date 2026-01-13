"""
Streamlit RAG Chat Application

This module provides a web-based chat interface for interacting with the RAG system.
Users can ask questions about statistical learning and receive answers based on
retrieved documents from the vector store.

Features:
    - Interactive chat interface
    - Persistent chat history
    - Cached backend resources for performance
    - Error handling and user feedback

Dependencies:
    - streamlit: Web framework
    - RAG_pipeline: Document processing and vector store
    - main: Query and response generation
    - User_Inputs: Configuration parameters
"""
import streamlit as st
import logging
import os
from typing import Tuple, Optional, Any
from RAG_pipeline import EmbeddingManager, VectorStore
from User_Inputs import (
    Embedding_Model_Name,
    collection_name,
    persist_directory
)
from main import get_response
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================================================================================
# Page Configuration
# ======================================================================================================================
st.set_page_config(
    page_title="Chat App - Statistical Learning RAG",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ======================================================================================================================
# Application Header
# ======================================================================================================================
st.title("💬 Ask Me About Statistical Learning")
st.caption("RAG Application | Developed by Deb")

# ======================================================================================================================
# Sidebar Configuration
# ======================================================================================================================
with st.sidebar:
    # Check if API key is available
    groq_api_key = st.text_input("Enter GROQ API KEY")  # os.getenv("GROQ_API_KEY")
    st.markdown("""[Get Your Groq Key Here](https://console.groq.com/keys)""")

    st.markdown("### 📬 Connect with me")
    st.markdown(
        """
        Debjyoti Sadhu (Deb) 
        
        🔗 [LinkedIn](https://www.linkedin.com/in/debjyoti-sadhu/)  
        📧 [Email Me](mailto:debjyotisadhu@email.com)
        """
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        """
        This RAG (Retrieval-Augmented Generation) application 
        answers questions based on statistical learning documents.
        
        **How it works:**
        1. Your question is converted to an embedding
        2. Similar documents are retrieved from the knowledge base
        3. An AI response is generated using the retrieved context
        """
    )

# ======================================================================================================================
# Initialize Chat History
# ======================================================================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ======================================================================================================================
# Display Chat History
# ======================================================================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================================================================================================================
# Backend Resource Initialization (Cached)
# ======================================================================================================================
@st.cache_resource(show_spinner=False,
    ttl=3600  # Cache for 1 hour
)
def get_rag_backend() -> Tuple[VectorStore, EmbeddingManager]:
    """
    Initialize and cache RAG backend resources.
    
    This function loads the embedding manager and vector store,
    which are expensive to initialize. The results are cached
    to improve performance across user interactions.
    
    Returns:
        Tuple[VectorStore, EmbeddingManager]: Initialized vector store and embedding manager
        
    Raises:
        RuntimeError: If backend initialization fails
    """
    try:
        logger.info("Initializing RAG backend resources")
        
        # Initialize embedding manager
        try:
            embedding_manager = EmbeddingManager(Embedding_Model_Name)
            logger.info("Embedding manager initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize embedding manager: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Initialize vector store
        try:
            vector_store = VectorStore(collection_name, persist_directory)
            logger.info("Vector store initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize vector store: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        return vector_store, embedding_manager
        
    except Exception as e:
        error_msg = f"Failed to initialize RAG backend: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

# ======================================================================================================================
# User Input Processing
# ======================================================================================================================
prompt = st.chat_input("Type your message...")

if prompt:
    # Validate input
    if not prompt.strip():
        st.warning("Please enter a non-empty message.")
    else:
        # Add user message to chat history
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.spinner("Thinking..."):
            try:
                # Get backend resources (cached)
                vector_store, embedding_manager = get_rag_backend()
                

                if not groq_api_key:
                    st.error(
                        "⚠️ **API Key Missing**\n\n"
                        "Please set the GROQ_API_KEY environment variable to use this application.\n\n"
                        "You can set it by running:\n"
                        "```bash\n"
                        "export GROQ_API_KEY='your-api-key-here'\n"
                        "```"
                    )
                    st.stop()
                
                # Generate response
                try:
                    response = get_response(
                        query=prompt,
                        vector_store=vector_store,
                        embedding_manager=embedding_manager,
                        groq_api_key=groq_api_key
                    )
                    
                    # Validate response
                    if not response or not response.strip():
                        response = (
                            "I apologize, but I couldn't generate a response. "
                            "Please try rephrasing your question or check if the vector store contains relevant documents."
                        )
                        logger.warning("Empty response generated")
                    
                except ValueError as e:
                    # Handle validation errors
                    error_msg = f"Invalid input: {str(e)}"
                    logger.error(error_msg)
                    response = (
                        f"⚠️ **Input Error**\n\n"
                        f"I encountered an issue with your query: {str(e)}\n\n"
                        "Please try rephrasing your question."
                    )
                    
                except RuntimeError as e:
                    # Handle runtime errors (API failures, etc.)
                    error_msg = f"Runtime error: {str(e)}"
                    logger.error(error_msg)
                    response = (
                        f"⚠️ **Error**\n\n"
                        f"I encountered an error while processing your request: {str(e)}\n\n"
                        "Please try again later or contact support if the issue persists."
                    )
                    
                except Exception as e:
                    # Handle unexpected errors
                    error_msg = f"Unexpected error: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    response = (
                        "⚠️ **Unexpected Error**\n\n"
                        "An unexpected error occurred. Please try again or contact support.\n\n"
                        f"Error details: {str(e)}"
                    )
                
            except RuntimeError as e:
                # Handle backend initialization errors
                error_msg = f"Backend initialization failed: {str(e)}"
                logger.error(error_msg)
                response = (
                    f"⚠️ **Initialization Error**\n\n"
                    f"Failed to initialize the RAG backend: {str(e)}\n\n"
                    "Please ensure:\n"
                    "- The vector store exists and is properly configured\n"
                    "- The embedding model can be loaded\n"
                    "- All required dependencies are installed"
                )
                
            except Exception as e:
                # Handle any other unexpected errors
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                response = (
                    "⚠️ **System Error**\n\n"
                    "An unexpected system error occurred. Please try refreshing the page.\n\n"
                    f"Error: {str(e)}"
                )
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

# ======================================================================================================================
# Additional UI Features
# ======================================================================================================================
# Add a clear chat button in the sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
