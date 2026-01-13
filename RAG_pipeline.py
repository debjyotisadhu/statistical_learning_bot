"""
RAG Pipeline Module

This module provides classes for document ingestion, text chunking, embedding generation,
and vector store management for a Retrieval-Augmented Generation (RAG) system.

Classes:
    IngestDocumentsAndChunk: Handles document loading and text chunking
    EmbeddingManager: Manages embedding model loading and generation
    VectorStore: Manages ChromaDB vector store operations
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import Any, List, Optional
import chromadb
import os
import numpy as np
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================================================================================
class IngestDocumentsAndChunk:
    """
    Handles document ingestion from a directory and text chunking.
    
    Supports PDF and TXT file formats. Loads documents from a specified directory,
    validates the input, and provides methods to split documents into chunks.
    
    Attributes:
        text_contents (List[str]): List of text content from chunked documents
        document_chunks (List[Document]): List of document chunk objects with metadata
        documents (List[Document]): Original loaded documents
    """
    
    def __init__(self, document_directory: str, type_of_documents: str = "pdf"):
        """
        Initialize document ingestion.
        
        Args:
            document_directory (str): Path to directory containing documents
            type_of_documents (str): Type of documents to load ("pdf" or "txt")
            
        Raises:
            ValueError: If document type is not supported or directory doesn't exist
            FileNotFoundError: If directory path is invalid
        """
        self.text_contents: Optional[List[str]] = None
        self.document_chunks: Optional[List[Any]] = None
        self.documents: Optional[List[Any]] = None
        
        # Validate document type
        if type_of_documents not in ["pdf", "txt"]:
            error_msg = f"Unsupported document type: {type_of_documents}. Only 'pdf' and 'txt' are supported."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate directory exists
        if not os.path.exists(document_directory):
            error_msg = f"Document directory not found: {document_directory}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not os.path.isdir(document_directory):
            error_msg = f"Path is not a directory: {document_directory}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Load documents based on type
        try:
            if type_of_documents == "pdf":
                loader = DirectoryLoader(
                    document_directory,
                    glob="*.pdf",
                    show_progress=True,
                    loader_cls=PyPDFLoader
                )
            else:  # txt
                loader = DirectoryLoader(
                    document_directory,
                    glob="*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={'encoding': 'utf-8'}
                )
            
            logger.info(f"Loading {type_of_documents} documents from {document_directory}")
            self.documents = loader.load()
            
            if not self.documents:
                warning_msg = f"No {type_of_documents} documents found in {document_directory}"
                logger.warning(warning_msg)
                raise ValueError(warning_msg)
            
            logger.info(f"Successfully loaded {len(self.documents)} documents")
            
        except Exception as e:
            error_msg = f"Error loading documents: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def create_chunks(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Split documents into chunks of specified size with overlap.
        
        Args:
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Raises:
            ValueError: If documents haven't been loaded or invalid chunk parameters
            RuntimeError: If chunking fails
        """
        if self.documents is None:
            error_msg = "Documents not loaded. Call __init__ first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if chunk_size <= 0:
            error_msg = f"chunk_size must be positive, got {chunk_size}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if chunk_overlap < 0:
            error_msg = f"chunk_overlap must be non-negative, got {chunk_overlap}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if chunk_overlap >= chunk_size:
            error_msg = f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
            splitted_docs = text_splitter.split_documents(self.documents)
            
            logger.info(f"Original documents: {len(self.documents)}")
            logger.info(f"Chunked documents: {len(splitted_docs)}")
            
            self.text_contents = [text.page_content for text in splitted_docs]
            self.document_chunks = splitted_docs
            
            if not self.text_contents:
                warning_msg = "No text content extracted from documents"
                logger.warning(warning_msg)
                raise ValueError(warning_msg)
            
        except Exception as e:
            error_msg = f"Error creating chunks: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


# ======================================================================================================================
class EmbeddingManager:
    """
    Manages embedding model loading and text embedding generation.
    
    Uses SentenceTransformer models to generate embeddings for text documents.
    Supports any model compatible with the SentenceTransformer library.
    
    Attributes:
        model_name (str): Name of the embedding model
        model (SentenceTransformer): Loaded embedding model
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager and load the model.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use
            
        Raises:
            ValueError: If model_name is empty
            RuntimeError: If model fails to load
        """
        if not model_name or not isinstance(model_name, str):
            error_msg = "model_name must be a non-empty string"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the SentenceTransformer model.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            error_msg = f"Failed to load embedding model '{self.model_name}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def generate_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings with shape (num_texts, embedding_dim)
            
        Raises:
            ValueError: If texts list is empty or model not loaded
            RuntimeError: If embedding generation fails
        """
        if self.model is None:
            error_msg = "Model not loaded. Initialize EmbeddingManager first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not texts:
            error_msg = "Texts list cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not isinstance(texts, list):
            error_msg = f"texts must be a list, got {type(texts)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            logger.info(f"Successfully generated embeddings with shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


# ======================================================================================================================
class VectorStore:
    """
    Manages ChromaDB vector store for storing and querying document embeddings.
    
    Handles collection creation, document addition, and provides access to the
    underlying ChromaDB collection for querying.
    
    Attributes:
        collection_name (str): Name of the ChromaDB collection
        persist_directory (str): Directory path for persisting the vector store
        client (chromadb.PersistentClient): ChromaDB client instance
        collection (chromadb.Collection): ChromaDB collection instance
    """
    
    def __init__(self, collection_name: str, persist_directory: str):
        """
        Initialize vector store and create/connect to collection.
        
        Args:
            collection_name (str): Name of the collection to create or access
            persist_directory (str): Directory path for persisting data
            
        Raises:
            ValueError: If collection_name or persist_directory is invalid
            RuntimeError: If vector store initialization fails
        """
        if not collection_name or not isinstance(collection_name, str):
            error_msg = "collection_name must be a non-empty string"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not persist_directory or not isinstance(persist_directory, str):
            error_msg = "persist_directory must be a non-empty string"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        """
        Initialize ChromaDB client and create/get collection.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info(f"Initializing vector store at {self.persist_directory}")
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={'description': 'document embedding for RAG'}
            )
            
            logger.info(f"Vector store initialized successfully: {self.collection_name}")
        except Exception as e:
            error_msg = f"Failed to initialize vector store: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def add_documents(self, documents: List[Any], embeddings: np.ndarray) -> None:
        """
        Add documents and their embeddings to the vector store in batch.
        
        This method batches all documents and adds them in a single operation
        for better performance compared to adding them one by one.
        
        Args:
            documents (List[Any]): List of document objects with page_content and metadata
            embeddings (np.ndarray): Array of embeddings corresponding to documents
            
        Raises:
            ValueError: If inputs are invalid or mismatched
            RuntimeError: If adding documents fails
        """
        if self.collection is None:
            error_msg = "Vector store not initialized. Call __init__ first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not documents:
            error_msg = "Documents list cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not isinstance(embeddings, np.ndarray):
            error_msg = f"embeddings must be a numpy array, got {type(embeddings)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(documents) != len(embeddings):
            error_msg = f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Prepare all data in batch
            ids = []
            embeddings_list = []
            documents_list = []
            metadatas = []
            
            logger.info(f"Preparing {len(documents)} documents for batch addition")
            
            for i, (document, embedding) in enumerate(zip(documents, embeddings)):
                # Generate unique document ID
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(doc_id)
                
                # Prepare metadata
                metadata = dict(document.metadata) if hasattr(document, 'metadata') else {}
                metadata['doc_index'] = i
                metadata['current_length'] = len(document.page_content) if hasattr(document, 'page_content') else 0
                metadatas.append(metadata)
                
                # Extract document content
                doc_content = document.page_content if hasattr(document, 'page_content') else str(document)
                documents_list.append(doc_content)
                
                # Convert embedding to list
                if isinstance(embedding, np.ndarray):
                    embeddings_list.append(embedding.tolist())
                else:
                    embeddings_list.append(list(embedding))
            
            # Batch add all documents at once (optimized)
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents_list,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(documents)} document embeddings to vector store")
            
        except Exception as e:
            error_msg = f"Error adding documents to vector store: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
