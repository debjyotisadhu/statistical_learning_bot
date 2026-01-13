"""
User Configuration File

This file contains all user-configurable parameters for the RAG pipeline.
Modify these values according to your requirements.

Configuration Sections:
    1. Document Settings: Directory and file type
    2. Embedding Settings: Model selection
    3. Vector Store Settings: Collection name and persistence directory
"""
# ======================================================================================================================
# Document Configuration
# ======================================================================================================================
# Type of documents to process (supported: "pdf" or "txt")
Type_of_Documents = "pdf"

# Directory path containing the documents to process
# Use raw string (r"...") for Windows paths to avoid escape sequence issues
Document_Directory = r"Books"

# ======================================================================================================================
# Embedding Model Configuration
# ======================================================================================================================
# Name of the SentenceTransformer model to use for generating embeddings
# Popular options:
#   - "all-MiniLM-L6-v2" (default, fast and efficient)
#   - "all-mpnet-base-v2" (higher quality, slower)
#   - "sentence-transformers/all-MiniLM-L12-v2" (larger, better quality)
Embedding_Model_Name = "all-MiniLM-L6-v2"

# ======================================================================================================================
# Vector Store Configuration
# ======================================================================================================================
# Name of the ChromaDB collection to create or use
# This identifies your document collection in the vector store
collection_name = "ISL_pdf_documents"

# Directory path where the vector store will be persisted
# The directory will be created if it doesn't exist
# Use raw string (r"...") for Windows paths
persist_directory = r"vector_store"

# ======================================================================================================================
