"""
RAG Document Base Creation Script

This script creates a RAG document base by:
1. Loading and chunking documents from a directory
2. Generating embeddings for the document chunks
3. Storing embeddings and documents in a vector store

Usage:
    python create_rag_document_base.py

The script uses configuration from User_Inputs.py for:
    - Document directory and type
    - Embedding model name
    - Vector store collection name and persistence directory
"""
import logging
import sys
from User_Inputs import (
    Document_Directory,
    Type_of_Documents,
    Embedding_Model_Name,
    collection_name,
    persist_directory
)
from RAG_pipeline import IngestDocumentsAndChunk, EmbeddingManager, VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to create the RAG document base.
    
    Executes the complete pipeline:
    1. Document ingestion and chunking
    2. Embedding generation
    3. Vector store creation and document storage
    
    Raises:
        SystemExit: If any step fails
    """
    try:
        # Step 1: Ingest and chunk documents
        logger.info("=" * 80)
        logger.info("Starting RAG Document Base Creation")
        logger.info("=" * 80)
        logger.info(f"Document Directory: {Document_Directory}")
        logger.info(f"Document Type: {Type_of_Documents}")
        logger.info(f"Embedding Model: {Embedding_Model_Name}")
        logger.info(f"Collection Name: {collection_name}")
        logger.info(f"Persist Directory: {persist_directory}")
        logger.info("=" * 80)
        
        logger.info("Step 1: Loading and chunking documents...")
        try:
            ingest_docs = IngestDocumentsAndChunk(Document_Directory, Type_of_Documents)
            ingest_docs.create_chunks(chunk_size=1000, chunk_overlap=200)
            logger.info("✓ Document ingestion and chunking completed successfully")
        except Exception as e:
            error_msg = f"Failed to ingest documents: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Step 2: Generate embeddings
        logger.info("Step 2: Generating embeddings...")
        try:
            embedding_manager = EmbeddingManager(Embedding_Model_Name)
            document_embeddings = embedding_manager.generate_embedding(ingest_docs.text_contents)
            logger.info("✓ Embedding generation completed successfully")
        except Exception as e:
            error_msg = f"Failed to generate embeddings: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Step 3: Create vector store and add documents
        logger.info("Step 3: Creating vector store and adding documents...")
        try:
            vector_store = VectorStore(collection_name, persist_directory)
            vector_store.add_documents(ingest_docs.document_chunks, document_embeddings)
            logger.info("✓ Vector store creation and document addition completed successfully")
        except Exception as e:
            error_msg = f"Failed to create vector store or add documents: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Success message
        logger.info("=" * 80)
        logger.info("RAG Document Base Creation Completed Successfully!")
        logger.info(f"Total documents processed: {len(ingest_docs.documents)}")
        logger.info(f"Total chunks created: {len(ingest_docs.document_chunks)}")
        logger.info(f"Vector store location: {persist_directory}")
        logger.info(f"Collection name: {collection_name}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
