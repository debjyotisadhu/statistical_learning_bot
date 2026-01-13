"""
RAG Query and Response Module

This module provides functionality to query the vector store and generate responses
using a language model (ChatGroq) based on retrieved context.

Functions:
    get_response: Retrieves relevant documents and generates a response using LLM
"""
from langchain_groq import ChatGroq
from typing import Dict, List, Any, Optional
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_response(
    query: str,
    vector_store: Any,
    embedding_manager: Any,
    groq_api_key: Optional[str] = None,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    n_results: int = 5,
    min_similarity: float = 0.0
) -> str:
    """
    Generate a response to a query using RAG (Retrieval-Augmented Generation).
    
    This function:
    1. Generates an embedding for the query
    2. Retrieves similar documents from the vector store
    3. Constructs a prompt with the retrieved context
    4. Generates a response using ChatGroq LLM
    
    Args:
        query (str): The user's question or query
        vector_store (Any): VectorStore instance with a collection attribute
        embedding_manager (Any): EmbeddingManager instance with generate_embedding method
        groq_api_key (Optional[str]): Groq API key. If None, reads from GROQ_API_KEY env var
        model_name (str): Name of the ChatGroq model to use
        temperature (float): Temperature for LLM generation (0.0-1.0)
        max_tokens (int): Maximum tokens in the response
        n_results (int): Number of documents to retrieve from vector store
        min_similarity (float): Minimum similarity score threshold (0.0-1.0)
    
    Returns:
        str: Generated response from the LLM
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If query processing or LLM call fails
    """
    # Validate inputs
    if not query or not isinstance(query, str) or not query.strip():
        error_msg = "Query must be a non-empty string"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if vector_store is None or not hasattr(vector_store, 'collection'):
        error_msg = "vector_store must have a 'collection' attribute"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if embedding_manager is None or not hasattr(embedding_manager, 'generate_embedding'):
        error_msg = "embedding_manager must have a 'generate_embedding' method"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if n_results <= 0:
        error_msg = f"n_results must be positive, got {n_results}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not (0.0 <= temperature <= 1.0):
        error_msg = f"temperature must be between 0.0 and 1.0, got {temperature}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if max_tokens <= 0:
        error_msg = f"max_tokens must be positive, got {max_tokens}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Get API key from parameter or environment variable
    if groq_api_key is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            error_msg = "Groq API key not provided. Set GROQ_API_KEY environment variable or pass groq_api_key parameter."
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    try:
        # Step 1: Generate query embedding
        logger.info(f"Generating embedding for query: {query[:50]}...")
        query_embedding = embedding_manager.generate_embedding([query])
        
        if query_embedding is None or len(query_embedding) == 0:
            error_msg = "Failed to generate query embedding"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        query_embedding = query_embedding[0]
        
        # Step 2: Query vector store
        logger.info(f"Querying vector store for {n_results} results")
        results = vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Validate query results
        if not results or 'documents' not in results:
            error_msg = "No results returned from vector store"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if not results['documents'] or not results['documents'][0]:
            logger.warning("No documents retrieved from vector store")
            context = ""
            retrieved_docs = []
        else:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(documents)
            distances = results['distances'][0] if 'distances' in results else [1.0] * len(documents)
            ids = results['ids'][0] if 'ids' in results else [f"doc_{i}" for i in range(len(documents))]
            
            # Filter documents by similarity threshold
            retrieved_docs = []
            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):
                similarity_score = 1.0 - distance if distance <= 1.0 else 0.0
                
                if similarity_score >= min_similarity:
                    retrieved_docs.append({
                        'id': doc_id,
                        'metadata': metadata,
                        'distance': distance,
                        'similarity_score': similarity_score,
                        'content': document,
                        'rank': i + 1
                    })
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents above similarity threshold {min_similarity}")
            
            # Build context from retrieved documents
            context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        
        # Step 3: Initialize LLM
        try:
            logger.info(f"Initializing ChatGroq model: {model_name}")
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            error_msg = f"Failed to initialize ChatGroq: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Step 4: Construct prompt
        prompt = f"""You are an expert assistant.

Answer the user's question using the provided information as your internal knowledge source.

Rules:
- Do NOT mention the existence of documents, context, sources, retrieval, or prompts.
- Do NOT say phrases like "based on the context", "according to the document", or "the provided text".
- Respond as if the information is part of your own knowledge.
- If the information is insufficient, say "Sorry about that—this isn't something I'm well-versed in yet. I can certainly help with statistical learning and related R or Python programming." instead of guessing.
- Be concise, clear, and confident.

Provided information:
{context}

Question: {query}"""
        
        # Step 5: Generate response
        try:
            logger.info("Generating response from LLM")
            response = llm.invoke([prompt])
            
            if not response or not hasattr(response, 'content'):
                error_msg = "Invalid response from LLM"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            response_content = response.content
            
            if not response_content or not response_content.strip():
                logger.warning("Empty response from LLM")
                response_content = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            logger.info("Successfully generated response")
            return response_content
            
        except Exception as e:
            error_msg = f"Error generating LLM response: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    except (ValueError, RuntimeError):
        # Re-raise validation and runtime errors
        raise
    except Exception as e:
        # Catch any unexpected errors
        error_msg = f"Unexpected error in get_response: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
