import numpy as np
from typing import List, Dict, Any, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
import logging
import streamlit as st

logger = logging.getLogger(__name__)


class EnhancedRAGRetriever:
    def __init__(self, doc_ops, embedding_manager, llm_client, config):
        self.doc_ops = doc_ops
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.config = config

    def compute_similarity_scores(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: Union[List[np.ndarray], np.ndarray],
    ) -> List[float]:
        """Compute cosine similarity scores between query and chunk embeddings."""
        # Handle empty embeddings
        # st.write(chunk_embeddings)
        if isinstance(chunk_embeddings, list):
            if not chunk_embeddings:
                return []
            # Convert list of arrays to matrix
            embeddings_matrix = np.vstack(chunk_embeddings)
        elif isinstance(chunk_embeddings, np.ndarray):
            if chunk_embeddings.size == 0:
                return []
            # If it's already a 2D array, use it directly
            if chunk_embeddings.ndim == 1:
                embeddings_matrix = chunk_embeddings.reshape(1, -1)
            else:
                embeddings_matrix = chunk_embeddings
        else:
            logger.error(f"Unexpected chunk_embeddings type: {type(chunk_embeddings)}")
            return []

        # Ensure query_embedding is properly shaped
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        try:
            # Compute cosine similarity
            similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
            return similarities.tolist()
        except Exception as e:
            logger.error(f"Error computing similarity scores: {e}")
            return []

    def rerank_chunks(
        self, chunks_with_scores: List[Tuple[Dict, float]], min_score: float = 0.1
    ) -> List[Dict]:
        """Rerank chunks by similarity score and filter by minimum score."""
        if not chunks_with_scores:
            return []

        # Filter by minimum score
        filtered_chunks = [
            (chunk, score) for chunk, score in chunks_with_scores if score >= min_score
        ]

        # Sort by score descending
        filtered_chunks.sort(key=lambda x: x[1], reverse=True)

        # Add relevance scores to chunks for display
        result_chunks = []
        for chunk, score in filtered_chunks:
            chunk_with_score = chunk.copy()
            chunk_with_score["relevance_score"] = score
            result_chunks.append(chunk_with_score)

        return result_chunks

    def retrieve_child_chunks(
        self, query: str, user_id: str, top_k: int = 200
    ) -> List[Dict]:
        """Retrieve initial child chunks based on query similarity."""
        try:
            # Get query embedding
            query_embedding = self.embedding_manager.encode_single(query)

            # Get all child chunks for the user
            child_chunks_data = self.doc_ops.get_child_chunks_with_embeddings(user_id)

            # FIX: Safely check if child_chunks_data is empty
            if child_chunks_data is None or len(child_chunks_data) == 0:
                logger.warning(f"No child chunks found for user {user_id}")
                return []

            # Separate chunk info and embeddings
            chunks_info = []
            embeddings = []

            for chunk_id, chunk_text, contextual_header, embedding in child_chunks_data:
                chunks_info.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text,  # Use consistent key naming
                        "text": chunk_text,  # Also keep 'text' for backward compatibility
                        "contextual_header": contextual_header,
                    }
                )
                embeddings.append(embedding)

            # Compute similarity scores
            similarity_scores = self.compute_similarity_scores(
                query_embedding, embeddings
            )

            if not similarity_scores:
                logger.warning("No similarity scores computed")
                return []

            # Combine chunks with scores
            chunks_with_scores = list(zip(chunks_info, similarity_scores))

            # Rerank and filter
            reranked_chunks = self.rerank_chunks(
                chunks_with_scores, min_score=self.config.MIN_RELEVANCE_SCORE
            )

            # Return top k chunks
            return reranked_chunks[:top_k]

        except Exception as e:
            logger.error(f"Error retrieving child chunks: {e}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving child chunks: {e}")
            return []

    def get_parent_chunks_from_children(self, child_chunks: List[Dict]) -> List[Dict]:
        """Get parent chunks corresponding to the child chunks."""
        if not child_chunks:
            return []

        parent_chunks = []
        seen_parent_ids = set()

        try:
            for child_chunk in child_chunks:
                parent_chunk = self.doc_ops.get_parent_chunk_by_child_id(
                    child_chunk["chunk_id"]
                )

                if parent_chunk and parent_chunk["chunk_id"] not in seen_parent_ids:
                    # Ensure consistent key naming
                    if "text" not in parent_chunk and "chunk_text" in parent_chunk:
                        parent_chunk["text"] = parent_chunk["chunk_text"]
                    elif "chunk_text" not in parent_chunk and "text" in parent_chunk:
                        parent_chunk["chunk_text"] = parent_chunk["text"]

                    parent_chunks.append(parent_chunk)
                    seen_parent_ids.add(parent_chunk["chunk_id"])

            return parent_chunks

        except Exception as e:
            logger.error(f"Error getting parent chunks: {e}")
            return []

    def multi_query_retrieval(self, original_query: str, user_id: str) -> List[Dict]:
        """Perform multi-query retrieval using original and generated alternative query."""
        logger.info("Starting multi-query retrieval")

        try:
            # Generate alternative query
            alternative_query = self.llm_client.generate_multi_query(original_query)
            logger.info(f"Generated alternative query: {alternative_query[:100]}...")

            # Retrieve child chunks for both queries
            original_child_chunks = self.retrieve_child_chunks(
                original_query, user_id, self.config.INITIAL_RETRIEVAL_COUNT
            )

            alternative_child_chunks = self.retrieve_child_chunks(
                alternative_query, user_id, self.config.INITIAL_RETRIEVAL_COUNT
            )

            logger.info(
                f"Retrieved {len(original_child_chunks)} original and {len(alternative_child_chunks)} alternative child chunks"
            )

            # Filter to configured amounts
            original_filtered = original_child_chunks[
                : self.config.FILTERED_CHILD_CHUNKS
            ]
            alternative_filtered = alternative_child_chunks[
                : self.config.FILTERED_CHILD_CHUNKS
            ]

            # Get parent chunks for both sets
            original_parents = self.get_parent_chunks_from_children(original_filtered)
            alternative_parents = self.get_parent_chunks_from_children(
                alternative_filtered
            )

            # Limit parent chunks
            original_parents = original_parents[: self.config.PARENT_CHUNKS_COUNT]
            alternative_parents = alternative_parents[: self.config.PARENT_CHUNKS_COUNT]

            # Combine parent chunks (removing duplicates)
            combined_parents = []
            seen_ids = set()

            for parent in original_parents + alternative_parents:
                if parent["chunk_id"] not in seen_ids:
                    combined_parents.append(parent)
                    seen_ids.add(parent["chunk_id"])

            logger.info(f"Retrieved {len(combined_parents)} unique parent chunks")
            return combined_parents

        except Exception as e:
            logger.error(f"Error in multi-query retrieval: {e}")
            return []

    def final_reranking(self, query: str, parent_chunks: List[Dict]) -> List[Dict]:
        """Perform final reranking of parent chunks against original query."""
        if not parent_chunks:
            logger.warning("No parent chunks to rerank")
            return []

        try:
            # Get query embedding
            query_embedding = self.embedding_manager.encode_single(query)

            # Get embeddings for parent chunks
            parent_texts = []
            for chunk in parent_chunks:
                # Use chunk_text or text, whichever is available
                text = chunk.get("chunk_text") or chunk.get("text", "")
                parent_texts.append(text)

            if not parent_texts:
                logger.warning("No parent texts found for reranking")
                return []

            # Get embeddings using the batch encode method
            parent_embeddings = self.embedding_manager.encode(parent_texts)

            # Compute similarity scores
            similarity_scores = self.compute_similarity_scores(
                query_embedding, parent_embeddings
            )

            if not similarity_scores:
                logger.warning("No similarity scores computed for final reranking")
                return []

            # Combine with scores
            chunks_with_scores = list(zip(parent_chunks, similarity_scores))

            # Rerank and filter
            reranked_chunks = self.rerank_chunks(
                chunks_with_scores, min_score=self.config.MIN_RELEVANCE_SCORE
            )

            # Return final top chunks
            final_chunks = reranked_chunks[: self.config.FINAL_CHUNKS_COUNT]

            logger.info(f"Final reranking produced {len(final_chunks)} chunks")
            return final_chunks

        except Exception as e:
            logger.error(f"Error in final reranking: {e}")
            return []

    def retrieve_and_generate(
        self, query: str, user_id: str
    ) -> Tuple[str, List[Dict], float]:
        """Complete RAG pipeline: retrieve relevant chunks and generate answer."""
        import time

        start_time = time.time()

        logger.info(f"Starting RAG retrieval for query: {query[:100]}...")

        try:
            # Step 1: Multi-query retrieval
            parent_chunks = self.multi_query_retrieval(query, user_id)

            if not parent_chunks:
                logger.warning("No relevant chunks found")
                processing_time = time.time() - start_time
                return (
                    "I couldn't find relevant information to answer your question. Please try rephrasing your query or upload more documents.",
                    [],
                    processing_time,
                )

            # Step 2: Final reranking
            final_chunks = self.final_reranking(query, parent_chunks)

            if not final_chunks:
                logger.warning("No chunks passed final reranking")
                processing_time = time.time() - start_time
                return (
                    "I couldn't find sufficiently relevant information to answer your question.",
                    [],
                    processing_time,
                )

            # Step 3: Generate answer
            answer = self.llm_client.generate_answer(query, final_chunks)

            processing_time = time.time() - start_time
            logger.info(f"RAG retrieval completed in {processing_time:.2f} seconds")

            return answer, final_chunks, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in retrieve_and_generate: {e}")
            return (
                f"An error occurred while processing your query: {str(e)}",
                [],
                processing_time,
            )
