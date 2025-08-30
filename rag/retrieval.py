import numpy as np
from typing import List, Dict, Any, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
import logging
import streamlit as st
import re

logger = logging.getLogger(__name__)


class EnhancedMultiQueryRAGRetriever:
    def __init__(self, doc_ops, embedding_manager, llm_client, config):
        self.doc_ops = doc_ops
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.config = config
        self.total_tokens_used = 0

    def decompose_complex_query(self, query: str) -> Tuple[List[str], Dict[str, int]]:
        """Break down complex queries into sub-questions."""
        prompt = f"""Break down this complex question into 2-4 simpler, independent sub-questions that together would fully answer the original question. Each sub-question should focus on a specific aspect or entity.

Original question: {query}

Sub-questions (one per line):"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that breaks down complex questions into simpler sub-questions. Each sub-question should be specific and focused.",
                },
                {"role": "user", "content": prompt},
            ]

            response = self.llm_client._make_chat_completion(
                messages=messages, max_tokens=300, temperature=0.3
            )

            decomposition = self.llm_client._extract_content(response).strip()
            token_usage = self.llm_client.get_token_usage(response)

            # Parse sub-questions (simple line-based parsing)
            sub_questions = []
            for line in decomposition.split("\n"):
                line = line.strip()
                # Remove numbering and clean up
                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                if line and len(line) > 10:  # Filter out very short lines
                    sub_questions.append(line)

            logger.info(
                f"Decomposed query into {len(sub_questions)} sub-questions: {sub_questions}"
            )
            return sub_questions, token_usage

        except Exception as e:
            logger.error(f"Failed to decompose query: {e}")
            return [query], {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    def compute_similarity_scores(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: Union[List[np.ndarray], np.ndarray],
    ) -> List[float]:
        """Compute cosine similarity scores between query and chunk embeddings."""
        if isinstance(chunk_embeddings, list):
            if not chunk_embeddings:
                return []
            embeddings_matrix = np.vstack(chunk_embeddings)
        elif isinstance(chunk_embeddings, np.ndarray):
            if chunk_embeddings.size == 0:
                return []
            if chunk_embeddings.ndim == 1:
                embeddings_matrix = chunk_embeddings.reshape(1, -1)
            else:
                embeddings_matrix = chunk_embeddings
        else:
            logger.error(f"Unexpected chunk_embeddings type: {type(chunk_embeddings)}")
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        try:
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

        filtered_chunks = [
            (chunk, score) for chunk, score in chunks_with_scores if score >= min_score
        ]
        filtered_chunks.sort(key=lambda x: x[1], reverse=True)

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
            query_embedding = self.embedding_manager.encode_single(query)
            child_chunks_data = self.doc_ops.get_child_chunks_with_embeddings(user_id)

            if child_chunks_data is None or len(child_chunks_data) == 0:
                logger.warning(f"No child chunks found for user {user_id}")
                return []

            chunks_info = []
            embeddings = []

            for chunk_id, chunk_text, contextual_header, embedding in child_chunks_data:
                chunks_info.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text,
                        "text": chunk_text,
                        "contextual_header": contextual_header,
                    }
                )
                embeddings.append(embedding)

            similarity_scores = self.compute_similarity_scores(
                query_embedding, embeddings
            )

            if not similarity_scores:
                logger.warning("No similarity scores computed")
                return []

            chunks_with_scores = list(zip(chunks_info, similarity_scores))
            reranked_chunks = self.rerank_chunks(
                chunks_with_scores, min_score=self.config.MIN_RELEVANCE_SCORE
            )

            return reranked_chunks[:top_k]

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

    def multi_question_retrieval(
        self, sub_questions: List[str], user_id: str
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """Retrieve chunks for multiple sub-questions and combine results."""
        logger.info(f"Performing retrieval for {len(sub_questions)} sub-questions")

        all_parent_chunks = []
        seen_chunk_ids = set()
        total_tokens = {"total_tokens": 0}

        try:
            for i, sub_question in enumerate(sub_questions):
                logger.info(f"Processing sub-question {i+1}: {sub_question[:50]}...")

                # Retrieve child chunks for this sub-question
                child_chunks = self.retrieve_child_chunks(
                    sub_question,
                    user_id,
                    self.config.INITIAL_RETRIEVAL_COUNT // len(sub_questions),
                )

                # Get corresponding parent chunks
                parent_chunks = self.get_parent_chunks_from_children(child_chunks)

                # Add unique parent chunks to collection
                for parent_chunk in parent_chunks[
                    : self.config.PARENT_CHUNKS_COUNT // len(sub_questions)
                ]:
                    if parent_chunk["chunk_id"] not in seen_chunk_ids:
                        # Add metadata about which sub-question retrieved this chunk
                        parent_chunk["retrieved_by_question"] = sub_question
                        parent_chunk["question_index"] = i
                        all_parent_chunks.append(parent_chunk)
                        seen_chunk_ids.add(parent_chunk["chunk_id"])

            logger.info(
                f"Retrieved {len(all_parent_chunks)} unique parent chunks from all sub-questions"
            )
            return all_parent_chunks, total_tokens

        except Exception as e:
            logger.error(f"Error in multi-question retrieval: {e}")
            return [], total_tokens

    def diversified_retrieval(
        self, query: str, user_id: str
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """Enhanced retrieval that combines query decomposition with traditional multi-query approach."""
        logger.info("Starting diversified retrieval")
        total_tokens = {"total_tokens": 0}

        try:
            # Step 1: Decompose the query into sub-questions
            sub_questions, decomp_tokens = self.decompose_complex_query(query)
            total_tokens["total_tokens"] += decomp_tokens.get("total_tokens", 0)

            # Step 2: Generate alternative phrasing for the original query
            alternative_query, alt_tokens = self.llm_client.generate_multi_query(query)
            total_tokens["total_tokens"] += alt_tokens.get("total_tokens", 0)

            # Step 3: Create comprehensive query list
            all_queries = [query, alternative_query] + sub_questions

            # Step 4: Retrieve chunks for all queries
            all_parent_chunks = []
            seen_chunk_ids = set()

            for i, search_query in enumerate(all_queries):
                logger.info(
                    f"Retrieving for query {i+1}/{len(all_queries)}: {search_query[:50]}..."
                )

                # Adjust retrieval count based on number of queries
                retrieval_count = max(
                    50, self.config.INITIAL_RETRIEVAL_COUNT // len(all_queries)
                )

                child_chunks = self.retrieve_child_chunks(
                    search_query, user_id, retrieval_count
                )
                parent_chunks = self.get_parent_chunks_from_children(child_chunks)

                # Take top chunks for this query
                parent_count = max(
                    2, self.config.PARENT_CHUNKS_COUNT // len(all_queries)
                )

                for parent_chunk in parent_chunks[:parent_count]:
                    if parent_chunk["chunk_id"] not in seen_chunk_ids:
                        parent_chunk["retrieved_by_query"] = search_query
                        parent_chunk["query_type"] = (
                            "original"
                            if i == 0
                            else "alternative" if i == 1 else "sub_question"
                        )
                        all_parent_chunks.append(parent_chunk)
                        seen_chunk_ids.add(parent_chunk["chunk_id"])

            logger.info(
                f"Diversified retrieval found {len(all_parent_chunks)} unique chunks"
            )
            return all_parent_chunks, total_tokens

        except Exception as e:
            logger.error(f"Error in diversified retrieval: {e}")
            return [], total_tokens

    def final_reranking_with_coverage(
        self, original_query: str, parent_chunks: List[Dict]
    ) -> List[Dict]:
        """Enhanced reranking that considers both relevance and query coverage."""
        if not parent_chunks:
            logger.warning("No parent chunks to rerank")
            return []

        try:
            # Get query embedding
            query_embedding = self.embedding_manager.encode_single(original_query)

            # Get parent chunk texts and embeddings
            parent_texts = []
            for chunk in parent_chunks:
                text = chunk.get("chunk_text") or chunk.get("text", "")
                parent_texts.append(text)

            if not parent_texts:
                logger.warning("No parent texts found for reranking")
                return []

            parent_embeddings = self.embedding_manager.encode(parent_texts)
            similarity_scores = self.compute_similarity_scores(
                query_embedding, parent_embeddings
            )

            if not similarity_scores:
                logger.warning("No similarity scores computed for final reranking")
                return []

            # Enhanced scoring that considers diversity
            enhanced_chunks = []
            for i, (chunk, similarity_score) in enumerate(
                zip(parent_chunks, similarity_scores)
            ):
                enhanced_chunk = chunk.copy()
                enhanced_chunk["relevance_score"] = similarity_score

                # Boost score slightly if retrieved by sub-question (promotes diversity)
                if chunk.get("query_type") == "sub_question":
                    enhanced_chunk["relevance_score"] *= 1.1

                enhanced_chunks.append(enhanced_chunk)

            # Sort by enhanced relevance score
            enhanced_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Filter by minimum score
            filtered_chunks = [
                chunk
                for chunk in enhanced_chunks
                if chunk["relevance_score"] >= self.config.MIN_RELEVANCE_SCORE
            ]

            # Return final top chunks
            final_chunks = filtered_chunks[: self.config.FINAL_CHUNKS_COUNT]
            logger.info(
                f"Final reranking with coverage produced {len(final_chunks)} chunks"
            )

            return final_chunks

        except Exception as e:
            logger.error(f"Error in final reranking with coverage: {e}")
            return []

    def retrieve_and_generate(
        self, query: str, user_id: str
    ) -> Tuple[str, List[Dict], float, int]:
        """Complete enhanced RAG pipeline with improved multi-document retrieval."""
        import time

        start_time = time.time()
        total_tokens = 0

        logger.info(f"Starting enhanced RAG retrieval for query: {query[:100]}...")

        try:
            # Step 1: Diversified retrieval (combines decomposition + multi-query)
            parent_chunks, retrieval_tokens = self.diversified_retrieval(query, user_id)

            if not parent_chunks:
                logger.warning("No relevant chunks found")
                processing_time = time.time() - start_time
                return (
                    "I couldn't find relevant information to answer your question. Please try rephrasing your query or upload more documents.",
                    [],
                    processing_time,
                    retrieval_tokens.get("total_tokens", 0),
                )

            # Step 2: Enhanced final reranking with coverage consideration
            final_chunks = self.final_reranking_with_coverage(query, parent_chunks)

            if not final_chunks:
                logger.warning("No chunks passed final reranking")
                processing_time = time.time() - start_time
                return (
                    "I couldn't find sufficiently relevant information to answer your question.",
                    [],
                    processing_time,
                    retrieval_tokens.get("total_tokens", 0),
                )

            # Step 3: Generate comprehensive answer
            answer, answer_tokens = self.llm_client.generate_answer(query, final_chunks)

            # Calculate total tokens used
            total_tokens = retrieval_tokens.get("total_tokens", 0) + answer_tokens.get(
                "total_tokens", 0
            )

            processing_time = time.time() - start_time
            logger.info(
                f"Enhanced RAG retrieval completed in {processing_time:.2f} seconds"
            )
            logger.info(f"Total tokens used: {total_tokens}")

            return answer, final_chunks, processing_time, total_tokens

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in enhanced retrieve_and_generate: {e}")
            return (
                f"An error occurred while processing your query: {str(e)}",
                [],
                processing_time,
                0,
            )
