# reranker.py

import logging
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
import re

logger = logging.getLogger(__name__)

class LocalReranker:
    """Local reranking using Ollama models"""
    
    def __init__(self, llm: ChatOllama):
        self.llm = llm
    
    def calculate_relevance_score(self, query: str, document: Document) -> float:
        """Calculate relevance score using LLM"""
        try:
            prompt = f"""
Rate the relevance of this document to the query on a scale of 0.0 to 1.0.
Only respond with a number between 0.0 and 1.0.

Query: {query}

Document: {document.page_content[:500]}...

Relevance score (0.0-1.0):"""
            
            response = self.llm.invoke(prompt)
            
            # Extract numeric score from response
            score_text = response.content.strip()
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is between 0 and 1
                return max(0.0, min(1.0, score))
            else:
                logger.warning(f"Could not parse relevance score: {score_text}")
                return 0.5  # Default neutral score
                
        except Exception as e:
            logger.error(f"Error calculating relevance score: {str(e)}")
            return 0.5  # Default neutral score
    
    def rerank_documents(self, query: str, documents: List[Document], 
                        min_score: float = 0.1, max_docs: int = 30) -> List[Document]:
        """Rerank documents based on relevance scores"""
        try:
            if not documents:
                return []
            
            logger.info(f"Reranking {len(documents)} documents...")
            
            # Calculate scores for each document
            scored_docs = []
            for doc in documents:
                score = self.calculate_relevance_score(query, doc)
                if score >= min_score:
                    scored_docs.append((doc, score))
            
            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top documents
            result_docs = [doc for doc, score in scored_docs[:max_docs]]
            
            logger.info(f"Reranking completed: {len(result_docs)} documents passed threshold")
            return result_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return documents[:max_docs]  # Fallback to original order


class SimpleReranker:
    """Simple keyword-based reranking for faster processing"""
    
    def __init__(self):
        pass
    
    def calculate_keyword_score(self, query: str, document: Document) -> float:
        """Calculate simple keyword-based relevance score"""
        try:
            query_words = set(query.lower().split())
            doc_words = set(document.page_content.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))
            
            if union == 0:
                return 0.0
            
            jaccard_score = intersection / union
            
            # Boost score if query words appear in context header
            context_header = document.metadata.get('context_header', '').lower()
            header_boost = 0.0
            for word in query_words:
                if word in context_header:
                    header_boost += 0.1
            
            final_score = min(1.0, jaccard_score + header_boost)
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating keyword score: {str(e)}")
            return 0.5
    
    def rerank_documents(self, query: str, documents: List[Document], 
                        min_score: float = 0.1, max_docs: int = 30) -> List[Document]:
        """Rerank documents using keyword-based scoring"""
        try:
            if not documents:
                return []
            
            logger.info(f"Simple reranking {len(documents)} documents...")
            
            # Calculate scores for each document
            scored_docs = []
            for doc in documents:
                score = self.calculate_keyword_score(query, doc)
                if score >= min_score:
                    scored_docs.append((doc, score))
            
            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top documents
            result_docs = [doc for doc, score in scored_docs[:max_docs]]
            
            logger.info(f"Simple reranking completed: {len(result_docs)} documents passed threshold")
            return result_docs
            
        except Exception as e:
            logger.error(f"Simple reranking failed: {str(e)}")
            return documents[:max_docs]


class HybridReranker:
    """Hybrid reranker combining both approaches"""
    
    def __init__(self, llm: ChatOllama):
        self.local_reranker = LocalReranker(llm)
        self.simple_reranker = SimpleReranker()
    
    def rerank_documents(self, query: str, documents: List[Document], 
                        min_score: float = 0.1, max_docs: int = 30,
                        use_llm: bool = True) -> List[Document]:
        """Rerank using hybrid approach"""
        try:
            if len(documents) <= 10 and use_llm:
                # Use LLM reranking for small sets
                return self.local_reranker.rerank_documents(query, documents, min_score, max_docs)
            else:
                # Use simple reranking for large sets or when LLM is disabled
                return self.simple_reranker.rerank_documents(query, documents, min_score, max_docs)
                
        except Exception as e:
            logger.error(f"Hybrid reranking failed: {str(e)}")
            return documents[:max_docs]