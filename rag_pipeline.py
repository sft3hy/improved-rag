# rag_pipeline.py

from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
from reranker import HybridReranker
import logging
import os
import tempfile
import re

logger = logging.getLogger(__name__)


class AdvancedRAGPipeline:
    def __init__(self, embedding_model="nomic-embed-text", llm_model="llama3.1:8b"):
        """
        Initialize the RAG pipeline with improved error handling and flexibility
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        try:
            # Initialize embedding model via Ollama
            self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            
            # Initialize ChatOllama for the LLM
            self.llm = ChatOllama(
                model=self.llm_model,
                temperature=0.1,
                top_p=0.9
            )
            
            logger.info(f"Initialized RAG pipeline with embedding model: {embedding_model}, LLM: {llm_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise

        self.setup_splitters()
        self.setup_vectorstore()
        self.setup_retrievers()
        self.setup_reranker()
        self.setup_qa_chain()

    def setup_splitters(self):
        """Setup text splitters optimized for contextual chunking"""
        # Child splitter for small, contextual chunks (200-250 chars + 100 char header)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Parent splitter for larger context chunks (up to 2500 chars + header)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Markdown splitter to preserve structure and context
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ],
            strip_headers=False
        )

    def setup_vectorstore(self):
        """Setup vector store with persistent storage"""
        try:
            # Create a temporary directory for Chroma if needed
            self.chroma_persist_dir = tempfile.mkdtemp(prefix="chroma_")
            
            self.vectorstore = Chroma(
                collection_name="rag_documents",
                embedding_function=self.embeddings,
                persist_directory=self.chroma_persist_dir
            )
            
            self.store = InMemoryStore()
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup vector store: {str(e)}")
            raise

    def setup_retrievers(self):
        """Setup advanced retrieval pipeline following best practices"""
        try:
            # Parent Document Retriever with optimized settings
            self.parent_retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=self.store,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
                k=20  # Return top 20 parent documents
            )
            
            # Multi-Query Retriever for generating alternative questions
            self.multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 200}),
                llm=self.llm
            )
            
            # Contextual Compression Retriever for reranking
            self.compressor = LLMChainExtractor.from_llm(self.llm)
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": 150})
            )
            
            logger.info("All retrievers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup retrievers: {str(e)}")
            raise
    
    def setup_reranker(self):
        """Setup hybrid reranker for document scoring"""
        try:
            self.reranker = HybridReranker(self.llm)
            logger.info("Reranker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup reranker: {str(e)}")
            raise

    def setup_qa_chain(self):
        """Setup the question-answering chain"""
        try:
            # Create a prompt template for RAG
            self.qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context documents.
Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Keep your answer concise but comprehensive.

Context:
{context}

Question: {question}

Answer:""")
            
            # Create the RAG chain
            self.qa_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("QA chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup QA chain: {str(e)}")
            raise

    def convert_to_markdown(self, content, source):
        """Convert content to markdown format for better chunking"""
        # If already markdown, return as is
        if source.endswith('.md'):
            return content
        
        # Basic conversion for other formats
        lines = content.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append('')
                continue
            
            # Convert common patterns to markdown
            if line.startswith('--- ') and line.endswith(' ---'):
                # Convert section headers
                header_text = line[4:-4].strip()
                markdown_lines.append(f"## {header_text}")
            elif line.isupper() and len(line) < 100:
                # Convert all-caps lines to headers
                markdown_lines.append(f"### {line.title()}")
            elif line.startswith('Table:'):
                # Mark tables
                markdown_lines.append(f"#### {line}")
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)

    def create_contextual_header(self, document, max_length=100):
        """Create contextual header for chunks (up to 100 chars)"""
        source = document.metadata.get('source', 'Unknown')
        
        # Extract meaningful context from document
        context_parts = []
        
        # Add source filename (without extension)
        if source != 'Unknown':
            filename = os.path.splitext(os.path.basename(source))[0]
            context_parts.append(filename)
        
        # Add any existing headers from markdown
        for key in ['Header 1', 'Header 2', 'Header 3', 'Header 4']:
            if key in document.metadata:
                header_text = document.metadata[key].strip()
                if header_text and len(header_text) < 50:
                    context_parts.append(header_text)
                    break
        
        # Create contextual header
        context_header = " > ".join(context_parts)
        
        # Truncate if too long
        if len(context_header) > max_length:
            context_header = context_header[:max_length-3] + "..."
        
        return context_header

    def add_contextual_headers(self, documents):
        """Add contextual headers to document chunks"""
        enhanced_docs = []
        
        for doc in documents:
            context_header = self.create_contextual_header(doc)
            
            # Create enhanced content with header
            enhanced_content = f"[{context_header}]\n{doc.page_content}"
            
            # Create new document with enhanced content
            enhanced_doc = Document(
                page_content=enhanced_content,
                metadata={**doc.metadata, "context_header": context_header}
            )
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs

    def filter_by_relevance(self, docs, min_score=0.1):
        """Filter documents by relevance score using reranker"""
        # Note: For now, we'll return all docs since the reranker handles filtering
        # In a full implementation, you'd use actual similarity scores from ChromaDB
        return docs

    def advanced_retrieve(self, query):
        """Advanced retrieval following the optimized pipeline"""
        try:
            logger.info("Starting advanced retrieval pipeline...")
            
            # Step 1: Generate alternative questions using multi-query retriever
            logger.info("Generating alternative queries...")
            try:
                # Get alternative queries (this returns documents, not just queries)
                multi_query_docs = self.multi_query_retriever.invoke(query)
                logger.info(f"Multi-query retriever found {len(multi_query_docs)} documents")
            except Exception as e:
                logger.warning(f"Multi-query retriever failed: {str(e)}")
                multi_query_docs = []
            
            # Step 2: Get small chunks from vector store (up to 200 chunks)
            logger.info("Retrieving small chunks from vector store...")
            small_chunk_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 200})
            original_small_chunks = small_chunk_retriever.invoke(query)
            logger.info(f"Retrieved {len(original_small_chunks)} small chunks")
            
            # Step 3: First reranking - filter small chunks (keep up to 150 with score >= 0.1)
            logger.info("First reranking: filtering small chunks...")
            filtered_small_chunks = self.reranker.rerank_documents(
                query, original_small_chunks, min_score=0.1, max_docs=150
            )
            logger.info(f"After first reranking: {len(filtered_small_chunks)} chunks")
            
            # Step 4: Use parent retriever to get larger parent chunks
            logger.info("Retrieving parent chunks...")
            try:
                parent_docs_original = self.parent_retriever.invoke(query)
                logger.info(f"Retrieved {len(parent_docs_original)} parent docs for original query")
            except Exception as e:
                logger.warning(f"Parent retriever failed: {str(e)}")
                parent_docs_original = []
            
            # Combine all parent documents (limit to prevent excessive processing)
            all_parent_docs = parent_docs_original[:20]  # Start with original query results
            
            # Step 5: Get additional parent docs if we have multi-query results
            if multi_query_docs and len(all_parent_docs) < 40:
                # Add unique documents from multi-query results
                seen_content = {doc.page_content for doc in all_parent_docs}
                for doc in multi_query_docs:
                    if doc.page_content not in seen_content and len(all_parent_docs) < 40:
                        all_parent_docs.append(doc)
                        seen_content.add(doc.page_content)
            
            logger.info(f"Combined parent documents: {len(all_parent_docs)}")
            
            # Step 6: Final reranking of parent documents (keep top 30 with score >= 0.1)
            logger.info("Final reranking of parent documents...")
            final_docs = self.reranker.rerank_documents(
                query, all_parent_docs, min_score=0.1, max_docs=30, use_llm=True
            )
            
            logger.info(f"Advanced retrieval completed: {len(final_docs)} final documents")
            
            # Calculate total tokens estimate for logging
            total_chars = sum(len(doc.page_content) for doc in final_docs)
            estimated_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 characters
            logger.info(f"Final context size: ~{estimated_tokens} tokens ({total_chars} characters)")
            
            return final_docs
            
        except Exception as e:
            logger.error(f"Advanced retrieval failed: {str(e)}")
            # Fallback to simple retrieval
            try:
                fallback_docs = self.vectorstore.similarity_search(query, k=10)
                logger.info(f"Fallback retrieval returned {len(fallback_docs)} documents")
                return fallback_docs
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {str(fallback_error)}")
                return []

    def process_and_add_documents(self, documents):
        """Process and add documents using advanced chunking strategy"""
        try:
            if not documents:
                logger.warning("No documents provided for processing")
                return
            
            all_splits = []
            for doc in documents:
                # Convert content to markdown for better structure
                markdown_content = self.convert_to_markdown(doc.page_content, doc.metadata.get('source', ''))
                
                # Create document with markdown content
                markdown_doc = Document(
                    page_content=markdown_content,
                    metadata=doc.metadata
                )
                
                # Split using markdown splitter first to preserve structure
                try:
                    md_header_splits = self.markdown_splitter.split_documents([markdown_doc])
                    if not md_header_splits:
                        md_header_splits = [markdown_doc]
                except Exception as e:
                    logger.warning(f"Markdown splitting failed for {doc.metadata.get('source', 'unknown')}: {str(e)}")
                    md_header_splits = [markdown_doc]
                
                # Add contextual headers to chunks
                enhanced_splits = self.add_contextual_headers(md_header_splits)
                all_splits.extend(enhanced_splits)
            
            # Add documents to parent retriever (this will create child and parent chunks)
            self.parent_retriever.add_documents(all_splits)
            
            logger.info(f"Successfully processed and added {len(all_splits)} enhanced document chunks from {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to process documents: {str(e)}")
            raise

    def retrieve_documents(self, query, retrieval_strategy="advanced"):
        """
        Retrieve documents using different strategies
        
        Args:
            query: The search query
            retrieval_strategy: "advanced", "parent", "multi_query", "compression", or "hybrid"
        """
        try:
            if retrieval_strategy == "advanced":
                return self.advanced_retrieve(query)
            elif retrieval_strategy == "parent":
                return self.parent_retriever.invoke(query)
            elif retrieval_strategy == "multi_query":
                return self.multi_query_retriever.invoke(query)
            elif retrieval_strategy == "compression":
                return self.compression_retriever.invoke(query)
            elif retrieval_strategy == "hybrid":
                # Combine results from different retrievers
                parent_docs = self.parent_retriever.invoke(query)
                multi_docs = self.multi_query_retriever.invoke(query)
                
                # Deduplicate and combine
                all_docs = parent_docs + multi_docs
                seen = set()
                unique_docs = []
                for doc in all_docs:
                    doc_hash = hash(doc.page_content)
                    if doc_hash not in seen:
                        seen.add(doc_hash)
                        unique_docs.append(doc)
                
                return unique_docs[:20]  # Limit to top 20 results
            else:
                # Default to advanced retrieval
                return self.advanced_retrieve(query)
                
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            # Fallback to basic vector search
            try:
                return self.vectorstore.similarity_search(query, k=10)
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {str(fallback_error)}")
                return []

    def generate_answer(self, question, context_docs):
        """Generate an answer based on the question and context documents"""
        try:
            if not context_docs:
                return "I couldn't find any relevant documents to answer your question."
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Generate answer using the QA chain
            answer = self.qa_chain.invoke({
                "context": context,
                "question": question
            })
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            return f"I encountered an error while generating the answer: {str(e)}"

    def get_vectorstore_stats(self):
        """Get statistics about the vector store"""
        try:
            # This is a simplified version - actual implementation depends on Chroma version
            collection = self.vectorstore._collection
            return {
                "document_count": collection.count(),
                "collection_name": collection.name
            }
        except Exception as e:
            logger.error(f"Failed to get vectorstore stats: {str(e)}")
            return {"error": str(e)}

    def cleanup(self):
        """Clean up resources and temporary files"""
        try:
            if hasattr(self, 'chroma_persist_dir') and os.path.exists(self.chroma_persist_dir):
                import shutil
                shutil.rmtree(self.chroma_persist_dir)
                logger.info("Cleaned up temporary Chroma directory")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary files: {str(e)}")