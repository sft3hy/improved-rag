# main.py

import streamlit as st
from rag_pipeline import AdvancedRAGPipeline
import database
from document_processor import DocumentProcessor
from langchain_core.documents import Document
import logging
import traceback
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Pipeline",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def initialize_document_processor():
    """Initialize document processor with caching"""
    return DocumentProcessor()

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize RAG pipeline with caching to avoid reinitialization"""
    try:
        return AdvancedRAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        logger.error(f"RAG pipeline initialization error: {traceback.format_exc()}")
        return None

def main():
    st.title("🔍 Advanced RAG Pipeline with Local Models")
    st.markdown("Upload documents and ask questions using local Ollama models")
    
    # Initialize database
    try:
        database.init_db()
    except Exception as e:
        st.error(f"Database initialization failed: {str(e)}")
        return
    
    # Initialize document processor
    doc_processor = initialize_document_processor()
    
    # Initialize RAG pipeline
    rag_pipeline = initialize_rag_pipeline()
    if rag_pipeline is None:
        st.error("Cannot proceed without RAG pipeline. Please check your Ollama installation and models.")
        st.info("Make sure you have Ollama running and the required models installed:")
        st.code("ollama pull nomic-embed-text\nollama pull llama3.1:8b")
        return
    
    # Advanced settings sidebar
    with st.sidebar:
        st.header("⚙️ Advanced Settings")
        
        # RAG Pipeline Settings
        with st.expander("🔧 RAG Settings"):
            chunk_size = st.slider("Child Chunk Size", 100, 500, 200)
            parent_chunk_size = st.slider("Parent Chunk Size", 1000, 4000, 2000)
            max_docs = st.slider("Max Final Documents", 10, 50, 30)
            min_score = st.slider("Min Relevance Score", 0.0, 0.5, 0.1)
            
            if st.button("🔄 Update Settings"):
                st.info("Settings updated! Note: Restart may be needed for some changes.")
        
        # Database Management
        with st.expander("🗄️ Database Management"):
            try:
                stats = database.get_database_stats()
                st.metric("Documents", stats.get('document_count', 0))
                st.metric("Queries", stats.get('query_count', 0))
                st.metric("DB Size", f"{stats.get('database_file_size', 0):,} bytes")
                
                if st.button("🗑️ Clear All Data", help="This will delete all documents and queries"):
                    if st.checkbox("I understand this will delete all data"):
                        database.clear_all_data()
                        st.success("All data cleared!")
                        st.experimental_rerun()
            except Exception as e:
                st.error(f"Could not load database stats: {str(e)}")
        
        st.header("📁 Document Management")
        
        # Show supported file types
        supported_extensions = doc_processor.get_supported_extensions()
        st.info(f"Supported formats: {', '.join(supported_extensions)}")
        
        # Show missing dependencies if any
        missing_deps = doc_processor.get_missing_dependencies()
        if missing_deps:
            st.warning("Install these for full functionality:")
            for dep in missing_deps:
                st.text(f"• {dep}")
        
        # Show existing documents
        try:
            existing_docs = database.get_all_documents()
            if existing_docs:
                st.subheader("Existing Documents:")
                for filename, _ in existing_docs:
                    st.text(f"• {filename}")
            else:
                st.info("No documents uploaded yet.")
        except Exception as e:
            st.error(f"Failed to load existing documents: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents", 
            type=supported_extensions, 
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(supported_extensions)}"
        )
        
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Read file content
                    file_content = uploaded_file.getvalue()
                    file_type = uploaded_file.type
                    
                    # Process the document
                    processed_text, success = doc_processor.process_document(
                        file_content, uploaded_file.name, file_type
                    )
                    
                    if success:
                        # Add to database
                        database.add_document(uploaded_file.name, processed_text)
                        st.success(f"✅ Successfully uploaded and processed {uploaded_file.name}")
                    else:
                        st.error(f"❌ Failed to process {uploaded_file.name}: {processed_text}")
                    
                except Exception as e:
                    st.error(f"❌ Failed to upload {uploaded_file.name}: {str(e)}")
                    logger.error(f"Upload error for {uploaded_file.name}: {traceback.format_exc()}")
            
            # Process all documents
            try:
                status_text.text("Processing documents for retrieval...")
                all_docs_from_db = database.get_all_documents()
                documents_to_process = [
                    Document(page_content=content, metadata={"source": filename}) 
                    for filename, content in all_docs_from_db
                ]
                rag_pipeline.process_and_add_documents(documents_to_process)
                progress_bar.progress(1.0)
                status_text.text("✅ All documents processed and ready for queries!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Failed to process documents: {str(e)}")
                logger.error(f"Document processing error: {traceback.format_exc()}")
    
    with col2:
        st.header("❓ Ask Questions")
        
        # Retrieval strategy selector
        retrieval_strategy = st.selectbox(
            "Choose retrieval strategy:",
            ["advanced", "parent", "multi_query", "compression", "hybrid"],
            index=0,
            help="Advanced: Optimized multi-stage retrieval (recommended)"
        )
        
        query = st.text_area(
            "Enter your question:", 
            height=100,
            placeholder="What would you like to know about your documents?"
        )
        
        col2_1, col2_2 = st.columns([1, 3])
        with col2_1:
            ask_button = st.button("🔍 Ask", type="primary")
        with col2_2:
            show_sources = st.checkbox("Show source documents", value=True)
        
        if ask_button and query:
            try:
                # Record start time for performance tracking
                start_time = time.time()
                
                # Show spinner while processing
                with st.spinner("Searching through your documents..."):
                    retrieved_docs = rag_pipeline.retrieve_documents(query, retrieval_strategy)
                
                # Calculate response time
                response_time_ms = int((time.time() - start_time) * 1000)
                
                # Add query to database with metrics
                database.add_query(query, response_time_ms, len(retrieved_docs))
                
                if retrieved_docs:
                    st.subheader("📋 Answer")
                    
                    # Show retrieval statistics
                    with st.expander("🔍 Retrieval Statistics", expanded=False):
                        total_chars = sum(len(doc.page_content) for doc in retrieved_docs)
                        estimated_tokens = total_chars // 4
                        st.metric("Documents Retrieved", len(retrieved_docs))
                        st.metric("Total Characters", f"{total_chars:,}")
                        st.metric("Estimated Tokens", f"{estimated_tokens:,}")
                        st.metric("Strategy Used", retrieval_strategy.title())
                    
                    # Generate answer using the retrieved documents
                    try:
                        with st.spinner("Generating answer..."):
                            answer = rag_pipeline.generate_answer(query, retrieved_docs)
                        
                        # Display answer with nice formatting
                        st.markdown("### 💡 Answer")
                        st.write(answer)
                        
                    except Exception as e:
                        st.warning("Could not generate a complete answer, showing retrieved documents instead.")
                        st.error(f"Answer generation error: {str(e)}")
                        logger.error(f"Answer generation error: {traceback.format_exc()}")
                    
                    # Show performance metrics
                    st.caption(f"⏱️ Response time: {response_time_ms}ms | 📄 Documents found: {len(retrieved_docs)}")
                    
                    if show_sources:
                        st.subheader("📚 Source Documents")
                        for i, doc in enumerate(retrieved_docs, 1):
                            # Extract context header if available
                            context_header = doc.metadata.get('context_header', '')
                            source = doc.metadata.get('source', 'Unknown')
                            
                            # Create expandable section for each document
                            with st.expander(f"Source {i}: {source}" + (f" - {context_header}" if context_header else "")):
                                # Show metadata
                                if doc.metadata:
                                    st.caption(f"**Metadata:** {doc.metadata}")
                                
                                # Show content
                                content = doc.page_content
                                if content.startswith('[') and ']\n' in content:
                                    # If content has contextual header, highlight it
                                    header_end = content.find(']\n') + 2
                                    header = content[:header_end]
                                    body = content[header_end:]
                                    st.markdown(f"**{header}**")
                                    st.write(body)
                                else:
                                    st.write(content)
                else:
                    st.warning("No relevant documents found for your query.")
                    st.info("Try:")
                    st.write("• Using different keywords")
                    st.write("• Making your question more specific")
                    st.write("• Checking if documents are properly uploaded")
                    
            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")
                logger.error(f"Query error: {traceback.format_exc()}")
    
    # Query history
    with st.expander("📝 Query History"):
        try:
            queries = database.get_recent_queries()
            if queries:
                for query_text, timestamp, response_time, num_results in queries:
                    metrics = f" ({response_time}ms, {num_results} docs)" if response_time and num_results else ""
                    st.text(f"{timestamp}: {query_text}{metrics}")
            else:
                st.text("No queries yet.")
        except Exception as e:
            st.error(f"Failed to load query history: {str(e)}")

if __name__ == "__main__":
    main()