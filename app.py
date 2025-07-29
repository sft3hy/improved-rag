import streamlit as st
import os
import time
from datetime import datetime
import logging
import traceback

# Import project modules
from config.settings import settings
from database.models import DatabaseManager
from database.operations import DocumentOperations, QueryOperations
from rag.embeddings import EmbeddingManager
from rag.chunking import DocumentChunker
from rag.llm_client import GroqLLMClient
from rag.retrieval import EnhancedRAGRetriever
from utils.document_processor import EnhancedDocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enhanced RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = (
        "default_user"  # In production, use proper authentication
    )

if "initialized" not in st.session_state:
    st.session_state.initialized = False


# Initialize system components
@st.cache_resource
def initialize_system():
    """Initialize all system components."""
    try:
        # Check API key
        if not settings.GROQ_API_KEY:
            st.error("GROQ_API_KEY environment variable is not set!")
            st.stop()

        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Initialize components
        db_manager = DatabaseManager(settings.DATABASE_PATH)
        doc_ops = DocumentOperations(db_manager)
        query_ops = QueryOperations(db_manager)

        embedding_manager = EmbeddingManager(settings.EMBEDDING_MODEL)
        document_chunker = DocumentChunker(
            child_chunk_size=settings.CHILD_CHUNK_SIZE,
            parent_chunk_size=settings.PARENT_CHUNK_SIZE,
            contextual_header_size=settings.CONTEXTUAL_HEADER_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        llm_client = GroqLLMClient(settings.GROQ_API_KEY, settings.LLM_MODEL)
        document_processor = EnhancedDocumentProcessor()

        retriever = EnhancedRAGRetriever(
            doc_ops, embedding_manager, llm_client, settings
        )

        logger.info("System initialized successfully")

        return {
            "db_manager": db_manager,
            "doc_ops": doc_ops,
            "query_ops": query_ops,
            "embedding_manager": embedding_manager,
            "document_chunker": document_chunker,
            "llm_client": llm_client,
            "document_processor": document_processor,
            "retriever": retriever,
        }

    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        st.error(f"System initialization failed: {e}")
        st.stop()


# Initialize system
if not st.session_state.initialized:
    with st.spinner("Initializing Enhanced RAG System..."):
        st.session_state.components = initialize_system()
        st.session_state.initialized = True

# Get components
components = st.session_state.components


def process_document_upload(uploaded_file):
    """Process uploaded document and store in database."""
    try:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Process file
            result = components["document_processor"].process_uploaded_file(
                uploaded_file, settings.MAX_UPLOAD_SIZE
            )

            if not result["success"]:
                st.error(result["error_message"])
                return False

            # Store document in database
            document_id = components["doc_ops"].insert_document(
                document_name=result["filename"],
                user_id=st.session_state.user_id,
                document_text=result["text_content"],
                file_size=result["file_size"],
                file_type=result["file_type"],
            )

            # Create chunks
            parent_chunks, child_chunks = components[
                "document_chunker"
            ].create_parent_child_chunks(result["text_content"], result["filename"])

            # Store parent chunks and generate embeddings
            parent_chunk_ids = []
            for parent_chunk in parent_chunks:
                embedding = components["embedding_manager"].encode_single(
                    parent_chunk["text"]
                )
                parent_id = components["doc_ops"].insert_chunk(
                    document_id=document_id,
                    chunk_text=parent_chunk["text"],
                    contextual_header=parent_chunk["contextual_header"],
                    chunk_type="parent",
                    embedding=embedding,
                    chunk_index=parent_chunk["index"],
                )
                parent_chunk_ids.append(parent_id)

            # Store child chunks with parent relationships
            for child_chunk in child_chunks:
                parent_id = parent_chunk_ids[child_chunk["parent_index"]]
                embedding = components["embedding_manager"].encode_single(
                    child_chunk["text"]
                )
                components["doc_ops"].insert_chunk(
                    document_id=document_id,
                    chunk_text=child_chunk["text"],
                    contextual_header=child_chunk["contextual_header"],
                    chunk_type="child",
                    embedding=embedding,
                    chunk_index=child_chunk["index"],
                    parent_chunk_id=parent_id,
                )

            # Mark document as processed
            components["doc_ops"].mark_document_processed(document_id)

            st.success(f"✅ Successfully processed {uploaded_file.name}")
            st.info(
                f"Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks"
            )
            return True

    except Exception as e:
        logger.error(f"Error processing document {uploaded_file.name}: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error processing document {uploaded_file.name}: {str(e)}")
        return False


def display_sources(sources):
    """Display sources in a formatted way."""
    if not sources:
        st.info("No sources found.")
        return

    with st.expander(f"📚 Sources Used ({len(sources)} chunks)", expanded=False):
        for i, source in enumerate(sources, 1):
            # Handle different possible source structures
            header = (
                source.get("contextual_header") or source.get("header") or f"Source {i}"
            )

            # Try different possible text keys
            text_content = (
                source.get("text")
                or source.get("text")
                or source.get("content")
                or "Content not available"
            )

            st.markdown(f"**Source {i}: {header}**")

            # Display truncated text
            if len(text_content) > 500:
                st.markdown(text_content[:500] + "...")
            else:
                st.markdown(text_content)

            # Display relevance score if available
            relevance_score = source.get("relevance_score") or source.get("score")
            if relevance_score is not None:
                st.caption(f"Relevance: {relevance_score:.3f}")

            # Display additional metadata if available
            chunk_id = source.get("chunk_id")
            if chunk_id:
                st.caption(f"Chunk ID: {chunk_id}")

            if i < len(sources):  # Don't add divider after last source
                st.divider()


def get_file_type_emoji(file_type):
    """Get emoji for file type."""
    file_type_emojis = {
        ".pdf": "📄",
        ".docx": "📝",
        ".doc": "📝",
        ".xlsx": "📊",
        ".xls": "📊",
        ".csv": "📊",
        ".pptx": "📋",
        ".ppt": "📋",
        ".html": "🌐",
        ".htm": "🌐",
        ".py": "🐍",
        ".js": "⚡",
        ".json": "🔧",
        ".eml": "📧",
        ".txt": "📃",
        ".md": "📝",
        ".xml": "🔧",
        ".yaml": "🔧",
        ".yml": "🔧",
    }
    return file_type_emojis.get(file_type.lower(), "📎")


def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/(1024*1024):.1f} MB"


def main():
    """Main application interface."""

    # Header
    st.title("🔍 Enhanced RAG System")
    st.markdown(
        "*Advanced Retrieval-Augmented Generation with Multi-Query and Parent-Child Chunking*"
    )

    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")

        # File upload with expanded file types
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=[
                # Text files
                "txt",
                "md",
                "rst",
                "log",
                "cfg",
                "ini",
                "conf",
                # Code files
                "py",
                "js",
                "html",
                "css",
                "json",
                "xml",
                "yaml",
                "yml",
                "java",
                "cpp",
                "c",
                "h",
                "php",
                "rb",
                "go",
                "rs",
                "sql",
                # Office documents
                "pdf",
                "docx",
                "doc",
                "pptx",
                "ppt",
                # Spreadsheets
                "xlsx",
                "xls",
                "csv",
                "tsv",
                # Email and web
                "eml",
                "htm",
                "xhtml",
                # Other formats
                "rtf",
            ],
            accept_multiple_files=True,
            help=f"Supports various file types. Max file size: {settings.MAX_UPLOAD_SIZE}MB",
        )

        if uploaded_files:
            # Show file summary
            file_types = {}
            total_size = 0

            for file in uploaded_files:
                file.seek(0, 2)  # Go to end of file
                file_size = file.tell()
                file.seek(0)  # Reset to beginning
                total_size += file_size

                # Get file extension
                file_ext = os.path.splitext(file.name)[1].lower()
                file_types[file_ext] = file_types.get(file_ext, 0) + 1

            # Display file summary
            st.info(
                f"📊 **{len(uploaded_files)} files selected** ({format_file_size(total_size)})"
            )

            for file_type, count in file_types.items():
                emoji = get_file_type_emoji(file_type)
                st.caption(f"{emoji} {count}x {file_type.upper()[1:]} files")

            # Process files button
            if st.button("Process All Files", type="primary"):
                success_count = 0
                failed_files = []

                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    progress_bar.progress((i + 1) / len(uploaded_files))

                    if process_document_upload(uploaded_file):
                        success_count += 1
                    else:
                        failed_files.append(uploaded_file.name)

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Show results
                if success_count > 0:
                    st.success(
                        f"✅ Successfully processed {success_count}/{len(uploaded_files)} files"
                    )

                if failed_files:
                    st.error(f"❌ Failed to process {len(failed_files)} files:")
                    for failed_file in failed_files:
                        st.caption(f"• {failed_file}")

                # Refresh the page to show new documents
                if success_count > 0:
                    st.rerun()

        st.divider()

        # Show user documents
        st.subheader("📄 Your Documents")
        user_docs = components["doc_ops"].get_user_documents(st.session_state.user_id)

        if user_docs:
            st.metric("Total Documents", len(user_docs))

            # Show recent documents
            st.caption("**Recent uploads:**")
            for doc in user_docs[:10]:  # Show last 10 documents
                status = "✅" if doc["processed"] else "⏳"
                file_ext = doc.get("file_type", "")
                emoji = get_file_type_emoji(file_ext)
                size_display = format_file_size(doc.get("file_size", 0))

                st.caption(f"{status} {emoji} {doc['document_name']} ({size_display})")
        else:
            st.info("No documents uploaded yet")
            st.caption("💡 Upload documents to get started!")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask Questions")

        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Ask anything about your uploaded documents...",
            help="Type your question and click Search to get AI-powered answers from your documents.",
        )

        # Search button
        search_button = st.button(
            "🔍 Search", type="primary", disabled=not query.strip()
        )

        if search_button:
            if not user_docs:
                st.warning("⚠️ Please upload some documents first!")
            else:
                with st.spinner("🤖 Searching and generating answer..."):
                    try:
                        # Perform retrieval and generation
                        start_time = time.time()
                        answer, sources, processing_time = components[
                            "retriever"
                        ].retrieve_and_generate(query, st.session_state.user_id)

                        # Store query in database
                        source_info = []
                        if sources:
                            for source in sources:
                                source_info.append(
                                    {
                                        "chunk_id": source.get("chunk_id"),
                                        "contextual_header": source.get(
                                            "contextual_header"
                                        )
                                        or source.get("header"),
                                        "relevance_score": source.get(
                                            "relevance_score", 0.0
                                        ),
                                    }
                                )

                        components["query_ops"].insert_query(
                            user_query=query,
                            answer_text=answer,
                            answer_sources=source_info,
                            user_id=st.session_state.user_id,
                            processing_time=processing_time,
                            chunks_used=len(sources) if sources else 0,
                        )

                        # Display results
                        st.subheader("📝 Answer")
                        st.write(answer)

                        # Display sources
                        if sources:
                            display_sources(sources)
                        else:
                            st.info("No specific sources were used for this answer.")

                        # Show processing time
                        st.caption(f"⏱️ Processing time: {processing_time:.2f} seconds")

                    except Exception as e:
                        logger.error(f"Error during retrieval: {e}")
                        logger.error(traceback.format_exc())
                        st.error(
                            f"❌ An error occurred while processing your query: {str(e)}"
                        )
                        st.caption(
                            "Please check the logs for more details or try a different question."
                        )

    with col2:
        st.header("📊 Recent Queries")

        # Show recent queries
        try:
            recent_queries = components["query_ops"].get_user_queries(
                st.session_state.user_id, limit=5
            )

            if recent_queries:
                for query_data in recent_queries:
                    # Truncate long queries for display
                    query_preview = (
                        query_data["user_query"][:50] + "..."
                        if len(query_data["user_query"]) > 50
                        else query_data["user_query"]
                    )

                    with st.expander(f"Q: {query_preview}"):
                        st.write("**Question:**")
                        st.write(query_data["user_query"])

                        st.write("**Answer:**")
                        answer_preview = (
                            query_data["answer_text"][:300] + "..."
                            if len(query_data["answer_text"]) > 300
                            else query_data["answer_text"]
                        )
                        st.write(answer_preview)

                        # Show metadata
                        chunks_used = query_data.get("chunks_used", 0)
                        processing_time = query_data.get("processing_time", 0)
                        timestamp = query_data.get("timestamp", "Unknown")

                        st.caption(
                            f"📚 Sources: {chunks_used} | ⏱️ Time: {processing_time:.2f}s"
                        )
                        st.caption(f"🕒 Asked: {timestamp}")
            else:
                st.info("No queries yet")
                st.caption("Ask a question to see your query history here!")

        except Exception as e:
            logger.error(f"Error loading recent queries: {e}")
            st.error(f"Error loading recent queries: {str(e)}")

        st.divider()

        # System stats
        st.subheader("📈 System Stats")
        try:
            total_docs = len(user_docs) if user_docs else 0
            processed_docs = (
                len([doc for doc in user_docs if doc.get("processed", False)])
                if user_docs
                else 0
            )
            recent_queries_count = (
                len(recent_queries) if "recent_queries" in locals() else 0
            )

            st.metric("Total Documents", total_docs)
            st.metric("Processed Documents", processed_docs)
            st.metric("Recent Queries", recent_queries_count)

            # Show processing status if applicable
            if total_docs > 0:
                processing_rate = (processed_docs / total_docs) * 100
                st.metric("Processing Rate", f"{processing_rate:.1f}%")

        except Exception as e:
            logger.error(f"Error calculating system stats: {e}")
            st.error("Error loading system statistics")


if __name__ == "__main__":
    main()
