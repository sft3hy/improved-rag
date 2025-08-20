import streamlit as st
import os
import time
from datetime import datetime
import logging
import traceback
import json
from datetime import datetime, date

# Import project modules
from config.settings import settings

client = ""
powered = ""
model = ""

if settings.TEST == "True":
    from database.models import DatabaseManager
    from database.operations import DocumentOperations, QueryOperations
    from rag.llm_client import GroqLLMClient

    client = GroqLLMClient()
    powered = "Powered by Groq and Llama 4 Scout"
    model = "Llama 4 Scout"

else:
    from database.models_cvdb import DatabaseManager
    from database.operations_cvdb import DocumentOperations, QueryOperations
    from rag.llm_client import SanctuaryLLMClient

    client = SanctuaryLLMClient()
    powered = "Powered by Sanctuary and Claude 3.5 Sonnet"
    model = "Claude 3.5 Sonnet"

from rag.embeddings import EmbeddingManager
from rag.chunking import DocumentChunker
from rag.retrieval import EnhancedRAGRetriever
from utils.document_processor import EnhancedDocumentProcessor
from utils.streamlit_utils import float_to_percent

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

# Load messages from database only once
if "messages_loaded" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages_loaded = False


# Initialize system components
@st.cache_resource
def initialize_system():
    """Initialize all system components."""
    try:
        # Check API key
        if not settings.GROQ_API_KEY:
            error_msg = "GROQ_API_KEY environment variable is not set!"
            logger.error(error_msg)
            st.error(error_msg)
            st.stop()

        # Create data directory if it doesn't exist
        try:
            os.makedirs("data", exist_ok=True)
        except Exception as e:
            error_msg = f"Failed to create data directory: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(
                f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
            )
            st.stop()

        # Initialize components
        try:
            if settings.TEST == "True":
                db_manager = DatabaseManager(settings.DATABASE_PATH)
            else:
                db_manager = DatabaseManager()

            doc_ops = DocumentOperations(db_manager)
            query_ops = QueryOperations(db_manager)
        except Exception as e:
            error_msg = f"Failed to initialize database components: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(
                f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
            )
            st.stop()

        try:
            embedding_manager = EmbeddingManager(settings.EMBEDDING_MODEL)
        except Exception as e:
            error_msg = f"Failed to initialize embedding manager: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(
                f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
            )
            st.stop()

        try:
            document_chunker = DocumentChunker(
                child_chunk_size=settings.CHILD_CHUNK_SIZE,
                parent_chunk_size=settings.PARENT_CHUNK_SIZE,
                contextual_header_size=settings.CONTEXTUAL_HEADER_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
        except Exception as e:
            error_msg = f"Failed to initialize document chunker: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(
                f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
            )
            st.stop()

        try:
            llm_client = client
        except Exception as e:
            error_msg = f"Failed to initialize LLM client: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(
                f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
            )
            st.stop()

        try:
            document_processor = EnhancedDocumentProcessor()
        except Exception as e:
            error_msg = f"Failed to initialize document processor: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(
                f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
            )
            st.stop()

        try:
            retriever = EnhancedRAGRetriever(
                doc_ops, embedding_manager, llm_client, settings
            )
        except Exception as e:
            error_msg = f"Failed to initialize RAG retriever: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(
                f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
            )
            st.stop()

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
        error_msg = f"Failed to initialize system: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```")
        st.stop()


# Initialize system
if not st.session_state.initialized:
    with st.spinner("Initializing Enhanced RAG System..."):
        st.session_state.components = initialize_system()
        st.session_state.initialized = True

# Get components
components = st.session_state.components


def load_chat_history():
    """Load all queries from the database and format for display."""
    if not st.session_state.messages_loaded:
        try:
            all_queries = components["query_ops"].get_all_queries(limit=100)
            for query_data in all_queries:
                # Add user's query
                st.session_state.messages.append(
                    {"role": "user", "content": query_data["user_query"]}
                )
                # Add assistant's answer
                raw_sources = query_data.get("answer_sources_used", [])
                if type(raw_sources) is list:
                    sources = raw_sources
                else:
                    sources = json.loads(raw_sources)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": query_data["content"],
                        "sources": sources,
                        "processing_time": query_data.get("processing_time"),
                        "tokens_used": query_data.get("tokens_used", 0),
                    }
                )
            st.session_state.messages_loaded = True
        except Exception as e:
            error_msg = f"Failed to load chat history: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(
                f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
            )
            # Don't stop the app, just show empty messages
            st.session_state.messages = []
            st.session_state.messages_loaded = True


def process_document_upload(uploaded_file):
    """Process uploaded document and store in database."""
    try:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Process file
            try:
                result = components["document_processor"].process_uploaded_file(
                    uploaded_file, settings.MAX_UPLOAD_SIZE
                )
            except Exception as e:
                error_msg = f"Failed to process file {uploaded_file.name}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(
                    f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                )
                return False

            if not result["success"]:
                st.error(result["error_message"])
                return False

            # Store document in database
            try:
                document_id = components["doc_ops"].insert_document(
                    document_name=result["filename"],
                    user_id=st.session_state.user_id,
                    document_text=result["text_content"],
                    file_size=result["file_size"],
                    file_type=result["file_type"],
                )
            except Exception as e:
                error_msg = f"Failed to insert document {uploaded_file.name} into database: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(
                    f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                )
                return False

            # Create chunks
            try:
                parent_chunks, child_chunks = components[
                    "document_chunker"
                ].create_parent_child_chunks(result["text_content"], result["filename"])
            except Exception as e:
                error_msg = (
                    f"Failed to create chunks for {uploaded_file.name}: {str(e)}"
                )
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(
                    f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                )
                return False

            # Store parent chunks and generate embeddings
            try:
                parent_chunk_ids = []
                for i, parent_chunk in enumerate(parent_chunks):
                    try:
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
                    except Exception as e:
                        error_msg = f"Failed to process parent chunk {i} for {uploaded_file.name}: {str(e)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        st.error(
                            f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                        )
                        return False
            except Exception as e:
                error_msg = f"Failed to process parent chunks for {uploaded_file.name}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(
                    f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                )
                return False

            # Store child chunks with parent relationships
            try:
                for i, child_chunk in enumerate(child_chunks):
                    try:
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
                    except Exception as e:
                        error_msg = f"Failed to process child chunk {i} for {uploaded_file.name}: {str(e)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        st.error(
                            f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                        )
                        return False
            except Exception as e:
                error_msg = (
                    f"Failed to process child chunks for {uploaded_file.name}: {str(e)}"
                )
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(
                    f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                )
                return False

            # Mark document as processed
            try:
                components["doc_ops"].mark_document_processed(document_id)
            except Exception as e:
                error_msg = f"Failed to mark document {uploaded_file.name} as processed: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(
                    f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                )
                return False

            st.success(f"✅ Successfully processed {uploaded_file.name}")
            st.info(
                f"Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks"
            )
            return True

    except Exception as e:
        error_msg = f"Error processing document {uploaded_file.name}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```")
        return False


def display_sources(sources):
    """
    Displays sources in a stylish and structured format in Streamlit.
    """
    if not sources:
        return

    try:
        with st.expander(f"📚 Sources Used ({len(sources)} Chunks)", expanded=False):
            for i, source in enumerate(sources, 1):
                try:
                    with st.expander(f"Chunk {i}"):
                        breadcrumb = source.get(
                            "source_breadcrumb", f"Source Chunk {i}"
                        )
                        header = source.get("contextual_header", "No Header Available")
                        text = source.get("chunk_text", "Content not available.")

                        st.write(f"📄 {header}")

                        # Truncate text inline
                        display_text = text[:300] + "..." if len(text) > 300 else text
                        st.markdown(f"> {display_text}")

                        # Inline metadata display
                        meta_parts = []
                        if score := source.get("relevance_score"):
                            meta_parts.append(
                                f"Similarity score: {float_to_percent(score)}"
                            )
                        if chunk_id := source.get("chunk_id"):
                            meta_parts.append(f"Chunk ID: `{chunk_id}`")

                        if meta_parts:
                            st.caption(" | ".join(meta_parts))

                        if i < len(sources):
                            st.divider()

                except Exception as e:
                    st.error(f"Error displaying source {i}: {str(e)}")
                    logger.error(f"Source display error: {e}", exc_info=True)

    except Exception as e:
        st.error(f"Error displaying sources: {str(e)}")
        logger.error(f"Sources display error: {e}", exc_info=True)


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
    try:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.1f} KB"
        else:
            return f"{size_bytes/(1024*1024):.1f} MB"
    except Exception as e:
        error_msg = f"Error formatting file size: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return "Unknown size"


def get_daily_token_usage():
    """Get today's token usage from session state."""
    try:
        return components["query_ops"].get_todays_total_tokens()
    except Exception as e:
        error_msg = f"Error getting daily token usage: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```")
        return 0


def recalculate_tokens():
    try:
        with st.sidebar:
            daily_tokens = get_daily_token_usage()
            token_limit = 500000
            progress_percentage = min(daily_tokens / token_limit, 1.0)
            st.write("**Daily Token Usage**")
            st.progress(progress_percentage)
            st.caption(
                f"{daily_tokens:,} / 500k tokens used ({progress_percentage*100:.1f}%)"
            )
            if progress_percentage > 0.8:
                st.warning("⚠️ Approaching daily limit!")
            elif progress_percentage >= 1.0:
                st.error("🚫 Daily limit reached!")
    except Exception as e:
        error_msg = f"Error calculating token usage: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```")


def main():
    """Main application interface."""

    # Header
    # Token usage display
    with st.sidebar:
        st.header("🔍 Enhanced RAG System")
        st.write(
            "*Advanced Retrieval-Augmented Generation with Multi-Query and Parent-Child Chunking*"
        )
        st.write(powered)

    recalculate_tokens()

    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")

        # File upload with expanded file types
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=[
                "txt",
                "md",
                "rst",
                "log",
                "cfg",
                "ini",
                "conf",
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
                "pdf",
                "docx",
                "doc",
                "pptx",
                "ppt",
                "xlsx",
                "xls",
                "csv",
                "tsv",
                "eml",
                "htm",
                "xhtml",
                "rtf",
            ],
            accept_multiple_files=True,
            help=f"Supports various file types. Max file size: {settings.MAX_UPLOAD_SIZE}MB",
        )

        if uploaded_files:
            try:
                file_types = {}
                total_size = 0
                for file in uploaded_files:
                    try:
                        file.seek(0, 2)
                        file_size = file.tell()
                        file.seek(0)
                        total_size += file_size
                        file_ext = os.path.splitext(file.name)[1].lower()
                        file_types[file_ext] = file_types.get(file_ext, 0) + 1
                    except Exception as e:
                        error_msg = f"Error processing file {file.name}: {str(e)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        st.error(
                            f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                        )
                        continue

                st.info(
                    f"📊 **{len(uploaded_files)} files selected** ({format_file_size(total_size)})"
                )

                for file_type, count in file_types.items():
                    emoji = get_file_type_emoji(file_type)
                    st.caption(f"{emoji} {count}x {file_type.upper()[1:]} files")

                if st.button("Process All Files", type="primary"):
                    success_count = 0
                    failed_files = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        if process_document_upload(uploaded_file):
                            success_count += 1
                        else:
                            failed_files.append(uploaded_file.name)
                    progress_bar.empty()
                    status_text.empty()
                    if success_count > 0:
                        st.success(
                            f"✅ Successfully processed {success_count}/{len(uploaded_files)} files"
                        )
                    if failed_files:
                        st.error(f"❌ Failed to process {len(failed_files)} files:")
                        for failed_file in failed_files:
                            st.caption(f"• {failed_file}")
                    if success_count > 0:
                        st.rerun()
            except Exception as e:
                error_msg = f"Error processing uploaded files: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(
                    f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                )

        st.divider()

        st.subheader("📄 Your Documents")
        try:
            user_docs = components["doc_ops"].get_user_documents(
                st.session_state.user_id
            )
            if user_docs:
                st.metric("Total Documents", len(user_docs))
                st.caption("**Recent uploads:**")
                for doc in user_docs[:10]:
                    try:
                        status = "✅" if doc["processed"] else "⏳"
                        file_ext = doc.get("file_type", "")
                        emoji = get_file_type_emoji(file_ext)
                        size_display = format_file_size(doc.get("file_size", 0))
                        st.caption(
                            f"{status} {emoji} {doc['document_name']} ({size_display})"
                        )
                    except Exception as e:
                        error_msg = f"Error displaying document {doc.get('document_name', 'Unknown')}: {str(e)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        st.caption(f"❌ Error displaying document: {str(e)}")
            else:
                st.info("No documents uploaded yet")
                st.caption("💡 Upload documents to get started!")
        except Exception as e:
            error_msg = f"Error loading user documents: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(
                f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
            )

        st.divider()

        with st.expander("📈 System Stats:"):
            try:
                user_docs = components["doc_ops"].get_user_documents(
                    st.session_state.user_id
                )
                total_docs = len(user_docs) if user_docs else 0
                processed_docs = (
                    len([doc for doc in user_docs if doc.get("processed", False)])
                    if user_docs
                    else 0
                )
                # Get total query count from all users
                total_queries = len(
                    components["query_ops"].get_all_queries()
                )  # Assumes you add a get_query_count method
                st.metric("Total Documents", total_docs)
                st.metric("Processed Documents", processed_docs)
                st.metric("Total Queries (All Users)", total_queries)
            except Exception as e:
                error_msg = f"Error calculating system stats: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(
                    f"Error loading system statistics\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                )

    # ---- Main content area - Chat interface ----

    # Load history from DB on first run
    load_chat_history()

    # Display chat messages from history
    try:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant":
                    if "sources" in message and message["sources"]:
                        display_sources(message["sources"])
                    if (
                        "processing_time" in message
                        and message["processing_time"] is not None
                    ):
                        tokens_info = ""
                        if "tokens_used" in message and message["tokens_used"]:
                            tokens_info = (
                                f" | 🔢 Tokens used: {message['tokens_used']:,}"
                            )
                        st.caption(
                            f"🧠 Model: {model} | ⏱️ Response time: {message['processing_time']:.2f} seconds {tokens_info} | 📚 Sources: {len(message["sources"]) if message["sources"] else 0}"
                        )
    except Exception as e:
        error_msg = f"Error displaying chat messages: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```")

    # Accept user input
    if query := st.chat_input("Ask anything about your uploaded documents..."):
        # Add user message to chat history and display
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        try:
            user_docs = components["doc_ops"].get_user_documents(
                st.session_state.user_id
            )
        except Exception as e:
            error_msg = f"Error checking user documents: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(
                f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
            )
            user_docs = []

        if not user_docs:
            st.warning("⚠️ Please upload some documents first!")
        else:
            with st.chat_message("assistant"):
                # Create a placeholder for dynamic status updates
                status_placeholder = st.empty()
                progress_placeholder = st.empty()

                try:
                    start_time = time.time()

                    # Phase 1: Initial retrieval
                    with status_placeholder.container():
                        st.info(
                            "🔍 **Phase 1/2:** Searching knowledge base for relevant information..."
                        )

                    # Simulate progress for retrieval phase (adjust timing based on your actual process)
                    progress_bar = progress_placeholder.progress(
                        0, text="Initializing search..."
                    )
                    progress_bar.progress(
                        25, text="Analyzing query and extracting keywords..."
                    )
                    progress_bar.progress(
                        50,
                        text="Searching through document chunks and generating response...",
                    )

                    # Actual retrieval call
                    answer, sources, processing_time, total_tokens = components[
                        "retriever"
                    ].retrieve_and_generate(query, st.session_state.user_id)

                    progress_bar.progress(
                        100, text="✅ Search and generation complete!"
                    )

                    # Phase 2: Processing sources
                    with status_placeholder.container():
                        st.info(
                            "📋 **Phase 2/2:** Processing and validating source information..."
                        )

                    progress_bar.progress(0, text="Extracting source metadata...")

                    source_info = []
                    if sources:
                        try:
                            total_sources = len(sources)
                            for i, source in enumerate(sources):
                                progress_percentage = int((i + 1) / total_sources * 100)
                                progress_bar.progress(
                                    progress_percentage,
                                    text=f"Processing source {i + 1} of {total_sources}...",
                                )

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
                                        "chunk_text": source.get("chunk_text"),
                                    }
                                )

                            progress_bar.progress(
                                100,
                                text=f"✅ Processed {total_sources} sources successfully!",
                            )

                        except Exception as e:
                            error_msg = f"Error processing sources: {str(e)}"
                            logger.error(error_msg)
                            logger.error(traceback.format_exc())

                            with status_placeholder.container():
                                st.error(f"❌ **Error in Phase 2:** {error_msg}")
                                with st.expander("📋 View Error Details"):
                                    st.code(traceback.format_exc(), language="python")

                            source_info = []
                            progress_bar.progress(
                                100, text="⚠️ Source processing completed with errors"
                            )
                    else:
                        progress_bar.progress(
                            100, text="ℹ️ No sources found for this query"
                        )

                    # Clear status indicators
                    status_placeholder.empty()
                    progress_placeholder.empty()

                    # Display and store assistant response
                    st.write(answer)
                    if sources:
                        display_sources(sources)
                    total_elapsed = time.time() - start_time

                    st.caption(
                        f"🧠 Model: {model} | ⏱️ Response time: {total_elapsed:.2f} seconds | 🔢 Tokens used: {total_tokens:,} | 📚 Sources: {len(sources) if sources else 0}"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "processing_time": processing_time,
                            "tokens_used": total_tokens,
                        }
                    )

                    try:
                        components["query_ops"].insert_query(
                            user_query=query,
                            answer_text=answer,
                            answer_sources=source_info,
                            user_id=st.session_state.user_id,
                            processing_time=total_elapsed,
                            chunks_used=len(sources) if sources else 0,
                            tokens_used=total_tokens,
                        )

                    except Exception as e:
                        error_msg = f"Error saving query to database: {str(e)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())

                        with status_placeholder.container():
                            st.warning(f"⚠️ Answer generated but database save failed")
                            with st.expander("📋 View Database Error Details"):
                                st.code(
                                    f"{error_msg}\n\n{traceback.format_exc()}",
                                    language="python",
                                )
                    recalculate_tokens()
                    st.rerun()

                except Exception as e:
                    # Handle any unexpected errors
                    status_placeholder.empty()
                    progress_placeholder.empty()

                    st.error(f"❌ **Unexpected Error:** {str(e)}")
                    with st.expander("📋 View Full Error Details"):
                        st.code(traceback.format_exc(), language="python")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"Fatal error in main application: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```")
        st.stop()
