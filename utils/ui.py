# utils/ui.py

import os
import streamlit as st
import logging
import traceback
from utils.file_handler import process_document_upload
from utils.streamlit_utils import (
    float_to_percent,
)  # Assuming this exists as in the original code

logger = logging.getLogger(__name__)


def get_file_type_emoji(file_type):
    """Get emoji for a given file type."""
    # (Same code as in your original app.py)
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
    """Format file size into a human-readable string."""
    # (Same code as in your original app.py)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/(1024*1024):.1f} MB"


def display_sources(sources):
    """Displays retrieved sources in an expander."""
    # (Same code as in your original app.py)
    if not sources:
        return
    with st.expander(f"📚 Sources Used ({len(sources)} Chunks)", expanded=False):
        for i, source in enumerate(sources, 1):
            try:
                with st.expander(f"Chunk {i}"):
                    header = source.get("contextual_header", "No Header")
                    text = source.get("chunk_text", "Content not available.")
                    display_text = text[:300] + "..." if len(text) > 300 else text
                    st.write(f"📄 {header}")
                    st.markdown(f"> {display_text}")
                    meta_parts = []
                    if score := source.get("relevance_score"):
                        meta_parts.append(
                            f"Similarity score: {float_to_percent(score)}"
                        )
                    if chunk_id := source.get("chunk_id"):
                        meta_parts.append(f"Chunk ID: `{chunk_id}`")
                    if meta_parts:
                        st.caption(" | ".join(meta_parts))
            except Exception as e:
                st.error(f"Error displaying source {i}: {str(e)}")


def display_sidebar(components, settings):
    """Renders the entire sidebar for document management and stats."""
    with st.sidebar:
        st.header("🔍 Enhanced RAG System")
        st.write("*Advanced Retrieval-Augmented Generation*")

        # Token Usage
        daily_tokens = components["query_ops"].get_todays_total_tokens()
        token_limit = 500000
        progress = min(daily_tokens / token_limit, 1.0)
        st.write("**Daily Token Usage**")
        st.progress(progress)
        st.caption(
            f"{daily_tokens:,} / {token_limit:,} tokens used ({progress*100:.1f}%)"
        )
        if progress > 0.8:
            st.warning("⚠️ Approaching daily limit!")

        st.divider()

        # File Uploader
        st.header("📁 Document Management")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help=f"Max file size: {settings.MAX_UPLOAD_SIZE}MB",
        )

        if uploaded_files:
            if st.button("Process All Files", type="primary"):
                progress_bar = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    process_document_upload(
                        uploaded_file, components, settings, st.session_state.user_id
                    )
                    progress_bar.progress((i + 1) / len(uploaded_files))
                st.rerun()

        st.divider()

        # Document List
        st.subheader("📄 Your Documents")
        try:
            user_docs = components["doc_ops"].get_user_documents(
                st.session_state.user_id
            )
            if user_docs:
                st.metric("Total Documents", len(user_docs))
                for doc in user_docs[:10]:
                    status = "✅" if doc["processed"] else "⏳"
                    emoji = get_file_type_emoji(doc.get("file_type", ""))
                    size = format_file_size(doc.get("file_size", 0))
                    st.caption(f"{status} {emoji} {doc['document_name']} ({size})")
            else:
                st.info("No documents uploaded yet.")
        except Exception as e:
            st.error(f"Error loading documents: {e}")


def display_chat_messages(model_name):
    """Displays the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                display_sources(message.get("sources", []))
                time_info = f"{message.get('processing_time', 0):.2f}"
                tokens_info = f"{message.get('tokens_used', 0):,}"
                st.caption(
                    f"🧠 Model: {model_name} | ⏱️ Response time: {time_info}s | 🔢 Tokens: {tokens_info}"
                )
