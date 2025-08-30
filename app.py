# app.py

import streamlit as st
import logging

# Import project modules
from config.settings import settings
from utils.system_init import initialize_system
from utils.ui import display_sidebar, display_chat_messages
from utils.chat_handler import load_chat_history, handle_user_query

# --- Page Configuration and Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

st.set_page_config(
    page_title="Enhanced RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model and Powering Info ---
if settings.TEST == "True":
    powered_by = "Powered by Groq and Llama 4 Scout"
    model_name = "Llama 4 Scout"
else:
    powered_by = "Powered by Sanctuary and Claude 3.5 Sonnet"
    model_name = "Claude 3.5 Sonnet"

# --- Session State Initialization ---
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- System Initialization ---
if not st.session_state.initialized:
    with st.spinner("Initializing Enhanced RAG System..."):
        st.session_state.components = initialize_system()
        st.session_state.initialized = True
components = st.session_state.components


def main():
    """Main application flow."""
    st.title("Enhanced RAG System")
    st.caption(powered_by)

    # --- Sidebar ---
    display_sidebar(components, settings)

    # --- Main Chat Area ---
    load_chat_history(components["query_ops"])
    display_chat_messages(model_name)

    # --- User Input ---
    if query := st.chat_input("Ask anything about your uploaded documents..."):
        user_docs = components["doc_ops"].get_user_documents(st.session_state.user_id)
        if not user_docs:
            st.warning("⚠️ Please upload some documents first!")
        else:
            handle_user_query(query, components, model_name)


if __name__ == "__main__":
    main()
