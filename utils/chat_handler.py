# utils/chat_handler.py

import json
import time
import logging
import traceback
import streamlit as st

logger = logging.getLogger(__name__)


def load_chat_history(query_ops):
    """Load all queries from the database and format them for display."""
    if st.session_state.get("messages_loaded", False):
        return
    try:
        st.session_state.messages = []
        all_queries = query_ops.get_all_queries(limit=100)
        for query_data in all_queries:
            st.session_state.messages.append(
                {"role": "user", "content": query_data["user_query"]}
            )

            raw_sources = query_data.get("answer_sources_used", "[]")
            sources = (
                json.loads(raw_sources) if isinstance(raw_sources, str) else raw_sources
            )

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
        st.error(f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```")


def handle_user_query(query, components, model_name):
    """Processes a user query, generates a response, and updates the state."""
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info("🔍 Searching knowledge base...")

        try:
            start_time = time.time()
            answer, sources, _, total_tokens = components[
                "retriever"
            ].retrieve_and_generate(query, st.session_state.user_id)
            total_elapsed = time.time() - start_time

            status_placeholder.empty()

            # Display response and save to DB
            st.write(answer)
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "processing_time": total_elapsed,
                "tokens_used": total_tokens,
            }
            st.session_state.messages.append(assistant_message)

            components["query_ops"].insert_query(
                user_query=query,
                answer_text=answer,
                answer_sources=sources,
                user_id=st.session_state.user_id,
                processing_time=total_elapsed,
                chunks_used=len(sources) if sources else 0,
                tokens_used=total_tokens,
            )
            st.rerun()

        except Exception as e:
            status_placeholder.empty()
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Query handling error: {traceback.format_exc()}")
