# utils/file_handler.py

import logging
import traceback
import streamlit as st

logger = logging.getLogger(__name__)


def process_document_upload(uploaded_file, components, settings, user_id):
    """Process a single uploaded document and store it in the database."""
    try:
        # Process file content
        result = components["document_processor"].process_uploaded_file(
            uploaded_file, settings.MAX_UPLOAD_SIZE
        )
        if not result["success"]:
            st.error(result["error_message"])
            return False

        # Store document metadata in database
        document_id = components["doc_ops"].insert_document(
            document_name=result["filename"],
            user_id=user_id,
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

        # Mark document as fully processed
        components["doc_ops"].mark_document_processed(document_id)

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
