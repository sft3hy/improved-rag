import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .models import DatabaseManager


class DocumentOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def insert_document(
        self,
        document_name: str,
        user_id: str,
        document_text: str,
        file_size: int,
        file_type: str,
    ) -> int:
        """Insert a new document and return its ID."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO documents (document_name, user_id, document_text, file_size, file_type)
            VALUES (?, ?, ?, ?, ?)
        """,
            (document_name, user_id, document_text, file_size, file_type),
        )

        document_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return document_id

    def insert_chunk(
        self,
        document_id: int,
        chunk_text: str,
        contextual_header: str,
        chunk_type: str,
        embedding: np.ndarray,
        chunk_index: int,
        parent_chunk_id: Optional[int] = None,
    ) -> int:
        """Insert a document chunk with its embedding."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Convert numpy array to bytes for storage
        embedding_bytes = embedding.tobytes()

        cursor.execute(
            """
            INSERT INTO document_chunks 
            (document_id, parent_chunk_id, chunk_text, contextual_header, 
             chunk_type, embedding, chunk_index)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                document_id,
                parent_chunk_id,
                chunk_text,
                contextual_header,
                chunk_type,
                embedding_bytes,
                chunk_index,
            ),
        )

        chunk_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return chunk_id

    def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT document_id, document_name, upload_timestamp, file_size, file_type, processed
            FROM documents WHERE user_id = ?
            ORDER BY upload_timestamp DESC
        """,
            (user_id,),
        )

        documents = []
        for row in cursor.fetchall():
            documents.append(
                {
                    "document_id": row[0],
                    "document_name": row[1],
                    "upload_timestamp": row[2],
                    "file_size": row[3],
                    "file_type": row[4],
                    "processed": row[5],
                }
            )

        conn.close()
        return documents

    def mark_document_processed(self, document_id: int):
        """Mark a document as processed."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE documents SET processed = TRUE WHERE document_id = ?",
            (document_id,),
        )

        conn.commit()
        conn.close()

    def get_child_chunks_with_embeddings(
        self, user_id: str
    ) -> List[Tuple[int, str, str, np.ndarray]]:
        """Get all child chunks with embeddings for a user."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT dc.chunk_id, dc.chunk_text, dc.contextual_header, dc.embedding
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.document_id
            WHERE d.user_id = ? AND dc.chunk_type = 'child'
        """,
            (user_id,),
        )

        chunks = []
        for row in cursor.fetchall():
            # Convert bytes back to numpy array
            embedding = np.frombuffer(row[3], dtype=np.float32).reshape(-1)
            chunks.append((row[0], row[1], row[2], embedding))

        conn.close()
        return chunks

    def get_parent_chunk_by_child_id(
        self, child_chunk_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get parent chunk for a given child chunk."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT parent.chunk_id, parent.chunk_text, parent.contextual_header
            FROM document_chunks child
            JOIN document_chunks parent ON child.parent_chunk_id = parent.chunk_id
            WHERE child.chunk_id = ?
        """,
            (child_chunk_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "chunk_id": row[0],
                "chunk_text": row[1],
                "contextual_header": row[2],
            }
        return None


class QueryOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def insert_query(
        self,
        user_query: str,
        answer_text: str,
        answer_sources: List[Dict],
        user_id: str,
        processing_time: float,
        chunks_used: int,
    ) -> int:
        """Insert a user query and its answer."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Convert sources to JSON string
        sources_json = json.dumps(answer_sources)

        cursor.execute(
            """
            INSERT INTO user_queries 
            (user_query, answer_text, answer_sources_used, user_id, processing_time, chunks_used)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                user_query,
                answer_text,
                sources_json,
                user_id,
                processing_time,
                chunks_used,
            ),
        )

        query_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return query_id

    def get_user_queries(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent queries for a user."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT query_id, user_query, answer_text, answer_sources_used, 
                   timestamp, processing_time, chunks_used
            FROM user_queries 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (user_id, limit),
        )

        queries = []
        for row in cursor.fetchall():
            # Parse sources JSON
            sources = json.loads(row[3]) if row[3] else []

            queries.append(
                {
                    "query_id": row[0],
                    "user_query": row[1],
                    "answer_text": row[2],
                    "answer_sources": sources,
                    "timestamp": row[4],
                    "processing_time": row[5],
                    "chunks_used": row[6],
                }
            )

        conn.close()
        return queries


class AdminOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def nuke_database(self):
        """
        Deletes all data from documents, document_chunks, and user_queries tables.
        This will leave the tables intact but empty, ready for deployment.
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        try:
            print("Attempting to delete all queries, documents, and chunks...")

            # The order of deletion is important due to foreign key constraints.
            # (document_chunks has a foreign key to documents)
            cursor.execute("DELETE FROM document_chunks;")
            cursor.execute("DELETE FROM documents;")
            cursor.execute("DELETE FROM user_queries;")

            # Reset the auto-increment counters for the primary keys so new
            # entries start from 1. This is good for a clean deployment state.
            print("Resetting table primary key sequences...")
            cursor.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('documents', 'document_chunks', 'user_queries');"
            )

            conn.commit()
            print("Database successfully nuked. All specified data has been deleted.")

        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            conn.rollback()
        finally:
            conn.close()
