import psycopg
from psycopg.rows import dict_row
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
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO documents (document_name, user_id, document_text, file_size, file_type)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING document_id
                """,
                    (document_name, user_id, document_text, file_size, file_type),
                )

                document_id = cursor.fetchone()[0]
                conn.commit()
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
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Convert numpy array to bytes for storage
                embedding_bytes = embedding.tobytes()

                cursor.execute(
                    """
                    INSERT INTO document_chunks 
                    (document_id, parent_chunk_id, chunk_text, contextual_header, 
                     chunk_type, embedding, chunk_index)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING chunk_id
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

                chunk_id = cursor.fetchone()[0]
                conn.commit()
                return chunk_id

    def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT document_id, document_name, upload_timestamp, file_size, file_type, processed
                    FROM documents WHERE user_id = %s
                    ORDER BY upload_timestamp DESC
                """,
                    (user_id,),
                )

                documents = cursor.fetchall()
                return [dict(row) for row in documents]

    def mark_document_processed(self, document_id: int):
        """Mark a document as processed."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE documents SET processed = TRUE WHERE document_id = %s",
                    (document_id,),
                )
                conn.commit()

    def get_child_chunks_with_embeddings(
        self, user_id: str
    ) -> List[Tuple[int, str, str, np.ndarray]]:
        """Get all child chunks with embeddings for a user."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT dc.chunk_id, dc.chunk_text, dc.contextual_header, dc.embedding
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.document_id
                    WHERE d.user_id = %s AND dc.chunk_type = 'child'
                """,
                    (user_id,),
                )

                chunks = []
                for row in cursor.fetchall():
                    # Convert bytes back to numpy array
                    embedding = np.frombuffer(row[3], dtype=np.float32).reshape(-1)
                    chunks.append((row[0], row[1], row[2], embedding))

                return chunks

    def get_parent_chunk_by_child_id(
        self, child_chunk_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get parent chunk for a given child chunk."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT parent.chunk_id, parent.chunk_text, parent.contextual_header
                    FROM document_chunks child
                    JOIN document_chunks parent ON child.parent_chunk_id = parent.chunk_id
                    WHERE child.chunk_id = %s
                """,
                    (child_chunk_id,),
                )

                row = cursor.fetchone()
                return dict(row) if row else None


class QueryOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def delete_query(self, query_id: int):
        """Delete a query by ID."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM user_queries WHERE query_id = %s", (query_id,)
                )
                conn.commit()
                print(f"Deleted query with id {query_id}")

    def insert_query(
        self,
        user_query: str,
        answer_text: str,
        answer_sources: List[Dict],
        user_id: str,
        processing_time: float,
        chunks_used: int,
        tokens_used: int = 0,
    ) -> int:
        """Insert a user query and its answer."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Convert sources to JSON string
                sources_json = json.dumps(answer_sources)

                cursor.execute(
                    """
                    INSERT INTO user_queries 
                    (user_query, answer_text, answer_sources_used, user_id, processing_time, chunks_used, tokens_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING query_id
                """,
                    (
                        user_query,
                        answer_text,
                        sources_json,
                        user_id,
                        processing_time,
                        chunks_used,
                        tokens_used,
                    ),
                )

                query_id = cursor.fetchone()[0]
                conn.commit()
                return query_id

    def get_user_queries(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent queries for a user."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT query_id, user_query, answer_text, answer_sources_used, 
                           timestamp, processing_time, chunks_used, tokens_used
                    FROM user_queries 
                    WHERE user_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """,
                    (user_id, limit),
                )

                queries = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    # Parse sources JSON
                    if row_dict["answer_sources_used"]:
                        try:
                            sources = json.loads(row_dict["answer_sources_used"])
                        except (json.JSONDecodeError, TypeError):
                            sources = []
                    else:
                        sources = []
                    row_dict["answer_sources"] = sources
                    queries.append(row_dict)

                return queries

    def get_all_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all recent queries from all users."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT query_id, user_query, answer_text, answer_sources_used,
                        timestamp, processing_time, chunks_used, user_id, tokens_used
                    FROM user_queries
                    ORDER BY timestamp ASC
                    LIMIT %s
                """,
                    (limit,),
                )

                queries = []
                for row in cursor.fetchall():
                    query_dict = dict(row)
                    # Parse sources JSON
                    if query_dict.get("answer_sources_used"):
                        try:
                            query_dict["answer_sources"] = json.loads(
                                query_dict["answer_sources_used"]
                            )
                        except (json.JSONDecodeError, TypeError):
                            query_dict["answer_sources"] = []
                    else:
                        query_dict["answer_sources"] = []

                    # For clarity, rename the key
                    query_dict["content"] = query_dict.pop("answer_text")

                    queries.append(query_dict)

                return queries

    def get_todays_total_tokens(self, user_id: Optional[str] = None) -> int:
        """
        Get the total number of tokens used today.

        Args:
            user_id: If provided, get tokens for specific user. If None, get total for all users.

        Returns:
            Total tokens used today
        """
        return self.db_manager.get_todays_total_tokens(user_id)


class UserOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def create_or_update_user(
        self, email: str, display_name: Optional[str] = None
    ) -> str:
        """
        Create a new user or update existing user's last login.

        Args:
            email: User's email address
            display_name: User's display name (optional)

        Returns:
            user_id: The user's ID (same as email in this implementation)
        """
        user_id = email  # Using email as user_id for simplicity

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Try to insert new user, or update last_login if exists
                cursor.execute(
                    """
                    INSERT INTO users (user_id, email, display_name, first_login, last_login)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id) 
                    DO UPDATE SET 
                        last_login = CURRENT_TIMESTAMP,
                        display_name = COALESCE(EXCLUDED.display_name, users.display_name),
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (user_id, email, display_name),
                )
                conn.commit()

        return user_id

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user information.

        Args:
            user_id: User's ID

        Returns:
            Dictionary with user information or None if not found
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, email, display_name, first_login, last_login,
                           total_queries, total_documents, is_active, created_at, updated_at
                    FROM users 
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                row = cursor.fetchone()

                if row:
                    return {
                        "user_id": row[0],
                        "email": row[1],
                        "display_name": row[2],
                        "first_login": row[3],
                        "last_login": row[4],
                        "total_queries": row[5],
                        "total_documents": row[6],
                        "is_active": row[7],
                        "created_at": row[8],
                        "updated_at": row[9],
                    }
                return None

    def update_user_stats(
        self, user_id: str, increment_queries: int = 0, increment_documents: int = 0
    ):
        """
        Update user statistics.

        Args:
            user_id: User's ID
            increment_queries: Number to add to total_queries
            increment_documents: Number to add to total_documents
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE users 
                    SET total_queries = total_queries + %s,
                        total_documents = total_documents + %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                    """,
                    (increment_queries, increment_documents, user_id),
                )
                conn.commit()

    def get_user_activity_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get detailed activity statistics for a user.

        Args:
            user_id: User's ID

        Returns:
            Dictionary with activity statistics
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get document count
                cursor.execute(
                    "SELECT COUNT(*) FROM documents WHERE user_id = %s", (user_id,)
                )
                doc_count = cursor.fetchone()[0]

                # Get query count
                cursor.execute(
                    "SELECT COUNT(*) FROM user_queries WHERE user_id = %s", (user_id,)
                )
                query_count = cursor.fetchone()[0]

                # Get today's token usage
                today_tokens = self.get_todays_total_tokens(user_id)

                # Get total token usage
                cursor.execute(
                    "SELECT COALESCE(SUM(tokens_used), 0) FROM user_queries WHERE user_id = %s",
                    (user_id,),
                )
                total_tokens = cursor.fetchone()[0]

                # Get recent activity (last 7 days)
                cursor.execute(
                    """
                    SELECT DATE(timestamp) as query_date, COUNT(*) as query_count
                    FROM user_queries 
                    WHERE user_id = %s AND timestamp >= CURRENT_DATE - INTERVAL '7 days'
                    GROUP BY DATE(timestamp)
                    ORDER BY query_date DESC
                    """,
                    (user_id,),
                )
                recent_activity = cursor.fetchall()

                return {
                    "document_count": doc_count,
                    "total_queries": query_count,
                    "today_tokens": today_tokens,
                    "total_tokens": total_tokens,
                    "recent_activity": recent_activity,
                }

    def get_all_users_summary(self) -> list:
        """
        Get a summary of all users for admin purposes.

        Returns:
            List of dictionaries with user summaries
        """
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT u.user_id, u.email, u.display_name, u.first_login, u.last_login,
                           u.total_queries, u.total_documents, u.is_active,
                           COALESCE(SUM(uq.tokens_used), 0) as total_tokens
                    FROM users u
                    LEFT JOIN user_queries uq ON u.user_id = uq.user_id
                    GROUP BY u.user_id, u.email, u.display_name, u.first_login, u.last_login,
                             u.total_queries, u.total_documents, u.is_active
                    ORDER BY u.last_login DESC
                    """
                )

                users = []
                for row in cursor.fetchall():
                    users.append(
                        {
                            "user_id": row[0],
                            "email": row[1],
                            "display_name": row[2],
                            "first_login": row[3],
                            "last_login": row[4],
                            "total_queries": row[5],
                            "total_documents": row[6],
                            "is_active": row[7],
                            "total_tokens": row[8],
                        }
                    )

                return users


class AdminOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def nuke_database(self):
        """
        Deletes all data from documents, document_chunks, and user_queries tables.
        This will leave the tables intact but empty, ready for deployment.
        """
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
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
                        "ALTER SEQUENCE documents_document_id_seq RESTART WITH 1;"
                    )
                    cursor.execute(
                        "ALTER SEQUENCE document_chunks_chunk_id_seq RESTART WITH 1;"
                    )
                    cursor.execute(
                        "ALTER SEQUENCE user_queries_query_id_seq RESTART WITH 1;"
                    )

                    conn.commit()
                    print(
                        "Database successfully nuked. All specified data has been deleted."
                    )

                except psycopg.Error as e:
                    print(f"An error occurred: {e}")
                    conn.rollback()
                    raise

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                stats = {}

                # Count documents
                cursor.execute("SELECT COUNT(*) FROM documents")
                stats["total_documents"] = cursor.fetchone()[0]

                # Count chunks
                cursor.execute("SELECT COUNT(*) FROM document_chunks")
                stats["total_chunks"] = cursor.fetchone()[0]

                # Count queries
                cursor.execute("SELECT COUNT(*) FROM user_queries")
                stats["total_queries"] = cursor.fetchone()[0]

                # Total tokens used
                cursor.execute("SELECT COALESCE(SUM(tokens_used), 0) FROM user_queries")
                stats["total_tokens_used"] = cursor.fetchone()[0]

                # Tokens used today
                stats["tokens_used_today"] = self.db_manager.get_todays_total_tokens()

                return stats
