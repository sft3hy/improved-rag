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

    def delete_query(self, query_id: int):
        """Delete a query by ID."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_queries WHERE query_id = ?", (query_id,))

        conn.commit()
        conn.close()
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
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Convert sources to JSON string
        sources_json = json.dumps(answer_sources)

        cursor.execute(
            """
            INSERT INTO user_queries 
            (user_query, answer_text, answer_sources_used, user_id, processing_time, chunks_used, tokens_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
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
                   timestamp, processing_time, chunks_used, tokens_used
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
                    "tokens_used": row[7],
                }
            )

        conn.close()
        return queries

    def get_all_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all recent queries from all users."""
        conn = self.db_manager.get_connection()
        # Use row factory to easily convert to dictionary
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT query_id, user_query, answer_text, answer_sources_used,
                timestamp, processing_time, chunks_used, user_id, tokens_used
            FROM user_queries
            ORDER BY timestamp ASC
            LIMIT ?
        """,
            (limit,),
        )

        queries = []
        for row in cursor.fetchall():
            query_dict = dict(row)
            # Parse sources JSON
            if query_dict.get("answer_sources_used"):
                query_dict["answer_sources"] = json.loads(
                    query_dict["answer_sources_used"]
                )
            else:
                query_dict["answer_sources"] = []

            # For clarity, rename the key
            query_dict["content"] = query_dict.pop("answer_text")

            queries.append(query_dict)

        conn.close()
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

        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
        existing_user = cursor.fetchone()

        if existing_user:
            # Update last login
            cursor.execute(
                """
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP,
                    display_name = COALESCE(?, display_name),
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
                """,
                (display_name, user_id),
            )
        else:
            # Create new user
            cursor.execute(
                """
                INSERT INTO users (user_id, email, display_name, first_login, last_login)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (user_id, email, display_name),
            )

        conn.commit()
        conn.close()
        return user_id

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user information.

        Args:
            user_id: User's ID

        Returns:
            Dictionary with user information or None if not found
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT user_id, email, display_name, first_login, last_login,
                   total_queries, total_documents, is_active, created_at, updated_at
            FROM users 
            WHERE user_id = ?
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()

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
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE users 
            SET total_queries = total_queries + ?,
                total_documents = total_documents + ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """,
            (increment_queries, increment_documents, user_id),
        )

        conn.commit()
        conn.close()

    def get_user_activity_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get detailed activity statistics for a user.

        Args:
            user_id: User's ID

        Returns:
            Dictionary with activity statistics
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Get document count
        cursor.execute("SELECT COUNT(*) FROM documents WHERE user_id = ?", (user_id,))
        doc_count = cursor.fetchone()[0]

        # Get query count
        cursor.execute(
            "SELECT COUNT(*) FROM user_queries WHERE user_id = ?", (user_id,)
        )
        query_count = cursor.fetchone()[0]

        # Get today's token usage
        today_tokens = self.db_manager.get_todays_total_tokens()

        # Get total token usage
        cursor.execute(
            "SELECT COALESCE(SUM(tokens_used), 0) FROM user_queries WHERE user_id = ?",
            (user_id,),
        )
        total_tokens = cursor.fetchone()[0]

        # Get recent activity (last 7 days)
        cursor.execute(
            """
            SELECT DATE(timestamp, 'localtime') as query_date, COUNT(*) as query_count
            FROM user_queries 
            WHERE user_id = ? AND timestamp >= DATE('now', '-7 days', 'localtime')
            GROUP BY DATE(timestamp, 'localtime')
            ORDER BY query_date DESC
            """,
            (user_id,),
        )
        recent_activity = cursor.fetchall()

        conn.close()

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
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

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

        conn.close()
        return users


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

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

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

        conn.close()
        return stats
