import psycopg
from typing import Optional
import os


class DatabaseManager:
    def __init__(
        self,
        host: str = os.getenv("PGSQL_HOST_IP"),
        port: int = os.getenv("PGSQL_HOST_PORT"),
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "dbname": "enhancedrag",
            "user": "slammy",
            "password": "cosmic",
        }
        self.init_database()

    def get_connection(self):
        """Get database connection."""
        return psycopg.connect(**self.connection_params)

    def init_database(self):
        """Initialize the database with required tables."""
        with psycopg.connect(**self.connection_params) as conn:
            with conn.cursor() as cursor:
                # Users table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        display_name TEXT,
                        first_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_queries INTEGER DEFAULT 0,
                        total_documents INTEGER DEFAULT 0,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Documents table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id SERIAL PRIMARY KEY,
                        document_name TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        document_text TEXT NOT NULL,
                        upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_size INTEGER,
                        file_type TEXT,
                        processed BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                    )
                """
                )

                # Document chunks table (for parent-child relationships)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id SERIAL PRIMARY KEY,
                        document_id INTEGER,
                        parent_chunk_id INTEGER,
                        chunk_text TEXT NOT NULL,
                        contextual_header TEXT,
                        chunk_type TEXT CHECK(chunk_type IN ('parent', 'child')),
                        embedding BYTEA,
                        chunk_index INTEGER,
                        FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE,
                        FOREIGN KEY (parent_chunk_id) REFERENCES document_chunks (chunk_id) ON DELETE CASCADE
                    )
                """
                )

                # User queries table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_queries (
                        query_id SERIAL PRIMARY KEY,
                        user_query TEXT NOT NULL,
                        answer_text TEXT NOT NULL,
                        answer_sources_used TEXT, -- JSON string of source references
                        user_id TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processing_time REAL,
                        chunks_used INTEGER,
                        tokens_used INTEGER DEFAULT 0,
                        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                    )
                """
                )

                # Check if tokens_used column exists and add it if it doesn't
                cursor.execute(
                    """
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='user_queries' AND column_name='tokens_used'
                    """
                )
                if not cursor.fetchone():
                    cursor.execute(
                        "ALTER TABLE user_queries ADD COLUMN tokens_used INTEGER DEFAULT 0"
                    )

                # Check if foreign key constraints exist and add them if they don't
                # Note: This is a simplified check - in production you might want more robust migration handling
                try:
                    cursor.execute(
                        """
                        SELECT constraint_name FROM information_schema.table_constraints 
                        WHERE table_name='documents' AND constraint_type='FOREIGN KEY'
                        """
                    )
                    fk_constraints = cursor.fetchall()
                    if not any(
                        "user" in str(constraint) for constraint in fk_constraints
                    ):
                        cursor.execute(
                            "ALTER TABLE documents ADD CONSTRAINT fk_documents_user FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE"
                        )
                except Exception as e:
                    # Foreign key might already exist or users table might not exist yet
                    print(f"Warning: Could not add foreign key constraint: {e}")
                    # Continue without failing - this is expected on first run

                # Create indexes for better performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON document_chunks(parent_chunk_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_queries_user_id ON user_queries(user_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON user_queries(timestamp)"
                )

                # No manual commit needed - context manager handles it

    def get_todays_total_tokens(self, user_id: Optional[str] = None) -> int:
        """
        Get the total number of tokens used today (local time).

        Args:
            user_id: If provided, get tokens for specific user. If None, get total for all users.

        Returns:
            Total tokens used today
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                if user_id:
                    cursor.execute(
                        """
                        SELECT COALESCE(SUM(tokens_used), 0)
                        FROM user_queries 
                        WHERE user_id = %s 
                        AND DATE(timestamp) = CURRENT_DATE
                        """,
                        (user_id,),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT COALESCE(SUM(tokens_used), 0)
                        FROM user_queries 
                        WHERE DATE(timestamp) = CURRENT_DATE
                        """
                    )

                total_tokens = cursor.fetchone()[0]
                return total_tokens
