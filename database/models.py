import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
import json


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Documents table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                document_text TEXT NOT NULL,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                file_type TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        """
        )

        # Document chunks table (for parent-child relationships)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                parent_chunk_id INTEGER,
                chunk_text TEXT NOT NULL,
                contextual_header TEXT,
                chunk_type TEXT CHECK(chunk_type IN ('parent', 'child')),
                embedding BLOB,
                chunk_index INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents (document_id),
                FOREIGN KEY (parent_chunk_id) REFERENCES document_chunks (chunk_id)
            )
        """
        )

        # User queries table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_queries (
                query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_query TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                answer_sources_used TEXT, -- JSON string of source references
                user_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processing_time REAL,
                chunks_used INTEGER
            )
        """
        )

        # Create indexes for better performance
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

        conn.commit()
        conn.close()

    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
