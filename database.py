# database.py

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = 'rag_database.db'

@contextmanager
def get_db_connection():
    """Context manager for database connections with proper error handling"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        yield conn
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def init_db():
    """Initialize the database and create tables if they don't exist."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create table for documents with additional metadata
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    file_size INTEGER,
                    upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_modified DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create table for user queries with more details
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    response_time_ms INTEGER,
                    num_results INTEGER
                )
            ''')
            
            # Create table for document chunks (for tracking processed documents)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    chunk_index INTEGER,
                    chunk_content TEXT,
                    chunk_metadata TEXT,
                    created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id)')
            
            conn.commit()
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

def add_document(filename, content):
    """Add a new document to the database or update existing one."""
    try:
        file_size = len(content.encode('utf-8'))
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if document already exists
            cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing document
                cursor.execute("""
                    UPDATE documents 
                    SET content = ?, file_size = ?, last_modified = CURRENT_TIMESTAMP 
                    WHERE filename = ?
                """, (content, file_size, filename))
                logger.info(f"Updated existing document: {filename}")
            else:
                # Insert new document
                cursor.execute("""
                    INSERT INTO documents (filename, content, file_size) 
                    VALUES (?, ?, ?)
                """, (filename, content, file_size))
                logger.info(f"Added new document: {filename}")
            
            conn.commit()
            
    except Exception as e:
        logger.error(f"Failed to add document {filename}: {str(e)}")
        raise

def get_all_documents():
    """Retrieve all documents from the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, content FROM documents ORDER BY upload_timestamp DESC")
            documents = cursor.fetchall()
            logger.info(f"Retrieved {len(documents)} documents from database")
            return documents
            
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {str(e)}")
        raise

def get_document_info():
    """Get information about all documents without content."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT filename, file_size, upload_timestamp, last_modified 
                FROM documents 
                ORDER BY upload_timestamp DESC
            """)
            return cursor.fetchall()
            
    except Exception as e:
        logger.error(f"Failed to retrieve document info: {str(e)}")
        raise

def delete_document(filename):
    """Delete a document from the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Deleted document: {filename}")
                return True
            else:
                logger.warning(f"Document not found: {filename}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to delete document {filename}: {str(e)}")
        raise

def add_query(query_text, response_time_ms=None, num_results=None):
    """Add a user query to the database with optional performance metrics."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO queries (query_text, response_time_ms, num_results) 
                VALUES (?, ?, ?)
            """, (query_text, response_time_ms, num_results))
            conn.commit()
            logger.info(f"Added query to database: {query_text[:50]}...")
            
    except Exception as e:
        logger.error(f"Failed to add query: {str(e)}")
        raise

def get_recent_queries(limit=10):
    """Retrieve recent queries from the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query_text, timestamp, response_time_ms, num_results 
                FROM queries 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            return cursor.fetchall()
            
    except Exception as e:
        logger.error(f"Failed to retrieve recent queries: {str(e)}")
        raise

def get_database_stats():
    """Get statistics about the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            # Count queries
            cursor.execute("SELECT COUNT(*) FROM queries")
            query_count = cursor.fetchone()[0]
            
            # Get total content size
            cursor.execute("SELECT SUM(file_size) FROM documents")
            total_size = cursor.fetchone()[0] or 0
            
            # Get database file size
            db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
            
            return {
                "document_count": doc_count,
                "query_count": query_count,
                "total_content_size": total_size,
                "database_file_size": db_size
            }
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {str(e)}")
        return {}

def clear_all_data():
    """Clear all data from the database (use with caution)."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_chunks")
            cursor.execute("DELETE FROM queries")
            cursor.execute("DELETE FROM documents")
            conn.commit()
            logger.info("All data cleared from database")
            
    except Exception as e:
        logger.error(f"Failed to clear database: {str(e)}")
        raise

def backup_database(backup_path=None):
    """Create a backup of the database."""
    try:
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"rag_database_backup_{timestamp}.db"
        
        with get_db_connection() as conn:
            backup_conn = sqlite3.connect(backup_path)
            conn.backup(backup_conn)
            backup_conn.close()
            
        logger.info(f"Database backed up to: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Failed to backup database: {str(e)}")
        raise