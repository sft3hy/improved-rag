# config.py

"""
Configuration settings for the Advanced RAG Pipeline
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.1:8b"
    temperature: float = 0.1
    top_p: float = 0.9

@dataclass
class ChunkingConfig:
    """Configuration for document chunking"""
    child_chunk_size: int = 200
    child_chunk_overlap: int = 20
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    max_context_header_length: int = 100
    
    # Separators for text splitting (in order of preference)
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]

@dataclass
class RetrievalConfig:
    """Configuration for document retrieval"""
    # Vector store settings
    initial_retrieval_k: int = 200
    first_filter_k: int = 150
    parent_docs_k: int = 20
    final_docs_k: int = 30
    
    # Relevance scoring
    min_relevance_score: float = 0.1
    use_llm_reranking: bool = True
    enable_multi_query: bool = True
    
    # Performance settings
    max_parallel_requests: int = 5
    request_timeout: int = 30

@dataclass
class DatabaseConfig:
    """Configuration for database"""
    db_path: str = "rag_database.db"
    backup_interval_hours: int = 24
    max_query_history: int = 100
    enable_performance_tracking: bool = True

@dataclass
class UIConfig:
    """Configuration for user interface"""
    page_title: str = "Advanced RAG Pipeline"
    page_icon: str = "🔍"
    default_retrieval_strategy: str = "advanced"
    show_debug_info: bool = False
    max_file_upload_mb: int = 200
    
    # Supported file extensions (will be filtered by available processors)
    supported_extensions: List[str] = None
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = [
                'txt', 'md', 'pdf', 'docx', 'pptx', 'xlsx', 'xls', 'csv'
            ]

@dataclass
class AppConfig:
    """Main application configuration"""
    model: ModelConfig = None
    chunking: ChunkingConfig = None
    retrieval: RetrievalConfig = None
    database: DatabaseConfig = None
    ui: UIConfig = None
    
    # Environment settings
    environment: str = "development"  # development, production
    log_level: str = "INFO"
    enable_telemetry: bool = False
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.ui is None:
            self.ui = UIConfig()

# Load configuration from environment variables or use defaults
def load_config_from_env() -> AppConfig:
    """Load configuration from environment variables"""
    config = AppConfig()
    
    # Model configuration
    config.model.embedding_model = os.getenv("EMBEDDING_MODEL", config.model.embedding_model)
    config.model.llm_model = os.getenv("LLM_MODEL", config.model.llm_model)
    config.model.temperature = float(os.getenv("LLM_TEMPERATURE", config.model.temperature))
    
    # Chunking configuration
    config.chunking.child_chunk_size = int(os.getenv("CHILD_CHUNK_SIZE", config.chunking.child_chunk_size))
    config.chunking.parent_chunk_size = int(os.getenv("PARENT_CHUNK_SIZE", config.chunking.parent_chunk_size))
    
    # Retrieval configuration
    config.retrieval.min_relevance_score = float(os.getenv("MIN_RELEVANCE_SCORE", config.retrieval.min_relevance_score))
    config.retrieval.final_docs_k = int(os.getenv("FINAL_DOCS_K", config.retrieval.final_docs_k))
    
    # Database configuration
    config.database.db_path = os.getenv("DATABASE_PATH", config.database.db_path)
    
    # Environment settings
    config.environment = os.getenv("ENVIRONMENT", config.environment)
    config.log_level = os.getenv("LOG_LEVEL", config.log_level)
    
    return config

# Global configuration instance
CONFIG = load_config_from_env()

# Retrieval strategy configurations
RETRIEVAL_STRATEGIES = {
    "advanced": {
        "name": "Advanced Pipeline",
        "description": "Multi-stage retrieval with reranking (recommended)",
        "icon": "🚀"
    },
    "parent": {
        "name": "Parent-Child",
        "description": "Retrieve small chunks, return parent documents",
        "icon": "📄"
    },
    "multi_query": {
        "name": "Multi-Query",
        "description": "Generate alternative queries for broader search",
        "icon": "🔍"
    },
    "compression": {
        "name": "Compression",
        "description": "LLM-based document compression and filtering",
        "icon": "🗜️"
    },
    "hybrid": {
        "name": "Hybrid",
        "description": "Combine multiple retrieval strategies",
        "icon": "🔀"
    }
}

# Document processing configurations
DOCUMENT_PROCESSORS = {
    'txt': {
        'name': 'Plain Text',
        'icon': '📝',
        'always_available': True
    },
    'md': {
        'name': 'Markdown',
        'icon': '📋',
        'always_available': True
    },
    'pdf': {
        'name': 'PDF Documents',
        'icon': '📕',
        'requires': 'PyPDF2'
    },
    'docx': {
        'name': 'Word Documents',
        'icon': '📘',
        'requires': 'python-docx'
    },
    'pptx': {
        'name': 'PowerPoint',
        'icon': '📊',
        'requires': 'python-pptx'
    },
    'xlsx': {
        'name': 'Excel Spreadsheets',
        'icon': '📗',
        'requires': 'openpyxl'
    },
    'xls': {
        'name': 'Excel Legacy',
        'icon': '📗',
        'requires': 'openpyxl'
    },
    'csv': {
        'name': 'CSV Files',
        'icon': '📊',
        'always_available': True
    }
}

def get_retrieval_strategy_info(strategy: str) -> Dict[str, Any]:
    """Get information about a retrieval strategy"""
    return RETRIEVAL_STRATEGIES.get(strategy, {
        "name": strategy.title(),
        "description": "Unknown strategy",
        "icon": "❓"
    })

def get_document_processor_info(extension: str) -> Dict[str, Any]:
    """Get information about a document processor"""
    return DOCUMENT_PROCESSORS.get(extension, {
        'name': f'{extension.upper()} Files',
        'icon': '📄',
        'always_available': False
    })