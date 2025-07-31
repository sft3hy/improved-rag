import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Model Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Chunking Configuration
    CHILD_CHUNK_SIZE = 250
    PARENT_CHUNK_SIZE = 2500
    CONTEXTUAL_HEADER_SIZE = 100
    CHUNK_OVERLAP = 50

    # Retrieval Configuration
    INITIAL_RETRIEVAL_COUNT = 200
    FILTERED_CHILD_CHUNKS = 150
    PARENT_CHUNKS_COUNT = 20
    FINAL_CHUNKS_COUNT = 30
    MIN_RELEVANCE_SCORE = 0.1

    # Database
    DATABASE_PATH = "data/rag_database.db"

    # UI Configuration
    MAX_UPLOAD_SIZE = 20  # MB


settings = Settings()
