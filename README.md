## 🚀 Features

- **Advanced RAG Pipeline**: Multi-stage retrieval with contextual headers and intelligent reranking
- **Multi-format Document Support**: PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx/.xls), CSV, TXT, Markdown
- **Local Models**: Runs entirely offline using Ollama models
- **Contextual Chunking**: Chunks include contextual headers (up to 100# Advanced RAG Pipeline with Local Models

A powerful Retrieval-Augmented Generation (RAG) system that works entirely with local models using Ollama. Upload documents in various formats and ask questions about their content using advanced retrieval techniques.

## 🚀 Features

- **Multi-format Document Support**: PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx/.xls), CSV, TXT, Markdown
- **Local Models**: Runs entirely offline using Ollama models
- **Advanced Retrieval**: Multiple retrieval strategies including parent-child, multi-query, and contextual compression
- **Interactive UI**: Clean Streamlit interface with progress tracking
- **Document Management**: SQLite database for document and query storage
- **Performance Metrics**: Response time and result count tracking

## 📋 Supported File Types

| Format | Extension | Library Used | Status |
|--------|-----------|--------------|--------|
| Text | `.txt` | Built-in | ✅ Always supported |
| Markdown | `.md` | Built-in | ✅ Always supported |
| PDF | `.pdf` | PyPDF2 | ✅ Supported |
| Word | `.docx` | python-docx | ✅ Supported |
| PowerPoint | `.pptx` | python-pptx | ✅ Supported |
| Excel | `.xlsx`, `.xls` | openpyxl, pandas | ✅ Supported |
| CSV | `.csv` | pandas | ✅ Supported |

## 🛠️ Prerequisites

1. **Python 3.8+**
2. **Ollama** - Install from [https://ollama.ai](https://ollama.ai)
3. **Required Ollama Models**:
   - `nomic-embed-text` (for embeddings)
   - `llama3.1:8b` (for text generation)

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone or download the project files
# Run the setup script
python setup.py
```

The setup script will:
- Check system requirements
- Install Python dependencies
- Download required Ollama models
- Create test files
- Verify installation

### Option 2: Manual Setup

1. **Install Ollama and Models**:
```bash
# Install Ollama (visit https://ollama.ai for instructions)
# Then pull the required models:
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

2. **Install Python Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Application**:
```bash
streamlit run main.py
```

## 📁 Project Structure

```
├── main.py                 # Main Streamlit application
├── rag_pipeline.py         # RAG pipeline implementation
├── document_processor.py   # Multi-format document processing
├── database.py            # SQLite database operations
├── requirements.txt       # Python dependencies
├── setup.py              # Automated setup script
└── README.md             # This file
```

## 🔧 Configuration

### Changing Models

Edit the model names in `rag_pipeline.py`:

```python
class AdvancedRAGPipeline:
    def __init__(self, embedding_model="nomic-embed-text", llm_model="llama3.1:8b"):
        # Change these to use different models
```

### Retrieval Settings

Modify retrieval parameters in `rag_pipeline.py`:

```python
# Chunk sizes
self.child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,    # Adjust chunk size
    chunk_overlap=50   # Adjust overlap
)

# Number of documents to retrieve
self.vectorstore.as_retriever(search_kwargs={"k": 6})
```

## 🎯 Usage

1. **Start the Application**:
   ```bash
   streamlit run main.py
   ```

2. **Upload Documents**:
   - Use the file uploader in the left column
   - Supports multiple files at once
   - Progress bar shows processing status

3. **Ask Questions**:
   - Type your question in the text area
   - Click "Ask" to get answers
   - Toggle "Show source documents" to see references

4. **View History**:
   - Expand "Query History" to see past questions
   - Includes performance metrics (response time, document count)

## 🔍 Advanced Features

### Multiple Retrieval Strategies

The system supports several retrieval approaches:

- **Compression Retrieval** (default): Uses LLM to compress and rank results
- **Parent Document**: Retrieves small chunks but returns larger parent documents
- **Multi-Query**: Generates multiple query variations for better coverage
- **Hybrid**: Combines multiple strategies

### Document Processing Intelligence

- **Automatic Format Detection**: Identifies file types automatically
- **Table Extraction**: Extracts tables from Word and PowerPoint files
- **Metadata Preservation**: Maintains source information and page numbers
- **Error Handling**: Graceful fallbacks for corrupted files

### Performance Monitoring

- Response time tracking
- Document count metrics
- Database statistics
- Query history with performance data

## 🐛 Troubleshooting

### Common Issues

1. **Import Error for Ollama**:
   ```
   ImportError: cannot import name 'Ollama' from 'langchain_ollama'
   ```
   **Solution**: The code uses `ChatOllama` instead of `Ollama` (already fixed)

2. **OllamaEmbeddings Validation Error**:
   ```
   Extra inputs are not permitted [type=extra_forbidden, input_value=True, input_type=bool]
   ```
   **Solution**: Removed `show_progress` parameter (already fixed)

3. **Models Not Found**:
   ```bash
   # Make sure models are installed
   ollama list
   # If missing, install them:
   ollama pull nomic-embed-text
   ollama pull llama3.1:8b
   ```

4. **Document Processing Fails**:
   - Check if required libraries are installed
   - Try converting to a simpler format (TXT/MD)
   - Check file permissions and corruption

### Performance Tips

1. **Large Documents**: 
   - Break into smaller files for better processing
   - Adjust chunk sizes in `rag_pipeline.py`

2. **Memory Usage**:
   - Restart Streamlit if memory usage grows
   - Clear database periodically: `database.clear_all_data()`

3. **Slow Responses**:
   - Use smaller models (e.g., `llama3.1:7b`)
   - Reduce number of retrieved documents
   - Check Ollama server performance

## 📊 Database Management

The system uses SQLite for data persistence:

```python
# Get database statistics
stats = database.get_database_stats()

# Backup database
backup_path = database.backup_database()

# Clear all data (use with caution)
database.clear_all_data()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Make sure Ollama is running with required models
4. Check the application logs for detailed error messages

## 🎉 Acknowledgments

- **Ollama** for local model serving
- **LangChain** for RAG framework
- **Streamlit** for the web interface
- **ChromaDB** for vector storage