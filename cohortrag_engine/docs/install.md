# 📚 Installation Guide

## Quick Installation (Recommended)

### Prerequisites
- **Python 3.8+** (Check with `python --version`)
- **Git** for repository cloning
- **4GB+ RAM** recommended for optimal performance
- **Google Gemini API key** ([Get one here](https://ai.google.dev/))

### 1. Clone Repository
```bash
git clone https://github.com/YourUsername/CohortHelperAI.git
cd CohortHelperAI/cohortrag_engine
```

### 2. Install Dependencies
```bash
# Option A: Basic installation
pip install -r requirements.txt

# Option B: With virtual environment (recommended)
python -m venv cohortrag_env
source cohortrag_env/bin/activate  # On Windows: cohortrag_env\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Create configuration file
echo "GEMINI_API_KEY=your_api_key_here" > .env
echo "DATA_DIR=./data" >> .env
echo "CHROMA_DB_PATH=./chroma_db" >> .env
```

### 4. Verify Installation
```bash
python main.py
# Should display the CohortRAG Engine menu
```

## Detailed Installation Options

### Docker Installation (Coming Soon)
```bash
# Pull official image
docker pull cohortrag/engine:latest

# Run with volume mounts
docker run -d \
  --name cohortrag \
  -e GEMINI_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -p 8000:8000 \
  cohortrag/engine:latest
```

### Development Installation
```bash
git clone https://github.com/YourUsername/CohortHelperAI.git
cd CohortHelperAI/cohortrag_engine

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install with development dependencies
pip install -r requirements.txt
pip install black isort pytest pytest-cov flake8

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests to verify
python tests/run_tests.py
```

## Configuration Guide

### Environment Variables
Create a `.env` file in the `cohortrag_engine` directory:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Paths (optional, defaults provided)
DATA_DIR=./data
CHROMA_DB_PATH=./chroma_db

# Model Configuration (optional)
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
LLM_MODEL=gemini-2.5-flash

# RAG Configuration (optional)
SIMILARITY_TOP_K=3
CONTEXT_WINDOW=4096
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Evaluation (optional)
ENABLE_EVALUATION=true
RAGAS_LOG_LEVEL=INFO
```

### Getting Your Gemini API Key
1. Go to [Google AI Studio](https://ai.google.dev/)
2. Sign in with your Google account
3. Click "Get API Key" and create a new key
4. Copy the key and add it to your `.env` file

**Note**: Gemini API has free tier limits. For production use, consider the paid tier.

### Directory Structure
After installation, your directory should look like:
```
cohortrag_engine/
├── .env                     # Your configuration
├── data/                    # Your documents go here
├── chroma_db/              # Vector database (created automatically)
├── core/                   # Core RAG components
├── utils/                  # Utilities and optimizations
├── models/                 # LLM and embedding models
├── tests/                  # Test suite
├── docs/                   # Documentation
├── main.py                 # Main application
├── config.py               # Configuration management
└── requirements.txt        # Python dependencies
```

## Troubleshooting

### Common Installation Issues

**1. Python Version Error**
```bash
# Error: Python 3.7 or older
# Solution: Install Python 3.8+
python --version  # Should be 3.8+
```

**2. Permission Errors**
```bash
# Error: Permission denied
# Solution: Use virtual environment or --user flag
pip install --user -r requirements.txt
```

**3. Dependency Conflicts**
```bash
# Error: Package conflicts
# Solution: Use fresh virtual environment
rm -rf venv  # Remove old environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**4. GPU/CUDA Issues**
```bash
# If you have GPU issues with embeddings
# Solution: Use CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### API Key Issues

**Invalid API Key**
- Verify your key is correct in `.env`
- Check key permissions in Google AI Studio
- Ensure no extra spaces in the key

**Rate Limiting**
- Gemini has free tier limits
- Consider upgrading to paid tier for production
- Enable caching to reduce API calls

### Performance Issues

**Slow Document Loading**
- Check available RAM (4GB+ recommended)
- Consider smaller chunk sizes for large documents
- Use async ingestion for multiple documents

**High Memory Usage**
- Reduce `CHUNK_SIZE` in configuration
- Process documents in smaller batches
- Monitor memory with the performance tools

## Next Steps

After successful installation:

1. **Add Your Documents**: Place educational content in the `data/` directory
2. **Run Ingestion**: Use option 1 in the main menu to process documents
3. **Test Queries**: Use option 2 to test question-answering
4. **Explore Features**: Try the production-ready options (8-14)
5. **Read Documentation**: Check out the other guides in `docs/`

## Getting Help

- **Documentation**: Check other files in the `docs/` directory
- **Issues**: Report problems on [GitHub Issues](https://github.com/YourUsername/CohortHelperAI/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/YourUsername/CohortHelperAI/discussions)
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to help

## System Requirements

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB
- **Storage**: 2GB for installation + space for your documents
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8+

### Recommended for Production
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: SSD with 10GB+ free space
- **Network**: Stable internet for API calls
- **Python**: 3.9+ for best performance

## License

CohortRAG Engine is licensed under the [Apache License 2.0](../LICENSE), making it suitable for both personal and commercial use in educational applications.