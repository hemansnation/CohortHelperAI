# 📁 Project Structure

## Overview

CohortRAG Engine is organized into clear, modular components that separate concerns and make the codebase easy to understand and extend.

```
cohortrag_engine/
├── 📁 core/                    # Core RAG functionality
├── 📁 utils/                   # Utilities and optimizations
├── 📁 models/                  # LLM and embedding models
├── 📁 tests/                   # Test suite
├── 📁 docs/                    # Documentation
├── 🐍 main.py                  # Main CLI application
├── ⚙️  config.py               # Configuration management
├── 📋 requirements.txt         # Python dependencies
└── 🔧 .env.example            # Configuration template
```

## Core Components

### 📁 `core/` - RAG Engine Core
The heart of the RAG system with all main functionality:

```
core/
├── ingestion.py           # Document processing and chunking
├── async_ingestion.py     # Production async document processing
├── retrieval.py           # Basic RAG query processing
├── cached_retrieval.py    # Production RAG with caching
└── evaluation.py          # RAGAS evaluation and quality assurance
```

**Key Classes:**
- `CohortRAGIngestionPipeline`: Document processing and vector storage
- `AsyncCohortRAGIngestion`: High-performance async ingestion
- `CohortRAGRetriever`: Basic RAG query processing
- `ProductionCohortRAGRetriever`: Production RAG with caching and monitoring
- `RAGASEvaluator`: Quality assurance and success metrics validation

### 📁 `utils/` - Utilities & Optimizations
Production-ready utilities for performance and monitoring:

```
utils/
├── chunking.py            # Text chunking strategies
├── query_expansion.py     # Query enhancement
├── reranking.py          # Result reranking
├── performance.py        # Async processing and monitoring
├── cost_modeling.py      # Cost tracking and optimization
├── benchmarks.py         # Performance benchmarking
└── success_metrics.py    # Production readiness validation
```

**Key Features:**
- **Performance**: Async processing, memory monitoring, benchmarking
- **Cost Optimization**: Token tracking, budget alerts, cost projections
- **Quality Assurance**: Success metrics validation against production targets

### 📁 `models/` - AI Model Integration
Clean abstractions for LLMs and embedding models:

```
models/
├── llm.py                # Large Language Model integration
└── embeddings.py         # Embedding model management
```

**Supported Models:**
- **LLM**: Gemini 2.5-flash (optimized for educational content)
- **Embeddings**: Nomic-embed-text-v1 (best open-source option for education)

### 📁 `tests/` - Quality Assurance
Comprehensive testing suite for reliability:

```
tests/
├── test_chunking.py      # Text processing tests
├── test_models.py        # Model integration tests
├── test_evaluation.py    # Evaluation framework tests
└── run_tests.py          # Test runner
```

## Configuration Management

### ⚙️ `config.py`
Centralized configuration with environment variable support:

```python
class CohortRAGConfig:
    # API Configuration
    gemini_api_key: str

    # Model Configuration
    embedding_model: str = "nomic-ai/nomic-embed-text-v1"
    llm_model: str = "gemini-2.5-flash"

    # RAG Configuration
    similarity_top_k: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 50
```

### 🔧 `.env.example`
Template for environment configuration with all options documented.

## Main Application

### 🐍 `main.py`
Interactive CLI with development and production options:

**Development Options (1-7):**
- Document ingestion
- Query testing
- System statistics
- Interactive Q&A
- Sample testing
- RAGAS evaluation
- Unit tests

**Production Options (8-14):**
- Async document ingestion
- Production query with caching
- Cache management & monitoring
- Performance benchmarking
- Cost optimization analysis
- Success metrics validation
- Production readiness check

## Documentation

### 📁 `docs/`
Comprehensive guides for all use cases:

```
docs/
├── install.md                      # Installation guide
├── self_host.md                    # Production deployment
├── migration.md                    # Vector store scaling
├── project_structure.md            # This file
├── SUCCESS_METRICS_VALIDATION.md   # Quality assurance
└── VECTOR_STORE_MIGRATION.md       # Technical migration guide
```

## Development Workflow

### 🔄 Adding New Features

1. **Core Functionality**: Add to `core/` directory
2. **Utilities**: Add to `utils/` for reusable components
3. **Tests**: Add comprehensive tests to `tests/`
4. **Documentation**: Update relevant guides in `docs/`
5. **CLI Integration**: Add menu options to `main.py`

### 🧪 Testing Strategy

```bash
# Run all tests
python tests/run_tests.py

# Test specific component
python -m pytest tests/test_chunking.py -v

# Performance testing
python main.py
# Select option 11: Performance benchmarking
```

### 📊 Quality Assurance

```bash
# Validate production readiness
python main.py
# Select option 14: Production readiness check

# Success metrics validation
python main.py
# Select option 13: Success metrics validation
```

## Architecture Principles

### 🏗 Modular Design
- **Separation of Concerns**: Each module has a single responsibility
- **Loose Coupling**: Components communicate through well-defined interfaces
- **High Cohesion**: Related functionality grouped together

### ⚡ Performance First
- **Async Processing**: Built for concurrent operations
- **Intelligent Caching**: Multiple caching strategies
- **Memory Optimization**: Efficient memory usage patterns

### 📈 Production Ready
- **Monitoring**: Comprehensive metrics and logging
- **Scalability**: Clear upgrade paths
- **Quality Assurance**: Automated validation against production targets

### 🎓 Educational Focus
- **Domain Optimization**: Specialized for educational content
- **Curriculum Awareness**: Understanding of educational structures
- **Assessment Integration**: Connection to learning evaluation

## Extension Points

### 🔌 Adding New Vector Stores
Extend `core/retrieval.py` with new retriever classes following the pattern:

```python
class NewVectorStoreRetriever(CohortRAGRetriever):
    def __init__(self, connection_params):
        # Initialize new vector store

    def query(self, question: str) -> RAGResponse:
        # Implement query logic
```

### 🤖 Adding New LLMs
Extend `models/llm.py` with new model integrations:

```python
class NewLLMModel:
    def complete(self, prompt: str) -> str:
        # Implement completion logic
```

### 📊 Adding New Metrics
Extend `utils/success_metrics.py` with new validation metrics:

```python
def _validate_new_metric(self) -> MetricResult:
    # Implement metric validation
```

## Dependencies

### Core Dependencies
- **LlamaIndex**: RAG framework foundation
- **ChromaDB**: Vector database for development
- **Gemini API**: Large language model
- **RAGAS**: Evaluation framework

### Production Dependencies
- **Redis**: Caching and session management
- **AsyncIO**: Concurrent processing
- **Prometheus**: Metrics collection
- **Pydantic**: Configuration validation

### Development Dependencies
- **Pytest**: Testing framework
- **Black**: Code formatting
- **isort**: Import sorting
- **Pre-commit**: Git hooks

## Getting Help

### 📚 Learning Resources
1. **Start Here**: [Installation Guide](install.md)
2. **Deploy**: [Self-Hosting Guide](self_host.md)
3. **Scale**: [Migration Guide](migration.md)
4. **Contribute**: [Contributing Guidelines](../CONTRIBUTING.md)

### 🤝 Community Support
- **Documentation**: Check `docs/` directory first
- **Discussions**: [GitHub Discussions](https://github.com/YourUsername/CohortHelperAI/discussions)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/YourUsername/CohortHelperAI/issues)
- **Code Review**: Submit PRs following [contribution guidelines](../CONTRIBUTING.md)

---

This structure enables rapid development while maintaining production-ready quality. Each component is designed for both educational effectiveness and operational excellence.