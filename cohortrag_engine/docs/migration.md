# 🔄 Vector Store Migration Guide

Comprehensive guide for migrating from ChromaDB to enterprise vector databases like Pinecone or Weaviate as your CohortRAG Engine scales.

## When to Migrate

### ChromaDB Limitations
- **Document Limit**: Performance degrades after 50M+ vectors
- **Concurrent Users**: Limited to ~100 simultaneous users
- **High Availability**: No built-in clustering or replication
- **Memory Usage**: Entire index loaded in memory
- **Backup/Recovery**: Limited enterprise backup options

### Migration Triggers
✅ **Migrate when you experience:**
- Document processing time >5 minutes for large batches
- Query latency >3 seconds consistently
- Memory usage >16GB for vector operations
- Need for >99.9% uptime
- Multi-region deployment requirements
- Team size >10 concurrent users

## Migration Options Overview

| **Vector Store** | **Best For** | **Pricing** | **Complexity** | **Features** |
|------------------|--------------|-------------|----------------|--------------|
| **Pinecone** | Quick migration, managed service | $0.096/1M queries | Low | Serverless, auto-scaling |
| **Weaviate** | Self-hosted, advanced features | Open source + hosting | Medium | GraphQL, hybrid search |
| **Qdrant** | High performance, Rust-based | Open source + cloud | Medium | Filtering, payloads |
| **Milvus** | Large scale, enterprise | Open source + cloud | High | Distributed, GPU support |

## Pinecone Migration

### Pre-Migration Assessment
```python
# Run this assessment before migrating
from core.retrieval import CohortRAGRetriever
from utils.cost_modeling import GeminiCostTracker

def assess_pinecone_migration():
    """Assess current system for Pinecone migration"""
    retriever = CohortRAGRetriever()
    stats = retriever.get_stats()

    # Current metrics
    total_vectors = stats['total_chunks']
    avg_queries_per_day = 1000  # Estimate from your logs

    # Pinecone cost calculation
    monthly_cost = (total_vectors / 1_000_000) * 0.096 * 30  # Storage
    monthly_cost += (avg_queries_per_day * 30 / 1_000_000) * 0.096  # Queries

    print(f"📊 Migration Assessment:")
    print(f"   Current vectors: {total_vectors:,}")
    print(f"   Estimated monthly Pinecone cost: ${monthly_cost:.2f}")
    print(f"   Recommended: {'Yes' if total_vectors > 1_000_000 else 'Evaluate need'}")

    return {
        'total_vectors': total_vectors,
        'monthly_cost': monthly_cost,
        'recommended': total_vectors > 1_000_000
    }
```

### Step 1: Pinecone Setup
```python
# 1. Install Pinecone
pip install pinecone-client

# 2. Setup Pinecone configuration
import pinecone
import os

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
)

# Create index
index_name = "cohortrag-education"
dimension = 768  # nomic-embed-text-v1 dimension

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        pod_type="p1.x1",  # Start small, scale up
        replicas=1
    )

print(f"✅ Pinecone index '{index_name}' ready")
```

### Step 2: Data Export from ChromaDB
```python
# utils/export_chroma.py
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

class ChromaExporter:
    """Export data from ChromaDB for migration"""

    def __init__(self, chroma_path: str = "./chroma_db/simple_store.pkl"):
        self.chroma_path = chroma_path

    def export_data(self, output_path: str = "./migration_data.json") -> Dict[str, Any]:
        """Export ChromaDB data for migration"""
        print("🔄 Exporting ChromaDB data...")

        # Load ChromaDB data
        with open(self.chroma_path, 'rb') as f:
            vector_store = pickle.load(f)

        # Prepare export data
        export_data = {
            'texts': vector_store.texts,
            'metadatas': vector_store.metadatas,
            'embeddings': [emb.tolist() for emb in vector_store.embeddings],
            'total_count': len(vector_store.texts),
            'export_timestamp': time.time()
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ Exported {export_data['total_count']:,} vectors to {output_path}")
        return export_data

# Usage
exporter = ChromaExporter()
exported_data = exporter.export_data()
```

### Step 3: Batch Upload to Pinecone
```python
# utils/pinecone_migrator.py
import pinecone
import time
from typing import List, Dict, Any
from tqdm import tqdm

class PineconeMigrator:
    """Handle migration to Pinecone"""

    def __init__(self, index_name: str):
        self.index = pinecone.Index(index_name)
        self.batch_size = 100  # Pinecone batch limit

    def migrate_data(self, export_file: str = "./migration_data.json"):
        """Migrate exported data to Pinecone"""
        print("🚀 Starting Pinecone migration...")

        # Load exported data
        with open(export_file, 'r') as f:
            data = json.load(f)

        texts = data['texts']
        metadatas = data['metadatas']
        embeddings = data['embeddings']

        # Batch upload
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(0, len(texts), self.batch_size),
                      desc="Uploading to Pinecone"):
            batch_texts = texts[i:i + self.batch_size]
            batch_metadatas = metadatas[i:i + self.batch_size]
            batch_embeddings = embeddings[i:i + self.batch_size]

            # Prepare vectors for Pinecone
            vectors = []
            for j, (text, metadata, embedding) in enumerate(
                zip(batch_texts, batch_metadatas, batch_embeddings)
            ):
                vector_id = f"vec_{i + j}"

                # Add text to metadata for retrieval
                metadata_with_text = {**metadata, 'text': text}

                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata_with_text
                })

            # Upload batch
            self.index.upsert(vectors=vectors)

            # Rate limiting
            time.sleep(0.1)

        print(f"✅ Migration complete! {len(texts):,} vectors uploaded")

        # Verify migration
        stats = self.index.describe_index_stats()
        print(f"📊 Pinecone index stats: {stats}")

# Usage
migrator = PineconeMigrator("cohortrag-education")
migrator.migrate_data()
```

### Step 4: Update CohortRAG Configuration
```python
# core/pinecone_retrieval.py
import pinecone
from typing import List, Dict, Any
import numpy as np

class PineconeRAGRetriever:
    """RAG retriever using Pinecone vector store"""

    def __init__(self, index_name: str, config=None):
        self.config = config or get_config()
        self.index = pinecone.Index(index_name)

        # Initialize embedding model (same as before)
        self.embedding_model = get_embedding_model(self.config)
        self.llm = get_llm_model(self.config)

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """Query using Pinecone index"""
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(question)

        # Search Pinecone
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Extract sources
        sources = []
        context_chunks = []

        for match in search_results['matches']:
            similarity_score = match['score']
            metadata = match['metadata']
            text = metadata.get('text', '')

            sources.append({
                'text': text,
                'similarity': similarity_score,
                'metadata': metadata
            })

            context_chunks.append(text)

        # Generate answer using LLM
        context = "\n\n".join(context_chunks)
        answer = self.llm.complete(
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ).text

        processing_time = time.time() - start_time

        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            processing_time=processing_time
        )

# Update configuration
# Add to config.py
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cohortrag-education")
```

### Step 5: Performance Validation
```python
# utils/migration_validator.py
import time
import statistics
from core.retrieval import CohortRAGRetriever
from core.pinecone_retrieval import PineconeRAGRetriever

class MigrationValidator:
    """Validate migration performance"""

    def __init__(self):
        self.chroma_retriever = CohortRAGRetriever()
        self.pinecone_retriever = PineconeRAGRetriever("cohortrag-education")

    def compare_performance(self, test_queries: List[str]) -> Dict[str, Any]:
        """Compare ChromaDB vs Pinecone performance"""
        print("🔍 Comparing ChromaDB vs Pinecone performance...")

        chroma_times = []
        pinecone_times = []
        accuracy_comparisons = []

        for query in test_queries:
            # Test ChromaDB
            start_time = time.time()
            chroma_response = self.chroma_retriever.query(query)
            chroma_time = time.time() - start_time
            chroma_times.append(chroma_time)

            # Test Pinecone
            start_time = time.time()
            pinecone_response = self.pinecone_retriever.query(query)
            pinecone_time = time.time() - start_time
            pinecone_times.append(pinecone_time)

            # Compare accuracy (simplified)
            chroma_sources = set(s['text'][:100] for s in chroma_response.sources)
            pinecone_sources = set(s['text'][:100] for s in pinecone_response.sources)
            overlap = len(chroma_sources.intersection(pinecone_sources))
            accuracy = overlap / max(len(chroma_sources), 1)
            accuracy_comparisons.append(accuracy)

        # Calculate statistics
        results = {
            'chroma': {
                'avg_latency': statistics.mean(chroma_times),
                'median_latency': statistics.median(chroma_times),
                'max_latency': max(chroma_times)
            },
            'pinecone': {
                'avg_latency': statistics.mean(pinecone_times),
                'median_latency': statistics.median(pinecone_times),
                'max_latency': max(pinecone_times)
            },
            'accuracy': {
                'avg_overlap': statistics.mean(accuracy_comparisons),
                'min_overlap': min(accuracy_comparisons)
            }
        }

        # Print comparison
        print(f"📊 Performance Comparison:")
        print(f"   ChromaDB avg latency: {results['chroma']['avg_latency']:.3f}s")
        print(f"   Pinecone avg latency: {results['pinecone']['avg_latency']:.3f}s")
        print(f"   Source overlap: {results['accuracy']['avg_overlap']:.2%}")

        return results

# Usage
validator = MigrationValidator()
test_queries = ["What is machine learning?", "Explain photosynthesis", "How does DNA work?"]
comparison = validator.compare_performance(test_queries)
```

## Weaviate Migration

### Step 1: Weaviate Setup
```python
# 1. Install Weaviate client
pip install weaviate-client

# 2. Setup Weaviate (Docker)
# docker-compose.weaviate.yml
version: '3.4'
services:
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8080'
    - --scheme
    - http
    image: semitechnologies/weaviate:1.21.2
    ports:
    - "8080:8080"
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'  # We'll use our own embeddings
      ENABLE_MODULES: 'backup-filesystem'
      BACKUP_FILESYSTEM_PATH: '/var/lib/weaviate/backups'
    volumes:
    - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

### Step 2: Weaviate Schema Creation
```python
# utils/weaviate_migrator.py
import weaviate
import json
from typing import List, Dict, Any

class WeaviateMigrator:
    """Handle migration to Weaviate"""

    def __init__(self, url: str = "http://localhost:8080"):
        self.client = weaviate.Client(url)
        self.class_name = "EducationalContent"

    def create_schema(self):
        """Create Weaviate schema for educational content"""
        schema = {
            "class": self.class_name,
            "description": "Educational content for RAG system",
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "The main content text"
                },
                {
                    "name": "file_name",
                    "dataType": ["string"],
                    "description": "Source file name"
                },
                {
                    "name": "file_type",
                    "dataType": ["string"],
                    "description": "Type of source file"
                },
                {
                    "name": "chunk_index",
                    "dataType": ["int"],
                    "description": "Index of chunk within document"
                },
                {
                    "name": "subject_area",
                    "dataType": ["string"],
                    "description": "Educational subject area"
                }
            ]
        }

        # Delete class if exists
        if self.client.schema.exists(self.class_name):
            self.client.schema.delete_class(self.class_name)

        # Create class
        self.client.schema.create_class(schema)
        print(f"✅ Created Weaviate schema for {self.class_name}")

    def migrate_data(self, export_file: str = "./migration_data.json"):
        """Migrate data to Weaviate"""
        print("🚀 Starting Weaviate migration...")

        # Load exported data
        with open(export_file, 'r') as f:
            data = json.load(f)

        texts = data['texts']
        metadatas = data['metadatas']
        embeddings = data['embeddings']

        # Batch upload
        batch_size = 100

        with self.client.batch.configure(batch_size=batch_size) as batch:
            for i, (text, metadata, embedding) in enumerate(
                tqdm(zip(texts, metadatas, embeddings), desc="Uploading to Weaviate")
            ):
                # Prepare properties
                properties = {
                    "text": text,
                    "file_name": metadata.get("file_name", "unknown"),
                    "file_type": metadata.get("file_type", "unknown"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "subject_area": metadata.get("subject_area", "general")
                }

                # Add to batch
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    vector=embedding
                )

        print(f"✅ Weaviate migration complete! {len(texts):,} objects uploaded")

# Usage
migrator = WeaviateMigrator()
migrator.create_schema()
migrator.migrate_data()
```

### Step 3: Weaviate RAG Implementation
```python
# core/weaviate_retrieval.py
import weaviate
from typing import List, Dict, Any

class WeaviateRAGRetriever:
    """RAG retriever using Weaviate"""

    def __init__(self, url: str = "http://localhost:8080", config=None):
        self.client = weaviate.Client(url)
        self.class_name = "EducationalContent"
        self.config = config or get_config()

        # Initialize models
        self.embedding_model = get_embedding_model(self.config)
        self.llm = get_llm_model(self.config)

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """Query using Weaviate with hybrid search"""
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(question)

        # Weaviate hybrid search (vector + keyword)
        result = (
            self.client.query
            .get(self.class_name, ["text", "file_name", "file_type", "chunk_index"])
            .with_hybrid(
                query=question,  # Keyword search
                vector=query_embedding,  # Vector search
                alpha=0.7  # Weight towards vector search
            )
            .with_additional(["score", "distance"])
            .with_limit(top_k)
            .do()
        )

        # Extract sources
        sources = []
        context_chunks = []

        if result.get("data", {}).get("Get", {}).get(self.class_name):
            for item in result["data"]["Get"][self.class_name]:
                text = item["text"]
                score = item.get("_additional", {}).get("score", 0)

                sources.append({
                    'text': text,
                    'similarity': score,
                    'metadata': {
                        'file_name': item.get('file_name'),
                        'file_type': item.get('file_type'),
                        'chunk_index': item.get('chunk_index')
                    }
                })

                context_chunks.append(text)

        # Generate answer
        context = "\n\n".join(context_chunks)
        answer = self.llm.complete(
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ).text

        processing_time = time.time() - start_time

        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            processing_time=processing_time
        )
```

## Migration Checklist

### Pre-Migration
- [ ] **Backup existing data** (ChromaDB + configuration)
- [ ] **Performance baseline** (current latency, accuracy, costs)
- [ ] **Test environment setup** (separate from production)
- [ ] **API keys and credentials** (Pinecone/Weaviate)
- [ ] **Migration timeline** (maintenance window planning)

### During Migration
- [ ] **Export ChromaDB data** (texts, embeddings, metadata)
- [ ] **Validate export integrity** (count, sample verification)
- [ ] **Setup target vector store** (schema, indexes, configuration)
- [ ] **Batch upload data** (with progress monitoring)
- [ ] **Performance testing** (latency, accuracy comparison)

### Post-Migration
- [ ] **Switch application configuration** (update retriever class)
- [ ] **End-to-end testing** (full user workflows)
- [ ] **Performance monitoring** (establish new baselines)
- [ ] **Cleanup old data** (after verification period)
- [ ] **Documentation updates** (new architecture, procedures)

## Cost Analysis

### Migration Costs
```python
def calculate_migration_costs(current_vectors: int, queries_per_month: int):
    """Calculate costs for different vector stores"""

    costs = {}

    # ChromaDB (self-hosted)
    costs['chromadb'] = {
        'monthly_hosting': 50,  # Server costs
        'maintenance_hours': 8,
        'total_monthly': 50 + (8 * 50)  # $50/hour developer time
    }

    # Pinecone
    costs['pinecone'] = {
        'storage': (current_vectors / 1_000_000) * 0.096 * 30,
        'queries': (queries_per_month / 1_000_000) * 0.096,
        'total_monthly': 0
    }
    costs['pinecone']['total_monthly'] = costs['pinecone']['storage'] + costs['pinecone']['queries']

    # Weaviate Cloud
    costs['weaviate_cloud'] = {
        'starter_plan': 25,  # Up to 1M vectors
        'standard_plan': 200,  # Up to 10M vectors
        'total_monthly': 25 if current_vectors < 1_000_000 else 200
    }

    # Weaviate Self-hosted
    costs['weaviate_self'] = {
        'monthly_hosting': 100,  # Larger server for Weaviate
        'maintenance_hours': 4,  # Easier than ChromaDB
        'total_monthly': 100 + (4 * 50)
    }

    return costs
```

### ROI Analysis
```python
def migration_roi_analysis(current_vectors: int):
    """Analyze ROI of migration"""

    # Current performance issues
    current_issues = {
        'slow_queries': 20,  # % of queries >3s
        'downtime_hours': 8,  # Hours per month
        'maintenance_hours': 16,  # Developer hours per month
    }

    # Cost of current issues
    issue_cost = (
        (current_issues['downtime_hours'] * 1000) +  # $1000/hour downtime
        (current_issues['maintenance_hours'] * 75)    # $75/hour developer time
    )

    print(f"💰 Monthly cost of current issues: ${issue_cost}")
    print(f"📊 Migration break-even calculation:")

    # Compare with migration costs
    costs = calculate_migration_costs(current_vectors, 100000)

    for option, cost in costs.items():
        monthly_savings = issue_cost - cost['total_monthly']
        months_to_breakeven = 5000 / max(monthly_savings, 1)  # $5k migration effort

        print(f"   {option}: ${cost['total_monthly']}/month, "
              f"breakeven in {months_to_breakeven:.1f} months")
```

## Rollback Plan

### Emergency Rollback
```bash
#!/bin/bash
# rollback_migration.sh - Emergency rollback script

echo "🚨 Starting emergency rollback..."

# 1. Stop new system
docker-compose -f production.yml down

# 2. Restore ChromaDB backup
tar -xzf backups/chroma_backup_pre_migration.tar.gz -C ./

# 3. Update configuration
cp config/chromadb.env .env

# 4. Restart with ChromaDB
docker-compose -f chromadb.yml up -d

# 5. Verify system health
python health_check.py

echo "✅ Rollback completed. System restored to ChromaDB."
```

### Gradual Rollback
```python
# Gradual traffic switching
class HybridRetriever:
    """Use both vector stores during transition"""

    def __init__(self, traffic_split: float = 0.5):
        self.chroma_retriever = CohortRAGRetriever()
        self.pinecone_retriever = PineconeRAGRetriever("cohortrag-education")
        self.traffic_split = traffic_split  # % to new system

    def query(self, question: str) -> RAGResponse:
        # Route traffic based on split
        if random.random() < self.traffic_split:
            return self.pinecone_retriever.query(question)
        else:
            return self.chroma_retriever.query(question)
```

## Monitoring Migration Success

### Key Metrics to Track
```python
class MigrationMonitor:
    """Monitor migration success metrics"""

    def __init__(self):
        self.metrics = {
            'latency_p95': [],
            'accuracy_scores': [],
            'error_rates': [],
            'cost_per_query': [],
            'user_satisfaction': []
        }

    def track_query_performance(self, query: str, response: RAGResponse):
        """Track individual query performance"""
        self.metrics['latency_p95'].append(response.processing_time)

        # Track costs
        cost = self.calculate_query_cost(response)
        self.metrics['cost_per_query'].append(cost)

    def daily_report(self) -> Dict[str, Any]:
        """Generate daily migration report"""
        return {
            'avg_latency': statistics.mean(self.metrics['latency_p95']),
            'avg_cost': statistics.mean(self.metrics['cost_per_query']),
            'error_rate': len([x for x in self.metrics['error_rates'] if x]) / len(self.metrics['error_rates']),
            'performance_trend': 'improving' if self.is_improving() else 'stable'
        }
```

---

This migration guide provides a comprehensive path for scaling your CohortRAG Engine. Choose the vector store that best fits your performance requirements, budget, and operational capabilities.

For additional support during migration, consult the [community discussions](https://github.com/YourUsername/CohortHelperAI/discussions) or create a [migration issue](https://github.com/YourUsername/CohortHelperAI/issues) for specific guidance.