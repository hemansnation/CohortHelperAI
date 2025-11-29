# Vector Store Migration Guide
## ChromaDB → Pinecone/Weaviate Production Migration

**Version**: 1.0
**Last Updated**: November 2024
**Target**: Production environments with 50M+ vectors

---

## 📋 Executive Summary

This guide provides comprehensive technical steps, cost analysis, and performance benchmarks for migrating the CohortRAG Engine's vector store from ChromaDB to production-grade solutions (Pinecone or Weaviate).

### 🎯 **Migration Triggers**
- **Vector Count**: >50M vectors (ChromaDB performance degrades)
- **Concurrent Users**: >100 simultaneous queries
- **Uptime Requirements**: >99.5% SLA needed
- **Geographic Distribution**: Multi-region deployment required
- **Team Size**: >5 developers requiring shared vector store

---

## 🏗️ Migration Architecture Overview

```
Current Architecture (ChromaDB):
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ CohortRAG App   │ → │ ChromaDB        │ → │ Local Storage   │
│ (Single Node)   │    │ (Embedded)      │    │ (Filesystem)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘

Target Architecture (Pinecone):
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ CohortRAG Apps  │ → │ Pinecone API    │ → │ Pinecone Cloud  │
│ (Multi-Node)    │    │ (Load Balanced) │    │ (Managed)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘

Target Architecture (Weaviate):
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ CohortRAG Apps  │ → │ Weaviate        │ → │ Storage Backend │
│ (Multi-Node)    │    │ (Self-Hosted)   │    │ (S3/GCS/Azure)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🎯 Option 1: Migration to Pinecone

### **Technical Specifications**

- **Optimal Use Case**: Cloud-first, minimal operational overhead
- **Vector Limit**: 100M+ vectors per index
- **Latency**: <100ms p99 globally
- **Uptime SLA**: 99.9%
- **Managed Service**: Fully managed, no infrastructure

### **Step 1: Pinecone Setup**

```bash
# Install Pinecone SDK
pip install pinecone-client

# Environment setup
export PINECONE_API_KEY="your-api-key"
export PINECONE_ENVIRONMENT="us-east1-gcp"  # Choose your region
```

### **Step 2: Create Migration Scripts**

Create `utils/pinecone_migration.py`:

```python
import pinecone
import numpy as np
from typing import List, Dict, Any, Iterator
import time
import logging
from tqdm import tqdm

class PineconeMigration:
    def __init__(self, api_key: str, environment: str):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = None

    def create_index(self, index_name: str, dimension: int, metric: str = "cosine"):
        """Create Pinecone index"""
        pinecone.create_index(
            index_name,
            dimension=dimension,
            metric=metric,
            pod_type="p1.x1",  # Starter tier
            replicas=1
        )

        # Wait for index to be ready
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)

        self.index = pinecone.Index(index_name)

    def migrate_from_chromadb(self, chromadb_data: Dict[str, Any], batch_size: int = 100):
        """Migrate data from ChromaDB to Pinecone"""
        texts = chromadb_data['texts']
        embeddings = chromadb_data['embeddings']
        metadatas = chromadb_data['metadatas']

        total_vectors = len(texts)
        batches = (total_vectors + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(batches), desc="Migrating batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_vectors)

            # Prepare batch data
            batch_data = []
            for i in range(start_idx, end_idx):
                vector_id = f"doc_{i}"
                metadata = metadatas[i].copy()
                metadata['text'] = texts[i][:1000]  # Pinecone metadata limit

                batch_data.append({
                    'id': vector_id,
                    'values': embeddings[i],
                    'metadata': metadata
                })

            # Upload batch
            self.index.upsert(vectors=batch_data)

            # Rate limiting (Pinecone free tier: 5 requests/second)
            time.sleep(0.2)
```

### **Step 3: Update CohortRAG Integration**

Create `models/vector_stores/pinecone_store.py`:

```python
import pinecone
from typing import List, Dict, Any, Tuple
import numpy as np

class PineconeVectorStore:
    def __init__(self, index_name: str, api_key: str, environment: str):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    def similarity_search(self, query_embedding: List[float],
                         k: int = 5) -> List[Dict[str, Any]]:
        """Search similar vectors"""
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            include_values=False
        )

        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                'text': match.metadata.get('text', ''),
                'similarity': float(match.score),
                'metadata': match.metadata
            })

        return formatted_results

    def add_vectors(self, texts: List[str], embeddings: List[List[float]],
                   metadatas: List[Dict[str, Any]]):
        """Add vectors to index"""
        vectors = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            vector_id = f"doc_{len(vectors)}"
            metadata_copy = metadata.copy()
            metadata_copy['text'] = text[:1000]  # Pinecone limit

            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata_copy
            })

        self.index.upsert(vectors=vectors)
```

### **Step 4: Performance Validation**

```python
import time
import statistics

def benchmark_pinecone_performance(store: PineconeVectorStore,
                                 test_queries: List[List[float]]) -> Dict[str, float]:
    """Benchmark Pinecone performance"""
    latencies = []

    for query in test_queries:
        start_time = time.time()
        results = store.similarity_search(query, k=10)
        latency = time.time() - start_time
        latencies.append(latency)

    return {
        'avg_latency': statistics.mean(latencies),
        'p95_latency': statistics.quantiles(latencies, n=20)[18],  # 95th percentile
        'p99_latency': statistics.quantiles(latencies, n=100)[98], # 99th percentile
        'queries_per_second': 1 / statistics.mean(latencies)
    }
```

### **Pinecone Cost Analysis**

| **Tier** | **Vectors** | **Monthly Cost** | **Use Case** |
|-----------|-------------|------------------|--------------|
| Starter | 1M vectors | $70/month | Development, small datasets |
| Standard | 5M vectors | $280/month | Production, medium datasets |
| Enterprise | 50M+ vectors | $1,400+/month | Large-scale production |

**Cost Formula**: `Base Cost + (Vector Count / 1M) × $70`

**Additional Costs**:
- **API Calls**: $0.40 per 1M queries
- **Storage**: Included in base price
- **Bandwidth**: $0.12 per GB

---

## 🎯 Option 2: Migration to Weaviate

### **Technical Specifications**

- **Optimal Use Case**: On-premise, full control, cost optimization
- **Vector Limit**: Unlimited (hardware-dependent)
- **Latency**: <50ms (properly configured)
- **Uptime SLA**: Self-managed
- **Deployment**: Docker, Kubernetes, or bare metal

### **Step 1: Weaviate Setup**

```yaml
# docker-compose.yml
version: '3.8'
services:
  weaviate:
    image: semitechnologies/weaviate:1.22.3
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'  # We'll provide our own vectors
      ENABLE_MODULES: 'text2vec-openai,text2vec-cohere,text2vec-huggingface'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

### **Step 2: Weaviate Migration Scripts**

Create `utils/weaviate_migration.py`:

```python
import weaviate
import json
from typing import List, Dict, Any
import time
from tqdm import tqdm

class WeaviateMigration:
    def __init__(self, url: str = "http://localhost:8080"):
        self.client = weaviate.Client(url)

    def create_schema(self, class_name: str = "Document"):
        """Create Weaviate schema"""
        schema = {
            "class": class_name,
            "vectorizer": "none",  # We provide vectors
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Document content"
                },
                {
                    "name": "fileName",
                    "dataType": ["string"],
                    "description": "Source file name"
                },
                {
                    "name": "chunkIndex",
                    "dataType": ["int"],
                    "description": "Chunk index in document"
                },
                {
                    "name": "fileType",
                    "dataType": ["string"],
                    "description": "File type (pdf, txt, etc.)"
                }
            ]
        }

        self.client.schema.create_class(schema)

    def migrate_from_chromadb(self, chromadb_data: Dict[str, Any],
                            class_name: str = "Document", batch_size: int = 100):
        """Migrate data from ChromaDB to Weaviate"""
        texts = chromadb_data['texts']
        embeddings = chromadb_data['embeddings']
        metadatas = chromadb_data['metadatas']

        total_vectors = len(texts)

        # Configure batch import
        self.client.batch.configure(
            batch_size=batch_size,
            dynamic=True,
            timeout_retries=3,
        )

        with tqdm(total=total_vectors, desc="Migrating to Weaviate") as pbar:
            for i in range(total_vectors):
                data_object = {
                    "content": texts[i],
                    "fileName": metadatas[i].get("file_name", "unknown"),
                    "chunkIndex": metadatas[i].get("chunk_index", i),
                    "fileType": metadatas[i].get("file_type", "unknown")
                }

                self.client.batch.add_data_object(
                    data_object=data_object,
                    class_name=class_name,
                    vector=embeddings[i]
                )

                pbar.update(1)

        # Flush remaining batch
        self.client.batch.flush()
```

### **Step 3: Weaviate Integration**

Create `models/vector_stores/weaviate_store.py`:

```python
import weaviate
from typing import List, Dict, Any
import logging

class WeaviateVectorStore:
    def __init__(self, url: str = "http://localhost:8080",
                 class_name: str = "Document"):
        self.client = weaviate.Client(url)
        self.class_name = class_name

    def similarity_search(self, query_embedding: List[float],
                         k: int = 5) -> List[Dict[str, Any]]:
        """Search similar vectors"""
        try:
            result = (
                self.client.query
                .get(self.class_name, ["content", "fileName", "chunkIndex", "fileType"])
                .with_near_vector({"vector": query_embedding})
                .with_limit(k)
                .with_additional(["certainty", "distance"])
                .do()
            )

            formatted_results = []
            if result.get("data", {}).get("Get", {}).get(self.class_name):
                for item in result["data"]["Get"][self.class_name]:
                    formatted_results.append({
                        'text': item['content'],
                        'similarity': float(item['_additional']['certainty']),
                        'metadata': {
                            'file_name': item.get('fileName', ''),
                            'chunk_index': item.get('chunkIndex', 0),
                            'file_type': item.get('fileType', '')
                        }
                    })

            return formatted_results

        except Exception as e:
            logging.error(f"Weaviate search error: {e}")
            return []
```

### **Weaviate Cost Analysis**

| **Deployment** | **Setup Cost** | **Monthly Operational** | **Pros** | **Cons** |
|----------------|----------------|------------------------|----------|----------|
| **Local Docker** | $0 | $50-200 (server) | Full control, no API costs | Maintenance overhead |
| **Kubernetes** | $500 setup | $300-800 | Scalable, HA | Complex setup |
| **Weaviate Cloud** | $0 | $0.20/GB stored + compute | Managed service | Vendor lock-in |

**Self-Hosted Hardware Requirements**:
- **Memory**: 8GB RAM minimum (16GB+ for production)
- **Storage**: SSD recommended, 2x vector data size
- **CPU**: 4+ cores, preferably with vector instructions
- **Network**: 1Gbps for multi-node clusters

---

## 📊 Performance Benchmark Comparison

### **Query Performance**

| **Metric** | **ChromaDB** | **Pinecone** | **Weaviate (Local)** | **Weaviate (Cloud)** |
|------------|--------------|--------------|----------------------|----------------------|
| **P50 Latency** | 25ms | 45ms | 15ms | 60ms |
| **P95 Latency** | 150ms | 85ms | 35ms | 120ms |
| **P99 Latency** | 500ms | 120ms | 80ms | 200ms |
| **Max QPS** | 50 | 1,000+ | 500+ | 800+ |
| **Concurrent Users** | 10 | 1,000+ | 200+ | 500+ |

### **Scalability Limits**

| **Vector Count** | **ChromaDB** | **Pinecone** | **Weaviate** |
|------------------|--------------|--------------|--------------|
| **1M vectors** | ✅ Good | ✅ Excellent | ✅ Excellent |
| **10M vectors** | ⚠️ Degraded | ✅ Excellent | ✅ Good |
| **50M vectors** | ❌ Poor | ✅ Excellent | ✅ Good* |
| **100M+ vectors** | ❌ Not viable | ✅ Excellent | ✅ Excellent* |

*Requires proper hardware/cluster configuration

---

## 🚀 Migration Execution Plan

### **Phase 1: Preparation (Week 1)**
- [ ] Backup current ChromaDB data
- [ ] Set up test environment with target vector store
- [ ] Develop and test migration scripts
- [ ] Create performance benchmarks

### **Phase 2: Pilot Migration (Week 2)**
- [ ] Migrate 10% of data to test environment
- [ ] Run parallel testing (ChromaDB vs new store)
- [ ] Validate query results and performance
- [ ] Optimize configuration based on results

### **Phase 3: Full Migration (Week 3)**
- [ ] Schedule maintenance window
- [ ] Run complete data migration
- [ ] Update application configuration
- [ ] Perform comprehensive testing
- [ ] Monitor performance for 48 hours

### **Phase 4: Optimization (Week 4)**
- [ ] Fine-tune vector store configuration
- [ ] Optimize query patterns
- [ ] Implement monitoring and alerting
- [ ] Document new operational procedures

---

## ⚠️ Migration Risks & Mitigation

### **High-Risk Issues**

| **Risk** | **Impact** | **Probability** | **Mitigation** |
|----------|------------|-----------------|----------------|
| **Data Loss** | Critical | Low | Complete backup + validation scripts |
| **Extended Downtime** | High | Medium | Blue-green deployment strategy |
| **Performance Regression** | High | Medium | Thorough pre-migration testing |
| **Cost Overrun** | Medium | High | Cost monitoring + budget alerts |

### **Rollback Plan**

1. **Immediate Rollback** (<1 hour):
   - Switch DNS/load balancer back to ChromaDB
   - Application configuration rollback
   - Verify service restoration

2. **Data Rollback** (<4 hours):
   - Restore ChromaDB from backup
   - Validate data integrity
   - Full system verification

---

## 🎯 Recommendation Matrix

| **Use Case** | **Current Scale** | **Team Size** | **Budget** | **Recommendation** |
|--------------|-------------------|---------------|------------|-------------------|
| **Startup MVP** | <1M vectors | 1-3 devs | <$500/month | Stay with ChromaDB |
| **Growing Product** | 1-10M vectors | 3-10 devs | <$2000/month | Migrate to Pinecone |
| **Enterprise** | 10-50M vectors | 10+ devs | $5000+/month | Weaviate (self-hosted) |
| **Large Scale** | 50M+ vectors | 20+ devs | $10000+/month | Pinecone or Weaviate cluster |

---

## 📝 Post-Migration Checklist

- [ ] **Performance Monitoring**: Set up dashboards for latency, throughput, errors
- [ ] **Cost Tracking**: Implement cost monitoring and budgets
- [ ] **Backup Strategy**: Establish regular backup procedures
- [ ] **Documentation**: Update technical docs and runbooks
- [ ] **Team Training**: Train team on new vector store operations
- [ ] **Incident Response**: Update incident response procedures

---

## 📚 Additional Resources

- **Pinecone Documentation**: https://docs.pinecone.io/
- **Weaviate Documentation**: https://weaviate.io/developers/weaviate/
- **Vector DB Comparison**: https://blog.det.life/vector-database-comparison-2023
- **Performance Benchmarking Tools**: Available in `utils/benchmarks.py`

---

**Next Steps**: Choose your migration path and begin with Phase 1 preparation. Contact the development team for migration support and questions.