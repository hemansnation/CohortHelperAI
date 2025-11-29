# 🐳 Docker Deployment Guide

Complete guide for deploying CohortRAG Engine using Docker for simplified installation and deployment.

## Quick Start with Docker

### Prerequisites
- Docker 20.10+ installed
- Docker Compose 2.0+ installed
- 4GB+ RAM available
- Your Gemini API key

### 1. Quick Run (Single Container)
```bash
# Create environment file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run CohortRAG Engine
docker run -it --rm \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  cohortrag/engine:latest
```

### 2. Full Stack with Docker Compose
```bash
# Clone repository
git clone https://github.com/YourUsername/CohortHelperAI.git
cd CohortHelperAI/cohortrag_engine

# Create environment file
cp .env.example .env
# Edit .env with your configuration

# Start full stack (app + Redis)
docker-compose up -d

# Access the CLI
docker-compose exec cohortrag cohortrag
```

## Docker Images Available

### Production Image: `cohortrag/engine:latest`
- **Size**: ~800MB optimized
- **Purpose**: Production deployment
- **Includes**: Core RAG engine, CLI tools
- **User**: Non-root `app` user for security

### Development Image: `cohortrag/engine:dev`
- **Size**: ~1.2GB with dev tools
- **Purpose**: Development and testing
- **Includes**: All dev dependencies, testing tools
- **User**: Non-root with development tools

### Jupyter Image: `cohortrag/engine:jupyter`
- **Size**: ~1.5GB with notebooks
- **Purpose**: Research and experimentation
- **Includes**: Jupyter Lab, notebooks, examples
- **Port**: 8888 exposed for web interface

## Deployment Scenarios

### Scenario 1: Development Environment
```bash
# Start development environment
docker-compose --profile development up -d cohortrag-dev

# Access development container
docker-compose exec cohortrag-dev bash

# Run tests
pytest tests/

# Format code
black . && isort .
```

### Scenario 2: Production Deployment
```bash
# Start production services
docker-compose up -d cohortrag redis

# Monitor logs
docker-compose logs -f cohortrag

# Scale if needed
docker-compose up -d --scale cohortrag=3
```

### Scenario 3: Research with Jupyter
```bash
# Start Jupyter environment
docker-compose --profile jupyter up -d jupyter

# Access Jupyter at http://localhost:8888
# Password: check logs with 'docker-compose logs jupyter'
```

### Scenario 4: Full Monitoring Stack
```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access services:
# - CohortRAG Engine: CLI via docker-compose exec
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

## Configuration

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key

# Paths (container defaults)
DATA_DIR=/app/data
CHROMA_DB_PATH=/app/chroma_db
LOG_DIR=/app/logs

# Redis caching
REDIS_URL=redis://redis:6379
ENABLE_CACHING=true
CACHE_TYPE=redis

# Performance
CHUNK_SIZE=512
SIMILARITY_TOP_K=3
ENABLE_ASYNC_PROCESSING=true

# Monitoring
ENABLE_MONITORING=true
LOG_LEVEL=INFO
```

### Volume Mounts
```yaml
volumes:
  - ./data:/app/data                    # Your documents
  - ./chroma_db:/app/chroma_db          # Vector database
  - ./logs:/app/logs                    # Application logs
  - ./.env:/app/.env:ro                 # Configuration
```

## CLI Commands in Container

### Using Docker Run
```bash
# Interactive CLI
docker run -it --rm \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  cohortrag/engine:latest cohortrag

# Run benchmark
docker run --rm \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  cohortrag/engine:latest cohortrag-benchmark --quick

# Validate success metrics
docker run --rm \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  cohortrag/engine:latest cohortrag-validate --readiness
```

### Using Docker Compose
```bash
# Interactive CLI
docker-compose exec cohortrag cohortrag

# Run specific commands
docker-compose exec cohortrag cohortrag-benchmark --comprehensive
docker-compose exec cohortrag cohortrag-validate --quick

# One-off commands
docker-compose run --rm cohortrag cohortrag-benchmark --output /app/logs/benchmark.json
```

## Building Custom Images

### Build Production Image
```bash
# Build from source
docker build -t cohortrag-custom:latest .

# Build with custom arguments
docker build \
  --build-arg INSTALL_DEV=true \
  --target development \
  -t cohortrag-custom:dev .
```

### Multi-Architecture Build
```bash
# Build for multiple platforms
docker buildx create --use
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t cohortrag/engine:latest \
  --push .
```

## Data Management

### Backup Strategy
```bash
# Backup vector database
docker-compose exec cohortrag tar -czf /app/logs/chroma_backup.tar.gz -C /app chroma_db

# Copy backup to host
docker cp cohortrag-engine:/app/logs/chroma_backup.tar.gz ./backups/

# Backup configuration
docker-compose exec cohortrag cp /app/.env /app/logs/config_backup.env
```

### Data Migration
```bash
# Export data for migration
docker-compose exec cohortrag python -c "
from utils.export_chroma import ChromaExporter
exporter = ChromaExporter()
exporter.export_data('/app/logs/migration_export.json')
"

# Import to new environment
docker run --rm \
  -v $(pwd)/migration_export.json:/app/migration_export.json \
  -v $(pwd)/new_chroma_db:/app/chroma_db \
  cohortrag/engine:latest python -c "
from utils.import_data import import_from_json
import_from_json('/app/migration_export.json')
"
```

## Performance Optimization

### Memory Optimization
```yaml
# docker-compose.yml
services:
  cohortrag:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    environment:
      - CHUNK_SIZE=256  # Smaller chunks for less memory
      - SIMILARITY_TOP_K=3
```

### CPU Optimization
```yaml
# docker-compose.yml
services:
  cohortrag:
    deploy:
      resources:
        limits:
          cpus: '2.0'
    environment:
      - MAX_WORKERS=4  # Async processing workers
      - BATCH_SIZE=50
```

### Redis Optimization
```yaml
# docker-compose.yml
services:
  redis:
    command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru
    deploy:
      resources:
        limits:
          memory: 1.5G
```

## Monitoring and Logging

### Application Logs
```bash
# View live logs
docker-compose logs -f cohortrag

# Search logs
docker-compose exec cohortrag grep "ERROR" /app/logs/*.log

# Log rotation (production)
docker-compose exec cohortrag logrotate /etc/logrotate.d/cohortrag
```

### Health Checks
```bash
# Check container health
docker-compose ps

# Manual health check
docker-compose exec cohortrag python -c "
from cohortrag_engine import CohortRAGEngine
engine = CohortRAGEngine()
print('Health check passed')
"
```

### Prometheus Metrics (with monitoring profile)
```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# View metrics
curl http://localhost:9090/metrics

# Grafana dashboards
# Access http://localhost:3000 (admin/admin)
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker-compose logs cohortrag

# Debug mode
docker-compose run --rm cohortrag bash
```

#### Memory Issues
```bash
# Monitor memory usage
docker stats cohortrag-engine

# Reduce memory usage
export CHUNK_SIZE=256
export SIMILARITY_TOP_K=2
docker-compose restart cohortrag
```

#### Permission Issues
```bash
# Fix volume permissions
sudo chown -R 1000:1000 data/ chroma_db/ logs/

# Check user in container
docker-compose exec cohortrag id
```

#### API Connection Issues
```bash
# Test API key
docker-compose exec cohortrag python -c "
import os
from models.llm import get_llm_model
from config import get_config
llm = get_llm_model(get_config())
print('API connection successful')
"
```

### Debug Mode
```bash
# Enable debug logging
echo "DEBUG=true" >> .env
echo "LOG_LEVEL=DEBUG" >> .env
docker-compose restart cohortrag

# Run in debug mode
docker-compose run --rm \
  -e DEBUG=true \
  cohortrag bash
```

## Security Best Practices

### Container Security
```yaml
# docker-compose.yml security settings
services:
  cohortrag:
    user: "1000:1000"  # Non-root user
    read_only: true     # Read-only filesystem
    tmpfs:
      - /tmp
      - /var/tmp
    security_opt:
      - no-new-privileges:true
```

### Network Security
```yaml
# Restrict network access
services:
  cohortrag:
    networks:
      - internal
  redis:
    networks:
      - internal
    # No external ports exposed
```

### Secrets Management
```bash
# Use Docker secrets (production)
echo "your_api_key" | docker secret create gemini_api_key -

# Reference in compose
services:
  cohortrag:
    secrets:
      - gemini_api_key
    environment:
      - GEMINI_API_KEY_FILE=/run/secrets/gemini_api_key
```

## Production Deployment

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml cohortrag

# Scale service
docker service scale cohortrag_cohortrag=3
```

### Kubernetes
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cohortrag-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cohortrag
  template:
    metadata:
      labels:
        app: cohortrag
    spec:
      containers:
      - name: cohortrag
        image: cohortrag/engine:latest
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: gemini
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: chroma-db
          mountPath: /app/chroma_db
```

### Container Registry
```bash
# Build and push to registry
docker build -t your-registry.com/cohortrag-engine:v1.0.0 .
docker push your-registry.com/cohortrag-engine:v1.0.0

# Deploy from registry
docker run -d \
  --name cohortrag-prod \
  --env-file .env \
  your-registry.com/cohortrag-engine:v1.0.0
```

## Getting Help

### Documentation
- **General Setup**: [Installation Guide](install.md)
- **Production Deployment**: [Self-Hosting Guide](self_host.md)
- **Project Structure**: [Project Structure](project_structure.md)

### Community Support
- **Docker Issues**: [GitHub Issues](https://github.com/YourUsername/CohortHelperAI/issues)
- **General Questions**: [GitHub Discussions](https://github.com/YourUsername/CohortHelperAI/discussions)
- **Docker Hub**: [Official Images](https://hub.docker.com/r/cohortrag/engine)

---

Docker deployment makes CohortRAG Engine accessible to users without Python expertise while providing production-ready scalability and monitoring capabilities.