# 🚀 Self-Hosting Guide

Complete guide for deploying CohortRAG Engine in production environments.

## Production Architecture

### Recommended Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Application   │    │   Vector Store  │
│     (nginx)     │───▶│   (CohortRAG)   │───▶│    (ChromaDB/   │
│                 │    │                 │    │   Pinecone)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Caching     │    │   Monitoring    │    │   Backup &      │
│   (Redis)       │    │  (Prometheus)   │    │   Storage       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Deployment Options

### Option 1: Docker Deployment (Recommended)

#### 1. Create Production Dockerfile
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data chroma_db logs

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV CHROMA_DB_PATH=/app/chroma_db

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "from core.retrieval import CohortRAGRetriever; CohortRAGRetriever().get_stats()" || exit 1

# Expose port (if adding web interface later)
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
```

#### 2. Docker Compose for Full Stack
```yaml
# docker-compose.yml
version: '3.8'

services:
  cohortrag:
    build: .
    container_name: cohortrag-engine
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./chroma_db:/app/chroma_db
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped
    networks:
      - cohortrag-network

  redis:
    image: redis:7-alpine
    container_name: cohortrag-redis
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - cohortrag-network

  nginx:
    image: nginx:alpine
    container_name: cohortrag-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - cohortrag
    restart: unless-stopped
    networks:
      - cohortrag-network

volumes:
  redis_data:

networks:
  cohortrag-network:
    driver: bridge
```

#### 3. Deploy with Docker Compose
```bash
# Set environment variables
export GEMINI_API_KEY=your_api_key_here

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f cohortrag

# Scale if needed
docker-compose up -d --scale cohortrag=3
```

### Option 2: Kubernetes Deployment

#### 1. Kubernetes Manifests
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cohortrag

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cohortrag-config
  namespace: cohortrag
data:
  DATA_DIR: "/app/data"
  CHROMA_DB_PATH: "/app/chroma_db"
  REDIS_URL: "redis://redis-service:6379"

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: cohortrag-secrets
  namespace: cohortrag
type: Opaque
data:
  GEMINI_API_KEY: <base64-encoded-api-key>

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cohortrag-deployment
  namespace: cohortrag
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
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: cohortrag-config
        - secretRef:
            name: cohortrag-secrets
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: chroma-volume
          mountPath: /app/chroma_db
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "from core.retrieval import CohortRAGRetriever; CohortRAGRetriever().get_stats()"
          initialDelaySeconds: 60
          periodSeconds: 30
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
      - name: chroma-volume
        persistentVolumeClaim:
          claimName: chroma-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: cohortrag-service
  namespace: cohortrag
spec:
  selector:
    app: cohortrag
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

#### 2. Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n cohortrag
kubectl get services -n cohortrag

# Scale deployment
kubectl scale deployment cohortrag-deployment --replicas=5 -n cohortrag

# Check logs
kubectl logs -f deployment/cohortrag-deployment -n cohortrag
```

### Option 3: Traditional Server Deployment

#### 1. System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.9+
sudo apt install -y python3.9 python3.9-venv python3.9-dev

# Install system dependencies
sudo apt install -y git nginx redis-server

# Create application user
sudo useradd -m -s /bin/bash cohortrag
sudo mkdir -p /opt/cohortrag
sudo chown cohortrag:cohortrag /opt/cohortrag
```

#### 2. Application Setup
```bash
# Switch to application user
sudo -u cohortrag bash

# Clone and setup application
cd /opt/cohortrag
git clone https://github.com/YourUsername/CohortHelperAI.git .
cd cohortrag_engine

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp .env.example .env
# Edit .env with your settings
```

#### 3. Create Systemd Service
```bash
# Create service file
sudo tee /etc/systemd/system/cohortrag.service > /dev/null <<EOF
[Unit]
Description=CohortRAG Engine
After=network.target

[Service]
Type=simple
User=cohortrag
Group=cohortrag
WorkingDirectory=/opt/cohortrag/cohortrag_engine
Environment=PATH=/opt/cohortrag/cohortrag_engine/venv/bin
ExecStart=/opt/cohortrag/cohortrag_engine/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable cohortrag
sudo systemctl start cohortrag

# Check status
sudo systemctl status cohortrag
```

## Production Configuration

### Environment Variables for Production
```env
# API Configuration
GEMINI_API_KEY=your_production_api_key

# Paths
DATA_DIR=/opt/cohortrag/data
CHROMA_DB_PATH=/opt/cohortrag/chroma_db
LOG_DIR=/opt/cohortrag/logs

# Performance
SIMILARITY_TOP_K=5
CONTEXT_WINDOW=8192
CHUNK_SIZE=1024
CHUNK_OVERLAP=100

# Caching
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600

# Production Features
ENABLE_EVALUATION=true
ENABLE_MONITORING=true
ENABLE_ASYNC_PROCESSING=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
CORS_ORIGINS=https://yourdomain.com
```

### Nginx Configuration
```nginx
# /etc/nginx/sites-available/cohortrag
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL Configuration
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=cohortrag:10m rate=10r/s;
    limit_req zone=cohortrag burst=20 nodelay;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;

    # Reverse proxy to application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long-running queries
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Static files (if any)
    location /static/ {
        alias /opt/cohortrag/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

## Monitoring & Logging

### Application Monitoring
```python
# monitoring.py - Add to your deployment
import logging
import psutil
import time
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# Metrics
QUERY_COUNT = Counter('cohortrag_queries_total', 'Total number of queries')
QUERY_LATENCY = Histogram('cohortrag_query_duration_seconds', 'Query latency')
MEMORY_USAGE = Gauge('cohortrag_memory_usage_bytes', 'Memory usage')
CACHE_HIT_RATE = Gauge('cohortrag_cache_hit_rate', 'Cache hit rate')

def setup_monitoring():
    """Setup monitoring and metrics collection"""
    # Start Prometheus metrics server
    start_http_server(9090)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/opt/cohortrag/logs/application.log'),
            logging.StreamHandler()
        ]
    )

    # Start metrics collection
    collect_system_metrics()

def collect_system_metrics():
    """Collect and report system metrics"""
    while True:
        # Memory usage
        memory_info = psutil.virtual_memory()
        MEMORY_USAGE.set(memory_info.used)

        # Log system status
        logging.info(f"Memory usage: {memory_info.percent}%")

        time.sleep(60)  # Collect every minute
```

### Log Analysis with ELK Stack
```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  elasticsearch:
    image: elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    networks:
      - monitoring

  logstash:
    image: logstash:7.17.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    networks:
      - monitoring
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:7.17.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    networks:
      - monitoring
    depends_on:
      - elasticsearch

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - monitoring

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - monitoring

volumes:
  es_data:
  grafana_data:

networks:
  monitoring:
```

## Backup & Recovery

### Automated Backup Script
```bash
#!/bin/bash
# backup.sh - Automated backup for CohortRAG

BACKUP_DIR="/opt/cohortrag/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup vector database
echo "Backing up ChromaDB..."
tar -czf "$BACKUP_DIR/chroma_db_$DATE.tar.gz" -C /opt/cohortrag chroma_db/

# Backup configuration
echo "Backing up configuration..."
cp /opt/cohortrag/cohortrag_engine/.env "$BACKUP_DIR/config_$DATE.env"

# Backup logs
echo "Backing up logs..."
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" -C /opt/cohortrag logs/

# Upload to cloud storage (optional)
# aws s3 sync "$BACKUP_DIR" s3://your-backup-bucket/cohortrag/

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR"
```

### Recovery Procedures
```bash
# Recovery script
#!/bin/bash
# recover.sh - Restore CohortRAG from backup

BACKUP_FILE=$1
RESTORE_DIR="/opt/cohortrag"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# Stop service
sudo systemctl stop cohortrag

# Restore from backup
echo "Restoring from $BACKUP_FILE..."
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"

# Fix permissions
sudo chown -R cohortrag:cohortrag "$RESTORE_DIR"

# Start service
sudo systemctl start cohortrag

echo "Recovery completed"
```

## Security Best Practices

### 1. API Key Management
- Use environment variables, never hardcode
- Rotate API keys regularly
- Use different keys for staging/production
- Monitor API usage for anomalies

### 2. Network Security
- Use HTTPS everywhere
- Implement proper firewall rules
- Use VPN for internal communications
- Regular security updates

### 3. Application Security
```python
# Security configuration
SECURITY_CONFIG = {
    'rate_limiting': {
        'queries_per_minute': 60,
        'burst_limit': 10
    },
    'input_validation': {
        'max_query_length': 1000,
        'allowed_file_types': ['.pdf', '.txt', '.md', '.docx']
    },
    'output_sanitization': {
        'filter_sensitive_info': True,
        'max_response_length': 5000
    }
}
```

### 4. Data Protection
- Encrypt sensitive data at rest
- Use secure backup practices
- Implement proper access controls
- Regular security audits

## Performance Optimization

### 1. Caching Strategy
```python
# Redis configuration for production
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'max_connections': 20,
    'socket_keepalive': True,
    'socket_keepalive_options': {},
    'health_check_interval': 30
}
```

### 2. Database Optimization
- Regular database maintenance
- Index optimization
- Query performance monitoring
- Connection pooling

### 3. Resource Management
```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "2Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## Troubleshooting

### Common Production Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux | grep python

# Solutions:
# - Reduce chunk size
# - Implement pagination
# - Add memory monitoring
```

#### Slow Query Performance
```bash
# Check query logs
tail -f /opt/cohortrag/logs/queries.log

# Solutions:
# - Enable caching
# - Optimize vector search
# - Add query preprocessing
```

#### API Rate Limits
```bash
# Monitor API usage
grep "rate_limit" /opt/cohortrag/logs/application.log

# Solutions:
# - Implement exponential backoff
# - Use caching aggressively
# - Consider API key rotation
```

### Health Checks
```python
def health_check():
    """Comprehensive health check for production"""
    checks = {
        'database': check_database_connection(),
        'api': check_gemini_api(),
        'memory': check_memory_usage(),
        'cache': check_cache_connection(),
        'disk': check_disk_space()
    }

    return {
        'status': 'healthy' if all(checks.values()) else 'unhealthy',
        'checks': checks,
        'timestamp': time.time()
    }
```

## Scaling Considerations

### Horizontal Scaling
- Load balancer configuration
- Session management (if stateful)
- Shared cache (Redis cluster)
- Database sharding strategies

### Vertical Scaling
- Memory optimization
- CPU utilization monitoring
- Storage performance tuning
- Network bandwidth management

## Support & Maintenance

### Regular Maintenance Tasks
1. **Daily**: Monitor logs and performance metrics
2. **Weekly**: Review backup integrity and security updates
3. **Monthly**: Capacity planning and performance optimization
4. **Quarterly**: Security audits and dependency updates

### Getting Help
- **Documentation**: Check all guides in `docs/`
- **Community**: [GitHub Discussions](https://github.com/YourUsername/CohortHelperAI/discussions)
- **Issues**: [GitHub Issues](https://github.com/YourUsername/CohortHelperAI/issues)
- **Enterprise Support**: Contact for production support options

---

This guide provides a foundation for production deployment. Customize based on your specific infrastructure, security requirements, and scale needs.