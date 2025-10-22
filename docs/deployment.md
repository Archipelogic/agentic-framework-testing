# Deployment Guide

## Overview

This guide covers deploying the Agentic AI Framework Testing Harness in various environments, from local development to cloud-based production systems.

## Running Benchmarks: Mock vs Production

## Overview

The testing harness supports two primary modes:
1. **Mock Mode** - Quick testing without API costs
2. **Live Mode** - Full testing with real AI models and APIs

## Quick Start with Unified Runner

The unified runner script handles all operations:

```bash
# Setup environment
./run.sh setup

# Run mock evaluation (no API keys needed)
./run.sh test

# Run live evaluation (requires API keys)
./run.sh test --live

# View latest report
./run.sh report

# Clean results
./run.sh clean
```

## Deployment Modes

### 1. Mock Mode
**Purpose**: Quick testing without API costs
**Requirements**: Minimal - Python 3.8+
**Use Cases**: Demonstrations, initial testing, development

```bash
# Quick mock deployment
./run.sh test --mock

# Or with specific samples
./run.sh test --mock --samples 50

# Or quick mode (3 frameworks only)
./run.sh test --mock --quick
```

### 2. Development Mode
**Purpose**: Feature development and testing
**Requirements**: API keys for at least one provider
**Use Cases**: Framework development, use case testing

```bash
# Set development environment
export ENVIRONMENT=development
export DEBUG=true

# Run with development config
python3 -m src.cli benchmark --config config/development.yaml
```

### 3. Production Mode
**Purpose**: Full-scale benchmarking and evaluation
**Requirements**: Complete infrastructure, all API keys
**Use Cases**: Official benchmarks, performance testing, reports

```bash
# Set production environment
export ENVIRONMENT=production
export LOG_LEVEL=INFO

# Run production benchmark
python3 -m src.cli benchmark --config config/production.yaml --parallel
```

## Local Deployment

### Standard Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd agentic-framework-testing

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run tests
pytest tests/

# 6. Start benchmarking
./run.sh test
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create results directory
RUN mkdir -p benchmark_results

# Entry point
ENTRYPOINT ["python3", "-m", "src.cli"]
CMD ["benchmark", "--help"]
```

Build and run:

```bash
# Build Docker image
docker build -t agentic-testing:latest .

# Run with environment file
docker run --env-file .env \
  -v $(pwd)/benchmark_results:/app/benchmark_results \
  agentic-testing:latest benchmark \
  --frameworks langgraph crewai \
  --use-cases all
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  benchmark:
    build: .
    image: agentic-testing:latest
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./benchmark_results:/app/benchmark_results
      - ./config:/app/config
    command: benchmark --config /app/config/production.yaml

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: benchmarks
      POSTGRES_USER: benchmark_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

Run with Docker Compose:

```bash
# Start all services
docker-compose up -d

# Run benchmark
docker-compose run benchmark

# View logs
docker-compose logs -f benchmark
```

## Cloud Deployment

### AWS Deployment

#### EC2 Instance

```bash
# 1. Launch EC2 instance (recommended: t3.xlarge or larger)
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t3.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-0123456789abcdef0 \
  --subnet-id subnet-6e7f829e

# 2. SSH into instance
ssh -i your-key.pem ec2-user@instance-ip

# 3. Install dependencies
sudo yum update -y
sudo yum install -y python3 python3-pip git

# 4. Clone and setup
git clone <repository-url>
cd agentic-framework-testing
pip3 install -r requirements.txt

# 5. Configure with AWS Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id agentic-testing/api-keys \
  --query SecretString \
  --output text > .env

# 6. Run benchmark
nohup python3 -m src.cli benchmark --config config/production.yaml &
```

#### AWS Lambda (Serverless)

Create `lambda_handler.py`:

```python
import json
import boto3
from run_evaluation import UnifiedBenchmarkRunner

def lambda_handler(event, context):
    """AWS Lambda handler for benchmarks."""
    
    # Parse request
    mode = event.get('mode', 'mock')
    samples = event.get('samples', 20)
    
    # Run benchmark
    runner = UnifiedBenchmarkRunner(
        mode=mode,
        samples=samples,
        test_cases_per_use_case=event.get('test_cases', 5),
        parallel=False  # Lambda handles parallelism
    )
    
    # Store results in S3
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket='benchmark-results',
        Key=f"results/{results['benchmark_id']}.json",
        Body=json.dumps(results)
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'benchmark_id': results['benchmark_id'],
            'summary': results['summary']
        })
    }
```

Deploy with SAM:

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  BenchmarkFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: lambda_handler.lambda_handler
      Runtime: python3.10
      Timeout: 900
      MemorySize: 3008
      Environment:
        Variables:
          OPENAI_API_KEY: !Ref OpenAIApiKey
      Policies:
        - S3CrudPolicy:
            BucketName: benchmark-results
```

### Google Cloud Platform

#### Cloud Run

Create `cloudbuild.yaml`:

```yaml
steps:
  # Build container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/agentic-testing', '.']
  
  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/agentic-testing']
  
  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'agentic-testing'
      - '--image=gcr.io/$PROJECT_ID/agentic-testing'
      - '--region=us-central1'
      - '--platform=managed'
      - '--memory=4Gi'
      - '--timeout=900'
      - '--set-env-vars=OPENAI_API_KEY=$$OPENAI_API_KEY'
```

Deploy:

```bash
# Submit build
gcloud builds submit --config=cloudbuild.yaml

# Or deploy directly
gcloud run deploy agentic-testing \
  --source . \
  --region us-central1 \
  --memory 4Gi \
  --timeout 900 \
  --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY"
```

### Azure Deployment

#### Azure Container Instances

```bash
# Create resource group
az group create --name agentic-testing-rg --location eastus

# Create container instance
az container create \
  --resource-group agentic-testing-rg \
  --name agentic-testing \
  --image agentic-testing:latest \
  --cpu 4 \
  --memory 8 \
  --environment-variables \
    OPENAI_API_KEY=$OPENAI_API_KEY \
    ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  --command-line "python3 -m src.cli benchmark --config config/production.yaml"
```

## Kubernetes Deployment

### Kubernetes Manifests

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-testing
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentic-testing
  template:
    metadata:
      labels:
        app: agentic-testing
    spec:
      containers:
      - name: benchmark
        image: agentic-testing:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-key
        volumeMounts:
        - name: results
          mountPath: /app/benchmark_results
      volumes:
      - name: results
        persistentVolumeClaim:
          claimName: benchmark-results-pvc
```

Create `k8s/job.yaml` for batch processing:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: benchmark-job
spec:
  template:
    spec:
      containers:
      - name: benchmark
        image: agentic-testing:latest
        command: ["python3", "-m", "src.cli", "benchmark"]
        args:
          - "--frameworks"
          - "all"
          - "--use-cases"
          - "all"
          - "--parallel"
      restartPolicy: OnFailure
  backoffLimit: 3
```

Deploy to Kubernetes:

```bash
# Create namespace
kubectl create namespace benchmarking

# Create secrets
kubectl create secret generic api-keys \
  --from-literal=openai-key=$OPENAI_API_KEY \
  --from-literal=anthropic-key=$ANTHROPIC_API_KEY \
  -n benchmarking

# Deploy application
kubectl apply -f k8s/ -n benchmarking

# Run benchmark job
kubectl apply -f k8s/job.yaml -n benchmarking

# Check status
kubectl get pods -n benchmarking
kubectl logs -f job/benchmark-job -n benchmarking
```

## Production Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Performance
MAX_PARALLEL_WORKERS=8
REQUEST_TIMEOUT=60
RETRY_ATTEMPTS=3
RATE_LIMIT_DELAY=1

# Monitoring
ENABLE_MONITORING=true
METRICS_PORT=8080
LOG_LEVEL=INFO

# Storage
RESULTS_STORAGE=s3
S3_BUCKET=benchmark-results
S3_REGION=us-east-1

# Database
DATABASE_URL=postgresql://user:pass@localhost/benchmarks
REDIS_URL=redis://localhost:6379
```

### Monitoring and Observability

#### Prometheus Metrics

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
benchmark_runs = Counter('benchmark_runs_total', 'Total benchmark runs', ['framework', 'use_case'])
benchmark_duration = Histogram('benchmark_duration_seconds', 'Benchmark duration')
active_benchmarks = Gauge('active_benchmarks', 'Currently running benchmarks')
framework_errors = Counter('framework_errors_total', 'Framework errors', ['framework', 'error_type'])

# Start metrics server
start_http_server(8080)
```

#### Logging Configuration

```python
# src/core/logging_config.py
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'json',
            'filename': 'logs/benchmark.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        '': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

## Scaling Considerations

### Horizontal Scaling

```python
# src/scaling/distributed_runner.py
from celery import Celery
from run_evaluation import UnifiedBenchmarkRunner

app = Celery('benchmarks', broker='redis://localhost:6379')

@app.task
def run_framework_benchmark(mode='mock', samples=20):
    """Run benchmark as Celery task."""
    runner = UnifiedBenchmarkRunner(mode=mode, samples=samples)
    return runner.run()

def run_distributed_benchmark(frameworks, use_cases):
    """Run distributed benchmark using Celery."""
    tasks = []
    
    for framework in frameworks:
        for use_case in use_cases:
            task = run_framework_benchmark.delay(
                framework, use_case, test_cases
            )
            tasks.append(task)
    
    # Collect results
    results = {}
    for task in tasks:
        result = task.get(timeout=300)
        results.update(result)
    
    return results
```

### Load Balancing

```nginx
# nginx.conf
upstream benchmark_api {
    least_conn;
    server benchmark1.example.com:8000;
    server benchmark2.example.com:8000;
    server benchmark3.example.com:8000;
}

server {
    listen 80;
    server_name api.benchmark.example.com;
    
    location / {
        proxy_pass http://benchmark_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
    }
}
```

## Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh

# Backup results
aws s3 sync benchmark_results/ s3://backup-bucket/results/ --delete

# Backup database
pg_dump $DATABASE_URL | gzip > backup_$(date +%Y%m%d).sql.gz
aws s3 cp backup_*.sql.gz s3://backup-bucket/db/

# Clean old backups
find . -name "backup_*.sql.gz" -mtime +30 -delete
```

### Disaster Recovery

```python
# src/recovery/disaster_recovery.py
class DisasterRecovery:
    """Handle disaster recovery scenarios."""
    
    def backup_checkpoint(self, benchmark_state):
        """Save benchmark checkpoint for recovery."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'state': benchmark_state,
            'completed_tasks': self.get_completed_tasks(),
            'pending_tasks': self.get_pending_tasks()
        }
        
        # Store in multiple locations
        self.save_to_s3(checkpoint)
        self.save_to_local(checkpoint)
        
        return checkpoint
    
    def restore_from_checkpoint(self, checkpoint_id):
        """Restore benchmark from checkpoint."""
        checkpoint = self.load_checkpoint(checkpoint_id)
        
        # Resume from where it left off
        self.restore_state(checkpoint['state'])
        self.requeue_pending_tasks(checkpoint['pending_tasks'])
        
        return checkpoint
```

## Security Considerations

1. **API Key Management**: Use secrets management services
2. **Network Security**: Use VPC and security groups
3. **Data Encryption**: Encrypt at rest and in transit
4. **Access Control**: Implement RBAC and audit logging
5. **Rate Limiting**: Protect against abuse

## Next Steps

- [Configure for production](./configuration.md)
- [Monitor and troubleshoot](./troubleshooting.md)
- [Best practices guide](./best-practices.md)
