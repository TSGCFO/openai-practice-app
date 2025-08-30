# Deployment & Configuration Guide

## Table of Contents

1. [Overview](#overview)
2. [Development Environment Setup](#development-environment-setup)
3. [Docker Configuration](#docker-configuration)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Configuration Management](#configuration-management)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Monitoring & Observability](#monitoring--observability)
8. [Scaling Strategies](#scaling-strategies)
9. [Backup & Recovery](#backup--recovery)
10. [Production Checklist](#production-checklist)

## Overview

This guide provides comprehensive instructions for deploying and configuring the AI Assistant system in various environments, from local development to production-scale Kubernetes clusters.

### Deployment Architecture

```mermaid
graph TB
    subgraph "Development"
        Dev[Docker Compose]
        LocalDB[(Local PostgreSQL)]
        LocalRedis[(Local Redis)]
    end
    
    subgraph "Staging"
        Stage[Kubernetes Staging]
        StageDB[(Staging RDS)]
        StageCache[(Staging ElastiCache)]
    end
    
    subgraph "Production"
        Prod[Kubernetes Production]
        ProdDB[(Production RDS)]
        ProdCache[(Production ElastiCache)]
        CDN[CloudFront CDN]
    end
    
    Dev --> Stage
    Stage --> Prod
```

## Development Environment Setup

### Prerequisites

```bash
# Required software versions
- Python 3.9+
- Node.js 18+
- Docker 24+
- Docker Compose 2.20+
- PostgreSQL 14+
- Redis 7+
- Git 2.30+
```

### Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/ai-assistant.git
cd ai-assistant

# 2. Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Install Node dependencies (for frontend)
cd frontend
npm install
cd ..

# 5. Copy environment configuration
cp .env.example .env
# Edit .env with your configuration

# 6. Initialize database
docker-compose up -d postgres redis
python scripts/init_db.py
python scripts/migrate.py

# 7. Start development servers
# Terminal 1: Start orchestrator
uvicorn orchestrator.main:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev

# Terminal 3: Start MCP server (example)
cd mcp_servers/example
uvicorn main:app --reload --port 8001
```

### Environment Configuration

```bash
# .env.example
# Application Configuration
APP_NAME=AI Assistant
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-change-in-production
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5000

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ai_assistant
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
REDIS_SSL=false

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.7

# OAuth Providers
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# MCP Servers
MCP_DISCOVERY_URLS=http://localhost:8001,http://localhost:8002
MCP_SERVICE_TOKEN=your-service-token

# Vector Database
VECTOR_DB_URL=http://localhost:6333
VECTOR_DB_API_KEY=

# Monitoring
PROMETHEUS_ENABLED=true
JAEGER_AGENT_HOST=localhost
JAEGER_AGENT_PORT=6831

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

## Docker Configuration

### Multi-stage Dockerfile for Orchestrator

```dockerfile
# orchestrator/Dockerfile
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Add local bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile for Frontend

```dockerfile
# frontend/Dockerfile
# Build stage
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy application code
COPY . .

# Build application
RUN npm run build

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Install Flask and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy built frontend assets from builder
COPY --from=builder /app/dist ./static
COPY --from=builder /app/templates ./templates

# Copy Flask application
COPY app.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:14-alpine
    container_name: ai_assistant_postgres
    environment:
      POSTGRES_DB: ai_assistant
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: ai_assistant_redis
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Orchestrator Service
  orchestrator:
    build:
      context: ./orchestrator
      dockerfile: Dockerfile
    container_name: ai_assistant_orchestrator
    env_file:
      - .env
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/ai_assistant
      REDIS_URL: redis://redis:6379/0
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./orchestrator:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  # Frontend Service
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: ai_assistant_frontend
    env_file:
      - .env
    environment:
      ORCHESTRATOR_URL: http://orchestrator:8000
      REDIS_URL: redis://redis:6379/0
    ports:
      - "5000:5000"
    depends_on:
      - orchestrator
    volumes:
      - ./frontend:/app

  # Example MCP Server
  mcp_example:
    build:
      context: ./mcp_servers/example
      dockerfile: Dockerfile
    container_name: ai_assistant_mcp_example
    env_file:
      - .env
    ports:
      - "8001:8001"
    depends_on:
      - redis

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: ai_assistant_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - orchestrator

  # Prometheus for monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: ai_assistant_prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    ports:
      - "9090:9090"

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: ai_assistant_grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3001:3000"
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: ai_assistant_network
```

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# k8s/00-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-assistant

---
# k8s/01-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-assistant-config
  namespace: ai-assistant
data:
  APP_NAME: "AI Assistant"
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  OPENAI_MODEL: "gpt-4"
  OPENAI_MAX_TOKENS: "4096"
  OPENAI_TEMPERATURE: "0.7"
  PROMETHEUS_ENABLED: "true"
  RATE_LIMIT_ENABLED: "true"
  RATE_LIMIT_PER_MINUTE: "60"
```

### Secrets Management

```yaml
# k8s/02-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-assistant-secrets
  namespace: ai-assistant
type: Opaque
stringData:
  SECRET_KEY: "your-secret-key"
  JWT_SECRET_KEY: "your-jwt-secret"
  DATABASE_URL: "postgresql://user:pass@postgres:5432/ai_assistant"
  REDIS_URL: "redis://redis:6379/0"
  OPENAI_API_KEY: "sk-..."
  GOOGLE_CLIENT_ID: "your-google-client-id"
  GOOGLE_CLIENT_SECRET: "your-google-client-secret"
```

### Orchestrator Deployment

```yaml
# k8s/deployments/orchestrator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestrator
  namespace: ai-assistant
  labels:
    app: orchestrator
    component: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: orchestrator
  template:
    metadata:
      labels:
        app: orchestrator
        component: backend
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: orchestrator
        image: ai-assistant/orchestrator:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: ai-assistant-config
        - secretRef:
            name: ai-assistant-secrets
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: tmp
        emptyDir: {}
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000

---
apiVersion: v1
kind: Service
metadata:
  name: orchestrator
  namespace: ai-assistant
  labels:
    app: orchestrator
spec:
  selector:
    app: orchestrator
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
```

### Frontend Deployment

```yaml
# k8s/deployments/frontend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: ai-assistant
  labels:
    app: frontend
    component: web
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
        component: web
    spec:
      containers:
      - name: frontend
        image: ai-assistant/frontend:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 5000
          name: http
        env:
        - name: ORCHESTRATOR_URL
          value: "http://orchestrator:8000"
        envFrom:
        - configMapRef:
            name: ai-assistant-config
        - secretRef:
            name: ai-assistant-secrets
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 20
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: ai-assistant
  labels:
    app: frontend
spec:
  selector:
    app: frontend
  ports:
  - port: 5000
    targetPort: 5000
    protocol: TCP
  type: ClusterIP
```

### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-assistant-ingress
  namespace: ai-assistant
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - ai-assistant.example.com
    - api.ai-assistant.example.com
    secretName: ai-assistant-tls
  rules:
  - host: ai-assistant.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 5000
  - host: api.ai-assistant.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: orchestrator
            port:
              number: 8000
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: orchestrator-hpa
  namespace: ai-assistant
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: orchestrator
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 2
        periodSeconds: 15
      selectPolicy: Max

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: frontend-hpa
  namespace: ai-assistant
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: frontend
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80
```

## Configuration Management

### Helm Chart Structure

```yaml
# helm/ai-assistant/Chart.yaml
apiVersion: v2
name: ai-assistant
description: AI Assistant with OpenAI and MCP
type: application
version: 1.0.0
appVersion: "1.0.0"

dependencies:
  - name: postgresql
    version: 12.1.2
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: 17.3.14
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
```

### Helm Values

```yaml
# helm/ai-assistant/values.yaml
global:
  image:
    registry: docker.io
    tag: latest
    pullPolicy: IfNotPresent

orchestrator:
  enabled: true
  replicaCount: 3
  image:
    repository: ai-assistant/orchestrator
  service:
    type: ClusterIP
    port: 8000
  resources:
    requests:
      cpu: 250m
      memory: 256Mi
    limits:
      cpu: 500m
      memory: 512Mi
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70

frontend:
  enabled: true
  replicaCount: 2
  image:
    repository: ai-assistant/frontend
  service:
    type: ClusterIP
    port: 5000
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 200m
      memory: 256Mi

postgresql:
  enabled: true
  auth:
    postgresPassword: postgres
    database: ai_assistant
  persistence:
    enabled: true
    size: 10Gi

redis:
  enabled: true
  auth:
    enabled: false
  persistence:
    enabled: true
    size: 5Gi

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: ai-assistant.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ai-assistant-tls
      hosts:
        - ai-assistant.example.com

config:
  openai:
    model: gpt-4
    maxTokens: 4096
    temperature: 0.7
  rateLimit:
    enabled: true
    perMinute: 60
    perHour: 1000
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy AI Assistant

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Orchestrator
      uses: docker/build-push-action@v4
      with:
        context: ./orchestrator
        push: true
        tags: |
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/orchestrator:latest
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/orchestrator:${{ github.sha }}
    
    - name: Build and push Frontend
      uses: docker/build-push-action@v4
      with:
        context: ./frontend
        push: true
        tags: |
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:latest
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup kubectl
      uses: azure/setup-kubectl@v3
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f k8s/
        kubectl set image deployment/orchestrator orchestrator=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/orchestrator:${{ github.sha }} -n ai-assistant
        kubectl set image deployment/frontend frontend=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:${{ github.sha }} -n ai-assistant
        kubectl rollout status deployment/orchestrator -n ai-assistant
        kubectl rollout status deployment/frontend -n ai-assistant
```

### GitLab CI Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_REGISTRY: registry.gitlab.com
  DOCKER_IMAGE_ORCHESTRATOR: $CI_REGISTRY_IMAGE/orchestrator
  DOCKER_IMAGE_FRONTEND: $CI_REGISTRY_IMAGE/frontend

test:
  stage: test
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pip install -r requirements-test.txt
    - pytest tests/ --cov=app --cov-report=xml
    - black --check .
    - flake8 .
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:orchestrator:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $DOCKER_IMAGE_ORCHESTRATOR:$CI_COMMIT_SHA ./orchestrator
    - docker tag $DOCKER_IMAGE_ORCHESTRATOR:$CI_COMMIT_SHA $DOCKER_IMAGE_ORCHESTRATOR:latest
    - docker push $DOCKER_IMAGE_ORCHESTRATOR:$CI_COMMIT_SHA
    - docker push $DOCKER_IMAGE_ORCHESTRATOR:latest

build:frontend:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $DOCKER_IMAGE_FRONTEND:$CI_COMMIT_SHA ./frontend
    - docker tag $DOCKER_IMAGE_FRONTEND:$CI_COMMIT_SHA $DOCKER_IMAGE_FRONTEND:latest
    - docker push $DOCKER_IMAGE_FRONTEND:$CI_COMMIT_SHA
    - docker push $DOCKER_IMAGE_FRONTEND:latest

deploy:staging:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_STAGING
    - kubectl set image deployment/orchestrator orchestrator=$DOCKER_IMAGE_ORCHESTRATOR:$CI_COMMIT_SHA -n ai-assistant-staging
    - kubectl set image deployment/frontend frontend=$DOCKER_IMAGE_FRONTEND:$CI_COMMIT_SHA -n ai-assistant-staging
    - kubectl rollout status deployment/orchestrator -n ai-assistant-staging
    - kubectl rollout status deployment/frontend -n ai-assistant-staging
  environment:
    name: staging
    url: https://staging.ai-assistant.example.com
  only:
    - develop

deploy:production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_PRODUCTION
    - kubectl set image deployment/orchestrator orchestrator=$DOCKER_IMAGE_ORCHESTRATOR:$CI_COMMIT_SHA -n ai-assistant
    - kubectl set image deployment/frontend frontend=$DOCKER_IMAGE_FRONTEND:$CI_COMMIT_SHA -n ai-assistant
    - kubectl rollout status deployment/orchestrator -n ai-assistant
    - kubectl rollout status deployment/frontend -n ai-assistant
  environment:
    name: production
    url: https://ai-assistant.example.com
  when: manual
  only:
    - main
```

## Monitoring & Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'orchestrator'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - ai-assistant
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: orchestrator
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

  - job_name: 'frontend'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - ai-assistant
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: frontend
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "AI Assistant Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{path}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      },
      {
        "title": "Token Usage",
        "targets": [
          {
            "expr": "sum(rate(openai_tokens_used_total[5m])) by (model)",
            "legendFormat": "{{model}}"
          }
        ]
      }
    ]
  }
}
```

## Scaling Strategies

### Vertical Pod Autoscaler

```yaml
# k8s/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: orchestrator-vpa
  namespace: ai-assistant
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: orchestrator
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: orchestrator
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: 2
        memory: 2Gi
```

### Cluster Autoscaler Configuration

```yaml
# k8s/cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  template:
    spec:
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/ai-assistant
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
```

## Backup & Recovery

### Database Backup

```yaml
# k8s/cronjob-backup.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: ai-assistant
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:14-alpine
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: password
            command:
            - /bin/sh
            - -c
            - |
              DATE=$(date +%Y%m%d_%H%M%S)
              pg_dump -h postgres -U postgres ai_assistant | gzip > /backup/ai_assistant_$DATE.sql.gz
              # Upload to S3
              aws s3 cp /backup/ai_assistant_$DATE.sql.gz s3://ai-assistant-backups/postgres/
              # Keep only last 30 days of backups
              find /backup -name "*.sql.gz" -mtime +30 -delete
            volumeMounts:
            - name: backup
              mountPath: /backup
          volumes:
          - name: backup
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

### Disaster Recovery Plan

```bash
#!/bin/bash
# scripts/disaster_recovery.sh

# Restore PostgreSQL from backup
restore_postgres() {
    local BACKUP_FILE=$1
    
    # Download backup from S3
    aws s3 cp s3://ai-assistant-backups/postgres/$BACKUP_FILE /tmp/
    
    # Restore database
    gunzip < /tmp/$BACKUP_FILE | psql -h $DATABASE_HOST -U $DATABASE_USER -d $DATABASE_NAME
    
    echo "Database restored from $BACKUP_FILE"
}

# Restore Redis from snapshot
restore_redis() {
    local SNAPSHOT_FILE=$1
    
    # Download snapshot from S3
    aws s3 cp s3://ai-assistant-backups/redis/$SNAPSHOT_FILE /data/dump.rdb
    
    # Restart Redis to load snapshot
    redis-cli SHUTDOWN NOSAVE
    redis-server --dir /data --dbfilename dump.rdb
    
    echo "Redis restored from $SNAPSHOT_FILE"
}

# Main recovery process
main() {
    echo "Starting disaster recovery..."
    
    # Restore PostgreSQL
    restore_postgres "ai_assistant_20240101_020000.sql.gz"
    
    # Restore Redis
    restore_redis "dump_20240101_020000.rdb"
    
    # Verify services
    kubectl get pods -n ai-assistant
    kubectl get svc -n ai-assistant
    
    echo "Disaster recovery completed"
}

main "$@"
```

## Production Checklist

### Pre-deployment Checklist

- [ ] **Security**
  - [ ] All secrets are stored in secret management system
  - [ ] TLS/SSL certificates are configured
  - [ ] Security headers are configured (HSTS, CSP, etc.)
  - [ ] Rate limiting is enabled
  - [ ] Input validation is implemented
  - [ ] SQL injection prevention is in place
  - [ ] XSS protection is enabled

- [ ] **Configuration**
  - [ ] Environment variables are properly set
  - [ ] Database connections are configured
  - [ ] Redis connections are configured
  - [ ] OpenAI API keys are set
  - [ ] OAuth providers are configured
  - [ ] CORS settings are correct

- [ ] **Monitoring**
  - [ ] Prometheus metrics are exposed
  - [ ] Grafana dashboards are created
  - [ ] Alerts are configured
  - [ ] Log aggregation is set up
  - [ ] Error tracking is enabled
  - [ ] APM is configured

- [ ] **Performance**
  - [ ] Database indexes are created
  - [ ] Caching is properly configured
  - [ ] CDN is set up for static assets
  - [ ] Connection pooling is configured
  - [ ] Autoscaling is configured
  - [ ] Resource limits are set

- [ ] **Reliability**
  - [ ] Health checks are implemented
  - [ ] Readiness probes are configured
  - [ ] Circuit breakers are in place
  - [ ] Retry logic is implemented
  - [ ] Graceful shutdown is handled
  - [ ] Backup strategy is defined

- [ ] **Testing**
  - [ ] Unit tests pass
  - [ ] Integration tests pass
  - [ ] Load testing completed
  - [ ] Security scanning completed
  - [ ] Penetration testing performed
  - [ ] Disaster recovery tested

### Post-deployment Verification

```bash
#!/bin/bash
# scripts/verify_deployment.sh

# Check pod status
echo "Checking pod status..."
kubectl get pods -n ai-assistant

# Check service endpoints
echo "Checking service endpoints..."
kubectl get endpoints -n ai-assistant

# Test health endpoints
echo "Testing health endpoints..."
curl -f https://ai-assistant.example.com/health
curl -f https://api.ai-assistant.example.com/health

# Check metrics
echo "Checking Prometheus metrics..."
curl -s https://api.ai-assistant.example.com/metrics | grep http_requests_total

# Verify database connection
echo "Verifying database connection..."
kubectl exec -it deployment/orchestrator -n ai-assistant -- python -c "
import psycopg2
conn = psycopg2.connect('$DATABASE_URL')
print('Database connection successful')
"

# Check Redis connection
echo "Checking Redis connection..."
kubectl exec -it deployment/orchestrator -n ai-assistant -- python -c "
import redis
r = redis.from_url('$REDIS_URL')
r.ping()
print('Redis connection successful')
"

echo "Deployment verification completed"
```

---

**Document Version**: 1.0.0  
**Last Updated**: December 2024  
**Deployment Standard**: Kubernetes 1.21+