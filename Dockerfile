# ─── Healthcare AI MLOps System ───────────────────────────────────
# Multi-stage Docker build for production FastAPI service

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim AS production

WORKDIR /app

# Security: Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY config/ ./config/
COPY monitoring/ ./monitoring/

# Create data directories
RUN mkdir -p data/raw data/processed data/drift && \
    chown -R mlops:mlops /app

# Switch to non-root user
USER mlops

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Run FastAPI with Uvicorn
CMD ["uvicorn", "api.app:", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
