# Multi-stage production Dockerfile for ResearchMind AI backend
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create application user and runtime data directories
RUN groupadd -r researchmind && useradd -r -g researchmind -d /app -s /sbin/nologin researchmind \
    && mkdir -p /app/data/vectors \
    && chown -R researchmind:researchmind /app

# Copy application source code
COPY --chown=researchmind:researchmind . /app

# Switch to non-root user for security
USER researchmind

# Expose backend API port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Production server entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
