# r3LAY Docker Image
# Base image for CPU-only deployments (Ollama backend)
FROM python:3.12-slim

# Unbuffered Python output for proper Docker logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY pyproject.toml README.md ./

# Install r3lay dependencies (no editable install for production)
RUN pip install --no-cache-dir .

# Copy application code
COPY r3lay/ ./r3lay/

# Create mount points and config directory
RUN mkdir -p /project /root/.r3lay/models

# Health check - verify Python can import r3lay
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import r3lay" || exit 1

# Default environment variables
ENV HF_HOME=/root/.cache/huggingface
ENV R3LAY_GGUF_FOLDER=/root/.r3lay/models

WORKDIR /project

ENTRYPOINT ["python", "-m", "r3lay.app"]
CMD ["/project"]
