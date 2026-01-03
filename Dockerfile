FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install huggingface-cli
RUN pip install --no-cache-dir huggingface_hub

# Copy application
COPY . /app/

# Install r3lay
RUN pip install --no-cache-dir -e .

# Create project mount point
RUN mkdir -p /project

# Default HF cache location
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /project

ENTRYPOINT ["python", "-m", "r3lay.app"]
CMD ["/project"]
