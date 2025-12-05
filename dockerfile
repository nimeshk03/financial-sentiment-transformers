# Use official Python slim image - smaller than full image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for some Python packages
# - gcc: C compiler for packages with C extensions
# - git: needed by some pip packages
# --no-install-recommends: skip optional packages to keep image small
# rm -rf /var/lib/apt/lists/*: clean up apt cache to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (Docker layer caching optimization)
# If requirements.txt hasn't changed, Docker reuses cached layer
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: don't store pip cache (smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (this layer rebuilds when code changes)
COPY src/ ./src/
COPY tests/ ./tests/
COPY app/ ./app/

# Create directories for mounted volumes
RUN mkdir -p data models outputs

# Expose Streamlit port
EXPOSE 8501

# Default command: run Streamlit dashboard
# --server.address=0.0.0.0: listen on all interfaces (required in Docker)
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]