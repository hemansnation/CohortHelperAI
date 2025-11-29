# CohortRAG Engine - Production Docker Image
# Multi-stage build for optimized production deployment

# =============================================================================
# Stage 1: Builder - Install dependencies and build package
# =============================================================================
FROM python:3.11-slim as builder

# Set build arguments
ARG INSTALL_DEV=false

# Set environment variables for build
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy package files
COPY pyproject.toml requirements.txt ./
COPY cohortrag_engine/ ./cohortrag_engine/
COPY LICENSE README.md ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install package dependencies
RUN pip install -r requirements.txt

# Install development dependencies if needed
RUN if [ "$INSTALL_DEV" = "true" ] ; then \
    pip install black isort pytest pytest-cov flake8 mypy ; \
    fi

# Build the package
RUN pip install -e .

# =============================================================================
# Stage 2: Production - Minimal runtime image
# =============================================================================
FROM python:3.11-slim as production

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PATH="/home/app/.local/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=app:app cohortrag_engine/ ./cohortrag_engine/
COPY --chown=app:app LICENSE README.md ./
COPY --chown=app:app pyproject.toml ./

# Create necessary directories
RUN mkdir -p /app/data /app/chroma_db /app/logs \
    && chown -R app:app /app

# Install the package in production mode
RUN pip install -e .

# Switch to app user
USER app

# Create volume mount points
VOLUME ["/app/data", "/app/chroma_db", "/app/logs"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "from cohortrag_engine import CohortRAGEngine; engine = CohortRAGEngine(); print('Health check passed')" || exit 1

# Default command
CMD ["cohortrag"]

# =============================================================================
# Stage 3: Development - Full development environment
# =============================================================================
FROM builder as development

# Install development tools
RUN pip install black isort pytest pytest-cov flake8 mypy pre-commit jupyter

# Install git for pre-commit hooks
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set development environment variables
ENV COHORTRAG_ENV=development
ENV DEBUG=true

# Switch to app user
USER app

# Set working directory
WORKDIR /app

# Default development command
CMD ["bash"]

# =============================================================================
# Stage 4: Jupyter - Development with Jupyter notebook support
# =============================================================================
FROM development as jupyter

# Expose Jupyter port
EXPOSE 8888

# Install Jupyter extensions
RUN pip install jupyter jupyterlab notebook ipywidgets

# Create Jupyter config directory
RUN mkdir -p /home/app/.jupyter

# Jupyter configuration
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/app/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /home/app/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /home/app/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = False" >> /home/app/.jupyter/jupyter_notebook_config.py

# Start Jupyter by default
CMD ["jupyter", "lab", "--allow-root"]