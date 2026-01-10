# Zeta-Life Docker Image
# =======================
# Provides reproducible environment for IPUESA experiments

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash zeta
USER zeta
WORKDIR /home/zeta/app

# Copy project files
COPY --chown=zeta:zeta . .

# Install Python dependencies
RUN pip install --user -e ".[full]"

# Set PATH for user-installed packages
ENV PATH="/home/zeta/.local/bin:${PATH}"

# Default command: run quickstart demo
CMD ["python", "demos/quickstart.py"]

# Labels
LABEL maintainer="IPUESA Research" \
      version="0.1.0" \
      description="Zeta-Life: Artificial Consciousness through Riemann Zeta Function"
