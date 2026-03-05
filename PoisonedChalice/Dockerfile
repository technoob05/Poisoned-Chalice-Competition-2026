FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install small set of system dependencies needed for some Python wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Copy and install Python dependencies
COPY pyproject.toml /app/pyproject.toml
RUN uv lock && uv sync --frozen

# Copy project files
COPY . /app

# Default entrypoint runs the experiment; pass CLI args when running the container
ENTRYPOINT ["uv", "run", "python"]