# Stage 1: Builder
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies and the package itself
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Stage 2: Final
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies (python3 and libraries needed for opencv/pillow if any)
RUN apt-get update && apt-get install -y \
    python3 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy src directory as requested (though code is also installed in venv)
COPY src /app/src

# Environment variables
ENV MODULE_NAME="nutrition_detector.api.app"
ENV VARIABLE_NAME="app"
ENV PORT="8000"
# Ensure python can find the src if we wanted to run from there (though venv takes precedence)
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Run the application
CMD ["uvicorn", "nutrition_detector.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
