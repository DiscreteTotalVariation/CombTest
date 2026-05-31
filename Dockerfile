FROM python:3.11-slim

# Install system dependencies for C compilation (GMP, OpenMP) and matplotlib backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    libgmp-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Use non-interactive matplotlib backend
ENV MPLBACKEND=Agg

# Copy code
COPY code/ ./code/

# Compile the C binary
RUN cd code && gcc -O3 -fopenmp -o ct_save ct_save.c -lgmp -lm

# Create data and paper output directories
RUN mkdir -p data paper/img

# Default: run the pipeline
ENTRYPOINT ["bash", "code/pipeline.sh"]
