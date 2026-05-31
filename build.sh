#!/bin/bash
# Build the Docker image for the Comb Test pipeline.
set -e
cd "$(dirname "$0")"
docker build -t comb-test .
echo "Docker image 'comb-test' built successfully."
