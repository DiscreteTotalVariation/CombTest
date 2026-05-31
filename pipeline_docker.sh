#!/bin/bash
# Run the Comb Test pipeline inside a Docker container.
# Output directories (data/ and paper/img/) are bind-mounted so results
# persist on the host. The container is removed after completion.
#
# Usage:
#   ./pipeline_docker.sh                          # experiments + plots
#   ./pipeline_docker.sh --plots-only             # plots only
#   ./pipeline_docker.sh --from-scratch           # full pipeline (Python)
#   ./pipeline_docker.sh --from-scratch --use-c   # full pipeline (C binary)
#   ./pipeline_docker.sh --from-scratch -a        # download archives + fit + experiments

set -e
cd "$(dirname "$0")"

IMAGE="comb-test"

# Build if image doesn't exist
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image '$IMAGE' not found. Building..."
    bash build.sh
fi

# Create output directories if they don't exist
mkdir -p data paper/img

# Run the pipeline in a disposable container.
# Mount data/ and paper/ as volumes so output persists on the host.
docker run --rm \
    -v "$(pwd)/data:/workspace/data" \
    -v "$(pwd)/paper:/workspace/paper" \
    "$IMAGE" "$@"

# Fix file permissions: Docker may create files as root.
# Change ownership to the current user.
if [ "$(id -u)" -ne 0 ]; then
    echo "Fixing file permissions..."
    sudo chown -R "$(id -u):$(id -g)" data/ paper/img/ 2>/dev/null || \
        echo "Warning: could not fix permissions. Files may be owned by root."
fi

echo "Pipeline complete. Results in data/ and paper/img/"
