#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t) OMP_NUM_THREADS="$2"; shift 2 ;;
        *)  args+=("$1"); shift ;;
    esac
done

if [[ -n "$OMP_NUM_THREADS" ]]; then
    export OMP_NUM_THREADS
fi

exec "$SCRIPT_DIR/ct_c" "${args[@]}"
