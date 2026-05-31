#!/bin/bash
cd "$(dirname "$0")"
python3 fit.py --exact-dir ../data/exact_distributions/ --cdf-dir ../data/cdf_exact/ --distribution beta -o ../data/fitted/cvm_beta/ --restarts 5 -v "$@"
