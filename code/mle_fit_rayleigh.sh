#!/bin/bash
cd "$(dirname "$0")"
python3 mle_fit.py --exact-dir ../data/exact_distributions/ --distribution rayleigh -o ../data/fitted/mle_rayleigh/ -v
