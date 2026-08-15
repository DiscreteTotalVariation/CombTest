#!/bin/bash
cd "$(dirname "$0")"
python3 mle_fit.py --exact-dir ../data/exact_distributions/ --distribution uniform -o ../data/fitted/mle_uniform/ -v
