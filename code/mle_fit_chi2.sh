#!/bin/bash
cd "$(dirname "$0")"
python3 mle_fit.py --exact-dir ../data/exact_distributions/ --distribution chi2 -o ../data/fitted/mle_chi2/ -v
