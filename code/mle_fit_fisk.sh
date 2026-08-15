#!/bin/bash
cd "$(dirname "$0")"
python3 mle_fit.py --exact-dir ../data/exact_distributions/ --distribution fisk -o ../data/fitted/mle_fisk/ -v
