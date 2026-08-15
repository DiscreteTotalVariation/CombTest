#!/bin/bash
cd "$(dirname "$0")"
python3 mle_fit.py --exact-dir ../data/exact_distributions/ --distribution burr12 -o ../data/fitted/mle_burr12/ -v
