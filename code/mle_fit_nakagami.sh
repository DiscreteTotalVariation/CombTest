#!/bin/bash
cd "$(dirname "$0")"
python3 mle_fit.py --exact-dir ../data/exact_distributions/ --distribution nakagami -o ../data/fitted/mle_nakagami/ -v
