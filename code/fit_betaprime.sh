#!/bin/bash
cd "$(dirname "$0")"
python3 fit.py --exact-dir ../data/exact_distributions/ --cdf-dir ../data/cdf_exact/ --distribution betaprime -o ../data/fitted/cvm_betaprime/ -v
