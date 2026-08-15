#!/bin/bash
# Generate and plot mean p-values for the banker's rounding experiment.
# Includes both Pearson's chi-squared and G-test.
# Safe to interrupt and resume — the generation script continues from existing data.

set -e
cd "$(dirname "$0")"

echo "=== Generating mean p-values (d=0, step=100) ==="
python3 generate_mean_p_values.py -d 0 --step 100 --N-max 3000 -p -g \
    -o ../data/mean_p_values_d_0_step_100.txt

echo ""
echo "=== Generating mean p-values (d=1, step=1000) ==="
python3 generate_mean_p_values.py -d 1 --step 1000 --N-max 160000 -p -g \
    -o ../data/mean_p_values_d_1_step_1000.txt

echo ""
echo "=== Plotting (d=0) ==="
python3 plot_mean_p_values.py -d 0 --step 100 \
    -i ../data/mean_p_values_d_0_step_100.txt \
    -o ../paper/img/mean_p_values_d_0_step_100.png

echo ""
echo "=== Plotting (d=1) ==="
python3 plot_mean_p_values.py -d 1 --step 1000 --upper-N 160000 \
    -i ../data/mean_p_values_d_1_step_1000.txt \
    -o ../paper/img/mean_p_values_d_1_step_1000.png

echo ""
echo "Done."
