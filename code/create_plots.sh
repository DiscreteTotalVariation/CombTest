#!/bin/bash
# Generate all plots used in the paper.
# Run from the code/ directory (cd is automatic).

set -e
cd "$(dirname "$0")"

echo "=== Fig 1a: DTV distribution histogram (N=20, n=4) ==="
python3 plot_dtv_distribution_histogram.py -N 20 -n 4 -o ../paper/img/dtv_distribution_histogram_N_20_n_4.png

echo ""
echo "=== Fig 1b: DTV distribution with beta/gamma overlay (N=50, n=10) ==="
python3 -c "from generate_beta_figures import generate_overlay_plot; generate_overlay_plot()"

echo ""
echo "=== Fig 2: Beta approximation KS statistic heatmap (2..450) ==="
echo "--- Step 1: Preparing heatmap data (incremental) ---"
python3 prepare_heatmap_data.py --N-max 450 --dist beta -o ../data/heatmap_ks_beta_450.npy
echo "--- Step 2: Plotting ---"
python3 plot_heatmap.py -i ../data/heatmap_ks_beta_450.npy -o ../paper/img/dtv_to_beta.png

echo ""
echo "=== Fig 3a: Mean p-values (d=0, step=100) ==="
if [ -f "../data/mean_p_values_d_0_step_100.txt" ]; then
    python3 plot_mean_p_values.py -d 0 --step 100 --lower-N 2 --upper-N 3000 -o ../paper/img/mean_p_values_d_0_step_100.png
else
    echo "SKIPPED: ../data/mean_p_values_d_0_step_100.txt not found"
fi

echo ""
echo "=== Fig 3b: Mean p-values (d=1, step=1000) ==="
if [ -f "../data/mean_p_values_d_1_step_1000.txt" ]; then
    python3 plot_mean_p_values.py -d 1 --step 1000 --lower-N 2 --upper-N 160000 -o ../paper/img/mean_p_values_d_1_step_1000.png
else
    echo "SKIPPED: ../data/mean_p_values_d_1_step_1000.txt not found"
fi

echo ""
echo "=== Fig 4: ADC alternating DNL power comparison ==="
echo "--- Step 1: Generating experiment data ---"
python3 experiment_adc_dnl_data.py -o ../data/adc_dnl_results.json
echo "--- Step 2: Plotting ---"
python3 plot_adc_dnl.py -i ../data/adc_dnl_results.json -o ../paper/img/adc_dnl_power_comparison.png

echo ""
echo "=== Done ==="
