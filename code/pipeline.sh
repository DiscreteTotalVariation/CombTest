#!/bin/bash
# Regenerate all experimental data and paper figures.
# Run from the code/ directory (cd is automatic).
#
# The full pipeline is:
#   0. Exact DTV distributions (ct.py or ct_save C binary) — very slow, days
#   0b. Exact CDFs (calculate_cdf_values.py)
#   0c. Fit distributions (fit.py for each distribution) — slow, hours
#   1-7. Experiments and plots
#
# Usage:
#   ./pipeline.sh              # experiments + plots (assumes exact/fitted data exists)
#   ./pipeline.sh --plots-only # only regenerate plots from existing data
#   ./pipeline.sh --from-scratch # full pipeline including fitting (very slow)
#   ./pipeline.sh --from-scratch --use-c # use compiled C binary for exact distributions
#   ./pipeline.sh --from-scratch -a # download archives instead of computing exact data

set -e
cd "$(dirname "$0")"

PLOTS_ONLY=false
FROM_SCRATCH=false
USE_C=false
ARCHIVE=false
CROP="--crop"

for arg in "$@"; do
    case "$arg" in
        --plots-only) PLOTS_ONLY=true ;;
        --from-scratch) FROM_SCRATCH=true ;;
        --use-c) USE_C=true ;;
        -a|--archive|--archived) ARCHIVE=true ;;
    esac
done

echo "========================================="
echo " Regenerating all data and figures"
echo " plots-only: $PLOTS_ONLY"
echo " from-scratch: $FROM_SCRATCH"
echo " use-c: $USE_C"
echo " archive: $ARCHIVE"
echo "========================================="

# -------------------------------------------------------
# 0. Download archives (with -a/--archive/--archived)
#    Downloads precomputed exact distributions, CDFs,
#    and fitted beta/gamma parameters from Dropbox,
#    replacing steps 0a, 0b, and 0c.
# -------------------------------------------------------
if [ "$ARCHIVE" = true ]; then
    echo ""
    echo "=== Step 0: Downloading precomputed data from archives ==="
    python3 download_data.py
fi

# -------------------------------------------------------
# 0. Precomputation (only with --from-scratch)
#    These produce ../data/exact_distributions/, ../data/cdf_exact/,
#    ../data/fitted/cvm_beta/, etc.
#    Required by all subsequent steps but very slow.
# -------------------------------------------------------
if [ "$FROM_SCRATCH" = true ]; then
    if [ "$ARCHIVE" = false ]; then
        echo ""
        echo "=== Step 0a: Exact DTV distributions ==="

        if [ "$USE_C" = true ]; then
            echo "Using compiled C binary (ct_save)."
            if [ ! -x ./ct_save ]; then
                echo "Compiling ct_save.c..."
                gcc -O3 -fopenmp -o ct_save ct_save.c -lgmp -lm
            fi
            echo "Computing N=1..500, n=1..500..."
            ./ct_save -N 500 -n 500 -u -o ../data/exact_distributions
            echo "Computing N=1..625, n=1..10..."
            ./ct_save -N 625 -n 10 -u -o ../data/exact_distributions
        else
            echo "Using Python (ct.py). WARNING: This takes days for large N."
            echo "Computing N=1..500, n=1..500 and N=1..625, n=1..10."
            python3 ct.py -N 500 -n 500 -u
            python3 ct.py -N 625 -n 10 -u
        fi

        echo ""
        echo "=== Step 0b: Exact CDFs ==="
        python3 calculate_cdf_values.py
    fi

    if [ "$ARCHIVE" = false ]; then
        echo ""
        echo "=== Step 0c: Fitting distributions ==="
        # Fits beta and gamma by default (the two used in the paper).
        # Use --all to fit all 29 distributions (not required for the paper).
        ./fit_all.sh
        # ./fit_all.sh --all  # uncomment to fit all 29 distributions
    else
        echo ""
        echo "=== Step 0c: Skipping fitting (beta/gamma included in archives) ==="
    fi
fi

# -------------------------------------------------------
# 1. Fig 1a: DTV distribution histogram (N=20, n=4)
# -------------------------------------------------------
echo ""
echo "=== Fig 1a: DTV distribution histogram (N=20, n=4) ==="
python3 plot_dtv_distribution_histogram.py -N 20 -n 4 \
    --xlim-upper 26 $CROP \
    -o ../paper/img/dtv_distribution_histogram_N_20_n_4.png

# -------------------------------------------------------
# 2. Fig 1b: DTV distribution with beta/gamma overlay (N=50, n=10)
# -------------------------------------------------------
echo ""
echo "=== Fig 1b: DTV + beta/gamma overlay (N=50, n=10) ==="
python3 -c "
from generate_beta_figures import generate_overlay_plot
generate_overlay_plot(xlim_upper=55, crop=True)
"

# -------------------------------------------------------
# 3. Fig 2: Beta approximation KS heatmaps (full + zoomed)
# -------------------------------------------------------
echo ""
echo "=== Fig 2a: Beta KS heatmap full (N,n from 2 to 500) ==="
python3 -c "
from generate_beta_figures import generate_heatmap
generate_heatmap('beta', N_max=500,
                 output_name='../paper/img/dtv_to_beta_full.png',
                 crop=True)
"
echo "=== Fig 2b: Beta KS heatmap zoomed (N>=100, n>=100) ==="
python3 -c "
from generate_beta_figures import generate_heatmap
generate_heatmap('beta', N_max=500, N_min=100, n_min=100,
                 vmax=0.015,
                 output_name='../paper/img/dtv_to_beta_zoomed.png',
                 crop=True)
"
echo "=== Fig 2c: Beta KS diagonal (N=n) ==="
python3 -c "
from generate_beta_figures import generate_diagonal_ks_plot
generate_diagonal_ks_plot('beta', N_max=500,
                          output_name='../paper/img/ks_diagonal_beta.png',
                          crop=True, compare_dist='gamma')
"

# -------------------------------------------------------
# 4. Fig 3: MC convergence (n=10, N=1..600)
#    Data generation is slow (~hours). Skip with --plots-only.
# -------------------------------------------------------
echo ""
echo "=== Fig 3: MC convergence ==="
if [ "$PLOTS_ONLY" = false ]; then
    echo "--- Generating MC convergence data ---"
    python3 experiment_mc_convergence_n10.py \
        --N-max 600 --n-repeats 100 --seed 13 \
        --output-dir ../data/mc_convergence_results
fi
echo "--- Plotting ---"
python3 plot_mc_convergence.py \
    --input-dir ../data/mc_convergence_results \
    --output ../paper/img/mc_convergence.png

# -------------------------------------------------------
# 5. Fig 4a: Mean p-values (d=0, banker's rounding, 1 decimal)
#    Data generation is slow (~hours). Skip with --plots-only.
# -------------------------------------------------------
echo ""
echo "=== Fig 4a: Mean p-values (d=0) ==="
if [ "$PLOTS_ONLY" = false ]; then
    echo "--- Generating data ---"
    python3 generate_mean_p_values.py -d 0 --step 100 --N-max 3000 -p -g --approx gamma \
        -o ../data/mean_p_values_d_0_step_100.txt
fi
echo "--- Plotting ---"
python3 plot_mean_p_values.py -d 0 --step 100 \
    --lower-N 10 --upper-N 3000 $CROP \
    -i ../data/mean_p_values_d_0_step_100.txt \
    -o ../paper/img/mean_p_values_d_0_step_100.png

# -------------------------------------------------------
# 6. Fig 4b: Mean p-values (d=1, banker's rounding, 2 decimals)
#    Data generation is slow (~many hours). Skip with --plots-only.
# -------------------------------------------------------
echo ""
echo "=== Fig 4b: Mean p-values (d=1) ==="
if [ "$PLOTS_ONLY" = false ]; then
    echo "--- Generating data ---"
    python3 generate_mean_p_values.py -d 1 --step 1000 --N-max 160000 -p -g --approx gamma \
        -o ../data/mean_p_values_d_1_step_1000.txt
fi
echo "--- Plotting ---"
python3 plot_mean_p_values.py -d 1 --step 1000 \
    --lower-N 10 --upper-N 160000 $CROP \
    -i ../data/mean_p_values_d_1_step_1000.txt \
    -o ../paper/img/mean_p_values_d_1_step_1000.png

# -------------------------------------------------------
# 7. Fig 5: ADC alternating DNL power comparison
# -------------------------------------------------------
echo ""
echo "=== Fig 5: ADC alternating DNL ==="
if [ "$PLOTS_ONLY" = false ]; then
    echo "--- Generating data ---"
    python3 experiment_adc_dnl_data.py --approx gamma -o ../data/adc_dnl_results.json
fi
echo "--- Plotting ---"
python3 plot_adc_dnl.py \
    -i ../data/adc_dnl_results.json $CROP \
    -o ../paper/img/adc_dnl_power_comparison.png

# -------------------------------------------------------
# 8. ADC n=256 experiment (reproduces paper numbers)
# -------------------------------------------------------
echo ""
echo "=== ADC n=256 experiment ==="
if [ "$PLOTS_ONLY" = false ]; then
    echo "--- Generating n=256 data ---"
    python3 experiment_adc_dnl_n256.py --approx gamma -o ../data/adc_dnl_all_results.json
fi

# -------------------------------------------------------
# 9. Beta approximation quality statistics (reproduces Section III numbers)
# -------------------------------------------------------
echo ""
echo "=== Beta approximation quality statistics ==="
if [ "$PLOTS_ONLY" = false ]; then
    python3 compute_beta_match_stats.py --N-max 500
fi

# -------------------------------------------------------
echo ""
echo "========================================="
echo " Done. All figures saved to ../paper/img/"
echo "========================================="
