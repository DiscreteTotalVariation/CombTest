#!/bin/bash
# Fit distributions to exact DTV data using CvM minimization.
#
# Usage:
#   ./fit_all.sh              # fit beta and gamma only (used in the paper)
#   ./fit_all.sh --all        # fit all 29 supported distributions
#
# Additional arguments are forwarded to fit.py (e.g., -p 8 for 8 processes).

set -e
cd "$(dirname "$0")"

EXACT_DIR=../data/exact_distributions/
CDF_DIR=../data/cdf_exact/
FIT_DIR=../data/fitted

FIT_ALL=false
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --all) FIT_ALL=true ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

# Beta and gamma are always fitted (used in the paper).
# Beta uses multi-start optimization (--restarts 5) to avoid local minima.
echo "=== Fitting beta distribution (multi-start, restarts=5) ==="
python3 fit.py --exact-dir "$EXACT_DIR" --cdf-dir "$CDF_DIR" \
    --distribution beta -o "$FIT_DIR/cvm_beta/" --restarts 5 -v "${EXTRA_ARGS[@]}"

echo ""
echo "=== Fitting gamma distribution (multi-start, restarts=5) ==="
python3 fit.py --exact-dir "$EXACT_DIR" --cdf-dir "$CDF_DIR" \
    --distribution gamma -o "$FIT_DIR/cvm_gamma/" --restarts 5 -v "${EXTRA_ARGS[@]}"

if [ "$FIT_ALL" = true ]; then
    # Fit all remaining 27 distributions (not required for the paper,
    # but useful for the full comparison in the repository).
    ALL_DISTS=(
        normal lognormal poisson binomial exponential weibull chi2
        nbinom uniform invgauss gengamma nakagami triang betaprime
        johnsonsb arcsine powerlaw bradford burr burr12 fisk genpareto
        genextreme rayleigh rice pareto f
    )
    for dist in "${ALL_DISTS[@]}"; do
        echo ""
        echo "=== Fitting $dist distribution ==="
        python3 fit.py --exact-dir "$EXACT_DIR" --cdf-dir "$CDF_DIR" \
            --distribution "$dist" -o "$FIT_DIR/cvm_$dist/" -v "${EXTRA_ARGS[@]}"
    done
fi

echo ""
echo "=== Fitting complete ==="
