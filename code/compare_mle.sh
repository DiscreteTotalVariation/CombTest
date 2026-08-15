#!/bin/bash
# Compare exact DTV distribution with MLE-fitted approximation(s)
#
# Usage examples:
#   ./compare_mle.sh -N 100 -n 5 -d normal
#   ./compare_mle.sh -N 100 -n 5 -d normal -d beta -d gamma
#   ./compare_mle.sh -N 100 -n 5 -d beta --cdf -o comparison.png
#   ./compare_mle.sh -N 100 -n 5 -d lognormal -f -o out.png
#
# Options:
#   -N          N parameter
#   -n          n parameter
#   -d          distribution name (can be repeated for multiple)
#   -f          force re-fitting even if fitted parameters exist
#   -o FILE     save plot to output image file
#   --cdf       plot CDF instead of PDF/PMF
#   --pdf       plot PDF/PMF (default)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/compare.py" --mle "$@"
