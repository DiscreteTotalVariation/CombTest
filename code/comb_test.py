#!/usr/bin/env python3
"""Comb Test implementation.

Provides both exact and gamma-approximated p-values for the DTV-based
uniformity test (Comb Test). The exact p-values use precomputed distributions
from the exact_distributions/ and cdf_exact/ directories. The gamma
approximation uses Monte Carlo sampling to estimate parameters.

Usage:
    from comb_test import comb_test, comb_test_gamma
    p_exact = comb_test(histogram)
    p_gamma = comb_test_gamma(histogram)
"""

import os
import numpy as np
from math import comb as math_comb
from scipy import stats

EXACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/exact_distributions")
CDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/cdf_exact")


def dtv(histogram):
    """Compute the discrete total variation of a histogram."""
    return sum(abs(int(histogram[i]) - int(histogram[i - 1])) for i in range(1, len(histogram)))


def load_exact_cdf(N, n):
    """Load precomputed exact CDF for given N, n.

    Returns (dtv_values, cdf_values) or None if not available.
    """
    path = os.path.join(CDF_DIR, f"N_{N}_n_{n}.txt")
    if not os.path.isfile(path):
        return None
    dtv_vals = []
    cdf_vals = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                dtv_vals.append(int(parts[0]))
                cdf_vals.append(float(parts[1]))
    return dtv_vals, cdf_vals


def comb_test(histogram):
    """Perform the Comb Test using the exact DTV distribution.

    Args:
        histogram: array-like of bin counts (must sum to N, length n)

    Returns:
        p-value (float), or None if exact distribution not available
    """
    histogram = list(map(int, histogram))
    N = sum(histogram)
    n = len(histogram)
    d = dtv(histogram)

    if d == 0:
        return 1.0

    result = load_exact_cdf(N, n)
    if result is None:
        return None

    dtv_vals, cdf_vals = result
    # p-value = 1 - CDF(d - 1) = P(DTV >= d)
    # Find the CDF value at d-1
    p_value = 1.0
    for v, c in zip(dtv_vals, cdf_vals):
        if v == d - 1:
            p_value = 1.0 - c
            break
        elif v >= d:
            # d-1 is not in the CDF (has 0 probability mass), use the last CDF value before d
            break
        p_value = 1.0 - c

    return max(0.0, min(1.0, p_value))


def estimate_gamma_params(N, n, num_samples=100000, seed=None):
    """Estimate gamma distribution parameters for DTV under uniformity.

    Generates random uniform histograms and computes their DTVs,
    then fits a gamma distribution using method of moments.

    Returns (shape, loc, scale) for scipy.stats.gamma.
    """
    rng = np.random.default_rng(seed)
    probs = np.full(n, 1.0 / n)
    dtvs = np.empty(num_samples)
    for i in range(num_samples):
        hist = rng.multinomial(N, probs)
        dtvs[i] = dtv(hist)

    # Fit gamma with loc=0 using method of moments
    mean_dtv = np.mean(dtvs)
    var_dtv = np.var(dtvs)
    if var_dtv < 1e-10:
        return 1.0, 0.0, max(mean_dtv, 1e-10)
    scale = var_dtv / mean_dtv
    shape = mean_dtv / scale
    return shape, 0.0, scale


def comb_test_gamma(histogram, num_samples=100000, gamma_params=None, seed=None):
    """Perform the Comb Test using a gamma distribution approximation.

    Args:
        histogram: array-like of bin counts
        num_samples: number of Monte Carlo samples for parameter estimation
        gamma_params: optional pre-computed (shape, loc, scale) tuple
        seed: optional random seed for reproducibility

    Returns:
        p-value (float)
    """
    histogram = list(map(int, histogram))
    N = sum(histogram)
    n = len(histogram)
    d = dtv(histogram)

    if d == 0:
        return 1.0

    if gamma_params is None:
        gamma_params = estimate_gamma_params(N, n, num_samples, seed=seed)

    shape, loc, scale = gamma_params
    # Use continuity correction: P(DTV >= d) ≈ 1 - Gamma_CDF(d - 0.5)
    p_value = 1.0 - stats.gamma.cdf(d - 0.5, a=shape, loc=loc, scale=scale)
    return max(0.0, min(1.0, p_value))


def pearson_chi2_test(histogram):
    """Perform Pearson's chi-squared test for uniformity.

    Returns p-value.
    """
    histogram = np.array(histogram, dtype=float)
    N = histogram.sum()
    n = len(histogram)
    expected = N / n
    chi2_stat = np.sum((histogram - expected) ** 2 / expected)
    p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=n - 1)
    return p_value


if __name__ == "__main__":
    # Quick demo
    import argparse
    parser = argparse.ArgumentParser(description="Comb Test demo")
    parser.add_argument("histogram", nargs="+", type=int,
                        help="Histogram bin counts")
    args = parser.parse_args()

    h = args.histogram
    print(f"Histogram: {h}")
    print(f"N={sum(h)}, n={len(h)}, DTV={dtv(h)}")

    p_exact = comb_test(h)
    if p_exact is not None:
        print(f"Comb Test (exact):  p = {p_exact:.6f}")
    else:
        print("Comb Test (exact):  not available for this N, n")

    p_gamma = comb_test_gamma(h)
    print(f"Comb Test (gamma):  p = {p_gamma:.6f}")

    p_chi2 = pearson_chi2_test(h)
    print(f"Pearson chi-sq:     p = {p_chi2:.6f}")
