#!/usr/bin/env python3
"""Analyze which continuous distribution best fits the DTV distribution.

Samples a subset of (N, n) pairs and determines the best-fitting
distribution using multiple metrics. Produces a summary report.
"""

import os
import sys
import numpy as np
from fractions import Fraction
from collections import Counter

from fit import load_distribution, load_cdf, SUPPORTED_DISTRIBUTIONS
from compare import DIST_MAP, DISCRETE_DISTS, load_fitted_params


def analyze_pair(N, n, verbose=False):
    """Analyze which distribution fits best for a given (N, n) pair.

    Returns dict with winner info or None.
    """
    fname = f"N_{N}_n_{n}.txt"
    exact_path = os.path.join("../data/exact_distributions", fname)
    cdf_path = os.path.join("../data/cdf_exact", fname)

    if not os.path.isfile(exact_path) or not os.path.isfile(cdf_path):
        return None

    values, counts = load_distribution(exact_path)
    cdf_values, cdf_probs = load_cdf(cdf_path)

    total = sum(counts)
    true_probs = np.array([float(Fraction(c, total)) for c in counts])
    values_arr = np.array(values, dtype=float)
    cdf_values_f = cdf_values.astype(float)

    results = []

    for dist_name in SUPPORTED_DISTRIBUTIONS:
        # Check MLE fit
        mle_path = os.path.join(f"../data/fitted/mle_{dist_name}", fname)
        cvm_path = os.path.join(f"../data/fitted/cvm_{dist_name}", fname)

        for fit_type, path in [("mle", mle_path), ("cvm", cvm_path)]:
            if not os.path.isfile(path):
                continue
            try:
                params, cvm_stat = load_fitted_params(path)
                dist_obj = DIST_MAP[dist_name](params)

                # CDF comparison
                theo_cdf = dist_obj.cdf(cdf_values_f)
                cvm = np.sum((cdf_probs - theo_cdf) ** 2)
                ks = np.max(np.abs(cdf_probs - theo_cdf))

                # PMF comparison (discretized)
                if dist_name in DISCRETE_DISTS:
                    fitted_pmf = dist_obj.pmf(values_arr.astype(int))
                else:
                    fitted_pmf = dist_obj.cdf(values_arr + 0.5) - dist_obj.cdf(values_arr - 0.5)
                fitted_pmf = np.maximum(fitted_pmf, 1e-300)

                l1 = np.sum(np.abs(true_probs - fitted_pmf))

                # KL divergence
                log_q = np.log(np.maximum(fitted_pmf, 1e-300))
                log_p = np.log(np.maximum(true_probs, 1e-300))
                kl = np.sum(true_probs * (log_p - log_q))

                results.append({
                    "dist": dist_name,
                    "fit_type": fit_type,
                    "cvm": cvm,
                    "ks": ks,
                    "l1": l1,
                    "kl": kl,
                })
            except Exception:
                continue

    if not results:
        return None

    # Find best by CvM
    best_cvm = min(results, key=lambda r: r["cvm"])
    # Find best by KS
    best_ks = min(results, key=lambda r: r["ks"])
    # Find best by L1
    best_l1 = min(results, key=lambda r: r["l1"])
    # Find best by KL
    best_kl = min(results, key=lambda r: r["kl"])

    return {
        "N": N, "n": n,
        "best_cvm": best_cvm["dist"],
        "best_ks": best_ks["dist"],
        "best_l1": best_l1["dist"],
        "best_kl": best_kl["dist"],
        "gamma_cvm": next((r["cvm"] for r in results
                           if r["dist"] == "gamma" and r["fit_type"] == "mle"), None),
        "best_cvm_val": best_cvm["cvm"],
        "best_cvm_dist": best_cvm["dist"],
    }


def main():
    # Sample a range of (N, n) pairs
    pairs = []

    # Systematic sampling
    for N in range(10, 201, 10):
        for n in range(2, min(N + 1, 201), max(1, N // 10)):
            pairs.append((N, n))

    # Add specific interesting pairs
    for N in [50, 100, 200, 300, 400]:
        for n in [5, 10, 20, 50, 100]:
            if n <= N:
                pairs.append((N, n))

    pairs = sorted(set(pairs))
    print(f"Analyzing {len(pairs)} (N, n) pairs...\n")

    winner_counts = {
        "cvm": Counter(),
        "ks": Counter(),
        "l1": Counter(),
        "kl": Counter(),
    }

    gamma_rank_data = []
    processed = 0

    for N, n in pairs:
        result = analyze_pair(N, n)
        if result is None:
            continue
        processed += 1
        winner_counts["cvm"][result["best_cvm"]] += 1
        winner_counts["ks"][result["best_ks"]] += 1
        winner_counts["l1"][result["best_l1"]] += 1
        winner_counts["kl"][result["best_kl"]] += 1

    print(f"Processed {processed} pairs.\n")

    for metric in ["cvm", "ks", "l1", "kl"]:
        print(f"\n--- Best distribution by {metric.upper()} ---")
        for dist, count in winner_counts[metric].most_common(10):
            pct = 100 * count / processed
            print(f"  {dist:<16} {count:>5} ({pct:5.1f}%)")

    # Overall winner across metrics
    print("\n\n=== OVERALL ANALYSIS ===")
    combined = Counter()
    for metric in winner_counts:
        for dist, count in winner_counts[metric].items():
            combined[dist] += count

    print("\nCombined ranking (sum of wins across all 4 metrics):")
    for dist, count in combined.most_common(10):
        total_possible = 4 * processed
        pct = 100 * count / total_possible
        print(f"  {dist:<16} {count:>5} / {total_possible} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
