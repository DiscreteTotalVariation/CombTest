#!/usr/bin/env python3
"""Compute beta approximation quality statistics for Section III of the paper.

For all (N,n) pairs in {2,...,N_max} with available exact CDF and fitted beta
parameters, compute:
  1. Critical value match rate (using continuity-corrected beta CDF at d+0.5)
  2. P-value error: |CDF_exact(d*) - CDF_beta(d*)| at the critical d*

These numbers appear in the paper's "Beta approximation quality" subsection.

Usage:
    python3 compute_beta_match_stats.py
    python3 compute_beta_match_stats.py --N-max 500
"""

import argparse
import os
import numpy as np
from scipy import stats
from tqdm import tqdm


def load_cdf(path):
    values, probs = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                values.append(int(float(parts[0])))
                probs.append(float(parts[1]))
    return values, probs


def load_params(path):
    with open(path) as f:
        return tuple(map(float, f.readline().strip().split()))


def main():
    parser = argparse.ArgumentParser(
        description="Compute beta approximation quality statistics.")
    parser.add_argument("--N-max", type=int, default=500)
    args = parser.parse_args()

    N_max = args.N_max
    cdf_dir = '../data/cdf_exact'
    beta_dir = '../data/fitted/cvm_beta'

    exact_match_all = 0
    within_1_all = 0
    p_errors_all = []
    p_errors_ge5n = []
    total = 0

    for N in tqdm(range(2, N_max + 1), desc="Beta match stats"):
        for n in range(2, N_max + 1):
            fname = 'N_%d_n_%d.txt' % (N, n)
            cdf_path = os.path.join(cdf_dir, fname)
            beta_path = os.path.join(beta_dir, fname)

            if not os.path.isfile(cdf_path) or not os.path.isfile(beta_path):
                continue

            try:
                vals, probs = load_cdf(cdf_path)
                if not vals:
                    continue

                total += 1

                # Find critical d*: where exact CDF closest to 0.95
                best_idx = 0
                best_diff = abs(probs[0] - 0.95)
                for i in range(1, len(probs)):
                    diff = abs(probs[i] - 0.95)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = i
                exact_d = vals[best_idx]
                exact_cdf_at_d = probs[best_idx]

                params = load_params(beta_path)
                bd = stats.beta(a=params[0], b=params[1],
                                loc=params[2], scale=params[3])

                # Critical value from beta: use CC (+0.5) at integer points
                all_d = np.arange(0, 2 * N + 1)
                beta_cdf_cc = bd.cdf(all_d + 0.5)
                beta_idx = int(np.argmin(np.abs(beta_cdf_cc - 0.95)))
                beta_d = all_d[beta_idx]
                cd = abs(int(exact_d) - int(beta_d))
                if cd == 0:
                    exact_match_all += 1
                if cd <= 1:
                    within_1_all += 1

                # P-value error: |CDF_exact(d*) - CDF_beta(d*)| at same point
                beta_cdf_at_d = bd.cdf(float(exact_d))
                p_err = abs(beta_cdf_at_d - exact_cdf_at_d)
                p_errors_all.append(p_err)
                if N >= 5 * n:
                    p_errors_ge5n.append(p_err)

            except Exception:
                continue

    ea = np.array(p_errors_all)
    eg = np.array(p_errors_ge5n)

    print()
    print("=" * 60)
    print("BETA APPROXIMATION QUALITY STATISTICS")
    print("=" * 60)
    print("N_max=%d, total valid pairs: %d" % (N_max, total))
    print()
    print("Critical value comparison (beta uses CC at d+0.5):")
    print("  Exact match:  %d/%d = %.1f%%" % (
        exact_match_all, total, 100 * exact_match_all / total))
    print("  Within 1:     %d/%d = %.1f%% (raw: %.4f%%)" % (
        within_1_all, total,
        round(100 * within_1_all / total, 1),
        100 * within_1_all / total))
    print()
    print("P-value error |CDF_exact(d*) - CDF_beta(d*)| at critical d*:")
    print("  All pairs (%d):" % len(ea))
    print("    Median:  %.6f" % np.median(ea))
    print("    <0.001:  %.1f%%" % (100 * np.sum(ea < 0.001) / len(ea)))
    print("    <0.005:  %.1f%%" % (100 * np.sum(ea < 0.005) / len(ea)))
    print("    Mean:    %.6f" % np.mean(ea))
    print("    Max:     %.6f" % np.max(ea))
    print()
    if len(eg) > 0:
        print("  N >= 5n (%d pairs):" % len(eg))
        print("    Median:  %.6f" % np.median(eg))
        print("    <0.001:  %.1f%%" % (100 * np.sum(eg < 0.001) / len(eg)))


if __name__ == '__main__':
    main()
