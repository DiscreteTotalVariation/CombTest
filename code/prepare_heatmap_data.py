#!/usr/bin/env python3
"""Prepare KS statistic data for the beta approximation heatmap.

Computes max |CDF_exact - CDF_beta| for each (N, n) pair and saves to a
NumPy file. Incremental: if the output file already exists, only computes
missing (N, n) pairs.
"""

import argparse
import os
import numpy as np
from scipy import stats


def load_cdf(path):
    values, probs = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                values.append(int(float(parts[0])))
                probs.append(float(parts[1]))
    return np.array(values), np.array(probs)


def load_fitted_params(path):
    with open(path) as f:
        return tuple(map(float, f.readline().strip().split()))


def compute_ks_for_pair(N, n, dist_name):
    """Compute KS statistic (max |CDF_exact - CDF_approx|) for one (N,n) pair."""
    cdf_path = "../data/cdf_exact/N_%d_n_%d.txt" % (N, n)
    fitted_path = "../data/fitted/cvm_%s/N_%d_n_%d.txt" % (dist_name, N, n)

    if not os.path.isfile(cdf_path) or not os.path.isfile(fitted_path):
        return None

    try:
        cdf_values, cdf_probs = load_cdf(cdf_path)
        params = load_fitted_params(fitted_path)

        if dist_name == "beta":
            dist_obj = stats.beta(a=params[0], b=params[1], loc=params[2], scale=params[3])
        elif dist_name == "gamma":
            dist_obj = stats.gamma(a=params[0], loc=params[1], scale=params[2])
        else:
            return None

        theo_cdf = dist_obj.cdf(cdf_values.astype(float))
        ks = np.max(np.abs(cdf_probs - theo_cdf))
        return ks
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Prepare KS statistic heatmap data (incremental).")
    parser.add_argument("--N-max", type=int, default=450, help="Maximum N value (default: 450)")
    parser.add_argument("--dist", type=str, default="beta", choices=["beta", "gamma"],
                        help="Distribution to fit (default: beta)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output .npy file path (default: heatmap_ks_{dist}_{N_max}.npy)")
    args = parser.parse_args()

    N_max = args.N_max
    dist_name = args.dist

    if args.output is None:
        output_path = "../data/heatmap_ks_%s_%d.npy" % (dist_name, N_max)
    else:
        output_path = args.output

    # Load existing data if available
    if os.path.isfile(output_path):
        grid = np.load(output_path)
        old_size = grid.shape[0]
        if old_size < N_max - 1:
            # Expand grid
            new_grid = np.full((N_max - 1, N_max - 1), np.nan)
            new_grid[:old_size, :old_size] = grid
            grid = new_grid
            print("Loaded existing data (%d x %d), expanding to %d x %d" % (old_size, old_size, N_max - 1, N_max - 1))
        else:
            print("Loaded existing data (%d x %d)" % (grid.shape[0], grid.shape[1]))
    else:
        grid = np.full((N_max - 1, N_max - 1), np.nan)
        print("Starting fresh (%d x %d)" % (N_max - 1, N_max - 1))

    # Count how many are missing
    total = (N_max - 1) ** 2
    already_done = np.sum(~np.isnan(grid))
    missing = total - int(already_done)
    print("Total cells: %d, already computed: %d, missing: %d" % (total, int(already_done), missing))

    if missing == 0:
        print("All cells already computed. Nothing to do.")
        return

    computed = 0
    for i, n in enumerate(range(2, N_max + 1)):
        row_had_updates = False
        for j, N in enumerate(range(2, N_max + 1)):
            if not np.isnan(grid[i, j]):
                continue
            ks = compute_ks_for_pair(N, n, dist_name)
            if ks is not None:
                grid[i, j] = ks
                row_had_updates = True
            computed += 1
        if (i + 1) % 50 == 0:
            print("  Row %d/%d (%d new cells computed so far)" % (i + 1, N_max - 1, computed))
            # Save periodically
            np.save(output_path, grid)

    np.save(output_path, grid)
    print("Saved: %s (%d new cells computed)" % (output_path, computed))


if __name__ == "__main__":
    main()
