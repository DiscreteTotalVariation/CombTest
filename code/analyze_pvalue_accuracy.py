#!/usr/bin/env python3
"""Analyze p-value accuracy of different distribution approximations for DTV.

Uses stratified sampling across all available (N,n) pairs to compare
gamma, nbinom, and beta approximations.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from fit import load_cdf
from compare import DIST_MAP, load_fitted_params

DISTS = ['gamma', 'nbinom', 'beta']


def get_all_pairs():
    """Get all (N,n) pairs that have CDF and all 3 fitted distributions."""
    pairs = []
    cdf_dir = '../data/cdf_exact'
    for fname in os.listdir(cdf_dir):
        if not fname.startswith('N_') or not fname.endswith('.txt'):
            continue
        parts = fname.replace('.txt', '').split('_')
        N, n = int(parts[1]), int(parts[3])
        # Check all dists available
        has_all = True
        for dist in DISTS:
            found = False
            for prefix in ['../data/fitted/mle_', '../data/fitted/cvm_']:
                if os.path.isfile(os.path.join(f'{prefix}{dist}', fname)):
                    found = True
                    break
            if not found:
                has_all = False
                break
        if has_all:
            pairs.append((N, n))
    return sorted(pairs)


def stratified_sample(pairs, max_per_cell=5):
    """Stratified sample: divide (N,n) space into grid cells, sample from each."""
    N_bins = list(range(0, 501, 25))  # 0-25, 25-50, ..., 475-500
    n_bins = list(range(0, 501, 25))

    cells = defaultdict(list)
    for N, n in pairs:
        Ni = min(N // 25, len(N_bins) - 1)
        ni = min(n // 25, len(n_bins) - 1)
        cells[(Ni, ni)].append((N, n))

    sampled = []
    for key, cell_pairs in cells.items():
        if len(cell_pairs) <= max_per_cell:
            sampled.extend(cell_pairs)
        else:
            rng = np.random.RandomState(42)
            idxs = rng.choice(len(cell_pairs), max_per_cell, replace=False)
            sampled.extend([cell_pairs[i] for i in idxs])

    return sorted(sampled)


def compute_pvalue_error(N, n):
    """Compute p-value error at the critical threshold for each distribution."""
    fname = f'N_{N}_n_{n}.txt'
    cdf_path = os.path.join('../data/cdf_exact', fname)
    cdf_values, cdf_probs = load_cdf(cdf_path)
    cdf_values_f = cdf_values.astype(float)

    # Find critical d where p closest to 0.05
    target = 0.95
    best_idx = np.argmin(np.abs(cdf_probs - target))
    d_crit = int(cdf_values[best_idx])
    exact_p = 1.0 - float(cdf_probs[best_idx])

    if exact_p > 0.15 or exact_p < 0.001:
        return None

    errors = {}
    for dist in DISTS:
        best_error = float('inf')
        for prefix in ['../data/fitted/mle_', '../data/fitted/cvm_']:
            path = os.path.join(f'{prefix}{dist}', fname)
            if os.path.isfile(path):
                try:
                    params, _ = load_fitted_params(path)
                    dist_obj = DIST_MAP[dist](params)
                    approx_p = 1.0 - float(dist_obj.cdf(d_crit))
                    error = abs(approx_p - exact_p)
                    best_error = min(best_error, error)
                except Exception:
                    pass
        errors[dist] = best_error if best_error < float('inf') else None

    return {'N': N, 'n': n, 'd_crit': d_crit, 'exact_p': exact_p, 'errors': errors}


def print_summary(results, label):
    """Print summary statistics."""
    print(f'\n{"="*75}')
    print(f'{label} ({len(results)} pairs)')
    print(f'{"="*75}')

    print(f'{"Distribution":<14} {"Mean":>10} {"Median":>10} {"P95":>10} {"Max":>10} {"<0.01":>7} {"<0.005":>7} {"<0.001":>7}')
    print('-' * 75)

    for dist in DISTS:
        errs = [r['errors'][dist] for r in results if r['errors'].get(dist) is not None]
        if not errs:
            print(f'{dist:<14} no data')
            continue
        n_v = len(errs)
        earr = np.array(errs)
        print(f'{dist:<14} {np.mean(earr):>10.6f} {np.median(earr):>10.6f} {np.percentile(earr,95):>10.6f} {np.max(earr):>10.6f} '
              f'{100*np.sum(earr<0.01)/n_v:>6.1f}% {100*np.sum(earr<0.005)/n_v:>6.1f}% {100*np.sum(earr<0.001)/n_v:>6.1f}%')

    # Winner count
    print()
    wins = {d: 0 for d in DISTS}
    for r in results:
        valid = {d: r['errors'][d] for d in DISTS if r['errors'].get(d) is not None}
        if valid:
            winner = min(valid, key=valid.get)
            wins[winner] += 1
    total = sum(wins.values())
    for d in DISTS:
        print(f'  {d} wins: {wins[d]} ({100*wins[d]/total:.1f}%)')


def main():
    print("Loading all available (N,n) pairs...")
    all_pairs = get_all_pairs()
    print(f"Total pairs with all 3 dists: {len(all_pairs)}")

    print("Stratified sampling...")
    sampled = stratified_sample(all_pairs, max_per_cell=5)
    print(f"Sampled pairs: {len(sampled)}")

    print("Computing p-value errors...")
    results_all = []
    results_5n = []
    results_lt5n = []

    for i, (N, n) in enumerate(sampled):
        if i % 500 == 0:
            print(f"  {i}/{len(sampled)}...")
        r = compute_pvalue_error(N, n)
        if r is None:
            continue
        results_all.append(r)
        if N >= 5 * n:
            results_5n.append(r)
        else:
            results_lt5n.append(r)

    print_summary(results_all, "ALL pairs")
    print_summary(results_5n, "N >= 5n only")
    print_summary(results_lt5n, "N < 5n only")

    # Breakdown by n ranges
    for lo, hi, label in [(2, 5, "n=2..5"), (6, 10, "n=6..10"), (11, 30, "n=11..30"),
                           (31, 100, "n=31..100"), (101, 465, "n=101..465")]:
        subset = [r for r in results_all if lo <= r['n'] <= hi]
        if subset:
            print_summary(subset, label)

    # Breakdown by N ranges
    for lo, hi, label in [(2, 25, "N=2..25"), (26, 50, "N=26..50"), (51, 100, "N=51..100"),
                           (101, 200, "N=101..200"), (201, 465, "N=201..465")]:
        subset = [r for r in results_all if lo <= r['N'] <= hi]
        if subset:
            print_summary(subset, label)

    # Breakdown by n/N ratio
    for lo, hi, label in [(0, 0.1, "n/N<0.1"), (0.1, 0.3, "0.1<=n/N<0.3"),
                           (0.3, 0.5, "0.3<=n/N<0.5"), (0.5, 1.0, "0.5<=n/N<=1.0")]:
        subset = [r for r in results_all if lo <= r['n']/r['N'] < hi + (0.001 if hi == 1.0 else 0)]
        if subset:
            print_summary(subset, label)

    # Save raw results as JSON
    output = {
        'all': {'count': len(results_all)},
        'ge5n': {'count': len(results_5n)},
        'lt5n': {'count': len(results_lt5n)},
    }
    for key, res_list in [('all', results_all), ('ge5n', results_5n), ('lt5n', results_lt5n)]:
        for dist in DISTS:
            errs = [r['errors'][dist] for r in res_list if r['errors'].get(dist) is not None]
            if errs:
                earr = np.array(errs)
                output[key][dist] = {
                    'mean': float(np.mean(earr)),
                    'median': float(np.median(earr)),
                    'p95': float(np.percentile(earr, 95)),
                    'max': float(np.max(earr)),
                    'lt_001': float(np.sum(earr < 0.001) / len(earr)),
                    'lt_005': float(np.sum(earr < 0.005) / len(earr)),
                    'lt_01': float(np.sum(earr < 0.01) / len(earr)),
                }

    with open('pvalue_accuracy_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved: pvalue_accuracy_results.json")


if __name__ == '__main__':
    main()
