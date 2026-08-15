#!/usr/bin/env python3
"""Full p-value accuracy analysis across ALL available (N,n) pairs.

Optimized: preloads fitted params, uses vectorized operations.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from compare import DIST_MAP, load_fitted_params

DISTS = ['gamma', 'nbinom', 'beta']


def load_cdf_fast(path):
    """Fast CDF loader."""
    vals, probs = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                vals.append(int(float(parts[0])))
                probs.append(float(parts[1]))
    return vals, probs


def process_pair(N, n, cdf_dir, fitted_cache):
    """Return p-value errors for one (N,n) pair, or None."""
    fname = f'N_{N}_n_{n}.txt'
    cdf_path = os.path.join(cdf_dir, fname)

    try:
        vals, probs = load_cdf_fast(cdf_path)
    except Exception:
        return None

    if not vals:
        return None

    # Find critical d where CDF closest to 0.95
    best_idx = 0
    best_diff = abs(probs[0] - 0.95)
    for i in range(1, len(probs)):
        diff = abs(probs[i] - 0.95)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    d_crit = vals[best_idx]
    exact_p = 1.0 - probs[best_idx]

    if exact_p > 0.15 or exact_p < 0.001:
        return None

    errors = {}
    d_crit_f = float(d_crit)

    for dist in DISTS:
        best_error = float('inf')

        for prefix in ['../data/fitted/mle_', '../data/fitted/cvm_']:
            key = (prefix, dist, N, n)
            if key in fitted_cache:
                dist_obj = fitted_cache[key]
                if dist_obj is None:
                    continue
            else:
                path = os.path.join(f'{prefix}{dist}', fname)
                if not os.path.isfile(path):
                    fitted_cache[key] = None
                    continue
                try:
                    params, _ = load_fitted_params(path)
                    dist_obj = DIST_MAP[dist](params)
                    fitted_cache[key] = dist_obj
                except Exception:
                    fitted_cache[key] = None
                    continue

            try:
                approx_p = 1.0 - float(dist_obj.cdf(d_crit_f))
                error = abs(approx_p - exact_p)
                best_error = min(best_error, error)
            except Exception:
                pass

        errors[dist] = best_error if best_error < float('inf') else None

    return {'N': N, 'n': n, 'exact_p': exact_p, 'errors': errors}


def print_summary(results, label):
    n_total = len(results)
    if n_total == 0:
        return
    print(f'\n{"="*80}')
    print(f'{label} ({n_total} pairs)')
    print(f'{"="*80}')
    print(f'{"Distribution":<14} {"Mean":>10} {"Median":>10} {"P95":>10} {"Max":>10} '
          f'{"<0.01":>7} {"<0.005":>7} {"<0.001":>7} {"Wins":>6}')
    print('-' * 80)

    wins = {d: 0 for d in DISTS}
    for r in results:
        valid = {d: r['errors'][d] for d in DISTS if r['errors'].get(d) is not None}
        if valid:
            wins[min(valid, key=valid.get)] += 1

    for dist in DISTS:
        errs = [r['errors'][dist] for r in results if r['errors'].get(dist) is not None]
        if not errs:
            print(f'{dist:<14} no data')
            continue
        earr = np.array(errs)
        nv = len(earr)
        total_wins = sum(wins.values())
        pct = 100 * wins[dist] / total_wins if total_wins > 0 else 0
        print(f'{dist:<14} {np.mean(earr):>10.6f} {np.median(earr):>10.6f} '
              f'{np.percentile(earr, 95):>10.6f} {np.max(earr):>10.6f} '
              f'{100*np.sum(earr<0.01)/nv:>6.1f}% {100*np.sum(earr<0.005)/nv:>6.1f}% '
              f'{100*np.sum(earr<0.001)/nv:>6.1f}% {pct:>5.1f}%')


def main():
    cdf_dir = '../data/cdf_exact'

    print("Scanning cdf_exact directory...")
    all_pairs = []
    for fname in os.listdir(cdf_dir):
        if not fname.startswith('N_') or not fname.endswith('.txt'):
            continue
        parts = fname.replace('.txt', '').split('_')
        try:
            N, n = int(parts[1]), int(parts[3])
            all_pairs.append((N, n))
        except (IndexError, ValueError):
            continue

    all_pairs.sort()
    print(f"Total CDF files: {len(all_pairs)}")
    print(f"N range: {min(p[0] for p in all_pairs)} to {max(p[0] for p in all_pairs)}")
    print(f"n range: {min(p[1] for p in all_pairs)} to {max(p[1] for p in all_pairs)}")

    # Process all pairs - don't cache dist objects (too much memory for 440K)
    results_all = []
    results_5n = []
    results_lt5n = []
    skipped = 0

    fitted_cache = {}  # Will grow but we only store None for missing files

    for i, (N, n) in enumerate(tqdm(all_pairs)):
        r = process_pair(N, n, cdf_dir, fitted_cache)
        if r is None:
            skipped += 1
            continue
        if all(r['errors'][d] is None for d in DISTS):
            skipped += 1
            continue
        results_all.append(r)
        if N >= 5 * n:
            results_5n.append(r)
        else:
            results_lt5n.append(r)

        # Clear cache periodically to avoid memory bloat
        if len(fitted_cache) > 50000:
            fitted_cache.clear()

    print(f"\nProcessed: {len(results_all)} valid pairs (skipped {skipped})")

    print_summary(results_all, "ALL pairs")
    print_summary(results_5n, "N >= 5n only")
    print_summary(results_lt5n, "N < 5n only")

    # By n/N ratio
    for lo, hi, label in [
        (0, 0.05, "n/N < 0.05"),
        (0.05, 0.1, "0.05 <= n/N < 0.1"),
        (0.1, 0.2, "0.1 <= n/N < 0.2"),
        (0.2, 0.3, "0.2 <= n/N < 0.3"),
        (0.3, 0.5, "0.3 <= n/N < 0.5"),
        (0.5, 0.75, "0.5 <= n/N < 0.75"),
        (0.75, 1.001, "0.75 <= n/N <= 1.0"),
        (1.001, 2, "1.0 < n/N < 2"),
        (2, 5, "2 <= n/N < 5"),
        (5, 100, "n/N >= 5"),
    ]:
        subset = [r for r in results_all if lo <= r['n'] / r['N'] < hi]
        if len(subset) >= 5:
            print_summary(subset, label)

    # By N ranges
    for lo, hi, label in [
        (1, 10, "N=1..10"),
        (11, 25, "N=11..25"),
        (26, 50, "N=26..50"),
        (51, 100, "N=51..100"),
        (101, 200, "N=101..200"),
        (201, 465, "N=201..465"),
        (466, 10000, "N>465"),
    ]:
        subset = [r for r in results_all if lo <= r['N'] <= hi]
        if len(subset) >= 5:
            print_summary(subset, label)

    # Save JSON summary
    output = {}
    for key, res_list in [('all', results_all), ('ge5n', results_5n), ('lt5n', results_lt5n)]:
        output[key] = {'count': len(res_list)}
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
        wins = {d: 0 for d in DISTS}
        for r in res_list:
            valid = {d: r['errors'][d] for d in DISTS if r['errors'].get(d) is not None}
            if valid:
                wins[min(valid, key=valid.get)] += 1
        output[key]['wins'] = wins

    # Add ratio breakdowns
    for lo, hi, rkey in [
        (0, 0.2, "ratio_lt02"),
        (0.2, 0.5, "ratio_02_05"),
        (0.5, 1.001, "ratio_05_1"),
        (1.001, 100, "ratio_gt1"),
    ]:
        subset = [r for r in results_all if lo <= r['n'] / r['N'] < hi]
        if subset:
            output[rkey] = {'count': len(subset)}
            for dist in DISTS:
                errs = [r['errors'][dist] for r in subset if r['errors'].get(dist) is not None]
                if errs:
                    earr = np.array(errs)
                    output[rkey][dist] = {
                        'mean': float(np.mean(earr)),
                        'median': float(np.median(earr)),
                    }

    with open('pvalue_accuracy_full_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved: pvalue_accuracy_full_results.json")


if __name__ == '__main__':
    main()
