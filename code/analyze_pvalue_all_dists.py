#!/usr/bin/env python3
"""P-value accuracy analysis for ALL fitted distributions across all (N,n) pairs."""

import os
import json
import numpy as np
import multiprocessing as mp
from scipy import stats

# All distributions with ~436K+ fitted files (good coverage)
DISTS_FULL = [
    'gamma', 'nbinom', 'beta', 'normal', 'lognormal', 'weibull', 'chi2',
    'exponential', 'invgauss', 'gengamma', 'nakagami', 'betaprime',
    'burr', 'burr12', 'fisk', 'genpareto', 'genextreme',
    'rayleigh', 'rice', 'pareto', 'f', 'poisson',
]

DIST_MAP = {
    'normal':     lambda p: stats.norm(loc=p[0], scale=p[1]),
    'lognormal':  lambda p: stats.lognorm(s=p[0], loc=p[1], scale=p[2]),
    'gamma':      lambda p: stats.gamma(a=p[0], loc=p[1], scale=p[2]),
    'beta':       lambda p: stats.beta(a=p[0], b=p[1], loc=p[2], scale=p[3]),
    'poisson':    lambda p: stats.poisson(mu=p[0]),
    'exponential':lambda p: stats.expon(loc=p[0], scale=p[1]),
    'weibull':    lambda p: stats.weibull_min(c=p[0], loc=p[1], scale=p[2]),
    'chi2':       lambda p: stats.chi2(df=p[0], loc=p[1], scale=p[2]),
    'nbinom':     lambda p: stats.nbinom(n=p[0], p=p[1]),
    'invgauss':   lambda p: stats.invgauss(mu=p[0], loc=p[1], scale=p[2]),
    'gengamma':   lambda p: stats.gengamma(a=p[0], c=p[1], loc=p[2], scale=p[3]),
    'nakagami':   lambda p: stats.nakagami(nu=p[0], loc=p[1], scale=p[2]),
    'betaprime':  lambda p: stats.betaprime(a=p[0], b=p[1], loc=p[2], scale=p[3]),
    'burr':       lambda p: stats.burr(c=p[0], d=p[1], loc=p[2], scale=p[3]),
    'burr12':     lambda p: stats.burr12(c=p[0], d=p[1], loc=p[2], scale=p[3]),
    'fisk':       lambda p: stats.fisk(c=p[0], loc=p[1], scale=p[2]),
    'genpareto':  lambda p: stats.genpareto(c=p[0], loc=p[1], scale=p[2]),
    'genextreme': lambda p: stats.genextreme(c=p[0], loc=p[1], scale=p[2]),
    'rayleigh':   lambda p: stats.rayleigh(loc=p[0], scale=p[1]),
    'rice':       lambda p: stats.rice(b=p[0], loc=p[1], scale=p[2]),
    'pareto':     lambda p: stats.pareto(b=p[0], loc=p[1], scale=p[2]),
    'f':          lambda p: stats.f(dfn=p[0], dfd=p[1], loc=p[2], scale=p[3]),
}

NPARAMS = {
    'normal': 2, 'lognormal': 3, 'gamma': 3, 'beta': 4, 'poisson': 1,
    'exponential': 2, 'weibull': 3, 'chi2': 3, 'nbinom': 2, 'invgauss': 3,
    'gengamma': 4, 'nakagami': 3, 'betaprime': 4, 'burr': 4, 'burr12': 4,
    'fisk': 3, 'genpareto': 3, 'genextreme': 3, 'rayleigh': 2, 'rice': 3,
    'pareto': 3, 'f': 4,
}


def load_fitted_params(path):
    with open(path) as f:
        lines = f.readlines()
    return tuple(map(float, lines[0].strip().split()))


def process_pair(args):
    N, n = args
    fname = f'N_{N}_n_{n}.txt'
    cdf_path = os.path.join('../data/cdf_exact', fname)

    try:
        vals, probs = [], []
        with open(cdf_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    vals.append(int(float(parts[0])))
                    probs.append(float(parts[1]))
        if not vals:
            return None
    except Exception:
        return None

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

    d_crit_f = float(d_crit)
    errors = {}

    for dist in DISTS_FULL:
        best_error = float('inf')
        for prefix in ['../data/fitted/mle_', '../data/fitted/cvm_']:
            path = os.path.join(f'{prefix}{dist}', fname)
            if not os.path.isfile(path):
                continue
            try:
                params = load_fitted_params(path)
                dist_obj = DIST_MAP[dist](params)
                approx_p = 1.0 - float(dist_obj.cdf(d_crit_f))
                error = abs(approx_p - exact_p)
                best_error = min(best_error, error)
            except Exception:
                pass
        errors[dist] = best_error if best_error < float('inf') else None

    return {'N': N, 'n': n, 'exact_p': exact_p, 'errors': errors}


def summarize(results, label, dists=None):
    if dists is None:
        dists = DISTS_FULL
    n_total = len(results)
    if n_total == 0:
        return

    print(f'\n{"="*120}')
    print(f'{label} ({n_total} pairs)')
    print(f'{"="*120}')
    print(f'{"Distribution":<14} {"Params":>6} {"Mean":>10} {"Median":>10} {"P95":>10} {"Max":>10} '
          f'{"<0.01":>7} {"<0.005":>7} {"<0.001":>7} {"Wins":>6}')
    print('-' * 120)

    wins = {d: 0 for d in dists}
    for r in results:
        valid = {d: r['errors'][d] for d in dists if r['errors'].get(d) is not None}
        if valid:
            wins[min(valid, key=valid.get)] += 1

    # Sort by median error
    dist_medians = []
    for dist in dists:
        errs = [r['errors'][dist] for r in results if r['errors'].get(dist) is not None]
        if errs:
            dist_medians.append((dist, np.median(errs)))
        else:
            dist_medians.append((dist, float('inf')))
    dist_medians.sort(key=lambda x: x[1])

    for dist, _ in dist_medians:
        errs = [r['errors'][dist] for r in results if r['errors'].get(dist) is not None]
        if not errs:
            print(f'{dist:<14} {NPARAMS.get(dist,"?"):>6} no data')
            continue
        earr = np.array(errs)
        nv = len(earr)
        total_wins = sum(wins.values())
        pct = 100 * wins[dist] / total_wins if total_wins > 0 else 0
        print(f'{dist:<14} {NPARAMS.get(dist,"?"):>6} {np.mean(earr):>10.6f} {np.median(earr):>10.6f} '
              f'{np.percentile(earr, 95):>10.6f} {np.max(earr):>10.6f} '
              f'{100*np.sum(earr<0.01)/nv:>6.1f}% {100*np.sum(earr<0.005)/nv:>6.1f}% '
              f'{100*np.sum(earr<0.001)/nv:>6.1f}% {pct:>5.1f}%')


def main():
    print("Scanning cdf_exact directory...")
    all_pairs = []
    for fname in os.listdir('../data/cdf_exact'):
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

    n_workers = max(1, mp.cpu_count() - 2)
    print(f"Using {n_workers} workers, {len(DISTS_FULL)} distributions")

    results_all = []
    skipped = 0
    done = 0

    with mp.Pool(n_workers) as pool:
        for r in pool.imap_unordered(process_pair, all_pairs, chunksize=128):
            done += 1
            if done % 20000 == 0:
                print(f"  {done}/{len(all_pairs)} processed, {len(results_all)} valid")
            if r is None:
                skipped += 1
                continue
            if all(r['errors'][d] is None for d in DISTS_FULL):
                skipped += 1
                continue
            results_all.append(r)

    print(f"\nProcessed: {len(results_all)} valid pairs (skipped {skipped})")

    results_5n = [r for r in results_all if r['N'] >= 5 * r['n']]
    results_lt5n = [r for r in results_all if r['N'] < 5 * r['n']]

    summarize(results_all, "ALL pairs")
    summarize(results_5n, "N >= 5n only")
    summarize(results_lt5n, "N < 5n only")

    # Key ratio ranges
    for lo, hi, label in [
        (0, 0.1, "n/N < 0.1"),
        (0.1, 0.3, "0.1 <= n/N < 0.3"),
        (0.3, 0.5, "0.3 <= n/N < 0.5"),
        (0.5, 1.001, "0.5 <= n/N <= 1.0"),
        (1.001, 5, "1.0 < n/N < 5"),
        (5, 100, "n/N >= 5"),
    ]:
        subset = [r for r in results_all if lo <= r['n'] / r['N'] < hi]
        if len(subset) >= 5:
            summarize(subset, label)

    # Save JSON with top distributions
    output = {
        'total_cdf_files': len(all_pairs),
        'valid_pairs': len(results_all),
        'skipped': skipped,
        'distributions_tested': DISTS_FULL,
    }

    for key, res_list in [('all', results_all), ('ge5n', results_5n), ('lt5n', results_lt5n)]:
        output[key] = {'count': len(res_list)}
        for dist in DISTS_FULL:
            errs = [r['errors'][dist] for r in res_list if r['errors'].get(dist) is not None]
            if errs:
                earr = np.array(errs)
                output[key][dist] = {
                    'count': len(earr),
                    'params': NPARAMS.get(dist),
                    'mean': float(np.mean(earr)),
                    'median': float(np.median(earr)),
                    'p95': float(np.percentile(earr, 95)),
                    'max': float(np.max(earr)),
                    'lt_001': float(np.sum(earr < 0.001) / len(earr)),
                    'lt_005': float(np.sum(earr < 0.005) / len(earr)),
                    'lt_01': float(np.sum(earr < 0.01) / len(earr)),
                }
        wins = {d: 0 for d in DISTS_FULL}
        for r in res_list:
            valid = {d: r['errors'][d] for d in DISTS_FULL if r['errors'].get(d) is not None}
            if valid:
                wins[min(valid, key=valid.get)] += 1
        output[key]['wins'] = wins

    # Ratio breakdowns
    for lo, hi, rkey in [
        (0, 0.1, "ratio_lt01"),
        (0.1, 0.3, "ratio_01_03"),
        (0.3, 0.5, "ratio_03_05"),
        (0.5, 1.001, "ratio_05_1"),
        (1.001, 5, "ratio_1_5"),
        (5, 100, "ratio_ge5"),
    ]:
        subset = [r for r in results_all if lo <= r['n'] / r['N'] < hi]
        if subset:
            output[rkey] = {'count': len(subset)}
            for dist in DISTS_FULL:
                errs = [r['errors'][dist] for r in subset if r['errors'].get(dist) is not None]
                if errs:
                    earr = np.array(errs)
                    output[rkey][dist] = {
                        'count': len(earr),
                        'mean': float(np.mean(earr)),
                        'median': float(np.median(earr)),
                        'p95': float(np.percentile(earr, 95)),
                    }
            wins = {d: 0 for d in DISTS_FULL}
            for r in subset:
                valid = {d: r['errors'][d] for d in DISTS_FULL if r['errors'].get(d) is not None}
                if valid:
                    wins[min(valid, key=valid.get)] += 1
            output[rkey]['wins'] = wins

    with open('pvalue_accuracy_all_dists_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved: pvalue_accuracy_all_dists_results.json")


if __name__ == '__main__':
    main()
