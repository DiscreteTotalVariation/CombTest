#!/usr/bin/env python3
"""Model selection for fitted distributions.

Ranks candidate continuous distributions that approximate the exact discrete
distribution using multiple metrics, and performs pairwise Vuong closeness tests.

Metrics computed:
  - Cross-entropy H(p, q)  [lower = better]
  - KL divergence KL(p||q)  [lower = better]
  - AIC (2k + 2*H(p,q))  [lower = better]
  - BIC (k*log(K) + 2*H(p,q))  [lower = better]
  - CvM statistic (sum of squared CDF differences)  [lower = better]
  - KS statistic (max |CDF difference|)  [lower = better]
  - Anderson-Darling statistic (tail-weighted CDF)  [lower = better]
  - L1 / Total Variation distance  [lower = better]
  - Chi-squared statistic  [lower = better]

Vuong's closeness test:
  Compares two models pairwise using the expected log-likelihood ratio
  under the true (known) distribution. Since the distribution is exact,
  the test uses the number of support points K as the effective sample size.

Usage:
  # Rank all fitted distributions for a given (N, n)
  python model_selection.py -N 100 -n 50

  # Use MLE fits instead of CvM fits
  python model_selection.py -N 100 -n 50 --mle

  # Only rank specific distributions
  python model_selection.py -N 100 -n 50 -d gamma -d lognormal -d beta

  # Pairwise Vuong test between two distributions
  python model_selection.py -N 100 -n 50 --vuong gamma lognormal

  # Rank and show top 5, sort by AIC
  python model_selection.py -N 100 -n 50 --top 5 --sort aic

  # Run over all (N,n) pairs, output CSV
  python model_selection.py --all --mle --sort aic --csv results.csv
"""

import argparse
import os
import sys
import numpy as np
from scipy import stats as sp_stats
from fractions import Fraction

from fit import load_distribution, load_cdf, SUPPORTED_DISTRIBUTIONS
from compare import DIST_MAP, DISCRETE_DISTS, load_fitted_params


# Number of FREE (optimized) parameters for each distribution.
NUM_PARAMS = {
    'normal': 2,
    'lognormal': 2,      # s, scale (loc=0 fixed)
    'gamma': 2,          # a, scale (loc=0 fixed)
    'beta': 3,           # a, b, scale (loc=0 fixed)
    'poisson': 1,        # mu
    'binomial': 1,       # p (n is fixed from data)
    'exponential': 1,    # scale (loc=0 fixed)
    'weibull': 2,        # c, scale (loc=0 fixed)
    'chi2': 2,           # df, scale (loc=0 fixed)
    'nbinom': 2,         # n, p
    'uniform': 2,        # loc, scale
    'invgauss': 2,       # mu, scale (loc=0 fixed)
    'gengamma': 3,       # a, c, scale (loc=0 fixed)
    'nakagami': 1,       # nu (scale derived from data, loc=0 fixed)
    'triang': 1,         # c (loc, scale fixed from data)
    'betaprime': 3,      # a, b, scale (loc=0 fixed)
    'johnsonsb': 4,      # a, b, loc, scale
    'arcsine': 2,        # loc, scale
    'powerlaw': 1,       # a (loc, scale fixed from data)
    'bradford': 1,       # c (loc, scale fixed from data)
    'burr': 3,           # c, d, scale (loc=0 fixed)
    'burr12': 3,         # c, d, scale (loc=0 fixed)
    'fisk': 2,           # c, scale (loc=0 fixed)
    'genpareto': 2,      # c, scale (loc=0 fixed)
    'genextreme': 3,     # c, loc, scale
    'rayleigh': 1,       # scale (loc=0 fixed)
    'rice': 2,           # b, scale (loc=0 fixed)
    'pareto': 2,         # b, scale (loc=0 fixed)
    'f': 3,              # dfn, dfd, scale (loc=0 fixed)
}


def get_available_distributions(N, n, mle=False):
    """Find which distributions have been fitted for this (N, n)."""
    fname = f"N_{N}_n_{n}.txt"
    prefix = "../data/fitted/mle_" if mle else "../data/fitted/cvm_"
    available = []
    for dist_name in SUPPORTED_DISTRIBUTIONS:
        path = os.path.join(f"{prefix}{dist_name}", fname)
        if os.path.isfile(path):
            available.append(dist_name)
    return available


def load_true_distribution(N, n):
    """Load exact distribution and CDF for a given (N, n)."""
    fname = f"N_{N}_n_{n}.txt"
    exact_path = os.path.join("../data/exact_distributions", fname)
    cdf_path = os.path.join("../data/cdf_exact", fname)

    if not os.path.isfile(exact_path):
        print(f"Error: {exact_path} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(cdf_path):
        print(f"Error: {cdf_path} not found", file=sys.stderr)
        sys.exit(1)

    values, counts = load_distribution(exact_path)
    cdf_values, cdf_probs = load_cdf(cdf_path)

    # Compute exact probabilities
    total = sum(counts)
    true_probs = np.array([float(Fraction(c, total)) for c in counts])

    return (np.array(values, dtype=float), true_probs,
            cdf_values.astype(float), cdf_probs)


def discretize_continuous(dist_obj, values):
    """Compute P(x - 0.5 < X <= x + 0.5) for each integer value x."""
    probs = dist_obj.cdf(values + 0.5) - dist_obj.cdf(values - 0.5)
    return np.maximum(probs, 1e-300)


def compute_log_probs(dist_name, params, values):
    """Compute log-probability for each support value.

    For continuous distributions: uses discretized CDF (probability mass in
    [x-0.5, x+0.5]) so that likelihoods are comparable across all model types.
    For discrete distributions: uses PMF directly.
    """
    dist_obj = DIST_MAP[dist_name](params)
    if dist_name in DISCRETE_DISTS:
        log_probs = dist_obj.logpmf(values.astype(int))
    else:
        probs = discretize_continuous(dist_obj, values)
        log_probs = np.log(probs)

    return np.where(np.isfinite(log_probs), log_probs, -700.0)


def compute_metrics(dist_name, params, values, true_probs, cdf_values, cdf_probs):
    """Compute all comparison metrics for a single fitted distribution."""
    dist_obj = DIST_MAP[dist_name](params)
    K = len(values)
    k = NUM_PARAMS[dist_name]

    # --- Information-theoretic metrics (discretized) ---
    log_q = compute_log_probs(dist_name, params, values)

    # Cross-entropy H(p, q) = -sum p_i * log q_i
    cross_entropy = -np.sum(true_probs * log_q)

    # Entropy H(p)
    log_p = np.log(np.maximum(true_probs, 1e-300))
    entropy = -np.sum(true_probs * log_p)

    # KL divergence
    kl_div = cross_entropy - entropy

    # AIC = 2k + 2 * H(p,q)
    aic = 2 * k + 2 * cross_entropy

    # BIC = k * log(K) + 2 * H(p,q)
    bic = k * np.log(K) + 2 * cross_entropy

    # --- CDF-based metrics ---
    theo_cdf = dist_obj.cdf(cdf_values)

    # CvM
    cvm = np.sum((cdf_probs - theo_cdf) ** 2)

    # KS
    ks = np.max(np.abs(cdf_probs - theo_cdf))

    # Anderson-Darling (tail-weighted)
    f_clamped = np.clip(theo_cdf, 1e-10, 1 - 1e-10)
    ad_weights = 1.0 / (f_clamped * (1 - f_clamped))
    ad = np.sum(ad_weights * (cdf_probs - theo_cdf) ** 2)

    # --- PMF-based metrics ---
    if dist_name in DISCRETE_DISTS:
        fitted_pmf = dist_obj.pmf(values.astype(int))
    else:
        fitted_pmf = discretize_continuous(dist_obj, values)
    fitted_pmf = np.maximum(fitted_pmf, 1e-300)

    # L1 / Total Variation
    l1 = np.sum(np.abs(true_probs - fitted_pmf))

    # Chi-squared
    chi2_stat = np.sum((true_probs - fitted_pmf) ** 2 / fitted_pmf)

    return {
        'cross_entropy': cross_entropy,
        'kl_div': kl_div,
        'aic': aic,
        'bic': bic,
        'cvm': cvm,
        'ks': ks,
        'ad': ad,
        'l1': l1,
        'chi2': chi2_stat,
        'n_params': k,
    }


def vuong_test(dist1_name, params1, dist2_name, params2, values, true_probs):
    """Vuong's closeness test between two models.

    Since p(x) is known exactly, we compute the expected per-observation
    log-likelihood ratio and its variance analytically. The effective sample
    size is K (number of support points).

    Returns dict with:
        mean_d:  E_p[log f(x) - log g(x)].  Positive => dist1 closer.
        std_d:   StdDev of per-point log-ratio under p.
        vuong_stat:  sqrt(K) * mean_d / std_d  (asymptotically N(0,1) under H0).
        p_value: two-sided p-value.
        corrected_stat / corrected_p:  AIC-corrected variant (Vuong 1989, eq. 5.7).
    """
    log_f = compute_log_probs(dist1_name, params1, values)
    log_g = compute_log_probs(dist2_name, params2, values)

    d = log_f - log_g
    mean_d = np.sum(true_probs * d)
    var_d = np.sum(true_probs * (d - mean_d) ** 2)
    std_d = np.sqrt(var_d) if var_d > 0 else 1e-300

    K = len(values)
    k1 = NUM_PARAMS[dist1_name]
    k2 = NUM_PARAMS[dist2_name]

    vuong_stat = np.sqrt(K) * mean_d / std_d
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(vuong_stat)))

    # AIC-corrected: penalize more complex model
    aic_penalty = (k1 - k2) / (2.0 * K)
    corrected_mean = mean_d - aic_penalty
    corrected_stat = np.sqrt(K) * corrected_mean / std_d
    corrected_p = 2 * (1 - sp_stats.norm.cdf(abs(corrected_stat)))

    return {
        'mean_d': mean_d,
        'std_d': std_d,
        'vuong_stat': vuong_stat,
        'p_value': p_value,
        'corrected_stat': corrected_stat,
        'corrected_p': corrected_p,
        'k1': k1,
        'k2': k2,
    }


METRIC_NAMES = ['cross_entropy', 'kl_div', 'aic', 'bic',
                'cvm', 'ks', 'ad', 'l1', 'chi2']
METRIC_SHORT = {
    'cross_entropy': 'H(p,q)',
    'kl_div': 'KL',
    'aic': 'AIC',
    'bic': 'BIC',
    'cvm': 'CvM',
    'ks': 'KS',
    'ad': 'AD',
    'l1': 'L1',
    'chi2': 'Chi2',
}


def print_ranking(results, sort_by='aic', top_n=None):
    """Pretty-print ranked results table."""
    sorted_results = sorted(results, key=lambda r: r['metrics'][sort_by])
    if top_n:
        sorted_results = sorted_results[:top_n]

    # Header
    header = f"{'Rank':>4}  {'Distribution':<14} {'k':>2}"
    for m in METRIC_NAMES:
        header += f"  {METRIC_SHORT[m]:>12}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(sorted_results, 1):
        line = f"{i:>4}  {r['dist_name']:<14} {r['metrics']['n_params']:>2}"
        for m in METRIC_NAMES:
            val = r['metrics'][m]
            if abs(val) < 0.001 or abs(val) > 99999:
                line += f"  {val:>12.4e}"
            else:
                line += f"  {val:>12.6f}"
        print(line)

    print(f"\n(Sorted by {METRIC_SHORT[sort_by]}; lower is better for all metrics)")


def print_vuong(dist1, dist2, result):
    """Pretty-print Vuong test result."""
    print(f"\nVuong's Closeness Test: {dist1} vs {dist2}")
    print("=" * 60)
    print(f"  {dist1}: {result['k1']} free parameters")
    print(f"  {dist2}: {result['k2']} free parameters")
    print()
    print(f"  E[log f(x) - log g(x)] = {result['mean_d']:+.6e}")
    print(f"  StdDev                  = {result['std_d']:.6e}")
    print()
    print(f"  Vuong statistic (raw)   = {result['vuong_stat']:+.4f}")
    print(f"  p-value (raw)           = {result['p_value']:.4e}")
    print()
    print(f"  Vuong statistic (AIC-corrected) = {result['corrected_stat']:+.4f}")
    print(f"  p-value (AIC-corrected)         = {result['corrected_p']:.4e}")
    print()

    alpha = 0.05
    if result['p_value'] < alpha:
        if result['vuong_stat'] > 0:
            winner = dist1
        else:
            winner = dist2
        print(f"  Conclusion (alpha={alpha}): {winner} is significantly closer")
        print(f"    to the true distribution (reject H0: equal closeness).")
    else:
        print(f"  Conclusion (alpha={alpha}): Cannot distinguish the two models")
        print(f"    (fail to reject H0: equal closeness).")

    if result['corrected_p'] < alpha:
        if result['corrected_stat'] > 0:
            winner_c = dist1
        else:
            winner_c = dist2
        print(f"  AIC-corrected:  {winner_c} is significantly closer.")
    else:
        print(f"  AIC-corrected:  Cannot distinguish (accounting for complexity).")


def write_csv(all_rows, csv_path):
    """Write results to CSV file."""
    import csv
    fieldnames = ['N', 'n', 'distribution', 'n_params'] + METRIC_NAMES
    with open(csv_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote {len(all_rows)} rows to {csv_path}")


def parse_Nn(fname):
    """Parse N and n from filename like N_100_n_50.txt."""
    base = fname.replace('.txt', '')
    parts = base.split('_')
    # N_{val}_n_{val}
    N = int(parts[1])
    n = int(parts[3])
    return N, n


def run_single(N, n, dist_names, mle, sort_by, top_n, vuong_pair):
    """Run analysis for a single (N, n) pair."""
    values, true_probs, cdf_values, cdf_probs = load_true_distribution(N, n)

    if vuong_pair:
        d1, d2 = vuong_pair
        fname = f"N_{N}_n_{n}.txt"
        prefix = "../data/fitted/mle_" if mle else "../data/fitted/cvm_"
        for d in [d1, d2]:
            p = os.path.join(f"{prefix}{d}", fname)
            if not os.path.isfile(p):
                print(f"Error: {p} not found", file=sys.stderr)
                sys.exit(1)
        params1, _ = load_fitted_params(os.path.join(f"{prefix}{d1}", fname))
        params2, _ = load_fitted_params(os.path.join(f"{prefix}{d2}", fname))

        vr = vuong_test(d1, params1, d2, params2, values, true_probs)
        print_vuong(d1, d2, vr)

        # Also show metrics for both
        m1 = compute_metrics(d1, params1, values, true_probs, cdf_values, cdf_probs)
        m2 = compute_metrics(d2, params2, values, true_probs, cdf_values, cdf_probs)
        print(f"\nIndividual metrics:")
        results = [
            {'dist_name': d1, 'metrics': m1},
            {'dist_name': d2, 'metrics': m2},
        ]
        print_ranking(results, sort_by=sort_by)
        return []

    # Rank all
    if dist_names is None:
        dist_names = get_available_distributions(N, n, mle=mle)
    if not dist_names:
        print(f"No fitted distributions found for N={N}, n={n}", file=sys.stderr)
        return []

    fname = f"N_{N}_n_{n}.txt"
    prefix = "../data/fitted/mle_" if mle else "../data/fitted/cvm_"

    results = []
    for dist_name in dist_names:
        path = os.path.join(f"{prefix}{dist_name}", fname)
        if not os.path.isfile(path):
            continue
        params, _ = load_fitted_params(path)
        metrics = compute_metrics(dist_name, params, values, true_probs,
                                  cdf_values, cdf_probs)
        results.append({'dist_name': dist_name, 'metrics': metrics})

    if results:
        print(f"\n=== N={N}, n={n} ({len(values)} support points, "
              f"{'MLE' if mle else 'CvM'} fits) ===\n")
        print_ranking(results, sort_by=sort_by, top_n=top_n)

    csv_rows = []
    for r in results:
        row = {'N': N, 'n': n, 'distribution': r['dist_name'],
               'n_params': r['metrics']['n_params']}
        for m in METRIC_NAMES:
            row[m] = r['metrics'][m]
        csv_rows.append(row)
    return csv_rows


def main():
    parser = argparse.ArgumentParser(
        description="Model selection: rank fitted distributions and Vuong tests")
    parser.add_argument('-N', type=int, help="N parameter")
    parser.add_argument('-n', type=int, help="n parameter")
    parser.add_argument('-d', '--distribution', action='append', default=None,
                        choices=SUPPORTED_DISTRIBUTIONS,
                        help="Restrict to these distributions (repeatable)")
    parser.add_argument('--mle', action='store_true',
                        help="Use MLE fits (fitted/mle_*) instead of CvM fits")
    parser.add_argument('--vuong', nargs=2, metavar=('DIST1', 'DIST2'),
                        help="Pairwise Vuong test between two distributions")
    parser.add_argument('--sort', default='aic', choices=METRIC_NAMES,
                        help="Metric to sort by (default: aic)")
    parser.add_argument('--top', type=int, default=None,
                        help="Show only top K distributions")
    parser.add_argument('--all', action='store_true',
                        help="Run over all (N,n) pairs found in exact_distributions/")
    parser.add_argument('--csv', default=None, metavar='PATH',
                        help="Write results to CSV file")

    args = parser.parse_args()

    if args.vuong:
        for d in args.vuong:
            if d not in SUPPORTED_DISTRIBUTIONS:
                print(f"Error: unknown distribution '{d}'", file=sys.stderr)
                sys.exit(1)

    if args.all:
        # Run over all (N, n) pairs
        fnames = sorted(os.listdir("../data/exact_distributions"))
        all_csv_rows = []
        for fname in fnames:
            if not fname.endswith('.txt'):
                continue
            try:
                N, n = parse_Nn(fname)
            except (ValueError, IndexError):
                continue
            rows = run_single(N, n, args.distribution, args.mle,
                              args.sort, args.top, args.vuong)
            all_csv_rows.extend(rows)

        if args.csv and all_csv_rows:
            write_csv(all_csv_rows, args.csv)
    else:
        if args.N is None or args.n is None:
            print("Error: -N and -n are required (or use --all)", file=sys.stderr)
            sys.exit(1)
        rows = run_single(args.N, args.n, args.distribution, args.mle,
                          args.sort, args.top, args.vuong)
        if args.csv and rows:
            write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
