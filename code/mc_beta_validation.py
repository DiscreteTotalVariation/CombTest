#!/usr/bin/env python3
"""Validate Monte Carlo + beta fitting against exact DTV distributions.

For several (N,n) pairs where we have exact results, compare:
1. Exact beta fit (from exact distribution)
2. MC beta fit (from Monte Carlo samples)
3. Pure MC empirical p-values

Shows that MC+beta is a valid approach for large N,n where exact is infeasible.
"""

import os
import numpy as np
from scipy import stats, optimize


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
        return list(map(float, f.readline().strip().split()))


def compute_dtv(histogram):
    """Compute DTV of a histogram: sum of |h_{i+1} - h_i| (adjacent differences)."""
    return sum(abs(histogram[i+1] - histogram[i]) for i in range(len(histogram) - 1))


def sample_dtv_mc(N, n, K=100000):
    """Generate K Monte Carlo DTV samples under H0 (uniform)."""
    dtvs = np.empty(K)
    for i in range(K):
        counts = np.random.multinomial(N, [1.0 / n] * n)
        dtvs[i] = np.sum(np.abs(np.diff(counts)))
    return dtvs


def fit_beta_to_samples(samples):
    """Fit 4-parameter beta to DTV samples using CvM minimization."""
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    ecdf = (np.arange(1, n + 1) - 0.5) / n

    def cvm_objective(params):
        a, b, loc, scale = params
        if a <= 0 or b <= 0 or scale <= 0:
            return 1e10
        try:
            theo_cdf = stats.beta.cdf(sorted_samples, a, b, loc=loc, scale=scale)
            return np.mean((ecdf - theo_cdf) ** 2)
        except Exception:
            return 1e10

    # Initial guess from method of moments
    mu = np.mean(samples)
    var = np.var(samples)
    lo = np.min(samples) - 0.5
    hi = np.max(samples) + 0.5
    scale0 = hi - lo
    mu_std = (mu - lo) / scale0
    var_std = var / (scale0 ** 2)
    if var_std > 0 and mu_std > 0 and mu_std < 1:
        common = mu_std * (1 - mu_std) / var_std - 1
        a0 = max(mu_std * common, 0.5)
        b0 = max((1 - mu_std) * common, 0.5)
    else:
        a0, b0 = 2.0, 2.0

    result = optimize.minimize(cvm_objective, [a0, b0, lo, scale0],
                               method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
    return result.x


def main():
    np.random.seed(42)
    K = 200000  # MC samples

    test_pairs = [
        (20, 4), (30, 6), (50, 10), (80, 8), (100, 10),
        (100, 20), (150, 15), (200, 10), (200, 20), (200, 50),
        (300, 10), (300, 30), (300, 100), (500, 50),
    ]

    print(f"{'N':>5s} {'n':>4s} | {'Exact p':>10s} {'MC-Beta p':>10s} {'MC-Emp p':>10s} "
          f"| {'|Ex-MCB|':>10s} {'|Ex-MCE|':>10s} | {'Exact d*':>8s} {'MCB d*':>8s} {'MCE d*':>8s}")
    print("-" * 110)

    results = []

    for N, n in test_pairs:
        cdf_path = f'../data/cdf_exact/N_{N}_n_{n}.txt'
        beta_path = f'../data/fitted/cvm_beta/N_{N}_n_{n}.txt'

        if not os.path.isfile(cdf_path) or not os.path.isfile(beta_path):
            print(f"{N:>5d} {n:>4d} | SKIPPED (no exact data)")
            continue

        # 1. Load exact CDF and exact beta fit
        cdf_values, cdf_probs = load_cdf(cdf_path)
        exact_params = load_fitted_params(beta_path)
        exact_beta = stats.beta(a=exact_params[0], b=exact_params[1],
                                loc=exact_params[2], scale=exact_params[3])

        # 2. Monte Carlo sampling
        mc_samples = sample_dtv_mc(N, n, K)

        # 3. Fit beta to MC samples
        mc_params = fit_beta_to_samples(mc_samples)
        mc_beta = stats.beta(a=mc_params[0], b=mc_params[1],
                             loc=mc_params[2], scale=mc_params[3])

        # 4. Compare p-values at the exact critical d (closest to alpha=0.05)
        # Find exact critical d
        target = 0.95
        best_idx = np.argmin(np.abs(cdf_probs - target))
        d_star = cdf_values[best_idx]
        exact_pval = 1.0 - cdf_probs[best_idx]

        # Exact beta p-value at d_star
        exact_beta_pval = 1.0 - exact_beta.cdf(d_star + 0.5)

        # MC beta p-value at d_star
        mc_beta_pval = 1.0 - mc_beta.cdf(d_star + 0.5)

        # MC empirical p-value at d_star
        mc_emp_pval = np.mean(mc_samples > d_star)

        # Also find critical d for each method
        # MC-beta critical d
        all_d = np.arange(0, 2 * N + 1)
        mc_beta_cdf = mc_beta.cdf(all_d + 0.5)
        mc_beta_d_star = all_d[np.argmin(np.abs(mc_beta_cdf - 0.95))]

        # MC empirical critical d
        mc_emp_cdf = np.array([np.mean(mc_samples <= d) for d in all_d[:d_star + 20]])
        mc_emp_d_star = np.arange(len(mc_emp_cdf))[np.argmin(np.abs(mc_emp_cdf - 0.95))]

        print(f"{N:>5d} {n:>4d} | {exact_pval:10.6f} {mc_beta_pval:10.6f} {mc_emp_pval:10.6f} "
              f"| {abs(exact_pval - mc_beta_pval):10.6f} {abs(exact_pval - mc_emp_pval):10.6f} "
              f"| {d_star:>8d} {mc_beta_d_star:>8d} {mc_emp_d_star:>8d}")

        results.append({
            'N': N, 'n': n, 'd_star': int(d_star),
            'exact_p': exact_pval,
            'mc_beta_p': mc_beta_pval,
            'mc_emp_p': mc_emp_pval,
            'mc_beta_d': int(mc_beta_d_star),
            'mc_emp_d': int(mc_emp_d_star),
        })

    # Summary
    if results:
        mc_beta_errors = [abs(r['exact_p'] - r['mc_beta_p']) for r in results]
        mc_emp_errors = [abs(r['exact_p'] - r['mc_emp_p']) for r in results]
        d_matches_mcb = sum(1 for r in results if r['d_star'] == r['mc_beta_d'])
        d_matches_mce = sum(1 for r in results if r['d_star'] == r['mc_emp_d'])
        d_within1_mcb = sum(1 for r in results if abs(r['d_star'] - r['mc_beta_d']) <= 1)
        d_within1_mce = sum(1 for r in results if abs(r['d_star'] - r['mc_emp_d']) <= 1)

        print(f"\n{'='*60}")
        print(f"Summary ({len(results)} pairs, K={K} MC samples)")
        print(f"{'='*60}")
        print(f"  MC-Beta  mean p-error: {np.mean(mc_beta_errors):.6f}")
        print(f"  MC-Empir mean p-error: {np.mean(mc_emp_errors):.6f}")
        print(f"  MC-Beta  d* exact match: {d_matches_mcb}/{len(results)}")
        print(f"  MC-Empir d* exact match: {d_matches_mce}/{len(results)}")
        print(f"  MC-Beta  d* within 1:    {d_within1_mcb}/{len(results)}")
        print(f"  MC-Empir d* within 1:    {d_within1_mce}/{len(results)}")


if __name__ == '__main__':
    main()
