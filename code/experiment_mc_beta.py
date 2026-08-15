#!/usr/bin/env python3
"""Validate Monte Carlo + beta fitting for DTV p-value computation.

Demonstrates that when exact DP computation is infeasible (large N, n),
Monte Carlo sampling combined with beta distribution fitting provides
accurate p-values. Compares MC-beta against exact results for (N, n)
pairs where both are available.
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


def sample_dtv_mc(N, n, K):
    """Generate K Monte Carlo DTV samples under H0 (uniform histogram)."""
    dtvs = np.empty(K)
    for i in range(K):
        counts = np.random.multinomial(N, [1.0 / n] * n)
        dtvs[i] = np.sum(np.abs(np.diff(counts)))
    return dtvs


def fit_beta_cvm(samples):
    """Fit 4-parameter beta to samples via Cramer-von Mises minimization."""
    sorted_s = np.sort(samples)
    n = len(sorted_s)
    ecdf = (np.arange(1, n + 1) - 0.5) / n

    def objective(params):
        a, b, loc, scale = params
        if a <= 0 or b <= 0 or scale <= 0:
            return 1e10
        try:
            tcdf = stats.beta.cdf(sorted_s, a, b, loc=loc, scale=scale)
            return np.mean((ecdf - tcdf) ** 2)
        except Exception:
            return 1e10

    mu = np.mean(samples)
    var = np.var(samples)
    lo = np.min(samples) - 0.5
    hi = np.max(samples) + 0.5
    sc = hi - lo
    mu_s = (mu - lo) / sc
    var_s = var / (sc ** 2)
    if var_s > 0 and 0 < mu_s < 1:
        c = mu_s * (1 - mu_s) / var_s - 1
        a0 = max(mu_s * c, 0.5)
        b0 = max((1 - mu_s) * c, 0.5)
    else:
        a0, b0 = 2.0, 2.0

    result = optimize.minimize(objective, [a0, b0, lo, sc],
                               method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-10,
                                        'fatol': 1e-12})
    return result.x


def main():
    np.random.seed(42)

    test_pairs = [
        (20, 4), (30, 6), (50, 10), (80, 8), (100, 10),
        (100, 20), (150, 15), (200, 10), (200, 20), (300, 30),
    ]
    K_values = [10000, 50000, 200000]

    print("Monte Carlo + Beta validation against exact DTV distribution")
    print("=" * 90)

    for K in K_values:
        print(f"\n--- K = {K} MC samples ---")
        print(f"{'N':>5s} {'n':>4s} | {'Exact p':>10s} {'MC-Beta p':>10s} "
              f"{'MC-Emp p':>10s} | {'|Err MCB|':>10s} {'|Err MCE|':>10s} | "
              f"{'d* ex':>6s} {'d* MCB':>6s}")
        print("-" * 85)

        errors_mcb = []
        errors_mce = []
        d_matches = 0
        d_within1 = 0
        total = 0

        for N, n in test_pairs:
            cdf_path = f'../data/cdf_exact/N_{N}_n_{n}.txt'
            beta_path = f'../data/fitted/cvm_beta/N_{N}_n_{n}.txt'
            if not os.path.isfile(cdf_path) or not os.path.isfile(beta_path):
                continue

            cdf_values, cdf_probs = load_cdf(cdf_path)

            # Exact critical d at alpha=0.05
            best_idx = np.argmin(np.abs(cdf_probs - 0.95))
            d_star = cdf_values[best_idx]
            exact_pval = 1.0 - cdf_probs[best_idx]

            # MC sampling
            mc = sample_dtv_mc(N, n, K)

            # Fit beta to MC samples
            mc_params = fit_beta_cvm(mc)
            mc_beta = stats.beta(a=mc_params[0], b=mc_params[1],
                                 loc=mc_params[2], scale=mc_params[3])

            # MC-beta p-value
            mc_beta_pval = 1.0 - mc_beta.cdf(d_star + 0.5)

            # MC empirical p-value
            mc_emp_pval = np.mean(mc > d_star)

            # MC-beta critical d
            all_d = np.arange(0, 2 * N + 1)
            mc_cdf = mc_beta.cdf(all_d + 0.5)
            mc_d = all_d[np.argmin(np.abs(mc_cdf - 0.95))]

            err_mcb = abs(exact_pval - mc_beta_pval)
            err_mce = abs(exact_pval - mc_emp_pval)
            errors_mcb.append(err_mcb)
            errors_mce.append(err_mce)

            if d_star == mc_d:
                d_matches += 1
            if abs(d_star - mc_d) <= 1:
                d_within1 += 1
            total += 1

            print(f"{N:5d} {n:4d} | {exact_pval:10.6f} {mc_beta_pval:10.6f} "
                  f"{mc_emp_pval:10.6f} | {err_mcb:10.6f} {err_mce:10.6f} | "
                  f"{d_star:>6d} {int(mc_d):>6d}")

        if total > 0:
            print(f"\nSummary (K={K}):")
            print(f"  MC-Beta  mean |error|: {np.mean(errors_mcb):.6f}")
            print(f"  MC-Empir mean |error|: {np.mean(errors_mce):.6f}")
            print(f"  MC-Beta  d* exact match: {d_matches}/{total}")
            print(f"  MC-Beta  d* within 1:    {d_within1}/{total}")


if __name__ == '__main__':
    main()
