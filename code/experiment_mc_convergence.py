#!/usr/bin/env python3
"""Experiment: MC convergence for beta approximation of DTV distribution.

Tests how many Monte Carlo samples K are needed for the beta approximation
to converge to the best achievable fit (i.e., the one obtained from the
exact distribution). Reports p-value error at alpha=0.05 as a function of K
for several (N, n) pairs.
"""

import os
import numpy as np
from scipy import stats
from scipy.optimize import minimize


def load_cdf(path):
    """Load exact CDF from file."""
    vals, cdfs = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                vals.append(int(float(p[0])))
                cdfs.append(float(p[1]))
    return np.array(vals), np.array(cdfs)


def fit_beta_to_cdf(vals, cdf_probs):
    """Fit beta distribution by minimizing CvM statistic against a CDF."""
    max_val = float(vals[-1]) if len(vals) > 0 else 1.0
    eps = 1e-10

    # Estimate initial parameters from CDF
    # Approximate mean and variance from CDF
    pmf = np.diff(np.concatenate([[0.0], cdf_probs]))
    mean = np.sum(vals * pmf)
    var = np.sum(vals**2 * pmf) - mean**2
    var = max(var, eps)

    m = np.clip(mean / max(max_val, eps), eps, 1 - eps)
    v = var / max(max_val, eps) ** 2
    if v <= 0 or v >= m * (1 - m):
        v = m * (1 - m) / 2
    sum_ab = m * (1 - m) / v - 1
    sum_ab = max(sum_ab, eps)
    init_a = max(m * sum_ab, eps)
    init_b = max((1 - m) * sum_ab, eps)

    def cvm_objective(log_params):
        a = np.exp(log_params[0])
        b = np.exp(log_params[1])
        sc = np.exp(log_params[2])
        theo = stats.beta.cdf(vals, a=a, b=b, loc=0, scale=sc)
        return np.sum((cdf_probs - theo) ** 2)

    result = minimize(cvm_objective,
                      x0=[np.log(init_a), np.log(init_b), np.log(max_val)],
                      method="Nelder-Mead")
    a = np.exp(result.x[0])
    b = np.exp(result.x[1])
    scale = np.exp(result.x[2])
    return (a, b, 0.0, scale)


def fit_beta_mc(N, n, K, rng):
    """Fit beta to DTV via MC sampling under H0."""
    dtvs = np.zeros(K, dtype=int)
    probs = [1.0 / n] * n
    for i in range(K):
        h = rng.multinomial(N, probs)
        dtvs[i] = int(np.sum(np.abs(np.diff(h))))

    unique_vals = np.unique(dtvs)
    counts = np.array([np.sum(dtvs == v) for v in unique_vals])
    cdf_probs = np.cumsum(counts) / K

    return fit_beta_to_cdf(unique_vals, cdf_probs)


def compute_pvalue_error(exact_vals, exact_cdfs, beta_params, alpha=0.05):
    """Compute absolute p-value error at alpha level."""
    # Find exact critical value (CDF closest to 1-alpha)
    target = 1 - alpha
    best_idx = np.argmin(np.abs(exact_cdfs - target))
    exact_crit = exact_vals[best_idx]
    exact_p = 1.0 - exact_cdfs[best_idx]

    # Beta p-value at same critical value
    bd = stats.beta(a=beta_params[0], b=beta_params[1],
                    loc=beta_params[2], scale=beta_params[3])
    beta_p = 1.0 - bd.cdf(exact_crit - 0.5)

    return abs(exact_p - beta_p)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MC convergence experiment for beta approximation")
    parser.add_argument("--K-values", type=int, nargs="+",
                        default=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
                        help="MC sample sizes to test")
    parser.add_argument("--n-repeats", type=int, default=50,
                        help="Number of repeats per K value")
    parser.add_argument("--seed", type=int, default=13,
                        help="Base random seed")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level")
    args = parser.parse_args()

    # Test cases: various (N, n) pairs
    test_cases = [
        (50, 5), (50, 10), (100, 5), (100, 10), (100, 20),
        (200, 10), (200, 20), (300, 10), (300, 30),
    ]

    K_values = args.K_values
    n_repeats = args.n_repeats
    base_seed = args.seed
    alpha = args.alpha

    print(f"MC convergence experiment")
    print(f"K values: {K_values}")
    print(f"Repeats per K: {n_repeats}")
    print(f"Seed: {base_seed}, alpha: {alpha}")
    print()

    # Header
    k_header = "".join(f"{'K=' + str(k):>12s}" for k in K_values)
    print(f"{'(N, n)':>12s} {'exact_beta':>12s} {k_header}")
    print("-" * (24 + 12 * len(K_values)))

    all_results = {}

    for N, n in test_cases:
        cdf_path = f"../data/cdf_exact/N_{N}_n_{n}.txt"
        if not os.path.isfile(cdf_path):
            print(f"({N:3d},{n:3d})  -- exact CDF not found, skipping")
            continue

        exact_vals, exact_cdfs = load_cdf(cdf_path)

        # Fit beta to exact CDF (ground truth best fit)
        exact_beta = fit_beta_to_cdf(exact_vals, exact_cdfs)
        exact_error = compute_pvalue_error(exact_vals, exact_cdfs, exact_beta, alpha)

        # MC fits for each K
        mc_errors = {}
        for K in K_values:
            errors = []
            for rep in range(n_repeats):
                rng = np.random.RandomState(base_seed + rep)
                mc_params = fit_beta_mc(N, n, K, rng)
                err = compute_pvalue_error(exact_vals, exact_cdfs, mc_params, alpha)
                errors.append(err)
            mc_errors[K] = errors

        # Print results (mean +/- std)
        exact_str = f"{exact_error:.6f}"
        mc_strs = []
        for K in K_values:
            errs = mc_errors[K]
            mc_strs.append(f"{np.mean(errs):.6f}")
        mc_line = "".join(f"{s:>12s}" for s in mc_strs)
        print(f"({N:3d},{n:3d})  {exact_str:>12s} {mc_line}")

        all_results[(N, n)] = {
            "exact_error": exact_error,
            "mc_errors": {K: mc_errors[K] for K in K_values},
        }

    # Summary statistics
    print()
    print("=== Summary: mean p-value error across all (N, n) pairs ===")
    k_header = "".join(f"{'K=' + str(k):>12s}" for k in K_values)
    print(f"{'Statistic':>12s} {'exact_beta':>12s} {k_header}")
    print("-" * (24 + 12 * len(K_values)))

    if all_results:
        exact_errors = [r["exact_error"] for r in all_results.values()]
        for stat_name, stat_fn in [("mean", np.mean), ("median", np.median),
                                    ("max", np.max)]:
            exact_str = f"{stat_fn(exact_errors):.6f}"
            mc_strs = []
            for K in K_values:
                all_errs = []
                for r in all_results.values():
                    all_errs.extend(r["mc_errors"][K])
                mc_strs.append(f"{stat_fn(all_errs):.6f}")
            mc_line = "".join(f"{s:>12s}" for s in mc_strs)
            print(f"{stat_name:>12s} {exact_str:>12s} {mc_line}")

        # Print std of MC errors (variability across repeats)
        print()
        print("=== MC variability: std of p-value error across repeats ===")
        print(f"{'(N, n)':>12s} {k_header}")
        print("-" * (12 + 12 * len(K_values)))
        for (N, n), r in sorted(all_results.items()):
            mc_stds = []
            for K in K_values:
                mc_stds.append(f"{np.std(r['mc_errors'][K]):.6f}")
            mc_line = "".join(f"{s:>12s}" for s in mc_stds)
            print(f"({N:3d},{n:3d})  {mc_line}")


if __name__ == "__main__":
    main()
