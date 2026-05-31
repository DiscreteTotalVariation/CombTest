#!/usr/bin/env python3
"""Generate ADC alternating DNL experiment data.

Simulates an ADC with alternating differential nonlinearity (DNL), where even
codes have slightly wider bins and odd codes have slightly narrower bins.
Computes power of test as a function of sample size N for several DNL amplitudes.
Saves results to a JSON file.
"""

import argparse
import json
import os
import numpy as np
from scipy import stats
from tqdm import tqdm


def compute_dtv(histogram):
    """Compute DTV: sum of |h_{i+1} - h_i| (adjacent bin differences)."""
    return int(np.sum(np.abs(np.diff(histogram))))


def load_cdf(path):
    """Load exact CDF from file."""
    values, probs = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                values.append(int(float(parts[0])))
                probs.append(float(parts[1]))
    return np.array(values), np.array(probs)


def load_beta_params(path):
    """Load fitted beta parameters from file."""
    with open(path) as f:
        return list(map(float, f.readline().strip().split()))


_approx = "gamma"  # set from --approx flag


def fit_gamma_mc(N, n, K=50000, rng=None):
    """Fit gamma distribution to DTV via Monte Carlo sampling under H0."""
    from scipy.optimize import minimize
    if rng is None:
        rng = np.random.default_rng()
    probs = np.full(n, 1.0 / n)
    hists = rng.multinomial(N, probs, size=K)
    dtvs = np.sum(np.abs(np.diff(hists, axis=1)), axis=1)
    unique_vals, counts = np.unique(dtvs, return_counts=True)
    cdf_probs = np.cumsum(counts) / K

    mean_dtv = np.mean(dtvs)
    var_dtv = np.var(dtvs)
    if var_dtv <= 0:
        var_dtv = 1.0
    init_a = mean_dtv**2 / var_dtv
    init_scale = var_dtv / mean_dtv

    def cvm_objective(log_params):
        a = np.exp(log_params[0])
        sc = np.exp(log_params[1])
        theo = stats.gamma.cdf(unique_vals, a=a, loc=0, scale=sc)
        return np.sum((cdf_probs - theo) ** 2)

    result = minimize(cvm_objective,
                      x0=[np.log(max(init_a, 1e-10)),
                          np.log(max(init_scale, 1e-10))],
                      method="Nelder-Mead")
    a = np.exp(result.x[0])
    scale = np.exp(result.x[1])
    return (a, 0.0, scale)


def fit_beta_mc(N, n, K=50000, rng=None):
    """Fit beta distribution to DTV via Monte Carlo sampling under H0."""
    from scipy.optimize import minimize
    if rng is None:
        rng = np.random.default_rng()
    probs = np.full(n, 1.0 / n)
    hists = rng.multinomial(N, probs, size=K)
    dtvs = np.sum(np.abs(np.diff(hists, axis=1)), axis=1)
    unique_vals, counts = np.unique(dtvs, return_counts=True)
    cdf_probs = np.cumsum(counts) / K
    max_val = float(unique_vals[-1]) if len(unique_vals) > 0 else 1.0
    eps = 1e-10
    pmf = np.diff(np.concatenate([[0.0], cdf_probs]))
    mean = np.sum(unique_vals * pmf)
    var = np.sum(unique_vals**2 * pmf) - mean**2
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
        theo = stats.beta.cdf(unique_vals, a=a, b=b, loc=0, scale=sc)
        return np.sum((cdf_probs - theo) ** 2)

    result = minimize(cvm_objective,
                      x0=[np.log(init_a), np.log(init_b), np.log(max_val)],
                      method="Nelder-Mead")
    a = np.exp(result.x[0])
    b = np.exp(result.x[1])
    scale = np.exp(result.x[2])
    return (a, b, 0.0, scale)


# Cache for MC-fitted params: (N, n) -> params
_mc_beta_cache = {}
_mc_gamma_cache = {}


def ct_pvalue(dtv_val, N, n):
    """Compute CT p-value P(DTV >= d) using exact CDF if available, else fitted approximation."""
    cdf_path = "../data/cdf_exact/N_%d_n_%d.txt" % (N, n)
    if os.path.isfile(cdf_path):
        cdf_values, cdf_probs = load_cdf(cdf_path)
        idx = np.searchsorted(cdf_values, dtv_val - 1, side="right") - 1
        if idx < 0:
            return 1.0
        return 1.0 - cdf_probs[idx]

    if _approx == "gamma":
        gamma_path = "../data/fitted/cvm_gamma/N_%d_n_%d.txt" % (N, n)
        if os.path.isfile(gamma_path):
            params = load_beta_params(gamma_path)
            gd = stats.gamma(a=params[0], loc=params[1], scale=params[2])
            return 1.0 - gd.cdf(dtv_val - 0.5)
        key = (N, n)
        if key not in _mc_gamma_cache:
            _mc_gamma_cache[key] = fit_gamma_mc(N, n, K=50000,
                                                rng=np.random.default_rng(seed=N * 1000 + n))
        params = _mc_gamma_cache[key]
        gd = stats.gamma(a=params[0], loc=params[1], scale=params[2])
        return 1.0 - gd.cdf(dtv_val - 0.5)
    else:
        beta_path = "../data/fitted/cvm_beta/N_%d_n_%d.txt" % (N, n)
        if os.path.isfile(beta_path):
            params = load_beta_params(beta_path)
            bd = stats.beta(a=params[0], b=params[1], loc=params[2], scale=params[3])
            return 1.0 - bd.cdf(dtv_val - 0.5)
        key = (N, n)
        if key not in _mc_beta_cache:
            _mc_beta_cache[key] = fit_beta_mc(N, n, K=50000,
                                              rng=np.random.default_rng(seed=N * 1000 + n))
        params = _mc_beta_cache[key]
        bd = stats.beta(a=params[0], b=params[1], loc=params[2], scale=params[3])
        return 1.0 - bd.cdf(dtv_val - 0.5)


def chi2_pvalue(histogram):
    """Compute Pearson's chi-squared p-value for uniformity."""
    N = int(np.sum(histogram))
    n = len(histogram)
    expected = N / n
    chi2_stat = np.sum((histogram - expected) ** 2 / expected)
    return 1.0 - stats.chi2.cdf(chi2_stat, df=n - 1)


def gtest_pvalue(histogram):
    """Compute G-test (log-likelihood ratio) p-value for uniformity."""
    N = int(np.sum(histogram))
    n = len(histogram)
    expected = N / n
    h_safe = np.where(histogram > 0, histogram, 0.5)
    G = 2 * np.sum(histogram * np.log(h_safe / expected))
    return 1.0 - stats.chi2.cdf(G, df=n - 1)


def generate_adc_histogram(N, n_codes, dnl_amplitude):
    """Simulate ADC with alternating DNL."""
    widths = np.array([1 + dnl_amplitude * (1 if i % 2 == 0 else -1)
                       for i in range(n_codes)], dtype=float)
    probs = widths / widths.sum()
    return np.random.multinomial(N, probs)


def compute_power(N, n_codes, dnl_amplitude, n_trials, alpha=0.05):
    """Compute power for CT, chi-squared, and G-test at given parameters."""
    ct_rejections = 0
    chi2_rejections = 0
    gtest_rejections = 0
    ct_valid = 0

    for _ in range(n_trials):
        h = generate_adc_histogram(N, n_codes, dnl_amplitude)
        dtv = compute_dtv(h)

        if chi2_pvalue(h) < alpha:
            chi2_rejections += 1

        if gtest_pvalue(h) < alpha:
            gtest_rejections += 1

        p = ct_pvalue(dtv, N, n_codes)
        if p is not None:
            ct_valid += 1
            if p < alpha:
                ct_rejections += 1

    return {
        "ct_power": ct_rejections / ct_valid if ct_valid > 0 else None,
        "chi2_power": chi2_rejections / n_trials,
        "gtest_power": gtest_rejections / n_trials,
        "ct_valid": ct_valid,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate ADC alternating DNL experiment data.")
    parser.add_argument("--n-codes", type=int, default=10, help="Number of ADC output codes (default: 10)")
    parser.add_argument("--n-trials", type=int, default=50000, help="Number of trials (default: 50000)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed (default: 13)")
    parser.add_argument("--approx", choices=["gamma", "beta"], default="gamma",
                        help="Approximation distribution for CT (default: gamma)")
    parser.add_argument("-o", "--output", type=str, default="../data/adc_dnl_results.json",
                        help="Output JSON file path (default: adc_dnl_results.json)")
    args = parser.parse_args()

    global _approx
    _approx = args.approx

    np.random.seed(args.seed)
    n_codes = args.n_codes
    n_trials = args.n_trials
    alpha = args.alpha

    print("ADC Alternating DNL Detection: CT vs Chi-squared vs G-test")
    print("n_codes=%d, n_trials=%d, alpha=%.2f" % (n_codes, n_trials, alpha))
    print()

    # Verify false positive rate
    print("=== Verifying false positive rate under H0 (uniform) ===")
    print("Expected: ~%.2f for all tests" % alpha)
    print("%5s | %8s | %8s | %8s" % ("N", "CT", "Chi2", "G-test"))
    print("-" * 40)
    for N in [50, 100, 200, 500]:
        r = compute_power(N, n_codes, 0.0, n_trials, alpha)
        ct = "%.4f" % r["ct_power"] if r["ct_power"] is not None else "N/A"
        print("%5d | %8s | %8.4f | %8.4f" % (N, ct, r["chi2_power"], r["gtest_power"]))
    print()

    # Run power experiment
    dnl_values = [0.05, 0.10, 0.20]
    N_range = list(range(20, 101, 10)) + list(range(120, 301, 20)) + \
              list(range(350, 501, 50))

    print("=== Running power experiment ===")
    results = {}
    total_configs = len(dnl_values) * len(N_range)

    pbar = tqdm(total=total_configs, desc="Power experiment")
    for dnl in dnl_values:
        results[str(dnl)] = {"N": [], "ct": [], "chi2": [], "gtest": []}
        for N in N_range:
            r = compute_power(N, n_codes, dnl, n_trials, alpha)
            results[str(dnl)]["N"].append(N)
            results[str(dnl)]["ct"].append(r["ct_power"])
            results[str(dnl)]["chi2"].append(r["chi2_power"])
            results[str(dnl)]["gtest"].append(r["gtest_power"])
            pbar.set_postfix(DNL=dnl, N=N)
            pbar.update(1)
    pbar.close()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: %s" % args.output)


if __name__ == "__main__":
    main()
