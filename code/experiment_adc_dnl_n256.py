#!/usr/bin/env python3
"""Reproduce all ADC-related numbers cited in the paper (n=10 and n=256).

Uses seed 13 throughout. Produces a JSON file with all results so that
every number in Section IV of the paper can be traced back to this script.

Numbers verified:
  n=10, N=200, eps=0.20: CT power, chi2 power, G-test power
  n=10 FPR at N in {50,100,200,500}: CT, chi2, G-test, MC-based CT
  n=256, N=500, eps=0.20: CT (MC-based) power, chi2 power
  n=256 G-test FPR at N in {500}: catastrophic FPR range
  n=256, N=500, eps=0.20, random DNL sigma=0.10: all tests similar power
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm


def compute_dtv(histogram):
    return int(np.sum(np.abs(np.diff(histogram))))


def fit_beta_mc(N, n, K=50000, seed=None):
    """Fit beta distribution to DTV via MC sampling under H0."""
    rng = np.random.default_rng(seed)
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


def fit_gamma_mc(N, n, K=50000, seed=None):
    """Fit gamma distribution to DTV via MC sampling under H0."""
    rng = np.random.default_rng(seed)
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


def load_cdf(path):
    values, probs = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                values.append(int(float(parts[0])))
                probs.append(float(parts[1]))
    return np.array(values), np.array(probs)


def load_beta_params(path):
    with open(path) as f:
        return list(map(float, f.readline().strip().split()))


# Cache for MC-fitted params
# The Monte Carlo fitting RNGs below are keyed on (N, n) rather than on --seed
# by design: the fitted null of a given (N, n) pair is a property of that pair,
# so keying it this way makes one fit shared and reproducible across every
# experiment and independent of the trial stream. --seed controls the trial
# stream only. Changing this changes every MC-fitted number in the paper.
_mc_beta_cache = {}
_mc_gamma_cache = {}
_approx = "gamma"  # set from --approx flag


def ct_pvalue_exact(dtv_val, N, n):
    """CT p-value from exact CDF (returns None if unavailable)."""
    cdf_path = "../data/cdf_exact/N_%d_n_%d.txt" % (N, n)
    if not os.path.isfile(cdf_path):
        return None
    cdf_values, cdf_probs = load_cdf(cdf_path)
    idx = np.searchsorted(cdf_values, dtv_val - 1, side="right") - 1
    if idx < 0:
        return 1.0
    return 1.0 - cdf_probs[idx]


def ct_pvalue_prefit_beta(dtv_val, N, n):
    """CT p-value from pre-fitted beta params (returns None if unavailable)."""
    beta_path = "../data/fitted/cvm_beta/N_%d_n_%d.txt" % (N, n)
    if not os.path.isfile(beta_path):
        return None
    params = load_beta_params(beta_path)
    bd = stats.beta(a=params[0], b=params[1], loc=params[2], scale=params[3])
    return 1.0 - bd.cdf(dtv_val - 0.5)


def ct_pvalue_prefit_gamma(dtv_val, N, n):
    """CT p-value from pre-fitted gamma params (returns None if unavailable)."""
    gamma_path = "../data/fitted/cvm_gamma/N_%d_n_%d.txt" % (N, n)
    if not os.path.isfile(gamma_path):
        return None
    params = load_beta_params(gamma_path)  # same format: space-separated line
    gd = stats.gamma(a=params[0], loc=params[1], scale=params[2])
    return 1.0 - gd.cdf(dtv_val - 0.5)


def ct_pvalue_mc(dtv_val, N, n):
    """CT p-value from MC-fitted beta (always available)."""
    key = (N, n)
    if key not in _mc_beta_cache:
        _mc_beta_cache[key] = fit_beta_mc(N, n, K=50000,
                                          seed=N * 1000 + n)
    params = _mc_beta_cache[key]
    bd = stats.beta(a=params[0], b=params[1], loc=params[2], scale=params[3])
    return 1.0 - bd.cdf(dtv_val - 0.5)


def ct_pvalue_mc_gamma(dtv_val, N, n):
    """CT p-value from MC-fitted gamma (always available)."""
    key = (N, n)
    if key not in _mc_gamma_cache:
        _mc_gamma_cache[key] = fit_gamma_mc(N, n, K=50000,
                                             seed=N * 1000 + n)
    params = _mc_gamma_cache[key]
    gd = stats.gamma(a=params[0], loc=params[1], scale=params[2])
    return 1.0 - gd.cdf(dtv_val - 0.5)


def ct_pvalue(dtv_val, N, n):
    """CT p-value: try exact, then pre-fitted approx, then MC approx."""
    p = ct_pvalue_exact(dtv_val, N, n)
    if p is not None:
        return p
    if _approx == "gamma":
        p = ct_pvalue_prefit_gamma(dtv_val, N, n)
        if p is not None:
            return p
        return ct_pvalue_mc_gamma(dtv_val, N, n)
    else:
        p = ct_pvalue_prefit_beta(dtv_val, N, n)
        if p is not None:
            return p
        return ct_pvalue_mc(dtv_val, N, n)


def chi2_pvalue(histogram):
    N = int(np.sum(histogram))
    n = len(histogram)
    expected = N / n
    chi2_stat = np.sum((histogram - expected) ** 2 / expected)
    return 1.0 - stats.chi2.cdf(chi2_stat, df=n - 1)


def gtest_pvalue(histogram):
    N = int(np.sum(histogram))
    n = len(histogram)
    expected = N / n
    h_safe = np.where(histogram > 0, histogram, 0.5)
    G = 2 * np.sum(histogram * np.log(h_safe / expected))
    return 1.0 - stats.chi2.cdf(G, df=n - 1)


def generate_adc_histogram(N, n_codes, dnl_amplitude, rng=None):
    """Simulate ADC with alternating DNL."""
    widths = np.array([1 + dnl_amplitude * (1 if i % 2 == 0 else -1)
                       for i in range(n_codes)], dtype=float)
    probs = widths / widths.sum()
    if rng is not None:
        return rng.multinomial(N, probs)
    return np.random.multinomial(N, probs)


def generate_random_dnl_histogram(N, n_codes, sigma, rng):
    """Simulate ADC with i.i.d. random DNL (Gaussian perturbation)."""
    widths = 1.0 + rng.normal(0, sigma, n_codes)
    widths = np.maximum(widths, 0.01)  # avoid negative widths
    probs = widths / widths.sum()
    return rng.multinomial(N, probs)


def run_power(N, n_codes, dnl_amplitude, n_trials, alpha, use_mc_ct=False):
    """Compute power for CT, chi2, G-test. Returns dict with all counts."""
    ct_rejections = 0
    mc_ct_rejections = 0
    gamma_rejections = 0
    chi2_rejections = 0
    gtest_rejections = 0
    ct_valid = 0
    mc_ct_valid = 0
    gamma_valid = 0

    for _ in tqdm(range(n_trials), desc="run_power N=%d eps=%.2f" % (N, dnl_amplitude), leave=False):
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

        p_gamma = ct_pvalue_prefit_gamma(dtv, N, n_codes)
        if p_gamma is not None:
            gamma_valid += 1
            if p_gamma < alpha:
                gamma_rejections += 1

        if use_mc_ct:
            p_mc = ct_pvalue_mc(dtv, N, n_codes)
            mc_ct_valid += 1
            if p_mc < alpha:
                mc_ct_rejections += 1

    return {
        "ct_power": ct_rejections / ct_valid if ct_valid > 0 else None,
        "chi2_power": chi2_rejections / n_trials,
        "gtest_power": gtest_rejections / n_trials,
        "mc_ct_power": mc_ct_rejections / mc_ct_valid if mc_ct_valid > 0 else None,
        "gamma_power": gamma_rejections / gamma_valid if gamma_valid > 0 else None,
        "ct_valid": ct_valid,
    }


def run_random_dnl_power(N, n_codes, sigma, n_trials, alpha, seed):
    """Power for i.i.d. random DNL."""
    rng = np.random.default_rng(seed)
    ct_rej = gamma_rej = chi2_rej = gtest_rej = 0
    for _ in tqdm(range(n_trials), desc="random_dnl N=%d sigma=%.2f" % (N, sigma), leave=False):
        h = generate_random_dnl_histogram(N, n_codes, sigma, rng)
        dtv = compute_dtv(h)
        if ct_pvalue(dtv, N, n_codes) < alpha:
            ct_rej += 1
        p_gamma = ct_pvalue_prefit_gamma(dtv, N, n_codes)
        if p_gamma is not None and p_gamma < alpha:
            gamma_rej += 1
        if chi2_pvalue(h) < alpha:
            chi2_rej += 1
        if gtest_pvalue(h) < alpha:
            gtest_rej += 1
    return {
        "ct_power": ct_rej / n_trials,
        "gamma_power": gamma_rej / n_trials,
        "chi2_power": chi2_rej / n_trials,
        "gtest_power": gtest_rej / n_trials,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all ADC numbers cited in the paper.")
    parser.add_argument("--n-trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--approx", choices=["gamma", "beta"], default="gamma",
                        help="Approximation distribution for CT (default: gamma)")
    parser.add_argument("-o", "--output", type=str,
                        default="../data/adc_dnl_all_results.json")
    args = parser.parse_args()

    global _approx
    _approx = args.approx

    n_trials = args.n_trials
    alpha = 0.05
    results = {}

    # ---------------------------------------------------------------
    # 1. n=10 FPR verification (Table in paper)
    # ---------------------------------------------------------------
    print("=" * 60)
    print("1. FPR verification, n=10, seed=%d, n_trials=%d" % (args.seed, n_trials))
    print("=" * 60)
    np.random.seed(args.seed)
    fpr_n10 = {}
    print("%5s | %8s | %8s | %8s | %8s | %8s" % ("N", "CT", "MC-CT", "Gamma", "Chi2", "G-test"))
    print("-" * 65)
    for N in [50, 100, 200, 500]:
        r = run_power(N, 10, 0.0, n_trials, alpha, use_mc_ct=True)
        fpr_n10[str(N)] = r
        ct = "%.4f" % r["ct_power"] if r["ct_power"] is not None else "N/A"
        mc = "%.4f" % r["mc_ct_power"] if r["mc_ct_power"] is not None else "N/A"
        gm = "%.4f" % r["gamma_power"] if r["gamma_power"] is not None else "N/A"
        print("%5d | %8s | %8s | %8s | %8.4f | %8.4f" % (
            N, ct, mc, gm, r["chi2_power"], r["gtest_power"]))
    results["fpr_n10"] = fpr_n10
    print()

    # ---------------------------------------------------------------
    # 2. n=10, N=200, eps=0.20 power (main result)
    # ---------------------------------------------------------------
    print("=" * 60)
    print("2. Power: n=10, N=200, eps=0.20")
    print("=" * 60)
    np.random.seed(args.seed)
    r10 = run_power(200, 10, 0.20, n_trials, alpha, use_mc_ct=True)
    results["power_n10_N200_eps020"] = r10
    print("  CT:      %.4f" % (r10["ct_power"] or 0))
    print("  MC-CT:   %.4f" % (r10["mc_ct_power"] or 0))
    print("  Gamma:   %.4f" % (r10["gamma_power"] or 0))
    print("  Chi2:    %.4f" % r10["chi2_power"])
    print("  G-test:  %.4f" % r10["gtest_power"])
    if r10["ct_power"] and r10["chi2_power"] > 0:
        print("  CT/chi2 improvement: %.0f%%" % (
            100 * (r10["ct_power"] - r10["chi2_power"]) / r10["chi2_power"]))
    if r10["ct_power"] and r10["gtest_power"] > 0:
        print("  CT/gtest improvement: %.0f%%" % (
            100 * (r10["ct_power"] - r10["gtest_power"]) / r10["gtest_power"]))
    print()

    # ---------------------------------------------------------------
    # 3. n=256, N=500, eps=0.20 power (MC-based CT)
    # ---------------------------------------------------------------
    print("=" * 60)
    print("3. Power: n=256, N=500, eps=0.20 (MC-based CT)")
    print("=" * 60)
    np.random.seed(args.seed)
    r256 = run_power(500, 256, 0.20, n_trials, alpha, use_mc_ct=True)
    results["power_n256_N500_eps020"] = r256
    ct_p = r256["ct_power"]
    mc_p = r256["mc_ct_power"]
    gm_p = r256["gamma_power"]
    print("  CT (%s): %.4f" % (args.approx, ct_p or 0))
    print("  MC-beta: %.4f" % (mc_p or 0))
    print("  Gamma:   %.4f" % (gm_p or 0))
    print("  Chi2:    %.4f" % r256["chi2_power"])
    print("  G-test:  %.4f" % r256["gtest_power"])
    if ct_p and r256["chi2_power"] > 0:
        print("  CT(%s)/chi2 improvement: %.0f%%" % (
            args.approx,
            100 * (ct_p - r256["chi2_power"]) / r256["chi2_power"]))
    if mc_p and r256["chi2_power"] > 0:
        print("  MC-beta/chi2 improvement: %.0f%%" % (
            100 * (mc_p - r256["chi2_power"]) / r256["chi2_power"]))
    print()

    # ---------------------------------------------------------------
    # 4. n=256 G-test FPR (catastrophic)
    # ---------------------------------------------------------------
    print("=" * 60)
    print("4. G-test FPR at n=256 (uniform, eps=0)")
    print("=" * 60)
    np.random.seed(args.seed)
    gtest_fprs_256 = {}
    print("%5s | %8s | %8s | %8s | %8s" % ("N", "MC-CT", "Gamma", "Chi2", "G-test"))
    print("-" * 50)
    for N in [200, 300, 500, 1000]:
        r = run_power(N, 256, 0.0, n_trials, alpha, use_mc_ct=True)
        gtest_fprs_256[str(N)] = r
        mc = "%.4f" % r["mc_ct_power"] if r["mc_ct_power"] is not None else "N/A"
        gm = "%.4f" % r["gamma_power"] if r["gamma_power"] is not None else "N/A"
        print("%5d | %8s | %8s | %8.4f | %8.4f" % (N, mc, gm, r["chi2_power"], r["gtest_power"]))
    results["fpr_n256"] = gtest_fprs_256
    gtest_vals = [v["gtest_power"] for v in gtest_fprs_256.values()]
    print("  G-test FPR range: %.2f -- %.2f" % (min(gtest_vals), max(gtest_vals)))
    print()

    # ---------------------------------------------------------------
    # 5. n=10, random DNL (sigma=0.10) — cited in the paper
    # ---------------------------------------------------------------
    print("=" * 60)
    print("5. Random DNL: n=10, N=200, sigma=0.10")
    print("=" * 60)
    rr10 = run_random_dnl_power(200, 10, 0.10, n_trials, alpha, seed=args.seed)
    results["random_dnl"] = rr10
    print("  CT:      %.4f" % rr10["ct_power"])
    print("  Gamma:   %.4f" % rr10["gamma_power"])
    print("  Chi2:    %.4f" % rr10["chi2_power"])
    print("  G-test:  %.4f" % rr10["gtest_power"])
    print()

    # ---------------------------------------------------------------
    # 6. Non-comb deviations (N=200, n=10): CT vs chi-square
    # ---------------------------------------------------------------
    print("=" * 60)
    print("6. Non-comb deviations: N=200, n=10")
    print("=" * 60)
    N_nc, n_nc = 200, 10
    noncomb = {}

    # 6a. Single inflated bin (bin 0 has 4x probability)
    rng_nc = np.random.default_rng(args.seed)
    ct_rej = gamma_rej = chi2_rej = 0
    for _ in tqdm(range(n_trials), desc="inflated bin", leave=False):
        probs = np.ones(n_nc)
        probs[0] *= 4.0
        probs /= probs.sum()
        h = rng_nc.multinomial(N_nc, probs)
        dtv_val = compute_dtv(h)
        if ct_pvalue(dtv_val, N_nc, n_nc) < alpha:
            ct_rej += 1
        p_gm = ct_pvalue_prefit_gamma(dtv_val, N_nc, n_nc)
        if p_gm is not None and p_gm < alpha:
            gamma_rej += 1
        if chi2_pvalue(h) < alpha:
            chi2_rej += 1
    noncomb["inflated_bin"] = {
        "ct_power": ct_rej / n_trials,
        "gamma_power": gamma_rej / n_trials,
        "chi2_power": chi2_rej / n_trials,
    }
    print("  Inflated bin: Chi2=%.3f, CT=%.3f, Gamma=%.3f" % (
        chi2_rej / n_trials, ct_rej / n_trials, gamma_rej / n_trials))

    # 6b. Monotonic trend (linearly increasing probabilities)
    rng_nc = np.random.default_rng(args.seed)
    ct_rej = gamma_rej = chi2_rej = 0
    for _ in tqdm(range(n_trials), desc="monotonic", leave=False):
        probs = np.linspace(0.5, 1.5, n_nc)
        probs /= probs.sum()
        h = rng_nc.multinomial(N_nc, probs)
        dtv_val = compute_dtv(h)
        if ct_pvalue(dtv_val, N_nc, n_nc) < alpha:
            ct_rej += 1
        p_gm = ct_pvalue_prefit_gamma(dtv_val, N_nc, n_nc)
        if p_gm is not None and p_gm < alpha:
            gamma_rej += 1
        if chi2_pvalue(h) < alpha:
            chi2_rej += 1
    noncomb["monotonic"] = {
        "ct_power": ct_rej / n_trials,
        "gamma_power": gamma_rej / n_trials,
        "chi2_power": chi2_rej / n_trials,
    }
    print("  Monotonic:    Chi2=%.3f, CT=%.3f, Gamma=%.3f" % (
        chi2_rej / n_trials, ct_rej / n_trials, gamma_rej / n_trials))

    # 6c. Block pattern (blocks of 5, bias=0.3)
    rng_nc = np.random.default_rng(args.seed)
    ct_rej = gamma_rej = chi2_rej = 0
    for _ in tqdm(range(n_trials), desc="block pattern", leave=False):
        probs = np.ones(n_nc)
        for i in range(n_nc):
            block_idx = i // 5
            if block_idx % 2 == 0:
                probs[i] *= 1.3
            else:
                probs[i] *= 0.7
        probs /= probs.sum()
        h = rng_nc.multinomial(N_nc, probs)
        dtv_val = compute_dtv(h)
        if ct_pvalue(dtv_val, N_nc, n_nc) < alpha:
            ct_rej += 1
        p_gm = ct_pvalue_prefit_gamma(dtv_val, N_nc, n_nc)
        if p_gm is not None and p_gm < alpha:
            gamma_rej += 1
        if chi2_pvalue(h) < alpha:
            chi2_rej += 1
    noncomb["block"] = {
        "ct_power": ct_rej / n_trials,
        "gamma_power": gamma_rej / n_trials,
        "chi2_power": chi2_rej / n_trials,
    }
    print("  Block:        Chi2=%.3f, CT=%.3f, Gamma=%.3f" % (
        chi2_rej / n_trials, ct_rej / n_trials, gamma_rej / n_trials))

    results["noncomb_N200_n10"] = noncomb
    print()

    # ---------------------------------------------------------------
    # Save all results
    # ---------------------------------------------------------------
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print("All results saved to: %s" % args.output)

    # ---------------------------------------------------------------
    # Summary of paper numbers
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("PAPER NUMBER VERIFICATION SUMMARY")
    print("=" * 60)

    # n=10 FPR ranges
    ct_fprs = [v["ct_power"] for v in fpr_n10.values() if v["ct_power"] is not None]
    chi2_fprs = [v["chi2_power"] for v in fpr_n10.values()]
    gtest_fprs = [v["gtest_power"] for v in fpr_n10.values()]
    mc_ct_fprs = [v["mc_ct_power"] for v in fpr_n10.values() if v["mc_ct_power"] is not None]
    print("n=10 FPR ranges:")
    print("  CT:      %.3f -- %.3f" % (min(ct_fprs), max(ct_fprs)))
    print("  Chi2:    %.3f -- %.3f" % (min(chi2_fprs), max(chi2_fprs)))
    print("  G-test:  %.3f -- %.3f" % (min(gtest_fprs), max(gtest_fprs)))
    print("  MC-CT:   %.3f -- %.3f" % (min(mc_ct_fprs), max(mc_ct_fprs)))
    se = max(np.sqrt(0.05 * 0.95 / n_trials), 0)
    print("  SE:      %.4f" % se)
    print()
    print("n=10, N=200, eps=0.20:")
    print("  CT=%.3f, Chi2=%.3f, G-test=%.3f" % (
        r10["ct_power"], r10["chi2_power"], r10["gtest_power"]))
    print()
    print("n=256, N=500, eps=0.20:")
    ct256 = r256["ct_power"] or 0
    mc256 = r256["mc_ct_power"] or 0
    print("  CT(%s)=%.3f, MC-beta=%.3f, Chi2=%.3f, G-test=%.3f" % (
        args.approx, ct256, mc256, r256["chi2_power"], r256["gtest_power"]))
    print()
    print("n=256, G-test FPR range: %.2f -- %.2f" % (
        min(gtest_vals), max(gtest_vals)))
    print()
    print("n=10, random DNL sigma=0.10:")
    print("  CT=%.3f, Gamma=%.3f, Chi2=%.3f, G-test=%.3f" % (
        rr10["ct_power"], rr10["gamma_power"], rr10["chi2_power"], rr10["gtest_power"]))
    print()
    print("Non-comb deviations (N=200, n=10):")
    for name, vals in noncomb.items():
        print("  %-15s Chi2=%.3f, CT=%.3f" % (
            name + ":", vals["chi2_power"], vals["ct_power"]))


if __name__ == "__main__":
    main()
