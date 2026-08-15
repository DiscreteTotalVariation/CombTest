#!/usr/bin/env python3
"""Experiment: ADC alternating DNL detection using CT vs chi-squared vs G-test.

Simulates an ADC with alternating differential nonlinearity (DNL), where even
codes have slightly wider bins and odd codes have slightly narrower bins.
This creates a comb-like histogram pattern that CT is designed to detect.

Computes power of test as a function of sample size N for several DNL amplitudes.
Generates a publication-quality figure for the paper.
"""

import os
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def ct_pvalue(dtv_val, N, n):
    """Compute CT p-value P(DTV >= d) using exact CDF if available, else beta approximation."""
    cdf_path = f"../data/cdf_exact/N_{N}_n_{n}.txt"
    if os.path.isfile(cdf_path):
        cdf_values, cdf_probs = load_cdf(cdf_path)
        # p = P(DTV >= d) = 1 - P(DTV <= d-1) = 1 - F(d-1)
        idx = np.searchsorted(cdf_values, dtv_val - 1, side="right") - 1
        if idx < 0:
            return 1.0
        return 1.0 - cdf_probs[idx]

    beta_path = f"../data/fitted/cvm_beta/N_{N}_n_{n}.txt"
    if os.path.isfile(beta_path):
        params = load_beta_params(beta_path)
        bd = stats.beta(a=params[0], b=params[1], loc=params[2], scale=params[3])
        # Continuity correction: P(DTV >= d) ≈ 1 - F_beta(d - 0.5)
        return 1.0 - bd.cdf(dtv_val - 0.5)

    return None


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
    # Avoid log(0) by replacing 0 counts with a tiny value
    h_safe = np.where(histogram > 0, histogram, 0.5)
    G = 2 * np.sum(histogram * np.log(h_safe / expected))
    return 1.0 - stats.chi2.cdf(G, df=n - 1)


def fit_beta_mc(N, n, K=50000, seed=13):
    """Fit beta distribution to DTV via Monte Carlo sampling under H0."""
    from scipy.optimize import minimize as sp_minimize
    rng = np.random.RandomState(seed)
    dtvs = np.zeros(K, dtype=int)
    for i in range(K):
        h = rng.multinomial(N, [1.0 / n] * n)
        dtvs[i] = int(np.sum(np.abs(np.diff(h))))

    unique_vals = np.unique(dtvs)
    counts = np.array([np.sum(dtvs == v) for v in unique_vals])
    cdf_probs = np.cumsum(counts) / K

    max_val = float(unique_vals[-1]) if len(unique_vals) > 0 else 1.0
    mean = np.mean(dtvs)
    var = np.var(dtvs)
    eps = 1e-10
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

    result = sp_minimize(cvm_objective,
                         x0=[np.log(init_a), np.log(init_b), np.log(max_val)],
                         method="Nelder-Mead")
    a = np.exp(result.x[0])
    b = np.exp(result.x[1])
    scale = np.exp(result.x[2])
    return (a, b, 0.0, scale)


def ct_mc_pvalue(dtv_val, beta_params):
    """Compute CT p-value using MC-fitted beta parameters."""
    bd = stats.beta(a=beta_params[0], b=beta_params[1],
                    loc=beta_params[2], scale=beta_params[3])
    return 1.0 - bd.cdf(dtv_val - 0.5)


def generate_adc_histogram(N, n_codes, dnl_amplitude):
    """Simulate ADC with alternating DNL.

    Even codes have bin width (1 + dnl_amplitude), odd codes (1 - dnl_amplitude).
    A uniform input signal is sampled N times.

    Args:
        N: number of samples
        n_codes: number of ADC output codes (histogram bins)
        dnl_amplitude: DNL amplitude (0 = ideal, 1 = max)

    Returns:
        histogram of ADC output codes
    """
    # Bin widths: proportional to (1 + dnl * (-1)^i)
    widths = np.array([1 + dnl_amplitude * (1 if i % 2 == 0 else -1)
                       for i in range(n_codes)], dtype=float)
    probs = widths / widths.sum()
    return np.random.multinomial(N, probs)


def compute_power(N, n_codes, dnl_amplitude, n_trials, alpha=0.05,
                   mc_beta_params=None):
    """Compute power for CT, MC-CT, chi-squared, and G-test at given parameters."""
    ct_rejections = 0
    mc_ct_rejections = 0
    chi2_rejections = 0
    gtest_rejections = 0
    ct_valid = 0

    for _ in range(n_trials):
        h = generate_adc_histogram(N, n_codes, dnl_amplitude)
        dtv = compute_dtv(h)

        # Chi-squared
        if chi2_pvalue(h) < alpha:
            chi2_rejections += 1

        # G-test
        if gtest_pvalue(h) < alpha:
            gtest_rejections += 1

        # CT (exact or pre-fitted beta)
        p = ct_pvalue(dtv, N, n_codes)
        if p is not None:
            ct_valid += 1
            if p < alpha:
                ct_rejections += 1

        # MC-based CT
        if mc_beta_params is not None:
            mc_p = ct_mc_pvalue(dtv, mc_beta_params)
            if mc_p < alpha:
                mc_ct_rejections += 1

    return {
        "ct_power": ct_rejections / ct_valid if ct_valid > 0 else None,
        "mc_ct_power": mc_ct_rejections / n_trials if mc_beta_params is not None else None,
        "chi2_power": chi2_rejections / n_trials,
        "gtest_power": gtest_rejections / n_trials,
        "ct_valid": ct_valid,
    }


def verify_false_positive_rate(n_codes=10, N_values=[50, 100, 200],
                                n_trials=10000, alpha=0.05):
    """Verify that all tests have correct false positive rate under H0."""
    print("=== Verifying false positive rate under H0 (uniform) ===")
    print(f"Expected: ~{alpha:.2f} for all tests")
    print(f"{'N':>5s} | {'CT':>8s} | {'MC-CT':>8s} | {'Chi2':>8s} | {'G-test':>8s}")
    print("-" * 55)

    for N in N_values:
        mc_params = fit_beta_mc(N, n_codes, K=50000, seed=13)
        r = compute_power(N, n_codes, 0.0, n_trials, alpha,
                          mc_beta_params=mc_params)
        ct = f"{r['ct_power']:.4f}" if r["ct_power"] is not None else "N/A"
        mc_ct = f"{r['mc_ct_power']:.4f}" if r["mc_ct_power"] is not None else "N/A"
        print(f"{N:5d} | {ct:>8s} | {mc_ct:>8s} | {r['chi2_power']:8.4f} | {r['gtest_power']:8.4f}")
    print()


def run_power_experiment(n_codes=10, dnl_values=[0.05, 0.10, 0.20],
                         N_range=None, n_trials=10000, alpha=0.05):
    """Run the full power experiment."""
    if N_range is None:
        N_range = list(range(20, 101, 10)) + list(range(120, 301, 20)) + \
                  list(range(350, 501, 50))

    # Pre-fit MC beta for each unique N
    mc_beta_cache = {}
    print("  Fitting MC beta distributions...")
    for N in N_range:
        if N not in mc_beta_cache:
            mc_beta_cache[N] = fit_beta_mc(N, n_codes, K=50000, seed=13)
    print(f"  Fitted {len(mc_beta_cache)} MC beta distributions.")

    results = {}
    total_configs = len(dnl_values) * len(N_range)
    done = 0

    for dnl in dnl_values:
        results[dnl] = {"N": [], "ct": [], "mc_ct": [], "chi2": [], "gtest": []}
        for N in N_range:
            r = compute_power(N, n_codes, dnl, n_trials, alpha,
                              mc_beta_params=mc_beta_cache[N])
            results[dnl]["N"].append(N)
            results[dnl]["ct"].append(r["ct_power"])
            results[dnl]["mc_ct"].append(r["mc_ct_power"])
            results[dnl]["chi2"].append(r["chi2_power"])
            results[dnl]["gtest"].append(r["gtest_power"])
            done += 1
            if done % 10 == 0:
                print(f"  Progress: {done}/{total_configs} "
                      f"(DNL={dnl:.2f}, N={N})", flush=True)

    return results


def generate_figure(results, output_path):
    """Generate publication-quality power comparison figure."""
    tick_fontsize = 22
    label_fontsize = 36

    matplotlib.rc("font", size=tick_fontsize)
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 8),
                             sharey=True)
    if len(results) == 1:
        axes = [axes]

    colors = {"ct": "#1f77b4", "mc_ct": "#9467bd", "chi2": "#ff7f0e", "gtest": "#2ca02c"}
    labels = {"ct": "CT (exact)", "mc_ct": "CT (MC beta)", "chi2": r"Pearson's $\chi^2$", "gtest": "G-test"}
    markers = {"ct": "o", "mc_ct": "D", "chi2": "s", "gtest": "^"}

    for ax, (dnl, data) in zip(axes, sorted(results.items())):
        N_arr = np.array(data["N"])
        for test in ["ct", "mc_ct", "chi2", "gtest"]:
            power = np.array(data[test])
            valid = power != None  # noqa
            ax.plot(N_arr[valid], power[valid],
                    color=colors[test], marker=markers[test],
                    markersize=3, linewidth=1.5, label=labels[test])

        ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=0.8,
                   alpha=0.5, label=r"$\alpha=0.05$")
        ax.set_xlabel("$N$", fontsize=label_fontsize)
        ax.set_title("DNL = %s" % str(dnl), fontsize=label_fontsize)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(N_arr[0] - 5, N_arr[-1] + 5)

    axes[0].set_ylabel("Power of test", fontsize=label_fontsize)
    axes[0].legend(fontsize=tick_fontsize, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary(results, alpha=0.05):
    """Print summary table of results."""
    print("\n=== Power of test summary ===")
    print(f"{'DNL':>5s} {'N':>5s} | {'CT':>8s} {'Chi2':>8s} {'G-test':>8s} | "
          f"{'CT-Chi2':>8s} {'CT-Gtest':>8s}")
    print("-" * 65)

    for dnl in sorted(results.keys()):
        data = results[dnl]
        for i, N in enumerate(data["N"]):
            ct = data["ct"][i]
            chi2 = data["chi2"][i]
            gt = data["gtest"][i]
            if ct is None:
                continue
            d1 = ct - chi2
            d2 = ct - gt
            print(f"{dnl:5.2f} {N:5d} | {ct:8.4f} {chi2:8.4f} {gt:8.4f} | "
                  f"{d1:+8.4f} {d2:+8.4f}")


def main():
    np.random.seed(42)
    n_codes = 10  # Number of ADC output codes
    n_trials = 50000  # High number for statistical significance
    alpha = 0.05

    print("ADC Alternating DNL Detection: CT vs Chi-squared vs G-test")
    print(f"n_codes={n_codes}, n_trials={n_trials}, alpha={alpha}")
    print()

    # Step 1: Verify false positive rate
    verify_false_positive_rate(n_codes, [50, 100, 200, 500], n_trials, alpha)

    # Step 2: Run power experiment
    dnl_values = [0.05, 0.10, 0.20]
    N_range = list(range(20, 101, 10)) + list(range(120, 301, 20)) + \
              list(range(350, 501, 50))

    print("=== Running power experiment ===")
    results = run_power_experiment(n_codes, dnl_values, N_range, n_trials, alpha)

    # Step 3: Print summary
    print_summary(results, alpha)

    # Step 4: Generate figure
    output_path = "../paper/img/adc_dnl_power_comparison.png"
    generate_figure(results, output_path)

    # Step 5: Compute key statistics for paper text
    print("\n=== Key statistics for paper ===")
    for dnl in dnl_values:
        data = results[dnl]
        # Find N where CT first exceeds alpha+0.05 (10%)
        for i, N in enumerate(data["N"]):
            ct = data["ct"][i]
            if ct is not None and ct > 0.10:
                print(f"DNL={dnl}: CT reaches 10% power at N={N}")
                break

        # Find average CT advantage over chi2
        advantages = []
        for i in range(len(data["N"])):
            ct = data["ct"][i]
            chi2 = data["chi2"][i]
            if ct is not None:
                advantages.append(ct - chi2)
        print(f"DNL={dnl}: Mean CT advantage over chi2: {np.mean(advantages):+.4f}")
        print(f"DNL={dnl}: Max CT advantage over chi2: {np.max(advantages):+.4f}")


if __name__ == "__main__":
    main()
