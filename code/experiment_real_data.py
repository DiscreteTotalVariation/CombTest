#!/usr/bin/env python3
"""Test CT on real-world data sources.

Searches for and tests on:
1. Python's random module (round-half-to-even is real, not simulated)
2. System entropy (/dev/urandom) last-digit uniformity
3. Financial data digit distribution (Benford-adjacent)
4. Real-world measurement rounding artifacts
"""

import os
import sys
import struct
import numpy as np
from scipy import stats


def compute_dtv(h):
    """Compute DTV: sum of |h_{i+1} - h_i|."""
    return int(np.sum(np.abs(np.diff(h))))


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


def ct_pvalue(dtv_val, N, n):
    """Compute CT p-value P(DTV >= d) using exact CDF if available, else beta approximation."""
    cdf_path = f'../data/cdf_exact/N_{N}_n_{n}.txt'
    if os.path.isfile(cdf_path):
        cdf_values, cdf_probs = load_cdf(cdf_path)
        # p = P(DTV >= d) = 1 - P(DTV <= d-1) = 1 - F(d-1)
        idx = np.searchsorted(cdf_values, dtv_val - 1, side='right') - 1
        if idx < 0:
            return 1.0
        return 1.0 - cdf_probs[idx]
    beta_path = f'../data/fitted/cvm_beta/N_{N}_n_{n}.txt'
    if os.path.isfile(beta_path):
        params = load_beta_params(beta_path)
        bd = stats.beta(a=params[0], b=params[1], loc=params[2], scale=params[3])
        # Continuity correction: P(DTV >= d) ≈ 1 - F_beta(d - 0.5)
        return 1.0 - bd.cdf(dtv_val - 0.5)
    return None


def chi2_pvalue(h):
    N = int(np.sum(h))
    n = len(h)
    expected = N / n
    chi2_stat = np.sum((h - expected) ** 2 / expected)
    return 1.0 - stats.chi2.cdf(chi2_stat, df=n - 1)


def gtest_pvalue(h):
    N = int(np.sum(h))
    n = len(h)
    expected = N / n
    h_safe = np.where(h > 0, h, 0.5)
    G = 2 * np.sum(h * np.log(h_safe / expected))
    return 1.0 - stats.chi2.cdf(G, df=n - 1)


def test_histogram(h, N, n, label):
    """Test a histogram and print results."""
    dtv = compute_dtv(h)
    ct_p = ct_pvalue(dtv, N, n)
    chi2_p = chi2_pvalue(h)
    gt_p = gtest_pvalue(h)

    ct_str = f"{ct_p:.6f}" if ct_p is not None else "N/A"
    print(f"  {label}")
    print(f"    Histogram: {h}")
    print(f"    DTV={dtv}, CT p={ct_str}, Chi2 p={chi2_p:.6f}, G-test p={gt_p:.6f}")

    reject_ct = ct_p is not None and ct_p < 0.05
    reject_chi2 = chi2_p < 0.05
    if reject_ct and not reject_chi2:
        print(f"    --> CT detects non-uniformity, chi-squared does not!")
    elif reject_chi2 and not reject_ct:
        print(f"    --> Chi-squared detects, CT does not")
    elif reject_ct and reject_chi2:
        print(f"    --> Both detect non-uniformity")
    else:
        print(f"    --> Neither detects non-uniformity")
    return ct_p, chi2_p, gt_p


def experiment_python_rounding():
    """Test Python's real round() function (uses round-half-to-even)."""
    print("\n=== Experiment 1: Python's round() function ===")
    print("Python's round() uses round-half-to-even (banker's rounding).")
    print("This should create a comb-like last-digit distribution.\n")

    n_bins = 10  # digits 0-9
    n_repetitions = 1000
    alpha = 0.05

    for N in [50, 100, 200, 500, 1000]:
        for n_decimals in [1, 2]:
            ct_rejections = 0
            chi2_rejections = 0
            ct_valid = 0

            for _ in range(n_repetitions):
                # Generate random numbers with extra decimals, round them
                values = np.random.uniform(0, 10 ** (n_decimals + 1), N)
                # Use Python's built-in round (round-half-to-even)
                rounded = np.array([round(v, n_decimals) for v in values])
                last_digits = (rounded * 10 ** n_decimals).astype(int) % 10
                h = np.bincount(last_digits, minlength=n_bins)[:n_bins]

                dtv = compute_dtv(h)
                if chi2_pvalue(h) < alpha:
                    chi2_rejections += 1
                p = ct_pvalue(dtv, N, n_bins)
                if p is not None:
                    ct_valid += 1
                    if p < alpha:
                        ct_rejections += 1

            ct_power = ct_rejections / ct_valid if ct_valid > 0 else None
            chi2_power = chi2_rejections / n_repetitions
            ct_str = f"{ct_power:.4f}" if ct_power is not None else "N/A"
            diff = f"{ct_power - chi2_power:+.4f}" if ct_power is not None else "N/A"
            print(f"  N={N:4d}, d={n_decimals}: "
                  f"CT power={ct_str}, Chi2 power={chi2_power:.4f}, "
                  f"diff={diff}")


def experiment_urandom():
    """Test /dev/urandom for digit uniformity."""
    print("\n=== Experiment 2: System entropy (/dev/urandom) ===")
    print("Should be uniform -> both tests should accept H0.\n")

    n_bins = 10
    for N in [100, 500, 1000]:
        raw = os.urandom(N * 4)
        values = [struct.unpack('I', raw[i*4:(i+1)*4])[0] % 10
                  for i in range(N)]
        h = np.bincount(values, minlength=n_bins)[:n_bins]
        test_histogram(h, N, n_bins, f"N={N}, /dev/urandom last digits")


def experiment_numpy_round():
    """Test NumPy's round function (also round-half-to-even)."""
    print("\n=== Experiment 3: NumPy np.round() (banker's rounding) ===")
    print("np.round also uses round-half-to-even.\n")

    n_bins = 10
    for N in [100, 500]:
        for n_dec in [1, 2]:
            values = np.random.uniform(0, 100, N)
            rounded = np.round(values, n_dec)
            last_digits = (rounded * 10**n_dec).astype(int) % 10
            h = np.bincount(last_digits, minlength=n_bins)[:n_bins]
            test_histogram(h, N, n_bins,
                          f"N={N}, np.round({n_dec} dec)")


def experiment_rounding_power():
    """Comprehensive power comparison for Python's real rounding."""
    print("\n=== Experiment 4: Power comparison with real Python round() ===")
    print("Using real round() function, not simulation.\n")

    n_bins = 10
    n_trials = 50000
    alpha = 0.05

    print(f"{'N':>5s} {'dec':>3s} | {'CT':>8s} {'Chi2':>8s} {'G-test':>8s} | "
          f"{'CT-Chi2':>8s}")
    print("-" * 55)

    for N in [50, 100, 200, 300, 500]:
        for n_dec in [1, 2, 3]:
            ct_rej = chi2_rej = gtest_rej = ct_valid = 0

            for _ in range(n_trials):
                values = np.random.uniform(0, 10**(n_dec+1), N)
                rounded = np.array([round(v, n_dec) for v in values])
                last_digits = (rounded * 10**n_dec).astype(int) % 10
                h = np.bincount(last_digits, minlength=n_bins)[:n_bins]

                dtv = compute_dtv(h)
                if chi2_pvalue(h) < alpha: chi2_rej += 1
                if gtest_pvalue(h) < alpha: gtest_rej += 1
                p = ct_pvalue(dtv, N, n_bins)
                if p is not None:
                    ct_valid += 1
                    if p < alpha: ct_rej += 1

            ct_pow = ct_rej / ct_valid if ct_valid > 0 else None
            chi2_pow = chi2_rej / n_trials
            gt_pow = gtest_rej / n_trials

            if ct_pow is not None:
                diff = ct_pow - chi2_pow
                print(f"{N:5d} {n_dec:3d} | {ct_pow:8.4f} {chi2_pow:8.4f} "
                      f"{gt_pow:8.4f} | {diff:+8.4f}")


def main():
    np.random.seed(42)

    experiment_python_rounding()
    experiment_urandom()
    experiment_numpy_round()
    experiment_rounding_power()


if __name__ == '__main__':
    main()
