#!/usr/bin/env python3
"""Compare CT vs chi-squared power for signal processing scenarios.

Tests several signal processing scenarios where histogram uniformity testing
is relevant, measuring power of test (probability of rejecting false H0).

Scenarios:
1. ADC differential nonlinearity (alternating DNL)
2. ADC differential nonlinearity (random DNL)
3. Quantization-induced patterns (re-quantization)
4. Dithering quality assessment
5. LSB patterns in steganography
6. PRNG quality testing (LCG vs good PRNG)
"""

import os
import numpy as np
from scipy import stats


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
    """Compute DTV: sum of |h_{i+1} - h_i| (adjacent bin differences)."""
    return np.sum(np.abs(np.diff(histogram)))


def ct_pvalue(dtv_val, N, n):
    """Compute CT p-value P(DTV >= d) using exact CDF if available, else beta approximation."""
    cdf_path = f'../data/cdf_exact/N_{N}_n_{n}.txt'
    beta_path = f'../data/fitted/cvm_beta/N_{N}_n_{n}.txt'

    # Try exact first
    if os.path.isfile(cdf_path):
        cdf_values, cdf_probs = load_cdf(cdf_path)
        # p = P(DTV >= d) = 1 - P(DTV <= d-1) = 1 - F(d-1)
        idx = np.searchsorted(cdf_values, dtv_val - 1, side='right') - 1
        if idx < 0:
            return 1.0
        return 1.0 - cdf_probs[idx]

    # Fall back to beta
    if os.path.isfile(beta_path):
        params = load_fitted_params(beta_path)
        beta_dist = stats.beta(a=params[0], b=params[1], loc=params[2], scale=params[3])
        # Continuity correction: P(DTV >= d) ≈ 1 - F_beta(d - 0.5)
        return 1.0 - beta_dist.cdf(dtv_val - 0.5)

    return None


def chi2_pvalue(histogram):
    """Compute Pearson's chi-squared p-value for uniformity."""
    N = np.sum(histogram)
    n_bins = len(histogram)
    expected = N / n_bins
    chi2_stat = np.sum((histogram - expected) ** 2 / expected)
    return 1.0 - stats.chi2.cdf(chi2_stat, df=n_bins - 1)


# =============================================================================
# Signal processing scenario generators
# =============================================================================

def scenario_adc_alternating_dnl(N, n_bins, dnl_amplitude):
    """ADC with alternating differential nonlinearity.

    Even codes are wider by +dnl_amplitude, odd codes narrower by -dnl_amplitude.
    This creates a comb-like histogram when sampling a ramp/uniform signal.
    """
    # Bin widths: 1 + dnl for even, 1 - dnl for odd
    widths = np.array([1 + dnl_amplitude * (1 if i % 2 == 0 else -1) for i in range(n_bins)])
    widths = widths / widths.sum()  # normalize to probabilities
    return np.random.multinomial(N, widths)


def scenario_adc_random_dnl(N, n_bins, dnl_std):
    """ADC with random DNL (Gaussian deviations in bin widths)."""
    widths = np.ones(n_bins) + np.random.normal(0, dnl_std, n_bins)
    widths = np.maximum(widths, 0.01)  # avoid negative widths
    widths = widths / widths.sum()
    return np.random.multinomial(N, widths)


def scenario_requantization(N, n_bins, original_levels):
    """Re-quantization: uniform data quantized to original_levels, then to n_bins.

    When original_levels is not a multiple of n_bins, the re-quantization
    creates a characteristic non-uniform pattern.
    """
    # Generate uniform samples quantized to original_levels
    samples = np.random.randint(0, original_levels, size=N)
    # Re-quantize to n_bins
    requantized = (samples * n_bins) // original_levels
    histogram = np.bincount(requantized, minlength=n_bins)[:n_bins]
    return histogram


def scenario_bad_dither(N, n_bins, dither_bits):
    """Quantization with insufficient dither.

    Proper dither should make quantization error uniform.
    Insufficient dither (fewer bits) creates periodic patterns.
    """
    # Continuous uniform signal
    signal = np.random.uniform(0, n_bins, N)
    # Add insufficient dither (should be uniform over [-0.5, 0.5] quantization step)
    # Instead, use dither with fewer levels
    n_dither_levels = 2 ** dither_bits
    dither = np.random.randint(0, n_dither_levels, N) / n_dither_levels - 0.5
    dithered = signal + dither
    # Quantize
    quantized = np.floor(dithered).astype(int) % n_bins
    histogram = np.bincount(quantized, minlength=n_bins)[:n_bins]
    return histogram


def scenario_lsb_stego(N, n_bins, embed_rate):
    """LSB replacement steganography detection.

    Natural images have non-uniform LSB histograms. LSB replacement
    makes pairs (2k, 2k+1) more balanced, detectable via uniformity
    of the "flipped" histogram.

    Here we simulate: n_bins categories of LSB pair ratios.
    Under no stego: non-uniform. Under stego: pushed toward uniform.
    We test: is the deviation pattern uniform?
    """
    # Simulate pixel value pairs
    # Natural image: each pair (2k, 2k+1) has some ratio r_k
    # Under embedding, r_k -> 0.5
    n_pairs = n_bins
    # Natural ratios: slightly different from 0.5
    natural_ratios = 0.5 + np.random.normal(0, 0.05, n_pairs)
    natural_ratios = np.clip(natural_ratios, 0.01, 0.99)

    # After embedding at rate embed_rate, ratios move toward 0.5
    stego_ratios = natural_ratios * (1 - embed_rate) + 0.5 * embed_rate

    # Generate counts for each pair
    samples_per_pair = N // n_pairs
    histogram = np.zeros(n_pairs, dtype=int)
    for i in range(n_pairs):
        # Count how many are even in this pair
        count_even = np.random.binomial(samples_per_pair, stego_ratios[i])
        # The "test histogram" is the deviation from expected
        histogram[i] = count_even

    return histogram


def scenario_lcg_prng(N, n_bins, a=1103515245, c=12345, m=2**31):
    """Linear congruential generator (LCG) last-digit histogram.

    LCGs have known weaknesses in lower-order bits.
    Test uniformity of last digits.
    """
    seed = np.random.randint(1, m)
    digits = np.empty(N, dtype=int)
    x = seed
    for i in range(N):
        x = (a * x + c) % m
        digits[i] = x % n_bins
    return np.bincount(digits, minlength=n_bins)[:n_bins]


def scenario_banker_rounding(N, n_bins, n_decimals):
    """Banker's rounding (round-half-to-even) last digit distribution.

    Already in the paper, included for comparison.
    """
    values = np.random.uniform(0, 10 ** (n_decimals + 1), N)
    rounded = np.round(values / 10, n_decimals) * 10
    last_digits = (rounded.astype(int)) % n_bins
    return np.bincount(last_digits, minlength=n_bins)[:n_bins]


# =============================================================================
# Power of test computation
# =============================================================================

def compute_power(scenario_fn, N, n_bins, n_trials, alpha, scenario_kwargs):
    """Compute power of CT and chi-squared for a given scenario."""
    ct_rejections = 0
    chi2_rejections = 0
    ct_valid = 0

    for _ in range(n_trials):
        histogram = scenario_fn(N, n_bins, **scenario_kwargs)

        # Ensure histogram has right shape
        if len(histogram) != n_bins:
            continue
        if np.sum(histogram) == 0:
            continue

        actual_N = int(np.sum(histogram))
        dtv = compute_dtv(histogram)

        # Chi-squared test
        chi2_p = chi2_pvalue(histogram)
        if chi2_p < alpha:
            chi2_rejections += 1

        # CT test
        ct_p = ct_pvalue(dtv, actual_N, n_bins)
        if ct_p is not None:
            ct_valid += 1
            if ct_p < alpha:
                ct_rejections += 1

    chi2_power = chi2_rejections / n_trials
    ct_power = ct_rejections / ct_valid if ct_valid > 0 else None

    return ct_power, chi2_power, ct_valid


def main():
    np.random.seed(42)
    alpha = 0.05
    n_trials = 10000

    print("=" * 90)
    print("Power of test comparison: CT vs Pearson's chi-squared")
    print(f"alpha = {alpha}, n_trials = {n_trials}")
    print("=" * 90)

    scenarios = [
        # (name, function, N, n_bins, kwargs_list_with_labels)
        ("ADC alternating DNL", scenario_adc_alternating_dnl, [
            (100, 10, {"dnl_amplitude": 0.01}, "DNL=0.01"),
            (100, 10, {"dnl_amplitude": 0.02}, "DNL=0.02"),
            (100, 10, {"dnl_amplitude": 0.05}, "DNL=0.05"),
            (100, 10, {"dnl_amplitude": 0.10}, "DNL=0.10"),
            (200, 10, {"dnl_amplitude": 0.01}, "N=200, DNL=0.01"),
            (200, 10, {"dnl_amplitude": 0.02}, "N=200, DNL=0.02"),
            (200, 10, {"dnl_amplitude": 0.05}, "N=200, DNL=0.05"),
            (500, 10, {"dnl_amplitude": 0.01}, "N=500, DNL=0.01"),
            (500, 10, {"dnl_amplitude": 0.02}, "N=500, DNL=0.02"),
            (50, 20, {"dnl_amplitude": 0.05}, "N=50,n=20, DNL=0.05"),
            (100, 20, {"dnl_amplitude": 0.02}, "N=100,n=20, DNL=0.02"),
            (100, 20, {"dnl_amplitude": 0.05}, "N=100,n=20, DNL=0.05"),
        ]),
        ("ADC random DNL", scenario_adc_random_dnl, [
            (100, 10, {"dnl_std": 0.02}, "std=0.02"),
            (100, 10, {"dnl_std": 0.05}, "std=0.05"),
            (100, 10, {"dnl_std": 0.10}, "std=0.10"),
            (200, 10, {"dnl_std": 0.05}, "N=200, std=0.05"),
            (200, 10, {"dnl_std": 0.10}, "N=200, std=0.10"),
        ]),
        ("Re-quantization", scenario_requantization, [
            (100, 10, {"original_levels": 13}, "13->10 levels"),
            (100, 10, {"original_levels": 17}, "17->10 levels"),
            (100, 10, {"original_levels": 23}, "23->10 levels"),
            (200, 10, {"original_levels": 13}, "N=200, 13->10"),
            (200, 10, {"original_levels": 17}, "N=200, 17->10"),
            (100, 8, {"original_levels": 10}, "10->8 levels"),
            (100, 8, {"original_levels": 13}, "13->8 levels"),
            (200, 8, {"original_levels": 10}, "N=200, 10->8"),
        ]),
        ("Bad dither", scenario_bad_dither, [
            (100, 10, {"dither_bits": 1}, "1-bit dither"),
            (100, 10, {"dither_bits": 2}, "2-bit dither"),
            (100, 10, {"dither_bits": 3}, "3-bit dither"),
            (200, 10, {"dither_bits": 1}, "N=200, 1-bit"),
            (200, 10, {"dither_bits": 2}, "N=200, 2-bit"),
        ]),
        ("LCG PRNG", scenario_lcg_prng, [
            (100, 10, {}, "default LCG"),
            (200, 10, {}, "N=200"),
            (500, 10, {}, "N=500"),
            (100, 8, {}, "n=8"),
            (100, 16, {}, "n=16"),
            # Bad LCG with small modulus
            (100, 10, {"a": 37, "c": 1, "m": 256}, "bad LCG (m=256)"),
            (200, 10, {"a": 37, "c": 1, "m": 256}, "N=200, bad LCG"),
            (100, 10, {"a": 37, "c": 1, "m": 1024}, "bad LCG (m=1024)"),
        ]),
        ("Banker's rounding", scenario_banker_rounding, [
            (100, 10, {"n_decimals": 1}, "d=1"),
            (100, 10, {"n_decimals": 2}, "d=2"),
            (200, 10, {"n_decimals": 1}, "N=200, d=1"),
            (200, 10, {"n_decimals": 2}, "N=200, d=2"),
            (500, 10, {"n_decimals": 2}, "N=500, d=2"),
            (500, 10, {"n_decimals": 3}, "N=500, d=3"),
        ]),
    ]

    all_results = []

    for scenario_name, scenario_fn, configs in scenarios:
        print(f"\n--- {scenario_name} ---")
        print(f"  {'Config':<25s} | {'CT power':>10s} | {'Chi2 power':>10s} | {'CT-Chi2':>8s} | {'Winner':>6s}")
        print("  " + "-" * 75)

        for N, n_bins, kwargs, label in configs:
            ct_pow, chi2_pow, ct_valid = compute_power(
                scenario_fn, N, n_bins, n_trials, alpha, kwargs
            )

            if ct_pow is not None:
                diff = ct_pow - chi2_pow
                winner = "CT" if diff > 0.005 else ("Chi2" if diff < -0.005 else "~tie")
                print(f"  {label:<25s} | {ct_pow:10.4f} | {chi2_pow:10.4f} | {diff:+8.4f} | {winner:>6s}")
            else:
                print(f"  {label:<25s} | {'N/A':>10s} | {chi2_pow:10.4f} | {'N/A':>8s} |")

            all_results.append({
                'scenario': scenario_name, 'label': label,
                'N': N, 'n': n_bins,
                'ct_power': ct_pow, 'chi2_power': chi2_pow,
                'ct_valid': ct_valid,
            })

    # Summary: which scenarios show CT advantage?
    print("\n" + "=" * 90)
    print("SUMMARY: Scenarios where CT significantly outperforms chi-squared (diff > 0.01)")
    print("=" * 90)
    for r in all_results:
        if r['ct_power'] is not None and (r['ct_power'] - r['chi2_power']) > 0.01:
            print(f"  {r['scenario']:25s} | {r['label']:25s} | CT={r['ct_power']:.4f} Chi2={r['chi2_power']:.4f} "
                  f"diff={r['ct_power']-r['chi2_power']:+.4f}")

    print("\nSUMMARY: Scenarios where chi-squared significantly outperforms CT (diff > 0.01)")
    print("=" * 90)
    for r in all_results:
        if r['ct_power'] is not None and (r['chi2_power'] - r['ct_power']) > 0.01:
            print(f"  {r['scenario']:25s} | {r['label']:25s} | CT={r['ct_power']:.4f} Chi2={r['chi2_power']:.4f} "
                  f"diff={r['chi2_power']-r['ct_power']:+.4f}")


if __name__ == '__main__':
    main()
