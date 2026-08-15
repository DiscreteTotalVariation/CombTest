#!/usr/bin/env python3
"""Additional simulation experiments for the Comb Test.

Tests the Comb Test against Pearson's chi-squared for various types of
non-uniform distributions that are comb-like or exhibit alternating patterns.

Scenarios:
  1. Comb distribution: alternating bins with higher/lower probabilities
  2. Benford's Law deviation: testing for digit frequency anomalies
  3. Alternating-bias distribution: even bins slightly favored over odd
  4. Sinusoidal perturbation: probabilities oscillate around uniform
  5. Random number generator quality: testing PRNG output digit uniformity
"""

import argparse
import numpy as np
from scipy import stats
import json

from comb_test import comb_test, comb_test_gamma, pearson_chi2_test, dtv, estimate_gamma_params


def generate_comb_histogram(N, n, delta, rng):
    """Generate a histogram from a comb distribution.

    Bin probabilities alternate between (1+delta)/n and (1-delta)/n.

    Args:
        N: sample size
        n: number of bins
        delta: perturbation strength (0 = uniform, 1 = max comb)
        rng: numpy random generator
    """
    probs = np.ones(n) / n
    for i in range(n):
        if i % 2 == 0:
            probs[i] *= (1 + delta)
        else:
            probs[i] *= (1 - delta)
    probs /= probs.sum()
    samples = rng.choice(n, size=N, p=probs)
    return np.bincount(samples, minlength=n)


def generate_sinusoidal_histogram(N, n, amplitude, frequency, rng):
    """Generate a histogram with sinusoidal probability perturbation.

    p_i = (1 + amplitude * sin(2*pi*frequency*i/n)) / n (normalized)
    """
    idx = np.arange(n)
    probs = 1.0 + amplitude * np.sin(2 * np.pi * frequency * idx / n)
    probs = np.maximum(probs, 0.001)
    probs /= probs.sum()
    samples = rng.choice(n, size=N, p=probs)
    return np.bincount(samples, minlength=n)


def generate_alternating_bias_histogram(N, n, bias, rng):
    """Generate histogram where even-indexed bins are biased.

    Even bins have probability (1+bias)/n, odd bins have (1-bias)/n.
    This models the round-half-to-even effect.
    """
    probs = np.ones(n)
    probs[0::2] *= (1 + bias)
    probs[1::2] *= (1 - bias)
    probs /= probs.sum()
    samples = rng.choice(n, size=N, p=probs)
    return np.bincount(samples, minlength=n)


def generate_block_bias_histogram(N, n, bias, block_size, rng):
    """Generate histogram where blocks of bins alternate between high and low.

    E.g., with block_size=2: [high, high, low, low, high, high, ...]
    """
    probs = np.ones(n)
    for i in range(n):
        block_idx = i // block_size
        if block_idx % 2 == 0:
            probs[i] *= (1 + bias)
        else:
            probs[i] *= (1 - bias)
    probs /= probs.sum()
    samples = rng.choice(n, size=N, p=probs)
    return np.bincount(samples, minlength=n)


def generate_single_spike_histogram(N, n, spike_idx, spike_factor, rng):
    """Generate histogram where one bin has higher probability.

    This is a non-comb deviation; chi-squared should be better here.
    """
    probs = np.ones(n)
    probs[spike_idx] *= spike_factor
    probs /= probs.sum()
    samples = rng.choice(n, size=N, p=probs)
    return np.bincount(samples, minlength=n)


def run_scenario(name, gen_func, N_values, n_bins, num_trials, alpha,
                 use_gamma=False, verbose=True):
    """Run a power comparison scenario.

    Args:
        gen_func: function(N, n, rng) -> histogram
    """
    results = {}

    for N in N_values:
        if verbose:
            print(f"  {name}: N={N} ...", end=" ", flush=True)

        gamma_params = None
        if use_gamma:
            gamma_params = estimate_gamma_params(N, n_bins, num_samples=200000)

        rng = np.random.default_rng(seed=42 + N)

        ct_rej = 0
        chi2_rej = 0
        ct_p_sum = 0.0
        chi2_p_sum = 0.0

        for _ in range(num_trials):
            hist = gen_func(N, n_bins, rng)

            if use_gamma:
                p_ct = comb_test_gamma(hist, gamma_params=gamma_params)
            else:
                p_ct = comb_test(hist)
                if p_ct is None:
                    if gamma_params is None:
                        gamma_params = estimate_gamma_params(N, n_bins, num_samples=200000)
                    p_ct = comb_test_gamma(hist, gamma_params=gamma_params)

            p_chi2 = pearson_chi2_test(hist)

            if p_ct <= alpha:
                ct_rej += 1
            if p_chi2 <= alpha:
                chi2_rej += 1
            ct_p_sum += p_ct
            chi2_p_sum += p_chi2

        results[N] = {
            "ct_power": ct_rej / num_trials,
            "chi2_power": chi2_rej / num_trials,
            "ct_mean_p": ct_p_sum / num_trials,
            "chi2_mean_p": chi2_p_sum / num_trials,
        }

        if verbose:
            r = results[N]
            print(f"CT={r['ct_power']:.4f}, Chi2={r['chi2_power']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Additional Comb Test simulation experiments")
    parser.add_argument("--N-min", type=int, default=100)
    parser.add_argument("--N-max", type=int, default=1000)
    parser.add_argument("--N-step", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--gamma", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    N_values = list(range(args.N_min, args.N_max + 1, args.N_step))
    n = args.n_bins
    alpha = args.alpha
    trials = args.trials
    use_gamma = args.gamma

    all_results = {}

    # Scenario 1: Comb distribution (delta=0.05)
    print("\n=== Scenario 1: Comb distribution (delta=0.05) ===")
    all_results["comb_0.05"] = run_scenario(
        "Comb(0.05)",
        lambda N, n, rng: generate_comb_histogram(N, n, 0.05, rng),
        N_values, n, trials, alpha, use_gamma)

    # Scenario 2: Comb distribution (delta=0.1)
    print("\n=== Scenario 2: Comb distribution (delta=0.1) ===")
    all_results["comb_0.1"] = run_scenario(
        "Comb(0.1)",
        lambda N, n, rng: generate_comb_histogram(N, n, 0.1, rng),
        N_values, n, trials, alpha, use_gamma)

    # Scenario 3: Alternating bias (even bins favored, bias=0.05)
    print("\n=== Scenario 3: Alternating bias (bias=0.05) ===")
    all_results["alt_bias_0.05"] = run_scenario(
        "AltBias(0.05)",
        lambda N, n, rng: generate_alternating_bias_histogram(N, n, 0.05, rng),
        N_values, n, trials, alpha, use_gamma)

    # Scenario 4: Sinusoidal (high frequency, amplitude=0.1)
    print("\n=== Scenario 4: Sinusoidal (freq=5, amp=0.1) ===")
    all_results["sin_f5_a0.1"] = run_scenario(
        "Sin(f=5,a=0.1)",
        lambda N, n, rng: generate_sinusoidal_histogram(N, n, 0.1, 5, rng),
        N_values, n, trials, alpha, use_gamma)

    # Scenario 5: Sinusoidal (low frequency, amplitude=0.1) - chi2 should win
    print("\n=== Scenario 5: Sinusoidal (freq=1, amp=0.1) ===")
    all_results["sin_f1_a0.1"] = run_scenario(
        "Sin(f=1,a=0.1)",
        lambda N, n, rng: generate_sinusoidal_histogram(N, n, 0.1, 1, rng),
        N_values, n, trials, alpha, use_gamma)

    # Scenario 6: Single spike (bin 0 has 2x probability) - chi2 should win
    print("\n=== Scenario 6: Single spike (factor=2.0) ===")
    all_results["spike_2.0"] = run_scenario(
        "Spike(2.0)",
        lambda N, n, rng: generate_single_spike_histogram(N, n, 0, 2.0, rng),
        N_values, n, trials, alpha, use_gamma)

    # Scenario 7: Block bias (blocks of 2, bias=0.1)
    print("\n=== Scenario 7: Block bias (block=2, bias=0.1) ===")
    all_results["block_b2_0.1"] = run_scenario(
        "Block(2,0.1)",
        lambda N, n, rng: generate_block_bias_histogram(N, n, 0.1, 2, rng),
        N_values, n, trials, alpha, use_gamma)

    # Print summary
    print(f"\n{'='*90}")
    print(f"SUMMARY TABLE: Power of test (alpha={alpha}, trials={trials})")
    print(f"{'='*90}")

    scenarios = list(all_results.keys())
    header = f"{'N':>6}"
    for s in scenarios:
        header += f" | {s[:12]:>12} CT/Chi2"
    print(header)
    print("-" * len(header))

    for N in N_values:
        line = f"{N:>6}"
        for s in scenarios:
            r = all_results[s][N]
            ct = r["ct_power"]
            chi2 = r["chi2_power"]
            line += f" | {ct:.3f}/{chi2:.3f}"
        print(line)

    # Identify where CT wins and where Chi2 wins
    print(f"\n{'='*60}")
    print("ANALYSIS: Where each test excels")
    print(f"{'='*60}")

    for s in scenarios:
        ct_wins = sum(1 for N in N_values
                      if all_results[s][N]["ct_power"] > all_results[s][N]["chi2_power"])
        chi2_wins = sum(1 for N in N_values
                        if all_results[s][N]["chi2_power"] > all_results[s][N]["ct_power"])
        avg_ct = np.mean([all_results[s][N]["ct_power"] for N in N_values])
        avg_chi2 = np.mean([all_results[s][N]["chi2_power"] for N in N_values])
        winner = "CT" if avg_ct > avg_chi2 else "Chi2"
        print(f"  {s:20s}: CT wins {ct_wins}/{len(N_values)}, "
              f"Chi2 wins {chi2_wins}/{len(N_values)}, "
              f"avg power CT={avg_ct:.4f} Chi2={avg_chi2:.4f} -> {winner}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
