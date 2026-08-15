#!/usr/bin/env python3
"""Simulate rounding experiments and compare Comb Test vs Pearson's chi-squared.

Tests different rounding methods on random numbers and measures the power
of both tests to detect non-uniformity in the last-digit histogram.

Rounding methods tested:
  - round-half-to-even (Python/NumPy default, aka banker's rounding)
  - round-half-up (traditional rounding)
  - round-half-down
  - truncation (round toward zero)
  - ceiling (round up)

For each method, random numbers are generated, rounded, and the last digit
histogram is tested for uniformity.
"""

import argparse
import numpy as np
from scipy import stats
import json
import sys
import os
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_HALF_DOWN, ROUND_FLOOR, ROUND_CEILING

from comb_test import comb_test, comb_test_gamma, pearson_chi2_test, dtv, estimate_gamma_params


def round_half_to_even(x, d_target):
    """Round using round-half-to-even (banker's rounding)."""
    return float(Decimal(str(x)).quantize(Decimal(10) ** (-d_target), rounding=ROUND_HALF_EVEN))


def round_half_up(x, d_target):
    """Round using round-half-up (traditional)."""
    return float(Decimal(str(x)).quantize(Decimal(10) ** (-d_target), rounding=ROUND_HALF_UP))


def round_half_down(x, d_target):
    """Round using round-half-down."""
    return float(Decimal(str(x)).quantize(Decimal(10) ** (-d_target), rounding=ROUND_HALF_DOWN))


def round_truncate(x, d_target):
    """Round by truncation (toward zero)."""
    return float(Decimal(str(x)).quantize(Decimal(10) ** (-d_target), rounding=ROUND_FLOOR))


def round_ceiling(x, d_target):
    """Round by ceiling (always up)."""
    return float(Decimal(str(x)).quantize(Decimal(10) ** (-d_target), rounding=ROUND_CEILING))


def get_last_digit(x, d_target):
    """Extract the last digit at decimal position d_target."""
    shifted = round(x * (10 ** d_target))
    return int(abs(shifted)) % 10


def generate_and_round(N, d_source, d_target, rounding_method, rng):
    """Generate N random numbers with d_source decimals and round to d_target.

    Returns histogram of last digits (10 bins).
    """
    # Generate random numbers with d_source decimal places
    max_val = 10 ** d_source
    raw_integers = rng.integers(0, max_val, size=N)
    raw_numbers = raw_integers / (10 ** d_source)

    # Round to d_target decimals
    if rounding_method == "half_to_even":
        # Use numpy's round which is half-to-even
        rounded = np.round(raw_numbers, d_target)
    elif rounding_method == "half_up":
        rounded = np.array([round_half_up(x, d_target) for x in raw_numbers])
    elif rounding_method == "half_down":
        rounded = np.array([round_half_down(x, d_target) for x in raw_numbers])
    elif rounding_method == "truncate":
        rounded = np.array([round_truncate(x, d_target) for x in raw_numbers])
    elif rounding_method == "ceiling":
        rounded = np.array([round_ceiling(x, d_target) for x in raw_numbers])
    else:
        raise ValueError(f"Unknown rounding method: {rounding_method}")

    # Get last digits and build histogram
    last_digits = np.array([get_last_digit(x, d_target) for x in rounded])
    hist = np.bincount(last_digits, minlength=10)
    return hist


def generate_and_round_fast(N, d_source, d_target, rounding_method, rng):
    """Fast vectorized version for half_to_even and half_up."""
    max_val = 10 ** d_source
    raw_integers = rng.integers(0, max_val, size=N)

    if rounding_method == "half_to_even":
        # numpy round is banker's rounding
        factor = 10 ** d_target
        shifted = raw_integers * factor / max_val
        rounded_shifted = np.round(shifted).astype(np.int64)
        last_digits = np.abs(rounded_shifted) % 10
    elif rounding_method == "half_up":
        factor = 10 ** d_target
        shifted = raw_integers * factor / max_val
        # half_up: floor(x + 0.5)
        rounded_shifted = np.floor(shifted + 0.5).astype(np.int64)
        last_digits = np.abs(rounded_shifted) % 10
    else:
        # Fall back to slow method for others
        raw_numbers = raw_integers / max_val
        rounded = np.array([
            round_half_to_even(x, d_target) if rounding_method == "half_to_even"
            else round_half_up(x, d_target) if rounding_method == "half_up"
            else round_half_down(x, d_target) if rounding_method == "half_down"
            else round_truncate(x, d_target) if rounding_method == "truncate"
            else round_ceiling(x, d_target)
            for x in raw_numbers
        ])
        last_digits = np.array([get_last_digit(x, d_target) for x in rounded])

    hist = np.bincount(last_digits, minlength=10)
    return hist


def run_power_experiment(N_values, d_source, d_target, rounding_method,
                         num_trials, alpha=0.05, use_gamma=False, verbose=True):
    """Run power experiment for a range of N values.

    Returns dict mapping N -> {ct_rejections, chi2_rejections, ct_power, chi2_power,
                                ct_mean_p, chi2_mean_p}
    """
    results = {}
    n_bins = 10  # last digit histogram always has 10 bins

    for N in N_values:
        if verbose:
            print(f"  N={N}, method={rounding_method}, d_source={d_source}, "
                  f"d_target={d_target} ...", end=" ", flush=True)

        # Pre-compute gamma params if using gamma approximation
        gamma_params = None
        if use_gamma:
            gamma_params = estimate_gamma_params(N, n_bins, num_samples=200000)

        rng = np.random.default_rng(seed=42 + N)

        ct_rejections = 0
        chi2_rejections = 0
        ct_p_sum = 0.0
        chi2_p_sum = 0.0

        for trial in range(num_trials):
            hist = generate_and_round_fast(N, d_source, d_target,
                                           rounding_method, rng)

            # Comb Test
            if use_gamma:
                p_ct = comb_test_gamma(hist, gamma_params=gamma_params)
            else:
                p_ct = comb_test(hist)
                if p_ct is None:
                    # Fall back to gamma if exact not available
                    if gamma_params is None:
                        gamma_params = estimate_gamma_params(N, n_bins, num_samples=200000)
                    p_ct = comb_test_gamma(hist, gamma_params=gamma_params)

            # Pearson's chi-squared test
            p_chi2 = pearson_chi2_test(hist)

            if p_ct <= alpha:
                ct_rejections += 1
            if p_chi2 <= alpha:
                chi2_rejections += 1

            ct_p_sum += p_ct
            chi2_p_sum += p_chi2

        ct_power = ct_rejections / num_trials
        chi2_power = chi2_rejections / num_trials

        results[N] = {
            "ct_rejections": ct_rejections,
            "chi2_rejections": chi2_rejections,
            "ct_power": ct_power,
            "chi2_power": chi2_power,
            "ct_mean_p": ct_p_sum / num_trials,
            "chi2_mean_p": chi2_p_sum / num_trials,
        }

        if verbose:
            print(f"CT power={ct_power:.4f}, Chi2 power={chi2_power:.4f}, "
                  f"CT mean p={ct_p_sum/num_trials:.4f}, "
                  f"Chi2 mean p={chi2_p_sum/num_trials:.4f}")

    return results


def run_type_i_error_experiment(N_values, num_trials, alpha=0.05,
                                use_gamma=False, verbose=True):
    """Measure Type I error rate (false positive) under truly uniform histograms.

    Generates uniform histograms and checks that both tests reject at approximately
    the alpha rate.
    """
    results = {}
    n_bins = 10

    for N in N_values:
        if verbose:
            print(f"  Type I error: N={N} ...", end=" ", flush=True)

        gamma_params = None
        if use_gamma:
            gamma_params = estimate_gamma_params(N, n_bins, num_samples=200000)

        rng = np.random.default_rng(seed=123 + N)

        ct_rejections = 0
        chi2_rejections = 0

        for trial in range(num_trials):
            # Generate truly uniform histogram
            bins = rng.choice(n_bins, size=N)
            hist = np.bincount(bins, minlength=n_bins)

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
                ct_rejections += 1
            if p_chi2 <= alpha:
                chi2_rejections += 1

        results[N] = {
            "ct_type_i": ct_rejections / num_trials,
            "chi2_type_i": chi2_rejections / num_trials,
        }

        if verbose:
            print(f"CT Type I={ct_rejections/num_trials:.4f}, "
                  f"Chi2 Type I={chi2_rejections/num_trials:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Rounding simulation: compare Comb Test vs Pearson's chi-squared")
    parser.add_argument("--methods", nargs="+",
                        default=["half_to_even", "half_up"],
                        choices=["half_to_even", "half_up", "half_down",
                                 "truncate", "ceiling"],
                        help="Rounding methods to test")
    parser.add_argument("--d-source", type=int, default=2,
                        help="Number of decimal places in source numbers (default: 2)")
    parser.add_argument("--d-target", type=int, default=1,
                        help="Number of decimal places after rounding (default: 1)")
    parser.add_argument("--N-min", type=int, default=100,
                        help="Minimum sample size")
    parser.add_argument("--N-max", type=int, default=1000,
                        help="Maximum sample size")
    parser.add_argument("--N-step", type=int, default=100,
                        help="Sample size step")
    parser.add_argument("--trials", type=int, default=10000,
                        help="Number of trials per N value")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level")
    parser.add_argument("--gamma", action="store_true",
                        help="Use gamma approximation instead of exact")
    parser.add_argument("--type-i", action="store_true",
                        help="Also run Type I error experiment")
    parser.add_argument("--output", default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    N_values = list(range(args.N_min, args.N_max + 1, args.N_step))

    all_results = {}

    for method in args.methods:
        print(f"\n=== Rounding method: {method} ===")
        print(f"  d_source={args.d_source}, d_target={args.d_target}, "
              f"trials={args.trials}, alpha={args.alpha}")
        results = run_power_experiment(
            N_values, args.d_source, args.d_target, method,
            args.trials, args.alpha, use_gamma=args.gamma)
        all_results[method] = {
            str(k): v for k, v in results.items()
        }

    if args.type_i:
        print(f"\n=== Type I error (uniform distribution) ===")
        type_i = run_type_i_error_experiment(
            N_values, args.trials, args.alpha, use_gamma=args.gamma)
        all_results["type_i_error"] = {
            str(k): v for k, v in type_i.items()
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: Power of test (rejection rate at alpha={args.alpha})")
    print(f"{'='*80}")

    for method in args.methods:
        print(f"\n--- {method} (d_source={args.d_source}, d_target={args.d_target}) ---")
        print(f"{'N':>6} | {'CT Power':>10} | {'Chi2 Power':>10} | "
              f"{'CT mean p':>10} | {'Chi2 mean p':>10} | {'Winner':>8}")
        print("-" * 70)
        for N in N_values:
            r = all_results[method][str(N)]
            winner = "CT" if r["ct_power"] > r["chi2_power"] else (
                "Chi2" if r["chi2_power"] > r["ct_power"] else "Tie")
            print(f"{N:>6} | {r['ct_power']:>10.4f} | {r['chi2_power']:>10.4f} | "
                  f"{r['ct_mean_p']:>10.4f} | {r['chi2_mean_p']:>10.4f} | {winner:>8}")


if __name__ == "__main__":
    main()
