#!/usr/bin/env python3
"""Generate mean p-values for the banker's rounding experiment.

Simulates histograms under round-half-to-even (banker's rounding) for each N,
computes test p-values, and outputs mean p-values per N. Resumes from existing
output if interrupted.

The flag -d is one less than the manuscript's r, the number of decimals the
rounding discards: r = d + 1.  Over the n = 10 last digits the law is
  even digits: (10^r + 1) / 10^(r+1)
  odd digits:  (10^r - 1) / 10^(r+1)
a period-2 comb of amplitude 10^-(r+1).
"""

import os
import argparse
import numpy as np
from scipy import stats
from tqdm import tqdm


def compute_dtv(histogram):
    """Compute DTV: sum of |h_{i+1} - h_i|."""
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


def fit_beta_mc(N, n, K=50000, seed=13):
    """Fit beta distribution to DTV via MC sampling under H0."""
    from scipy.optimize import minimize as sp_minimize
    rng_mc = np.random.RandomState(seed)
    probs_uniform = np.full(n, 1.0 / n)
    hists_mc = rng_mc.multinomial(N, probs_uniform, size=K)
    dtvs_mc = np.sum(np.abs(np.diff(hists_mc, axis=1)), axis=1)

    unique_vals, counts = np.unique(dtvs_mc, return_counts=True)
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

    result = sp_minimize(cvm_objective,
                         x0=[np.log(init_a), np.log(init_b), np.log(max_val)],
                         method="Nelder-Mead")
    a = np.exp(result.x[0])
    b = np.exp(result.x[1])
    scale = np.exp(result.x[2])
    return (a, b, 0.0, scale)


# Cache for MC-fitted distributions
_mc_beta_cache = {}
_mc_gamma_cache = {}
_approx = "gamma"  # set from --approx flag


def fit_gamma_mc(N, n, K=50000, seed=13):
    """Fit gamma distribution to DTV via MC sampling under H0."""
    from scipy.optimize import minimize as sp_minimize
    rng_mc = np.random.RandomState(seed)
    probs_uniform = np.full(n, 1.0 / n)
    hists_mc = rng_mc.multinomial(N, probs_uniform, size=K)
    dtvs_mc = np.sum(np.abs(np.diff(hists_mc, axis=1)), axis=1)

    unique_vals, counts = np.unique(dtvs_mc, return_counts=True)
    cdf_probs = np.cumsum(counts) / K

    mean_dtv = np.mean(dtvs_mc)
    var_dtv = np.var(dtvs_mc)
    if var_dtv <= 0:
        var_dtv = 1.0
    init_a = mean_dtv**2 / var_dtv
    init_scale = var_dtv / mean_dtv

    def cvm_objective(log_params):
        a = np.exp(log_params[0])
        sc = np.exp(log_params[1])
        theo = stats.gamma.cdf(unique_vals, a=a, loc=0, scale=sc)
        return np.sum((cdf_probs - theo) ** 2)

    result = sp_minimize(cvm_objective,
                         x0=[np.log(max(init_a, 1e-10)),
                             np.log(max(init_scale, 1e-10))],
                         method="Nelder-Mead")
    a = np.exp(result.x[0])
    scale = np.exp(result.x[1])
    return (a, 0.0, scale)


def get_beta_dist(N, n):
    """Get beta distribution for (N, n): from file or MC fit (cached)."""
    beta_path = f"../data/fitted/cvm_beta/N_{N}_n_{n}.txt"
    if os.path.isfile(beta_path):
        params = load_beta_params(beta_path)
        return stats.beta(a=params[0], b=params[1], loc=params[2], scale=params[3])

    if (N, n) not in _mc_beta_cache:
        params = fit_beta_mc(N, n)
        _mc_beta_cache[(N, n)] = stats.beta(a=params[0], b=params[1],
                                             loc=params[2], scale=params[3])
    return _mc_beta_cache[(N, n)]


def get_gamma_dist(N, n):
    """Get gamma distribution for (N, n): from file or MC fit (cached)."""
    gamma_path = f"../data/fitted/cvm_gamma/N_{N}_n_{n}.txt"
    if os.path.isfile(gamma_path):
        params = load_beta_params(gamma_path)
        return stats.gamma(a=params[0], loc=params[1], scale=params[2])

    if (N, n) not in _mc_gamma_cache:
        params = fit_gamma_mc(N, n)
        _mc_gamma_cache[(N, n)] = stats.gamma(a=params[0], loc=params[1],
                                               scale=params[2])
    return _mc_gamma_cache[(N, n)]


def get_approx_dist(N, n):
    """Get the configured approximation distribution."""
    if _approx == "gamma":
        return get_gamma_dist(N, n)
    return get_beta_dist(N, n)


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


def banker_rounding_probs(n, d):
    """Bin probabilities for banker's rounding discarding r = d + 1 decimals.

    A tie occurs with probability 10^-r and always goes to the even neighbor,
    so for n = 10 the even digits are slightly more probable:
      P(even digit) = (10^r + 1) / 10^(r+1)
      P(odd digit)  = (10^r - 1) / 10^(r+1)
    The weights below are unnormalized; the division by their sum yields these.
    """
    base = 10 ** (d + 1)
    probs = np.zeros(n)
    probs[0::2] = base + 1  # even digits
    probs[1::2] = base - 1  # odd digits
    return probs / probs.sum()


def load_existing(output_path):
    """Load previously computed results for resuming."""
    existing = {}
    if os.path.isfile(output_path):
        with open(output_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    existing[int(parts[0])] = line.strip()
    return existing


def main():
    parser = argparse.ArgumentParser(
        description="Generate mean p-values for banker's rounding experiment")
    parser.add_argument("-d", type=int, default=0,
                        help="One less than the manuscript's r, the number of "
                             "decimals discarded by the rounding (default: 0, "
                             "i.e. r=1)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of bins / last digits (default: 10)")
    parser.add_argument("--repeat", type=int, default=1000000,
                        help="Number of histograms per N (default: 1000000)")
    parser.add_argument("--N-min", type=int, default=2,
                        help="Minimum N (default: 2)")
    parser.add_argument("--N-max", type=int, default=3000,
                        help="Maximum N (default: 3000)")
    parser.add_argument("--step", type=int, default=100,
                        help="Step size for N (default: 100)")
    parser.add_argument("-p", "--pearson", action="store_true",
                        help="Include Pearson's chi-squared test")
    parser.add_argument("-g", "--gtest", action="store_true",
                        help="Include G-test")
    parser.add_argument("--seed", type=int, default=13,
                        help="Random seed (default: 13)")
    parser.add_argument("--approx", choices=["gamma", "beta"], default="gamma",
                        help="Approximation distribution for CT (default: gamma)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (default: mean_p_values_d_{d}_step_{step}.txt)")
    args = parser.parse_args()

    global _approx
    _approx = args.approx

    if not args.pearson and not args.gtest:
        args.pearson = True
        args.gtest = True

    if args.output is None:
        args.output = f"../data/mean_p_values_d_{args.d}_step_{args.step}.txt"

    n = args.n
    d = args.d
    repeat = args.repeat
    probs = banker_rounding_probs(n, d)

    # Build N range: 2..99 individually, then step
    Ns = list(range(args.N_min, min(100, args.N_max + 1)))
    if args.N_max >= 100:
        Ns += list(range(100, args.N_max + 1, args.step))
    Ns = sorted(set(Ns))

    # Build header
    columns = ["N", "ct"]
    if args.pearson:
        columns.append("chi2")
    if args.gtest:
        columns.append("gtest")

    existing = load_existing(args.output)
    print(f"Banker's rounding mean p-value generation")
    print(f"d={d}, n={n}, repeat={repeat}, seed={args.seed}")
    print(f"N range: {Ns[0]}..{Ns[-1]} ({len(Ns)} values)")
    print(f"Tests: {', '.join(columns[1:])}")
    print(f"Output: {args.output}")
    print(f"Existing results: {len(existing)}")
    print(flush=True)

    rng = np.random.RandomState(args.seed)

    with open(args.output, "w") as fo:
        # Write existing results first (preserve order)
        for N_val in tqdm(Ns, desc="Mean p-values"):
            if N_val in existing:
                # Check if existing has enough columns
                parts = existing[N_val].split()
                if len(parts) == len(columns):
                    fo.write(existing[N_val] + "\n")
                    continue
                # Column count changed — recompute

            # Generate histograms
            hists = rng.multinomial(N_val, probs, size=repeat)

            # CT p-values
            dtvs = np.sum(np.abs(np.diff(hists, axis=1)), axis=1)
            ct_ps = np.zeros(repeat)
            ct_p_func = None

            # Try exact CDF first, then fitted beta, then MC beta
            cdf_path = f"../data/cdf_exact/N_{N_val}_n_{n}.txt"
            if os.path.isfile(cdf_path):
                cdf_values, cdf_probs = load_cdf(cdf_path)
                for i in range(repeat):
                    idx = np.searchsorted(cdf_values, dtvs[i] - 1, side="right") - 1
                    ct_ps[i] = 1.0 - cdf_probs[idx] if idx >= 0 else 1.0
            else:
                dist = get_approx_dist(N_val, n)
                ct_ps = 1.0 - dist.cdf(dtvs - 0.5)

            row = [str(N_val), f"{np.nanmean(ct_ps):.10f}"]

            # Chi-squared
            if args.pearson:
                expected = N_val / n
                chi2_stats = np.sum((hists - expected) ** 2 / expected, axis=1)
                chi2_ps = 1.0 - stats.chi2.cdf(chi2_stats, df=n - 1)
                row.append(f"{np.mean(chi2_ps):.10f}")

            # G-test
            if args.gtest:
                expected = N_val / n
                h_safe = np.where(hists > 0, hists, 0.5)
                G_stats = 2 * np.sum(hists * np.log(h_safe / expected), axis=1)
                gtest_ps = 1.0 - stats.chi2.cdf(G_stats, df=n - 1)
                row.append(f"{np.mean(gtest_ps):.10f}")

            fo.write(" ".join(row) + "\n")
            fo.flush()

            if N_val % 100 == 0 or N_val <= 10:
                print(f"  N={N_val}: {' '.join(row[1:])}", flush=True)

    print(f"\nDone. Saved to {args.output}")


if __name__ == "__main__":
    main()
