#!/usr/bin/env python3
"""Experiment: MC convergence for beta approximation of DTV distribution.

For fixed n=10 and N=1..600, tests how many Monte Carlo samples K are needed
for the beta approximation to converge to the best achievable fit (the one
obtained from the exact distribution). Saves per-(N,K) results to a directory.

Uses multiprocessing to parallelize across N values.
"""

import os
import argparse
import json
import numpy as np
import multiprocessing as mp
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm


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
    probs = np.full(n, 1.0 / n)
    hists = rng.multinomial(N, probs, size=K)
    dtvs = np.sum(np.abs(np.diff(hists, axis=1)), axis=1)

    unique_vals, counts = np.unique(dtvs, return_counts=True)
    cdf_probs = np.cumsum(counts) / K

    return fit_beta_to_cdf(unique_vals, cdf_probs)


def compute_pvalue_error(exact_vals, exact_cdfs, beta_params, alpha=0.05):
    """Compute absolute p-value error at alpha level."""
    target = 1 - alpha
    best_idx = np.argmin(np.abs(exact_cdfs - target))
    exact_crit = exact_vals[best_idx]
    exact_p = 1.0 - exact_cdfs[best_idx]

    bd = stats.beta(a=beta_params[0], b=beta_params[1],
                    loc=beta_params[2], scale=beta_params[3])
    beta_p = 1.0 - bd.cdf(exact_crit - 0.5)

    return abs(exact_p - beta_p)


# Global config set by main before forking
_CONFIG = {}


def process_one_N(N):
    """Process a single N value: fit exact beta, then MC beta for all K values."""
    n = _CONFIG["n"]
    K_values = _CONFIG["K_values"]
    n_repeats = _CONFIG["n_repeats"]
    base_seed = _CONFIG["seed"]
    alpha = _CONFIG["alpha"]
    output_dir = _CONFIG["output_dir"]

    cdf_path = f"../data/cdf_exact/N_{N}_n_{n}.txt"
    if not os.path.isfile(cdf_path):
        return None

    exact_vals, exact_cdfs = load_cdf(cdf_path)

    # Fit beta to exact CDF (ground truth best fit)
    exact_beta = fit_beta_to_cdf(exact_vals, exact_cdfs)
    exact_error = compute_pvalue_error(exact_vals, exact_cdfs, exact_beta, alpha)

    # MC fits for each K
    result = {
        "N": N,
        "n": n,
        "exact_beta_params": list(exact_beta),
        "exact_error": exact_error,
        "K_values": {},
    }

    for K in K_values:
        errors = []
        for rep in range(n_repeats):
            rng = np.random.RandomState(base_seed + rep)
            mc_params = fit_beta_mc(N, n, K, rng)
            err = compute_pvalue_error(exact_vals, exact_cdfs, mc_params, alpha)
            errors.append(err)
        result["K_values"][str(K)] = {
            "errors": errors,
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "median": float(np.median(errors)),
            "max": float(np.max(errors)),
        }

    # Save per-N result
    out_path = os.path.join(output_dir, f"N_{N}.json")
    with open(out_path, "w") as f:
        json.dump(result, f)

    return N


def init_worker(config):
    """Initialize worker with shared config."""
    global _CONFIG
    _CONFIG = config


def discover_all_n(cdf_dir="../data/cdf_exact"):
    """Discover all available n values from exact CDF files."""
    n_values = set()
    for fname in os.listdir(cdf_dir):
        if fname.startswith("N_") and fname.endswith(".txt"):
            parts = fname.replace(".txt", "").split("_")
            if len(parts) == 4:
                try:
                    n_values.add(int(parts[3]))
                except ValueError:
                    pass
    return sorted(n_values)


def run_for_n(n_val, args, n_workers):
    """Run the experiment for a single n value."""
    output_dir = os.path.join(args.output_dir, f"n_{n_val}")

    config = {
        "n": n_val,
        "N_min": args.N_min,
        "N_max": args.N_max,
        "K_values": args.K_values,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "alpha": args.alpha,
        "output_dir": output_dir,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n--- n={n_val}, N={config['N_min']}..{config['N_max']} ---", flush=True)

    N_range = list(range(config["N_min"], config["N_max"] + 1))

    with mp.Pool(n_workers, initializer=init_worker, initargs=(config,)) as pool:
        for result_N in tqdm(pool.imap_unordered(process_one_N, N_range),
                             total=len(N_range), desc=f"n={n_val}", leave=True):
            pass

    print(f"  Done (n={n_val}). Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="MC convergence experiment for beta approximation of DTV")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of bins (default: 10; ignored if --all-n)")
    parser.add_argument("--all-n", action="store_true",
                        help="Run for all n values found in cdf_exact/")
    parser.add_argument("--n-list", type=int, nargs="+", default=None,
                        help="Run for specific n values (e.g., --n-list 4 10 100)")
    parser.add_argument("--N-min", type=int, default=1,
                        help="Minimum N (default: 1)")
    parser.add_argument("--N-max", type=int, default=600,
                        help="Maximum N (default: 600)")
    parser.add_argument("--K-values", type=int, nargs="+",
                        default=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
                        help="MC sample sizes to test")
    parser.add_argument("--n-repeats", type=int, default=100,
                        help="Number of repeats per K value (default: 100)")
    parser.add_argument("--seed", type=int, default=13,
                        help="Base random seed (default: 13)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument("--output-dir", type=str, default="../data/mc_convergence_results",
                        help="Output directory (default: mc_convergence_results)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: cpu_count - 2)")
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else max(1, mp.cpu_count() - 2)

    print(f"K values: {args.K_values}")
    print(f"Repeats per K: {args.n_repeats}")
    print(f"Seed: {args.seed}, alpha: {args.alpha}")
    print(f"Workers: {n_workers}")

    if args.all_n:
        n_values = discover_all_n()
        print(f"Discovered {len(n_values)} n values: {n_values[:10]}...")
        for n_val in n_values:
            run_for_n(n_val, args, n_workers)
    elif args.n_list:
        print(f"Running for n values: {args.n_list}")
        for n_val in args.n_list:
            run_for_n(n_val, args, n_workers)
    else:
        # Single n mode (backward compatible: results in output_dir directly)
        config = {
            "n": args.n,
            "N_min": args.N_min,
            "N_max": args.N_max,
            "K_values": args.K_values,
            "n_repeats": args.n_repeats,
            "seed": args.seed,
            "alpha": args.alpha,
            "output_dir": args.output_dir,
        }

        os.makedirs(config["output_dir"], exist_ok=True)
        with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"n={config['n']}, N={config['N_min']}..{config['N_max']}")
        print(f"Output: {config['output_dir']}/")
        print(flush=True)

        N_range = list(range(config["N_min"], config["N_max"] + 1))

        with mp.Pool(n_workers, initializer=init_worker, initargs=(config,)) as pool:
            for result_N in tqdm(pool.imap_unordered(process_one_N, N_range),
                                 total=len(N_range), desc=f"n={config['n']}", leave=True):
                pass

        print(f"\nDone. Results saved to {config['output_dir']}/")


if __name__ == "__main__":
    main()
