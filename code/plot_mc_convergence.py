#!/usr/bin/env python3
"""Plot MC convergence results from experiment_mc_convergence_n10.py.

Reads per-N JSON files from the results directory and generates a figure
showing mean p-value error as a function of K for different N values.
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot MC convergence results")
    parser.add_argument("--input-dir", type=str, default="../data/mc_convergence_results",
                        help="Input directory with per-N JSON files (default: mc_convergence_results)")
    parser.add_argument("--output", type=str, default="../paper/img/mc_convergence.png",
                        help="Output figure path (default: paper/img/mc_convergence.png)")
    parser.add_argument("--fig-width", type=float, default=14,
                        help="Canvas width in inches (default: 14)")
    parser.add_argument("--fig-height", type=float, default=6,
                        help="Canvas height in inches (default: 6)")
    parser.add_argument("--stack", action="store_true",
                        help="Stack the two panels vertically instead of side by side, "
                             "for placement in a single column")
    parser.add_argument("--font-scale", type=float, default=1.0,
                        help="Multiplier for tick, label and title sizes; raise it when "
                             "the figure is printed narrower (default: 1.0)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_path = args.output

    # Load config
    config_path = os.path.join(input_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    K_values = config["K_values"]
    n = config["n"]

    # Load all results
    results = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.startswith("N_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(input_dir, fname)) as f:
            results.append(json.load(f))

    results.sort(key=lambda r: r["N"])
    print(f"Loaded {len(results)} results (n={n})")

    if not results:
        print("No results found.")
        return

    # Compute aggregate stats: for each K, mean and std of mean-error across all N
    K_strs = [str(K) for K in K_values]

    # Also compute per-N mean error for each K, to plot as heatmap or line plot
    N_arr = np.array([r["N"] for r in results])
    exact_errors = np.array([r["exact_error"] for r in results])

    # For each K, compute mean error across repeats for each N
    mean_errors = {}
    std_errors = {}
    for K_str in K_strs:
        means = []
        stds = []
        for r in results:
            if K_str in r["K_values"]:
                means.append(r["K_values"][K_str]["mean"])
                stds.append(r["K_values"][K_str]["std"])
            else:
                means.append(np.nan)
                stds.append(np.nan)
        mean_errors[K_str] = np.array(means)
        std_errors[K_str] = np.array(stds)

    # --- Figure: mean p-value error vs K, aggregated across all N ---
    tick_fontsize = 14 * args.font_scale
    label_fontsize = 18 * args.font_scale
    title_fontsize = label_fontsize
    matplotlib.rc("font", size=tick_fontsize)

    rows, cols = (2, 1) if args.stack else (1, 2)
    fig, axes = plt.subplots(rows, cols, figsize=(args.fig_width, args.fig_height))

    # Panel (a): Mean error vs K (aggregate across all N)
    ax = axes[0]
    agg_means = []
    agg_stds = []
    agg_medians = []
    agg_p95 = []
    for K_str in K_strs:
        valid = ~np.isnan(mean_errors[K_str])
        errs = mean_errors[K_str][valid]
        agg_means.append(np.mean(errs))
        agg_stds.append(np.std(errs))
        agg_medians.append(np.median(errs))
        agg_p95.append(np.percentile(errs, 95))

    agg_means = np.array(agg_means)
    agg_medians = np.array(agg_medians)
    agg_p95 = np.array(agg_p95)

    # Exact-beta baseline (aggregate)
    valid_exact = exact_errors > 0
    exact_mean = np.mean(exact_errors[valid_exact]) if np.any(valid_exact) else 0
    exact_median = np.median(exact_errors[valid_exact]) if np.any(valid_exact) else 0

    ax.plot(K_values, agg_means, "o-", color="#1f77b4", linewidth=2,
            markersize=5, label="Mean error (MC)")
    ax.plot(K_values, agg_medians, "s-", color="#2ca02c", linewidth=2,
            markersize=5, label="Median error (MC)")
    ax.plot(K_values, agg_p95, "^-", color="#ff7f0e", linewidth=2,
            markersize=5, label="95th percentile (MC)")
    ax.axhline(y=exact_mean, color="#1f77b4", linestyle="--", linewidth=1,
               alpha=0.7, label="Mean (exact fit)")
    ax.axhline(y=exact_median, color="#2ca02c", linestyle="--", linewidth=1,
               alpha=0.7, label="Median (exact fit)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$K$ (MC samples)", fontsize=label_fontsize)
    ax.set_ylabel("$p$-value error at $\\alpha=0.05$", fontsize=label_fontsize)
    ax.set_title(f"Aggregate over all $N$ ($n={n}$)", fontsize=title_fontsize)
    ax.legend(fontsize=tick_fontsize - 2, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # Panel (b): Mean error vs N for selected K values
    ax = axes[1]
    K_show = [500, 2000, 10000, 50000, 100000]
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]
    for K_val, color in zip(K_show, colors):
        K_str = str(K_val)
        if K_str not in mean_errors:
            continue
        valid = ~np.isnan(mean_errors[K_str])
        ax.plot(N_arr[valid], mean_errors[K_str][valid],
                linewidth=1.2, alpha=0.8, color=color,
                label=f"$K={K_val:,}$")

    # Exact beta baseline
    ax.plot(N_arr, exact_errors, "k--", linewidth=1.5, alpha=0.6,
            label="Exact fit")

    ax.set_xlabel("$N$", fontsize=label_fontsize)
    ax.set_ylabel("Mean $p$-value error", fontsize=label_fontsize)
    ax.set_title(f"Error vs $N$ for selected $K$ ($n={n}$)", fontsize=title_fontsize)
    ax.legend(fontsize=tick_fontsize - 2, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Print summary table
    print(f"\n{'K':>8s} {'Mean':>10s} {'Median':>10s} {'P95':>10s}")
    print("-" * 42)
    for i, K in enumerate(K_values):
        print(f"{K:>8d} {agg_means[i]:>10.6f} {agg_medians[i]:>10.6f} {agg_p95[i]:>10.6f}")
    print(f"{'exact':>8s} {exact_mean:>10.6f} {exact_median:>10.6f}")


if __name__ == "__main__":
    main()
