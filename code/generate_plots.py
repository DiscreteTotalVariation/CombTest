#!/usr/bin/env python3
"""Generate plots for the paper and report from simulation results."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def plot_power_comparison(results_file, output_path, title, method_key="half_to_even"):
    """Plot power comparison between CT and Pearson's chi-squared."""
    with open(results_file) as f:
        data = json.load(f)

    results = data[method_key]
    N_values = sorted([int(k) for k in results.keys()])

    ct_power = [results[str(N)]["ct_power"] for N in N_values]
    chi2_power = [results[str(N)]["chi2_power"] for N in N_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_values, ct_power, 'b-o', markersize=4, linewidth=2, label='Comb Test')
    ax.plot(N_values, chi2_power, 'r-s', markersize=4, linewidth=2, label="Pearson's chi-squared")
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='alpha = 0.05')
    ax.set_xlabel('Sample size N', fontsize=12)
    ax.set_ylabel('Power (rejection rate)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_mean_p_comparison(results_file, output_path, title, method_key="half_to_even"):
    """Plot mean p-value comparison."""
    with open(results_file) as f:
        data = json.load(f)

    results = data[method_key]
    N_values = sorted([int(k) for k in results.keys()])

    ct_p = [results[str(N)]["ct_mean_p"] for N in N_values]
    chi2_p = [results[str(N)]["chi2_mean_p"] for N in N_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_values, ct_p, 'b-o', markersize=4, linewidth=2, label='Comb Test')
    ax.plot(N_values, chi2_p, 'r-s', markersize=4, linewidth=2, label="Pearson's chi-squared")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Expected under H0')
    ax.set_xlabel('Sample size N', fontsize=12)
    ax.set_ylabel('Mean p-value', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_additional_scenarios(results_file, output_path):
    """Plot power comparison across all scenarios."""
    with open(results_file) as f:
        data = json.load(f)

    scenarios = [k for k in data.keys()]
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        if idx >= 8:
            break
        ax = axes[idx]
        results = data[scenario]
        N_values = sorted([int(k) for k in results.keys()])

        ct_power = [results[str(N)]["ct_power"] for N in N_values]
        chi2_power = [results[str(N)]["chi2_power"] for N in N_values]

        ax.plot(N_values, ct_power, 'b-o', markersize=3, linewidth=1.5, label='CT')
        ax.plot(N_values, chi2_power, 'r-s', markersize=3, linewidth=1.5, label='Chi2')
        ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel('N', fontsize=9)
        ax.set_ylabel('Power', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    for idx in range(n_scenarios, 8):
        axes[idx].set_visible(False)

    fig.suptitle('Power Comparison: Comb Test vs Chi-squared across Scenarios',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_type_i_error(results_file, output_path):
    """Plot Type I error rates."""
    with open(results_file) as f:
        data = json.load(f)

    if "type_i_error" not in data:
        print("No Type I error data found")
        return

    results = data["type_i_error"]
    N_values = sorted([int(k) for k in results.keys()])

    ct_err = [results[str(N)]["ct_type_i"] for N in N_values]
    chi2_err = [results[str(N)]["chi2_type_i"] for N in N_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_values, ct_err, 'b-o', markersize=4, linewidth=2, label='Comb Test')
    ax.plot(N_values, chi2_err, 'r-s', markersize=4, linewidth=2, label="Pearson's chi-squared")
    ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=2, label='Nominal alpha = 0.05')
    ax.set_xlabel('Sample size N', fontsize=12)
    ax.set_ylabel('Type I error rate', fontsize=12)
    ax.set_title('Type I Error Rate (False Positive Rate under Uniform Distribution)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_distribution_fit_comparison(output_path):
    """Plot CvM comparison for gamma vs nbinom for various (N,n) pairs."""
    from fit import load_distribution, load_cdf
    from compare import DIST_MAP, load_fitted_params

    pairs = []
    for N in range(10, 201, 5):
        for n in [5, 10, 20]:
            if n <= N:
                pairs.append((N, n))

    data = {n_val: {"N": [], "gamma_cvm": [], "nbinom_cvm": [], "gengamma_cvm": []}
            for n_val in [5, 10, 20]}

    for N, n_val in pairs:
        fname = f"N_{N}_n_{n_val}.txt"
        cdf_path = os.path.join("../data/cdf_exact", fname)
        if not os.path.isfile(cdf_path):
            continue
        cdf_values, cdf_probs = load_cdf(cdf_path)
        cdf_values_f = cdf_values.astype(float)

        row = {"N": N}
        for dist in ["gamma", "nbinom", "gengamma"]:
            best_cvm = float('inf')
            for prefix in ["../data/fitted/mle_", "../data/fitted/cvm_"]:
                path = os.path.join(f"{prefix}{dist}", fname)
                if os.path.isfile(path):
                    try:
                        params, _ = load_fitted_params(path)
                        dist_obj = DIST_MAP[dist](params)
                        theo_cdf = dist_obj.cdf(cdf_values_f)
                        cvm = np.sum((cdf_probs - theo_cdf) ** 2)
                        best_cvm = min(best_cvm, cvm)
                    except:
                        pass
            row[dist] = best_cvm if best_cvm < float('inf') else None

        if all(row[d] is not None for d in ["gamma", "nbinom"]):
            data[n_val]["N"].append(N)
            data[n_val]["gamma_cvm"].append(row["gamma"])
            data[n_val]["nbinom_cvm"].append(row["nbinom"])
            if row["gengamma"] is not None:
                data[n_val]["gengamma_cvm"].append(row["gengamma"])
            else:
                data[n_val]["gengamma_cvm"].append(float('nan'))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, n_val in enumerate([5, 10, 20]):
        ax = axes[idx]
        d = data[n_val]
        if not d["N"]:
            continue
        ax.semilogy(d["N"], d["gamma_cvm"], 'r-o', markersize=3, label='Gamma')
        ax.semilogy(d["N"], d["nbinom_cvm"], 'b-s', markersize=3, label='Neg. Binomial')
        ax.semilogy(d["N"], d["gengamma_cvm"], 'g-^', markersize=3, label='Gen. Gamma')
        ax.set_xlabel('N')
        ax.set_ylabel('CvM statistic (log scale)')
        ax.set_title(f'CvM Fit Quality (n={n_val})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Distribution Fit Quality: CvM Statistic for DTV Distribution', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    os.makedirs("report_plots", exist_ok=True)

    # Generate plots from available results
    if os.path.isfile("rounding_results_d1.json"):
        plot_power_comparison("rounding_results_d1.json",
                              "report_plots/power_rounding_d1.png",
                              "Power: Round-half-to-even detection (d=1)")
        plot_mean_p_comparison("rounding_results_d1.json",
                               "report_plots/mean_p_rounding_d1.png",
                               "Mean p-value: Round-half-to-even (d=1)")
        plot_type_i_error("rounding_results_d1.json",
                          "report_plots/type_i_error.png")

    if os.path.isfile("rounding_results_d2.json"):
        plot_power_comparison("rounding_results_d2.json",
                              "report_plots/power_rounding_d2.png",
                              "Power: Round-half-to-even detection (d=2)")

    if os.path.isfile("rounding_results_large.json"):
        plot_power_comparison("rounding_results_large.json",
                              "report_plots/power_rounding_large.png",
                              "Power: Round-half-to-even detection (large N, d=1)")
        plot_type_i_error("rounding_results_large.json",
                          "report_plots/type_i_error_large.png")

    if os.path.isfile("../data/additional_test_results.json"):
        plot_additional_scenarios("../data/additional_test_results.json",
                                 "report_plots/additional_scenarios.png")

    # Distribution fit comparison
    plot_distribution_fit_comparison("report_plots/distribution_fit_comparison.png")
