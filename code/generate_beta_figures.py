#!/usr/bin/env python3
"""Generate figures for the paper showing beta distribution approximation of DTV.

Produces:
  1. dtv_distribution_histogram_and_beta_N_50_n_10.png - PDF overlay
  2. dtv_to_beta.png - Heatmap of KS statistic across (N,n) grid
  3. dtv_to_gamma_beta_comparison.png - Side-by-side gamma vs beta heatmap
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from tqdm import tqdm


def auto_crop(path):
    import cv2
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(255 - gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]
        cv2.imwrite(path, img)


def load_distribution(path):
    values, counts = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                values.append(int(parts[0]))
                counts.append(int(parts[1]))
    return np.array(values), np.array(counts, dtype=float)


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
        return tuple(map(float, f.readline().strip().split()))


def generate_overlay_plot(xlim_lower=None, xlim_upper=None,
                          ylim_lower=None, ylim_upper=None, crop=False,
                          output_name=None, fig_height=8, fig_width=13):
    """Generate DTV histogram with fitted beta overlay for N=50, n=10."""
    N, n = 50, 10
    exact_path = "../data/exact_distributions/N_%d_n_%d.txt" % (N, n)
    beta_path = "../data/fitted/cvm_beta/N_%d_n_%d.txt" % (N, n)
    gamma_path = "../data/fitted/cvm_gamma/N_%d_n_%d.txt" % (N, n)

    values, counts = load_distribution(exact_path)
    total = counts.sum()
    pmf = counts / total

    beta_params = load_fitted_params(beta_path)
    gamma_params = load_fitted_params(gamma_path)

    beta_dist = stats.beta(a=beta_params[0], b=beta_params[1],
                           loc=beta_params[2], scale=beta_params[3])
    gamma_dist = stats.gamma(a=gamma_params[0], loc=gamma_params[1],
                             scale=gamma_params[2])

    tick_fontsize = 22
    label_fontsize = 36

    matplotlib.rc("font", size=tick_fontsize)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.bar(values, pmf, width=0.8, edgecolor="black", label="Exact DTV distribution")

    x_smooth = np.linspace(max(values.min() - 1, 0), values.max() + 1, 500)
    ax.plot(x_smooth, beta_dist.pdf(x_smooth), "g-", linewidth=2.5,
            label="Fitted beta distribution")
    ax.plot(x_smooth, gamma_dist.pdf(x_smooth), "r-", linewidth=2, alpha=0.7,
            label="Fitted gamma distribution")

    ax.set_xlabel("DTV", fontsize=label_fontsize)
    ax.set_ylabel("PMF", fontsize=label_fontsize)
    ax.legend(fontsize=tick_fontsize)
    if xlim_lower is not None or xlim_upper is not None:
        ax.set_xlim(xlim_lower if xlim_lower is not None else 0,
                    xlim_upper if xlim_upper is not None else 2 * N)
    else:
        ax.set_xlim(0, 2 * N)
    if ylim_lower is not None or ylim_upper is not None:
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(ylim_lower if ylim_lower is not None else cur_lo,
                    ylim_upper if ylim_upper is not None else cur_hi)
    fig.tight_layout()

    out = output_name or "../paper/img/dtv_distribution_histogram_and_beta_N_50_n_10.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if crop:
        auto_crop(out)
    print("Saved: %s" % out)


def compute_ks_for_pair(N, n, dist_name):
    """Compute KS statistic (max |CDF_exact - CDF_approx|) for one (N,n) pair."""
    cdf_path = "../data/cdf_exact/N_%d_n_%d.txt" % (N, n)
    fitted_path = "../data/fitted/cvm_%s/N_%d_n_%d.txt" % (dist_name, N, n)

    if not os.path.isfile(cdf_path) or not os.path.isfile(fitted_path):
        return None

    try:
        cdf_values, cdf_probs = load_cdf(cdf_path)
        params = load_fitted_params(fitted_path)

        if dist_name == "beta":
            dist_obj = stats.beta(a=params[0], b=params[1], loc=params[2], scale=params[3])
        elif dist_name == "gamma":
            dist_obj = stats.gamma(a=params[0], loc=params[1], scale=params[2])
        else:
            return None

        theo_cdf = dist_obj.cdf(cdf_values.astype(float))
        ks = np.max(np.abs(cdf_probs - theo_cdf))
        return ks
    except Exception:
        return None


def generate_diagonal_ks_plot(dist_name="beta", N_max=500,
                              output_name=None, crop=False,
                              compare_dist=None):
    """Plot KS statistic along the diagonal N=n, showing decay with growing N.

    If compare_dist is given (e.g. "gamma"), a second series is overlaid.
    """
    if output_name is None:
        output_name = "../paper/img/ks_diagonal_%s.png" % dist_name

    Ns = list(range(2, N_max + 1))
    ks_vals = []
    valid_Ns = []
    for N in tqdm(Ns, desc="diagonal KS (%s)" % dist_name):
        ks = compute_ks_for_pair(N, N, dist_name)
        if ks is not None:
            valid_Ns.append(N)
            ks_vals.append(ks)

    cmp_vals = []
    cmp_Ns = []
    if compare_dist:
        for N in tqdm(Ns, desc="diagonal KS (%s)" % compare_dist):
            ks = compute_ks_for_pair(N, N, compare_dist)
            if ks is not None:
                cmp_Ns.append(N)
                cmp_vals.append(ks)

    tick_fontsize = 22
    label_fontsize = 28

    matplotlib.rc("font", size=tick_fontsize)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(valid_Ns, ks_vals, ".", markersize=3, color="navy", alpha=0.7,
            label="Beta" if compare_dist else None)
    if compare_dist and cmp_vals:
        ax.plot(cmp_Ns, cmp_vals, ".", markersize=3, color="crimson", alpha=0.7,
                label="Gamma")
        ax.legend(fontsize=tick_fontsize)
    ax.set_xlabel("$N = n$", fontsize=label_fontsize)
    ax.set_ylabel("KS statistic", fontsize=label_fontsize)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if crop:
        auto_crop(output_name)
    print("Saved: %s" % output_name)


def generate_heatmap(dist_name, N_max=300, N_min=2, n_min=2,
                     vmax=0.05, output_name=None, crop=False):
    """Generate KS-statistic heatmap for a distribution across (N,n) grid."""
    if output_name is None:
        output_name = "../paper/img/dtv_to_%s.png" % dist_name

    N_range = range(N_min, N_max + 1)
    n_range = range(n_min, N_max + 1)
    grid = np.full((len(n_range), len(N_range)), np.nan)

    for i, n in enumerate(tqdm(n_range, desc="%s heatmap" % dist_name)):
        for j, N in enumerate(N_range):
            ks = compute_ks_for_pair(N, n, dist_name)
            if ks is not None:
                grid[i, j] = ks

    # Replace NaN with 0 to avoid white artifacts from missing data
    grid = np.nan_to_num(grid, nan=0.0)

    tick_fontsize = 22
    label_fontsize = 36

    matplotlib.rc("font", size=tick_fontsize)
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   extent=[N_min, N_max, n_min, N_max],
                   vmin=0, vmax=vmax, cmap="jet")
    ax.set_xlabel("$N$", fontsize=label_fontsize)
    ax.set_ylabel("$n$", fontsize=label_fontsize)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("KS statistic", fontsize=label_fontsize)
    fig.tight_layout()
    fig.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if crop:
        auto_crop(output_name)
    print("Saved: %s" % output_name)


def generate_comparison_heatmap(N_max=300, crop=False):
    """Generate side-by-side gamma vs beta KS heatmap."""
    N_range = range(2, N_max + 1)
    n_range = range(2, N_max + 1)
    grid_gamma = np.full((N_max - 1, N_max - 1), np.nan)
    grid_beta = np.full((N_max - 1, N_max - 1), np.nan)

    for i, n in enumerate(tqdm(n_range, desc="comparison heatmap")):
        for j, N in enumerate(N_range):
            ks_g = compute_ks_for_pair(N, n, "gamma")
            ks_b = compute_ks_for_pair(N, n, "beta")
            if ks_g is not None:
                grid_gamma[i, j] = ks_g
            if ks_b is not None:
                grid_beta[i, j] = ks_b

    tick_fontsize = 22
    label_fontsize = 36

    matplotlib.rc("font", size=tick_fontsize)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    im1 = ax1.imshow(grid_gamma, origin="lower", aspect="auto",
                     extent=[2, N_max, 2, N_max], vmin=0, vmax=0.05, cmap="jet")
    ax1.set_xlabel("$N$", fontsize=label_fontsize)
    ax1.set_ylabel("$n$", fontsize=label_fontsize)
    ax1.set_title("Gamma approximation", fontsize=label_fontsize)
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(grid_beta, origin="lower", aspect="auto",
                     extent=[2, N_max, 2, N_max], vmin=0, vmax=0.05, cmap="jet")
    ax2.set_xlabel("$N$", fontsize=label_fontsize)
    ax2.set_ylabel("$n$", fontsize=label_fontsize)
    ax2.set_title("Beta approximation", fontsize=label_fontsize)
    fig.colorbar(im2, ax=ax2)

    fig.suptitle("Max |CDF error| (KS statistic) for DTV distribution approximation",
                 fontsize=label_fontsize, y=1.02)
    fig.tight_layout()
    out = "../paper/img/dtv_to_gamma_beta_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if crop:
        auto_crop(out)
    print("Saved: %s" % out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate beta figures.")
    parser.add_argument("--crop", action="store_true",
                        help="Auto-crop white margins from output images")
    args = parser.parse_args()

    print("=== Generating overlay plot ===")
    generate_overlay_plot(crop=args.crop)

    print("\n=== Generating beta KS heatmap ===")
    generate_heatmap("beta", N_max=500,
                     output_name="../paper/img/dtv_to_beta.png",
                     crop=args.crop)

    print("\n=== Generating gamma vs beta comparison heatmap ===")
    generate_comparison_heatmap(N_max=300, crop=args.crop)

    print("\nDone.")
