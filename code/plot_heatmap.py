#!/usr/bin/env python3
"""Plot the KS statistic heatmap from precomputed data.

Reads the .npy file produced by prepare_heatmap_data.py.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def interpolate_missing(grid):
    """Fill NaN cells by averaging their valid neighbors."""
    rows, cols = grid.shape
    filled = grid.copy()
    nan_mask = np.isnan(grid)
    for i in range(rows):
        for j in range(cols):
            if not nan_mask[i, j]:
                continue
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(grid[ni, nj]):
                    neighbors.append(grid[ni, nj])
            if neighbors:
                filled[i, j] = np.mean(neighbors)
    return filled


def plot_heatmap(input_path, output_path, interpolate=False):
    grid = np.load(input_path)
    if interpolate:
        grid = interpolate_missing(grid)
    N_max = grid.shape[0] + 1  # grid is (N_max-1) x (N_max-1), indexed from 2

    tick_fontsize = 22
    label_fontsize = 36

    matplotlib.rc("font", size=tick_fontsize)
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   extent=[2, N_max + 1, 2, N_max + 1],
                   vmin=0, vmax=0.05, cmap="jet")
    ax.set_xlabel("$N$", fontsize=label_fontsize)
    ax.set_ylabel("$n$", fontsize=label_fontsize)
    cbar = fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot KS statistic heatmap from precomputed data.")
    parser.add_argument("-i", "--input", type=str, default="../data/heatmap_ks_beta_450.npy",
                        help="Input .npy file path (default: heatmap_ks_beta_450.npy)")
    parser.add_argument("-o", "--output", type=str, default="../paper/img/dtv_to_beta.png",
                        help="Output image path (default: ../paper/img/dtv_to_beta.png)")
    parser.add_argument("--interpolate", action="store_true",
                        help="Interpolate missing cells from neighbors")
    args = parser.parse_args()

    plot_heatmap(args.input, args.output, interpolate=args.interpolate)
