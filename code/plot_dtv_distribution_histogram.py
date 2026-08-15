#!/usr/bin/env python3
"""Plot the exact DTV distribution histogram for given N and n."""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


EXACT_DISTRIBUTIONS_DIR = "../data/exact_distributions"


def auto_crop(path):
    import cv2
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(255 - gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]
        cv2.imwrite(path, img)


def load_exact_dtv_distribution(N, n, input_dir=EXACT_DISTRIBUTIONS_DIR):
    input_path = os.path.join(input_dir, "N_%d_n_%d.txt" % (N, n))
    if not os.path.isfile(input_path):
        raise FileNotFoundError("Distribution file not found: %s" % input_path)
    distribution = []
    with open(input_path) as f:
        for line in f:
            a, b = map(int, line.strip().split())
            distribution.append((a, b))
    return distribution


def plot_dtv_distribution_histogram(N, n, save_path, show=False,
                                    xlim_lower=None, xlim_upper=None,
                                    ylim_lower=None, ylim_upper=None,
                                    crop=False, normalize=False, fig_height=8,
                                    fig_width=13):
    distribution = load_exact_dtv_distribution(N=N, n=n)

    bins = [a for a, b in distribution]
    quantities = [b for a, b in distribution]

    if normalize:
        total = sum(quantities)
        heights = [q / total for q in quantities]
        ylabel = "PMF"
    else:
        heights = quantities
        ylabel = "Appearance"

    tick_fontsize = 22
    label_fontsize = 36

    matplotlib.rc("font", size=tick_fontsize)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(bins, height=heights, width=1.0, align="center", edgecolor="black")
    plt.ticklabel_format(style="plain")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    plt.xlabel("DTV", fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)

    if xlim_lower is not None or xlim_upper is not None:
        cur_lo, cur_hi = ax.get_xlim()
        ax.set_xlim(xlim_lower if xlim_lower is not None else cur_lo,
                    xlim_upper if xlim_upper is not None else cur_hi)
    if ylim_lower is not None or ylim_upper is not None:
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(ylim_lower if ylim_lower is not None else cur_lo,
                    ylim_upper if ylim_upper is not None else cur_hi)

    plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300)

    if show:
        plt.show()

    plt.clf()
    if crop:
        auto_crop(save_path)
    print("Saved: %s" % save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot exact DTV distribution histogram.")
    parser.add_argument("-N", type=int, default=20, help="Number of items (default: 20)")
    parser.add_argument("-n", type=int, default=4, help="Number of bins (default: 4)")
    parser.add_argument("--xlim-lower", type=float, default=None,
                        help="Lower x-axis limit (default: auto)")
    parser.add_argument("--xlim-upper", type=float, default=None,
                        help="Upper x-axis limit (default: auto)")
    parser.add_argument("--ylim-lower", type=float, default=None,
                        help="Lower y-axis limit (default: auto)")
    parser.add_argument("--ylim-upper", type=float, default=None,
                        help="Upper y-axis limit (default: auto)")
    parser.add_argument("--crop", action="store_true",
                        help="Auto-crop white margins from the output image")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize bar heights to sum to 1 (PMF) instead of raw counts")
    parser.add_argument("--fig-height", type=float, default=8,
                        help="Canvas height in inches; at a fixed width this changes "
                             "the printed height only, not the printed font size "
                             "(default: 8)")
    parser.add_argument("--fig-width", type=float, default=13,
                        help="Canvas width in inches. Printed font size scales with "
                             "printed_width/fig_width, so raise this in step with the "
                             "printed width to keep labels the same size (default: 13)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (default: dtv_distribution_histogram_N_{N}_n_{n}.png)")
    args = parser.parse_args()

    if args.output is None:
        save_path = "dtv_distribution_histogram_N_%d_n_%d.png" % (args.N, args.n)
    else:
        save_path = args.output

    plot_dtv_distribution_histogram(N=args.N, n=args.n, save_path=save_path,
                                    xlim_lower=args.xlim_lower,
                                    xlim_upper=args.xlim_upper,
                                    ylim_lower=args.ylim_lower,
                                    ylim_upper=args.ylim_upper,
                                    crop=args.crop,
                                    normalize=args.normalize,
                                    fig_height=args.fig_height,
                                    fig_width=args.fig_width)
