#!/usr/bin/env python3
"""Plot ADC alternating DNL power comparison figure.

Reads results JSON from experiment_adc_dnl_data.py and generates a publication-quality figure.
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def auto_crop(path):
    import cv2
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(255 - gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]
        cv2.imwrite(path, img)


def plot_adc_dnl(input_path, output_path,
                 xlim_lower=None, xlim_upper=None,
                 ylim_lower=None, ylim_upper=None,
                 crop=False):
    with open(input_path) as f:
        results = json.load(f)

    tick_fontsize = 22
    label_fontsize = 36

    matplotlib.rc("font", size=tick_fontsize)
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 8),
                             sharey=True)
    if len(results) == 1:
        axes = [axes]

    colors = {"ct": "#1f77b4", "chi2": "#ff7f0e", "gtest": "#2ca02c"}
    labels = {"ct": "CT (exact)", "chi2": r"Pearson's $\chi^2$", "gtest": "G-test"}
    markers = {"ct": "o", "chi2": "s", "gtest": "^"}

    # Determine which tests are available in the data
    sample_data = next(iter(results.values()))
    tests = [t for t in ["ct", "chi2", "gtest"] if t in sample_data]

    for ax, (dnl_str, data) in zip(axes, sorted(results.items(), key=lambda x: float(x[0]))):
        N_arr = np.array(data["N"])
        for test in tests:
            power = np.array(data[test], dtype=object)
            valid = np.array([p is not None for p in power])
            power_valid = np.array([p for p in power if p is not None], dtype=float)
            ax.plot(N_arr[valid], power_valid,
                    color=colors[test], marker=markers[test],
                    markersize=3, linewidth=1.5, label=labels[test])

        ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=0.8,
                   alpha=0.5, label=r"$\alpha=0.05$")
        ax.set_xlabel("$N$", fontsize=label_fontsize)
        ax.set_title(r"$\varepsilon$ = %.2f" % float(dnl_str), fontsize=label_fontsize)
        ax.grid(True, alpha=0.3)
        lo = xlim_lower if xlim_lower is not None else N_arr[0] - 5
        hi = xlim_upper if xlim_upper is not None else N_arr[-1] + 5
        ax.set_xlim(lo, hi)
        if ylim_lower is not None or ylim_upper is not None:
            cur_lo, cur_hi = ax.get_ylim()
            ax.set_ylim(ylim_lower if ylim_lower is not None else cur_lo,
                        ylim_upper if ylim_upper is not None else cur_hi)

    axes[0].set_ylabel("Statistical power", fontsize=label_fontsize)
    axes[0].legend(fontsize=tick_fontsize, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if crop:
        auto_crop(output_path)
    print("Saved: %s" % output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ADC alternating DNL power comparison.")
    parser.add_argument("-i", "--input", type=str, default="../data/adc_dnl_results.json",
                        help="Input JSON file path (default: adc_dnl_results.json)")
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
    parser.add_argument("-o", "--output", type=str, default="../paper/img/adc_dnl_power_comparison.png",
                        help="Output image path (default: paper/img/adc_dnl_power_comparison.png)")
    args = parser.parse_args()

    plot_adc_dnl(args.input, args.output,
                 xlim_lower=args.xlim_lower, xlim_upper=args.xlim_upper,
                 ylim_lower=args.ylim_lower, ylim_upper=args.ylim_upper,
                 crop=args.crop)
