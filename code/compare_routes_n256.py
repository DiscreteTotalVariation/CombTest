#!/usr/bin/env python3
"""Paired comparison of the four CT p-value routes at a single (N, n) pair.

All routes are evaluated on the *same* stream of alternative histograms, so the
resulting rejection rates are directly comparable. This reproduces the
beta-versus-gamma statement of Section III-A of the paper: at N=500, n=256 the
Cramer-von Mises-fitted beta places its critical value one DTV step too low and
therefore overstates power, whereas the Cramer-von Mises-fitted gamma
reproduces the exact critical value and the exact rejection rate.

The exact route doubles as a self-check: its rejection rate must equal the
ct_power reported by experiment_adc_dnl_n256.py for the same setting.
"""

import argparse
import sys

import numpy as np

from experiment_adc_dnl_n256 import (
    compute_dtv,
    ct_pvalue_exact,
    ct_pvalue_mc,
    ct_pvalue_prefit_beta,
    ct_pvalue_prefit_gamma,
    generate_adc_histogram,
)

ROUTES = [
    ("exact", "exact CDF", ct_pvalue_exact),
    ("cvm_beta", "CvM-fitted beta", ct_pvalue_prefit_beta),
    ("cvm_gamma", "CvM-fitted gamma", ct_pvalue_prefit_gamma),
    ("mc_beta", "MC-fitted beta", ct_pvalue_mc),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--eps", type=float, default=0.20,
                    help="alternating DNL amplitude")
    ap.add_argument("--trials", type=int, default=50000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    # generate_adc_histogram is the only consumer of the global NumPy RNG, so
    # seeding it here replays the same histograms as experiment_adc_dnl_n256.py.
    np.random.seed(args.seed)
    dtvs = np.array([compute_dtv(generate_adc_histogram(args.N, args.n, args.eps))
                     for _ in range(args.trials)])

    lines = ["N=%d, n=%d, eps=%.2f, trials=%d, alpha=%.2f, seed=%d"
             % (args.N, args.n, args.eps, args.trials, args.alpha, args.seed),
             "%-10s %-18s %8s %14s" % ("route", "fit", "power", "critical DTV")]

    unique = np.unique(dtvs)
    for key, label, pvalue in ROUTES:
        pv = {int(d): pvalue(int(d), args.N, args.n) for d in unique}
        if any(v is None for v in pv.values()):
            lines.append("%-10s %-18s %8s %14s" % (key, label, "n/a", "n/a"))
            continue
        rejected = [d for d in unique if pv[int(d)] < args.alpha]
        power = sum(1 for d in dtvs if pv[int(d)] < args.alpha) / args.trials
        crit = min(rejected) if rejected else None
        lines.append("%-10s %-18s %8.5f %14s"
                     % (key, label, power, crit if crit is not None else "none"))

    text = "\n".join(lines)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
