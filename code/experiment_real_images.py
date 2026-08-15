#!/usr/bin/env python3
"""Comb test on real image data at scales where exact computation is infeasible.

Well-posedness note.  Uniformity is *not* a plausible null for the raw
intensity histogram of a natural image, so testing that histogram would say
nothing about the test.  What is well posed, and is exactly the situation of
IEEE Std 1241, is the *output code* histogram of a quantiser driven by a real
signal: an ideal quantiser matched to the signal produces uniformly occupied
codes, and a hardware imperfection perturbs that uniformity.

Accordingly, for each real image we
  1. map its intensities to [0,1) by the rank (probability-integral) transform,
     which is what a companding / histogram-equalising front end does; the
     resulting ideal 8-bit code distribution is uniform up to the unavoidable
     one pixel per code when n does not divide the pixel count (at most 0.0018
     in l1 over these nine images, against a smallest tested eps of 0.05),
  2. re-quantise with alternating bin widths W_k = W_ideal (1 + (-1)^k eps),
     the LSB capacitor-mismatch model of Section IV-B2, and
  3. test the n = 256 code histogram of random pixel samples of size N.

Everything about the signal (its values, ties and empirical distribution) comes
from the real image; only the quantiser imperfection is controlled, which is
unavoidable because measuring power requires ground truth.  N is far beyond the
reach of the exact dynamic program, so the Monte Carlo gamma route of Section III
is used.  Note that here the gamma is fitted by matching moments (see
gamma_params below), not by minimizing the Cramer-von Mises statistic as in
Section III-B; the manuscript states this explicitly.  Matching moments is the
better choice for a tail-based test, not merely the cheaper one: at n = 256,
N = 500, where the exact critical DTV 427 is known, the moment-matched gamma
returns 427 at every one of forty seeds, whereas the Cramer-von Mises fit of the
same family returns 426 at twelve of them -- the same one-unit anticonservatism
for which Section III-A prefers gamma to beta.  W^2 weights the whole support; the test uses only the far
right tail.
"""

import argparse
import json

import numpy as np
from scipy import stats

import skimage.color as skcolor
import skimage.data as skdata

SEED = 13
IMAGES = ["camera", "coins", "moon", "page", "text", "brick",
          "gravel", "clock", "cell"]


def load_gray_uint8(name):
    im = getattr(skdata, name)()
    im = np.asarray(im)
    if im.ndim == 3:
        im = skcolor.rgb2gray(im) * 255.0
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)
    return im


def rank_uniform(image, rng):
    """Rank transform of the real intensities to (0,1), ties broken at random.

    The output is an exactly uniform empirical distribution whose ordering is
    that of the real image.
    """
    v = image.ravel().astype(np.float64)
    jitter = rng.random(v.size)
    order = np.lexsort((jitter, v))
    u = np.empty(v.size)
    u[order] = (np.arange(v.size) + 0.5) / v.size
    return u


def quantise_alternating(u, n, eps):
    """Quantise u in [0,1) with bin widths proportional to 1 + (-1)^k eps."""
    w = 1.0 + eps * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    edges = np.concatenate(([0.0], np.cumsum(w / w.sum())))
    edges[-1] = 1.0
    return np.clip(np.searchsorted(edges, u, side="right") - 1, 0, n - 1)


def dtv_rows(h):
    return np.abs(np.diff(h, axis=-1)).sum(axis=-1)


def gamma_params(N, n, mc, rng):
    d = dtv_rows(rng.multinomial(N, np.full(n, 1.0 / n), size=mc).astype(np.int64))
    m, v = d.mean(), d.var()
    scale = v / m
    return m / scale, 0.0, scale


def chi2_p_rows(h):
    N = h.sum(axis=-1, keepdims=True)
    n = h.shape[-1]
    e = N / n
    return 1.0 - stats.chi2.cdf(((h - e) ** 2 / e).sum(axis=-1), df=n - 1)


def g_p_rows(h):
    N = h.sum(axis=-1, keepdims=True)
    n = h.shape[-1]
    e = N / n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(h > 0, h * np.log(np.where(h > 0, h, 1) / e), 0.0)
    return 1.0 - stats.chi2.cdf(2.0 * t.sum(axis=-1), df=n - 1)


def run(N, n, eps, trials, alpha, mc, seed):
    """Pool trials over all real images; return rejection rates."""
    rng = np.random.default_rng(seed)
    shape, loc, scale = gamma_params(N, n, mc, rng)
    ct_crit = int(np.ceil(stats.gamma.ppf(1.0 - alpha, a=shape, loc=loc,
                                          scale=scale) + 0.5))

    rej = dict(CT=0, X2=0, G=0)
    total = 0
    per_trial = max(1, trials // len(IMAGES))
    for name in IMAGES:
        img = load_gray_uint8(name)
        u = rank_uniform(img, rng)
        codes = quantise_alternating(u, n, eps)
        hs = np.empty((per_trial, n), dtype=np.int64)
        for t in range(per_trial):
            idx = rng.integers(0, codes.size, size=N)
            hs[t] = np.bincount(codes[idx], minlength=n)
        rej["CT"] += int((dtv_rows(hs) >= ct_crit).sum())
        rej["X2"] += int((chi2_p_rows(hs) <= alpha).sum())
        rej["G"] += int((g_p_rows(hs) <= alpha).sum())
        total += per_trial
    return {k: rej[k] / total for k in rej}, ct_crit, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--N", type=int, nargs="+",
                    default=[500, 2000, 10000, 50000])
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--trials", type=int, default=9000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--mc", type=int, default=50000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    results = {}
    print(f"real images: {', '.join(IMAGES)}")
    print(f"n={args.n}, alpha={args.alpha}, trials per setting={args.trials}, "
          f"MC samples for the gamma fit={args.mc}\n")
    print(f"{'N':>7} {'eps':>6} {'crit':>6} {'CT':>8} {'X2':>8} {'G':>8}")
    print("-" * 48)
    for N in args.N:
        for eps in args.eps:
            r, crit, total = run(N, args.n, eps, args.trials, args.alpha,
                                 args.mc, SEED + N + int(1000 * eps))
            results[f"N{N}_eps{eps}"] = dict(N=N, eps=eps, crit=crit,
                                             trials=total, **r)
            tag = "  (size)" if eps == 0.0 else ""
            print(f"{N:>7} {eps:>6.2f} {crit:>6d} {r['CT']:>8.3f} "
                  f"{r['X2']:>8.3f} {r['G']:>8.3f}{tag}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
