#!/usr/bin/env python3
"""Power of CT and competitors across alternative shapes at a fixed L1 distance.

Answers reviewer requests for (a) an explicit characterisation of the
directions of departure from uniformity in which DTV is most/least sensitive
and (b) order-aware baselines beyond chi-square and the G-test.

All alternatives are normalised to the same L1 distance from uniformity,
||p - u||_1 = sum_i |p_i - 1/n|, so that only the *shape* of the departure
differs between rows.

Tests compared:
  CT   - comb test (exact null CDF when available, otherwise MC gamma)
  X2   - Pearson chi-square (asymptotic)
  G    - likelihood-ratio G-test (asymptotic)
  AC   - alternating contrast S = sum_i (-1)^i x_i, two-sided.  This is the
         locally most powerful (Neyman smooth) test for the *known* period-2
         comb alternative, i.e. an oracle competitor for the first row.
  vN   - discrete von Neumann ratio sum (x_i-x_{i-1})^2 / sum (x_i-xbar)^2,
         the L2 analogue of DTV.
  RUN  - Wald-Wolfowitz runs test applied to the sign of x_i - N/n.
  CO   - Paninski's coincidence count N - #{occupied bins}, the statistic that
         is minimax-optimal for uniformity testing in the sparse regime; it is
         the modern distribution-testing baseline and is order-agnostic.

Three of the four order-aware baselines have an exact null under the multinomial
and are used that way by default: AC is a linear contrast and collapses to a
binomial, CO depends only on the occupancy pattern, and RUN admits an exact
dynamic program because its comparison threshold is the fixed mean N/n.  Only vN,
a ratio of two quadratic forms in the counts, has no exact null within reach at
these sizes, so it is calibrated by Monte Carlo under H0 (same N, n), which makes
its level valid rather than asymptotic.  The Monte Carlo route is retained for
all four (--baseline-crits mc) and both sets of critical values are always
printed, so the two can be compared; they agree at the (N, n) pairs in the paper.

Because the statistics are discrete, either route takes the smallest critical
value whose attained size does not exceed alpha, so the baselines end up slightly
conservative rather than exactly sized -- this is a property of the statistics,
not of the calibration, and it survives the exact computation.  Row 1 of the
output reports every attained size.

Reading the 1-alpha point off the null instead would not be the neutral choice
it looks like.  On a coarse statistic that point falls inside an atom, the test
rejects on the whole atom, and the attained size lands *above* alpha, so the
quantile rule is anticonservative rather than merely less conservative.  The run
header printed by calibration_report() quantifies this on the exact null: at
N=200, n=10 the run count has ten attainable values, and the quantile rule
attains 0.161 against a nominal 0.05 where the rule used here attains 0.034.
"""

import argparse
import json
from fractions import Fraction
from math import comb, factorial

import numpy as np
from scipy import stats

from comb_test import load_exact_cdf, estimate_gamma_params

SEED = 13


def dtv(h):
    return int(np.abs(np.diff(np.asarray(h, dtype=np.int64))).sum())


def make_ct_critical(N, n, alpha, mc_samples, seed):
    """Return (critical DTV value, source) for a level-alpha one-sided CT.

    The exact null CDF is used when it has been precomputed, otherwise the
    gamma approximation with the continuity correction of Section III is used.
    Working with a critical value instead of a p-value per trial keeps the
    50,000-trial power runs fast.
    """
    res = load_exact_cdf(N, n)
    if res is not None:
        vals, cdf = np.asarray(res[0]), np.asarray(res[1])
        # smallest d with P(DTV >= d) <= alpha, i.e. 1 - F(d-1) <= alpha
        for i, v in enumerate(vals):
            tail = 1.0 - (cdf[i - 1] if i > 0 else 0.0)
            if tail <= alpha:
                return int(v), "exact"
        return int(vals[-1]) + 1, "exact"
    shape, loc, scale = estimate_gamma_params(N, n, num_samples=mc_samples,
                                              seed=seed)
    q = stats.gamma.ppf(1.0 - alpha, a=shape, loc=loc, scale=scale)
    return int(np.ceil(q + 0.5)), "gamma"


# --------------------------------------------------------------------------
# alternatives, all parameterised by the L1 distance eps1 = ||p - u||_1
# --------------------------------------------------------------------------

def shape_to_p(shape, eps1):
    """Turn an arbitrary shape vector into probabilities with

        sum_i p_i = 1   and   sum_i |p_i - 1/n| = eps1

    by centring the shape and rescaling it.  Only the shape, never the L1
    distance, differs between the alternatives built this way.
    """
    n = len(shape)
    s = np.asarray(shape, dtype=float)
    s = s - s.mean()
    scale = eps1 / (np.abs(s).sum() / n)
    p = (1.0 + scale * s) / n
    if p.min() <= 0:
        raise ValueError("eps1 too large for this shape")
    return p


def p_comb(n, eps1, period=2):
    """Block-alternating probabilities with the given period."""
    idx = np.arange(n)
    return shape_to_p(np.where((idx % period) < period / 2, 1.0, -1.0), eps1)


def p_sinusoid(n, eps1, freq):
    idx = np.arange(n)
    return shape_to_p(np.cos(2.0 * np.pi * freq * idx / n), eps1)


def p_single_jump(n, eps1):
    """Lower on the first half, higher on the second half (one jump only)."""
    s = np.ones(n)
    s[: n // 2] = -1.0
    return shape_to_p(s, eps1)


def p_ramp(n, eps1):
    """Monotone linear trend (smooth, order-1 departure)."""
    return shape_to_p(np.arange(n, dtype=float), eps1)


def p_spike(n, eps1):
    """One bin inflated, the rest deflated uniformly."""
    s = np.zeros(n)
    s[n // 2] = 1.0
    return shape_to_p(s, eps1)


def p_missing_ball(n, eps1, frac=0.25):
    """Kipnis-style missing ball: a contiguous block is depleted."""
    k = max(1, int(round(frac * n)))
    s = np.zeros(n)
    s[:k] = -1.0
    return shape_to_p(s, eps1)


def p_random(n, eps1, rng_seed=0):
    """i.i.d. random (unstructured) perturbation."""
    rng = np.random.default_rng(rng_seed)
    return shape_to_p(rng.normal(size=n), eps1)


# --------------------------------------------------------------------------
# test statistics
# --------------------------------------------------------------------------

def stat_ac(h):
    n = len(h)
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return abs(float(np.dot(sign, h)))


def stat_vn(h):
    h = np.asarray(h, dtype=float)
    den = np.sum((h - h.mean()) ** 2)
    if den <= 0:
        return 0.0
    return float(np.sum(np.diff(h) ** 2) / den)


def stat_runs(h):
    """Number of runs in the signs of x_i - N/n, ties dropped.

    The run count itself is returned; the two-sidedness is handled by the
    critical pair (ru_lo, ru_hi), because both too few runs (clustering) and too
    many (alternation) are evidence against uniformity.
    """
    h = np.asarray(h, dtype=float)
    s = np.sign(h - h.mean())
    s = s[s != 0]
    if len(s) < 2:
        return 0.0
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    return float(runs)


def stat_coin(h):
    """Paninski's coincidence count C = N - #{occupied bins}.

    Under uniformity the occupancy is as spread out as possible, so C is
    stochastically smallest; any departure creates extra coincidences.  This is
    the statistic that is minimax-optimal for uniformity testing in the sparse
    regime N = O(sqrt(n)), and it is order-agnostic by construction.
    """
    h = np.asarray(h, dtype=np.int64)
    return float(int(h.sum()) - int(np.count_nonzero(h)))


def stat_dtv(h):
    return float(dtv(h))


def chi2_p(h):
    h = np.asarray(h, dtype=float)
    N, n = h.sum(), len(h)
    e = N / n
    return float(1.0 - stats.chi2.cdf(np.sum((h - e) ** 2 / e), df=n - 1))


def g_p(h):
    h = np.asarray(h, dtype=float)
    N, n = h.sum(), len(h)
    e = N / n
    nz = h > 0
    g = 2.0 * np.sum(h[nz] * np.log(h[nz] / e))
    return float(1.0 - stats.chi2.cdf(g, df=n - 1))


# --------------------------------------------------------------------------
# exact null distributions of the order-aware baselines
#
# Each pmf below is a dict {statistic value: probability}.  vN is absent on
# purpose: it is a ratio of two quadratic forms, so an exact dynamic program has
# to carry the joint accumulator (sum of squared differences, sum of squares)
# and its state count grows like N^5.9 -- about 2e11 states at N=200, n=10.
# --------------------------------------------------------------------------

def _upper_crit_exact(pmf, alpha):
    """Smallest support value whose exact upper-tail mass does not exceed alpha.

    Same conservative rule as the Monte Carlo search in calibrate().
    """
    vals = sorted(pmf)
    crit, tail = float(vals[-1]) + 1.0, 0.0
    for v in reversed(vals):
        tail += float(pmf[v])
        if tail > alpha:
            break
        crit = float(v)
    return crit


def _two_sided_crits_exact(pmf, half):
    """Equal-tailed conservative two-sided critical pair (lo, hi)."""
    vals = sorted(pmf)
    run, lo = 0.0, float(vals[0]) - 1.0
    for v in vals:
        if run + float(pmf[v]) > half:
            break
        run += float(pmf[v])
        lo = float(v)
    run, hi = 0.0, float(vals[-1]) + 1.0
    for v in reversed(vals):
        if run + float(pmf[v]) > half:
            break
        run += float(pmf[v])
        hi = float(v)
    return lo, hi


def exact_null_ac(N, n):
    """Exact null of AC = |2A - N|, A = sum of the even-indexed bins.

    AC is a linear contrast of the counts, so grouping the bins by their sign
    collapses the multinomial and A ~ Binomial(N, ceil(n/2)/n).
    """
    pmf = {}
    x = np.arange(N + 1)
    for xi, p in zip(x, stats.binom.pmf(x, N, ((n + 1) // 2) / n)):
        v = float(abs(2 * int(xi) - N))
        pmf[v] = pmf.get(v, 0.0) + float(p)
    return pmf


def exact_null_coin(N, n):
    """Exact null of CO = N - K, K = number of occupied bins.

    CO is a function of the occupancy pattern alone, so K follows the classical
    occupancy distribution P(K=k) = C(n,k) k! S(N,k) / n^N with S the Stirling
    numbers of the second kind, evaluated here in exact integer arithmetic.
    """
    S = [[0] * (n + 1) for _ in range(N + 1)]
    S[0][0] = 1
    for i in range(1, N + 1):
        for j in range(1, min(i, n) + 1):
            S[i][j] = j * S[i - 1][j] + S[i - 1][j - 1]
    den = n ** N
    return {float(N - k): Fraction(comb(n, k) * factorial(k) * S[N][k], den)
            for k in range(1, n + 1)}


def exact_null_runs(N, n):
    """Exact null of the run count of sign(x_i - N/n) by dynamic programming.

    There is no closed form, but the threshold is the *fixed* mean N/n and not a
    random sample mean, so the sign of a bin depends on that bin alone.  The
    multinomial is generated bin by bin, x_i ~ Binomial(N - used, 1/(n - i + 1)),
    and the state (last non-zero sign, runs so far, number of non-zero signs
    capped at two) is carried as a vector over the number of balls used.  Ties
    x_i = N/n drop out of the sign sequence, matching stat_runs.  O(n^2 N^2).
    """
    t = N / n
    index, states = {}, []

    def state_id(st):
        if st not in index:
            index[st] = len(index)
            states.append(st)
        return index[st]

    cur = {state_id((0, 0, 0)): np.zeros(N + 1)}
    cur[0][0] = 1.0
    for i in range(1, n + 1):
        m = n - i + 1
        # M?[used, used + x] = P(x balls in bin i), split by the sign of x - N/n
        mats = {1: np.zeros((N + 1, N + 1)), 0: np.zeros((N + 1, N + 1)),
                -1: np.zeros((N + 1, N + 1))}
        for used in range(N + 1):
            left = N - used
            x = np.arange(left + 1)
            w = stats.binom.pmf(x, left, 1.0 / m)
            for xi, wi in zip(x[w > 0], w[w > 0]):
                mats[1 if xi > t else 0 if xi == t else -1][used, used + xi] += wi
        keys = list(cur)
        stacked = np.array([cur[k] for k in keys])
        moved = {sg: stacked @ mats[sg] for sg in mats}
        nxt = {}

        def add(st, vec):
            j = state_id(st)
            nxt[j] = nxt[j] + vec if j in nxt else vec.copy()

        for row, k in enumerate(keys):
            sign, runs, nnz = states[k]
            add((sign, runs, nnz), moved[0][row])       # tie: sign dropped
            for sg in (1, -1):
                if sign == 0:
                    add((sg, 1, 1), moved[sg][row])
                else:
                    add((sg, runs if sg == sign else runs + 1,
                         min(nnz + 1, 2)), moved[sg][row])
        cur = nxt

    pmf = {}
    for k, vec in cur.items():
        sign, runs, nnz = states[k]
        v = 0.0 if nnz < 2 else float(runs)             # stat_runs returns 0.0
        pmf[v] = pmf.get(v, 0.0) + float(vec[N])
    return pmf


def exact_nulls(N, n):
    """The three exact nulls.  The RUN dynamic program is the slow one, so the
    caller computes this once and passes it to everything that needs it."""
    return dict(AC=exact_null_ac(N, n), CO=exact_null_coin(N, n),
                RUN=exact_null_runs(N, n))


def exact_crits(N, n, alpha, nulls=None):
    """Exact critical values for the three baselines that admit an exact null."""
    nulls = nulls if nulls is not None else exact_nulls(N, n)
    lo, hi = _two_sided_crits_exact(nulls["RUN"], alpha / 2.0)
    return dict(ac=_upper_crit_exact(nulls["AC"], alpha),
                co=_upper_crit_exact(nulls["CO"], alpha),
                ru_lo=lo, ru_hi=hi)


def _quantile(pmf, p):
    """The p-quantile of a discrete law: smallest support value with F(v) >= p."""
    run = 0.0
    for v in sorted(pmf):
        run += float(pmf[v])
        if run >= p:
            return float(v)
    return float(max(pmf))


def _size(pmf, lo=None, hi=None):
    """Exact mass of the rejection region {X <= lo} union {X >= hi}."""
    return float(sum(p for v, p in pmf.items()
                     if (lo is not None and v <= lo)
                     or (hi is not None and v >= hi)))


def calibration_report(N, n, alpha, nulls=None):
    """Exact size of the conservative rule against the ordinary quantile rule.

    Every discrete baseline here is calibrated by the smallest critical value
    whose exact size does not exceed alpha.  The obvious alternative is to read
    the 1-alpha point off the null and reject there.  On a coarse statistic that
    point falls *inside* an atom, so the test rejects on the whole atom and the
    attained size lands above alpha: the quantile rule is anticonservative, not
    conservative, which is why it is not used.  Both rules are applied here to
    the same exact null so the two sizes can be compared directly.  At N=200,
    n=10 the run count has ten attainable values and the gap is large: the
    quantile rule attains 0.161 against a nominal 0.05.
    """
    nulls = nulls if nulls is not None else exact_nulls(N, n)
    # the dynamic programs enumerate the whole integer range, so a value is
    # attainable only if it actually carries mass (R=1 is impossible, say)
    n_attainable = lambda pmf: sum(1 for p in pmf.values() if p > 0)
    out = {}
    for name in ("AC", "CO"):
        pmf = nulls[name]
        c, q = _upper_crit_exact(pmf, alpha), _quantile(pmf, 1.0 - alpha)
        out[name] = dict(attainable=n_attainable(pmf), conservative=[c],
                         conservative_size=_size(pmf, hi=c),
                         quantile=[q], quantile_size=_size(pmf, hi=q))
    pmf = nulls["RUN"]
    lo, hi = _two_sided_crits_exact(pmf, alpha / 2.0)
    qlo, qhi = _quantile(pmf, alpha / 2.0), _quantile(pmf, 1.0 - alpha / 2.0)
    out["RUN"] = dict(attainable=n_attainable(pmf), conservative=[lo, hi],
                      conservative_size=_size(pmf, lo=lo, hi=hi),
                      quantile=[qlo, qhi],
                      quantile_size=_size(pmf, lo=qlo, hi=qhi))
    return out


# --------------------------------------------------------------------------
# MC calibration of the order-aware baselines
# --------------------------------------------------------------------------

def calibrate(N, n, alpha, n_null, seed):
    """Return Monte Carlo critical values for AC, vN, RUN and CO."""
    rng = np.random.default_rng(seed)
    probs = np.full(n, 1.0 / n)
    hs = rng.multinomial(N, probs, size=n_null)

    ac = np.array([stat_ac(h) for h in hs])
    vn = np.array([stat_vn(h) for h in hs])
    ru = np.array([stat_runs(h) for h in hs])
    co = np.array([stat_coin(h) for h in hs])

    # AC and vN: one-sided upper tail (large |contrast| / large roughness).
    # The statistics are discrete, so pick the smallest critical value whose
    # attained size does not exceed alpha (conservative, i.e. valid).
    def upper_crit(x, a):
        for c in np.unique(x):
            if np.mean(x >= c) <= a:
                return float(c)
        return float(np.max(x) + 1.0)

    ac_crit = upper_crit(ac, alpha)
    vn_crit = upper_crit(vn, alpha)
    co_crit = upper_crit(co, alpha)
    # runs: two-sided, equal-tailed, again conservative because of discreteness
    lo_cands = [c for c in np.unique(ru) if np.mean(ru <= c) <= alpha / 2.0]
    hi_cands = [c for c in np.unique(ru) if np.mean(ru >= c) <= alpha / 2.0]
    ru_lo = float(max(lo_cands)) if lo_cands else float(np.min(ru) - 1.0)
    ru_hi = float(min(hi_cands)) if hi_cands else float(np.max(ru) + 1.0)
    return dict(ac=ac_crit, vn=vn_crit, co=co_crit, ru_lo=ru_lo, ru_hi=ru_hi)


TESTS = ["CT", "X2", "G", "AC", "vN", "RUN", "CO"]


def run_case(N, n, probs, trials, alpha, crit, ct_crit, seed):
    rng = np.random.default_rng(seed)
    hs = rng.multinomial(N, probs, size=trials)
    rej = {k: 0 for k in TESTS}
    for h in hs:
        if dtv(h) >= ct_crit:
            rej["CT"] += 1
        if chi2_p(h) <= alpha:
            rej["X2"] += 1
        if g_p(h) <= alpha:
            rej["G"] += 1
        if stat_ac(h) >= crit["ac"]:
            rej["AC"] += 1
        if stat_vn(h) >= crit["vn"]:
            rej["vN"] += 1
        r = stat_runs(h)
        if r <= crit["ru_lo"] or r >= crit["ru_hi"]:
            rej["RUN"] += 1
        if stat_coin(h) >= crit["co"]:
            rej["CO"] += 1
    return {k: rej[k] / trials for k in TESTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--eps1", type=float, default=0.20,
                    help="L1 distance from uniformity")
    ap.add_argument("--trials", type=int, default=50000)
    ap.add_argument("--null-samples", type=int, default=200000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--baseline-crits", choices=["exact", "mc"], default="exact",
                    help="critical values for AC, RUN and CO: exact null "
                         "(default) or Monte Carlo. vN is always Monte Carlo.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    N, n, eps1, alpha = args.N, args.n, args.eps1, args.alpha

    cases = [
        ("uniform (H0, size)", np.full(n, 1.0 / n)),
        ("comb, period 2", p_comb(n, eps1, 2)),
        ("comb, period 4", p_comb(n, eps1, 4)),
        ("comb, period 6", p_comb(n, eps1, 6)),
        ("sinusoid, period 4", p_sinusoid(n, eps1, n / 4.0)),
        ("sinusoid, period n", p_sinusoid(n, eps1, 1)),
        ("monotone ramp", p_ramp(n, eps1)),
        ("single jump", p_single_jump(n, eps1)),
        ("missing block", p_missing_ball(n, eps1)),
        ("single spike", p_spike(n, eps1)),
        ("unstructured", p_random(n, eps1, rng_seed=SEED)),
    ]

    ct_crit, ct_src = make_ct_critical(N, n, alpha, 50000, SEED)
    # vN needs the Monte Carlo null sample either way, so both routes are always
    # computed and reported; only which one is *used* depends on the flag.
    crit_mc = calibrate(N, n, alpha, args.null_samples, SEED)
    nulls = exact_nulls(N, n)
    crit_ex = exact_crits(N, n, alpha, nulls)
    calib = calibration_report(N, n, alpha, nulls)
    crit = dict(crit_mc)
    if args.baseline_crits == "exact":
        crit.update(crit_ex)                    # vN stays Monte Carlo

    print(f"N={N}, n={n}, ||p-u||_1={eps1}, trials={args.trials}, "
          f"alpha={alpha}, CT null: {ct_src}, CT critical DTV: {ct_crit}")
    print(f"exact critical values (AC, RUN, CO): {crit_ex}")
    print(f"MC-calibrated critical values:       {crit_mc}")
    agree = all(crit_ex[k] == crit_mc[k] for k in crit_ex)
    print(f"routes agree on AC, RUN, CO: {agree};  "
          f"used: {args.baseline_crits} (+ MC for vN)")
    print(f"calibration rule, exact size at alpha={alpha} -- the conservative "
          f"rule used here vs. the 1-alpha point of the same null:")
    for name, d in calib.items():
        print(f"  {name:<4}{d['attainable']:>4} attainable values   "
              f"conservative {str(d['conservative']):<14} size "
              f"{d['conservative_size']:.4f}   "
              f"quantile {str(d['quantile']):<14} size "
              f"{d['quantile_size']:.4f}")
    header = f"{'alternative':<22}" + "".join(f"{t:>8}" for t in TESTS)
    print(header)
    print("-" * len(header))

    results = {}
    for i, (name, probs) in enumerate(cases):
        # sanity check on the L1 distance
        l1 = float(np.abs(probs - 1.0 / n).sum())
        r = run_case(N, n, probs, args.trials, alpha, crit, ct_crit,
                     SEED + 1000 * (i + 1))
        results[name] = dict(l1=l1, **r)
        print(f"{name:<22}" + "".join(f"{r[t]:>8.3f}" for t in TESTS)
              + f"   (L1={l1:.3f})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(dict(N=N, n=n, eps1=eps1, alpha=alpha,
                           trials=args.trials, crit=crit, results=results,
                           baseline_crits=args.baseline_crits,
                           crit_exact=crit_ex, crit_mc=crit_mc,
                           calibration=calib),
                      f, indent=2)
        print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
