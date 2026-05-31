# Distribution Fitting: Notes and Best Practices

## Overview

The script `fit.py` fits continuous distributions to exact DTV distributions by
minimizing a discrete Cramer-von Mises (CvM) statistic against exact CDFs.
The beta distribution is the primary approximation used in the comb test.

## Multi-Start Optimization

The CvM objective is non-convex, and Nelder-Mead can converge to local minima.
This produces fitted parameters that, while giving a reasonable absolute fit, are
noticeably worse than neighboring (N, n) pairs in the grid.

**Symptom:** Isolated cells in the KS heatmap with values 5-20x higher than their
neighbors ("speckles"). The exact distributions and CDFs are correct in these
cases; only the fitted parameters are suboptimal.

**Cause:** The optimizer finds a different basin of attraction. For example, a
speckle cell might have `b ~ 200` while all neighbors have `b ~ 380`, with
correspondingly different `scale` values. Both are valid local minima of the CvM
objective, but the neighbor solution achieves a much lower CvM statistic.

**Solution:** Multi-start optimization (`--restarts N`, default 5). The
moment-based initial guess is augmented with N random perturbations in
log-parameter space (std = 0.5). The result with the lowest CvM statistic is
selected. This eliminates the vast majority of local-minimum artifacts.

Multi-start optimization is used for the beta and gamma distributions
(the two used in the paper, both fitted via `fit_all.sh` with `--restarts 5`).
Other distributions use single-start optimization.

### Example

```bash
# Standard fit (multi-start enabled by default)
python3 fit.py --exact-dir ../data/exact_distributions/ \
    --cdf-dir ../data/cdf_exact/ \
    --distribution beta -o ../data/fitted/cvm_beta/ -v

# More restarts for higher confidence (slower)
python3 fit.py --exact-dir ../data/exact_distributions/ \
    --cdf-dir ../data/cdf_exact/ \
    --distribution beta -o ../data/fitted/cvm_beta/ --restarts 10 -v -w
```

Use `-w` (overwrite) to refit files that already exist.

## Diagnostics

After fitting, the KS heatmap (`generate_beta_figures.py`) provides a visual
check: a clean heatmap without isolated bright spots indicates good fits across
the grid. Any remaining speckles can be refit with more restarts or by using
neighbor parameters as initial guesses.

## Parameters

For the beta distribution, each output file contains:

```
a b loc scale
cvm_statistic
```

where `a, b` are shape parameters, `loc` is the location (typically 0), and
`scale` is the scale. The beta CDF is `scipy.stats.beta.cdf(x, a=a, b=b, loc=loc, scale=scale)`.

In practice, `loc = 0` for nearly all (N, n) pairs since DTV is non-negative
and the CvM optimizer does not benefit from a negative location shift.
