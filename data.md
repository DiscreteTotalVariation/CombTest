# Data Directory Documentation

This document describes the precomputed data used in the Comb Test project. All data is stored in the `data/` directory. It relates to the distribution of the Discrete Total Variation (DTV) statistic under the uniform null hypothesis for histogram configurations (N, n), where N is the sample size and n is the number of bins.

All scripts that generate or consume this data are in the `code/` directory and reference data via `../data/` relative paths.

## File Naming Convention

All per-(N,n) files use the same naming convention:

```
N_{N}_n_{n}.txt
```

For example, `N_50_n_10.txt` contains data for N=50 samples and n=10 bins.

---

## 1. `data/exact_distributions/`

**What it contains:** The exact probability mass function (PMF) of DTV under the uniform null hypothesis, computed via dynamic programming (Algorithm 1 in the paper).

**File count:** ~450,000 files (N up to 625 for small n, N up to 500 for n up to 500).

**Format:** Two columns per line — DTV value (integer) and its unnormalized count (arbitrary-precision integer):

```
0 49120458506088132224064306071170476903628800
1 0
2 293800479053261192096351707223526090636288000
3 3326278753982347470145376087342920346419200000
...
```

The count for DTV value d is the number of ordered histograms (x_1, ..., x_n) with sum N and DTV equal to d, weighted by the multinomial coefficient. The probability is obtained by dividing by the sum of all counts (which equals n^N, the total number of equally likely ordered outcomes).

Lines with count 0 may be present (e.g., DTV=1 is often impossible).

**How they are created:** The Python script `code/ct.py` or the C implementation `code/ct_save.c` implements the DP recurrence from the paper. The complexity is O(nN^4) time and O(N^3) memory using arbitrary-precision integers. The C version uses GMP for arbitrary precision and OpenMP for parallelization, and is significantly faster.

**Coverage:** All pairs in the {2,...,500} x {2,...,500} grid plus {2,...,625} x {2,...,10} are present with valid data.

---

## 2. `data/cdf_exact/`

**What it contains:** The exact cumulative distribution function (CDF) of DTV, derived from the exact distributions.

**File count:** ~450,000 files (mirrors `exact_distributions/`).

**Format:** Two columns per line — DTV value (integer) and cumulative probability (float):

```
0 4.912045850608813e-07
2 3.429209375593493e-06
3 3.669199691541697e-05
4 0.00014731502202626785
...
98 1.0
99 1.0
100 1.0
```

Only DTV values with nonzero probability are listed (e.g., DTV=1 is skipped if its count is zero). The CDF is computed as the running sum of counts divided by the total count.

**How they are created:** The script `code/calculate_cdf_values.py` reads each file from `exact_distributions/`, computes the cumulative sum, divides by the total, and writes the result. The function `calculate_exact_cdf(N, n)` handles one (N, n) pair; `main()` processes all available files.

---

## 3. `data/fitted/cvm_{distribution}/`

**What it contains:** Parameters of continuous (or discrete) distributions fitted to the exact DTV distribution by minimizing a discrete Cramer-von Mises statistic against the exact CDF.

**Directories:** 29 CvM-fitted distributions, each in its own subdirectory under `data/fitted/`. The `fitted/previous/` subdirectory contains archived earlier fits (cvm_beta and cvm_gamma only).

| Directory | Distribution | Parameters |
|-----------|-------------|------------|
| `cvm_beta` | Four-parameter beta | a, b, loc, scale |
| `cvm_gamma` | Gamma | shape, loc, scale |
| `cvm_normal` | Normal | loc, scale |
| `cvm_nbinom` | Negative binomial | n, p |
| `cvm_weibull` | Weibull | shape, loc, scale |
| `cvm_chi2` | Chi-squared | df, loc, scale |
| `cvm_exponential` | Exponential | loc, scale |
| `cvm_lognormal` | Lognormal | shape, loc, scale |
| `cvm_genextreme` | Generalized extreme value | shape, loc, scale |
| `cvm_rice` | Rice | shape, loc, scale |
| `cvm_burr12` | Burr Type XII | c, d, loc, scale |
| `cvm_f` | F-distribution | dfn, dfd, loc, scale |
| `cvm_nakagami` | Nakagami | shape, loc, scale |
| `cvm_invgauss` | Inverse Gaussian | shape, loc, scale |
| `cvm_poisson` | Poisson | mu |
| `cvm_pareto` | Pareto | shape, loc, scale |
| `cvm_genpareto` | Generalized Pareto | shape, loc, scale |
| `cvm_gengamma` | Generalized gamma | a, c, loc, scale |
| `cvm_rayleigh` | Rayleigh | loc, scale |
| `cvm_burr` | Burr | c, d, loc, scale |
| `cvm_fisk` | Fisk (log-logistic) | shape, loc, scale |
| `cvm_johnsonsb` | Johnson SB | a, b, loc, scale |
| `cvm_betaprime` | Beta prime | a, b, loc, scale |
| `cvm_arcsine` | Arcsine | loc, scale |
| `cvm_powerlaw` | Power law | shape, loc, scale |
| `cvm_bradford` | Bradford | shape, loc, scale |
| `cvm_triang` | Triangular | shape, loc, scale |
| `cvm_uniform` | Uniform | loc, scale |
| `cvm_binomial` | Binomial | n, p |

**File count:** ~449,000 for beta and gamma (fitted across the full grid including N up to 625 for small n); ~444,000 for other distributions.

**Format:** Two lines:
- Line 1: Space-separated distribution parameters (number of values depends on the distribution)
- Line 2: The Cramer-von Mises goodness-of-fit statistic (sum of squared CDF differences)

Example (`fitted/cvm_beta/N_50_n_10.txt`):
```
8.457046663035548 33.83279684539579 0.0 109.90568262830892
1.4564329643613226e-05
```

This means: beta(a=8.457, b=33.833, loc=0.0, scale=109.906) with CvM = 1.46e-05.

**How they are created:** The script `code/fit.py` contains `fit_distribution()`, which uses `scipy.optimize.minimize` (Nelder-Mead) to minimize the discrete CvM statistic W^2 = sum_i (F_exact(x_i) - F_approx(x_i))^2. Initial parameter estimates are derived from the distribution's mean and variance. For the beta and gamma distributions, multi-start optimization is used (`--restarts N`, default 5) to avoid local minima: the moment-based initialization is augmented with N random perturbations, and the result with the lowest CvM statistic is selected. The script `code/fit_all.sh` orchestrates fitting: by default it fits beta and gamma (the two used in the paper); pass `--all` to fit all 29 distributions.

**Key result:** The beta distribution achieves the lowest p-value error at alpha=0.05 across the (N, n) grid and is used as the approximation in the comb test.

**Note on local minima:** Single-start Nelder-Mead optimization can converge to local minima, producing fitted parameters that differ markedly from neighbors in the (N,n) grid. Multi-start optimization (`--restarts 5`) eliminates the vast majority of these. See `code/fitting.md` for details.

---

## 4. Experiment Results

These files are generated by experiment scripts and consumed by plotting scripts.

### `data/adc_dnl_results.json`

ADC alternating DNL power comparison results (n=10). Contains per-epsilon, per-N power values for CT (exact), CT (MC+beta), Pearson's chi-squared, and G-test.

**Created by:** `code/experiment_adc_dnl_data.py`
**Used by:** `code/plot_adc_dnl.py`

### `data/adc_dnl_all_results.json`

Comprehensive ADC experiment reproducing all paper numbers (n=10 and n=256): power values, false positive rates, and random DNL scenarios.

**Created by:** `code/experiment_adc_dnl_n256.py`

### `data/adc_dnl_results_n256.json`

Earlier n=256-only ADC DNL results (superseded by `adc_dnl_all_results.json`).

### `data/additional_test_results.json`

Power comparison for non-standard scenarios (sinusoidal, block bias, single spike, etc.).

**Created by:** `code/simulate_additional_tests.py`
**Used by:** `code/generate_plots.py`

### `data/mean_p_values_d_{d}_step_{step}.txt`

Mean p-values for banker's rounding experiments. Space-separated columns: N, CT mean p-value, Pearson mean p-value, G-test mean p-value.

- `d=0, step=100`: 1-decimal rounding, N from 10 to 3000
- `d=1, step=1000`: 2-decimal rounding, N from 10 to 160000

**Created by:** `code/generate_mean_p_values.py`
**Used by:** `code/plot_mean_p_values.py`

### `data/mc_convergence_results/`

Per-N JSON files with MC convergence data (errors at various K values across repeats), plus `config.json` with experiment parameters.

**Created by:** `code/experiment_mc_convergence_n10.py`
**Used by:** `code/plot_mc_convergence.py`

### `data/heatmap_ks_beta_450.npy`

NumPy array of max KS statistics |CDF_exact - CDF_beta| for a grid of (N,n) pairs. NaN for missing pairs. Supports incremental computation.

**Created by:** `code/prepare_heatmap_data.py`
**Used by:** `code/plot_heatmap.py`

### Analysis Results (JSON)

These are generated by one-off analysis scripts, not part of the main pipeline.

| File | Description | Created by |
|------|-------------|------------|
| `pvalue_accuracy_results.json` | Stratified p-value error (gamma vs beta vs nbinom) | `code/analyze_pvalue_accuracy.py` |
| `pvalue_accuracy_all_results.json` | Full p-value accuracy across all pairs | `code/analyze_pvalue_accuracy_all.py` |
| `pvalue_accuracy_full_results.json` | Optimized single-thread variant | `code/analyze_pvalue_accuracy_full.py` |
| `pvalue_accuracy_all_dists_results.json` | P-value accuracy across 22+ distributions | `code/analyze_pvalue_all_dists.py` |
| `rounding_results_d1.json` | Rounding simulation (1 decimal) | `code/simulate_rounding.py` |
| `rounding_results_d2.json` | Rounding simulation (2 decimals) | `code/simulate_rounding.py` |
| `rounding_results_large.json` | Rounding simulation (large N) | `code/simulate_rounding.py` |

---

## Summary

| Path | Contents | Format | Created by |
|------|----------|--------|------------|
| `data/exact_distributions/` | Exact DTV PMF counts | `dtv_value count` (per line) | `code/ct.py` or `code/ct_save.c` |
| `data/cdf_exact/` | Exact DTV CDF | `dtv_value cdf_prob` (per line) | `code/calculate_cdf_values.py` |
| `data/fitted/cvm_{dist}/` | CvM-fitted parameters | params (line 1), CvM stat (line 2) | `code/fit.py` |
| `data/adc_dnl_results.json` | ADC power comparison (n=10) | JSON | `code/experiment_adc_dnl_data.py` |
| `data/adc_dnl_all_results.json` | Full ADC paper numbers (n=10 + n=256) | JSON | `code/experiment_adc_dnl_n256.py` |
| `data/l1_matched_power_N*_n*.txt` | Table I: L1-matched power over ten departure shapes plus the uniform null | JSON | `code/experiment_l1_matched_power.py` |
| `data/real_images_results.txt` | Section IV-C3: size and power at n=256, N up to 5e4, nine test images | JSON | `code/experiment_real_images.py` |
| `data/route_comparison_N500_n256.txt` | Section III-A: paired exact / CvM-beta / CvM-gamma / MC-beta critical values and power | text table | `code/compare_routes_n256.py` |
| `data/mean_p_values_*.txt` | Banker's rounding p-values | space-separated text | `code/generate_mean_p_values.py` |
| `data/mc_convergence_results/` | MC convergence data | per-N JSON files | `code/experiment_mc_convergence_n10.py` |
| `data/heatmap_ks_*.npy` | KS heatmap grid | NumPy array | `code/prepare_heatmap_data.py` |
| `data/pvalue_accuracy_*.json` | P-value accuracy analysis | JSON | `code/analyze_pvalue_*.py` |
| `data/rounding_results_*.json` | Rounding simulation results | JSON | `code/simulate_rounding.py` |

Superseded files, kept but written by no current script — do not cite these:
`data/adc_dnl_results_n256.json`, `data/l1_matched_N200_n10.{json,txt}` and
`data/l1_matched_N500_n256.{json,txt}` (a six-test predecessor of Table I, without the
coincidence-test column), `data/real_images_n256.json` (a copy of
`real_images_results.txt`) and `data/real_images_n256.txt` (a plain-text rendering of it).
