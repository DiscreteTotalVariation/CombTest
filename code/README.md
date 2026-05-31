# Code Directory

All scripts for the Comb Test project. Run scripts from this directory (`cd code/`); data paths are relative (`../data/`), paper output goes to `../paper/img/`.

## Pipeline

| Script | Description |
|--------|-------------|
| `pipeline.sh` | Master pipeline (see modes below) |
| `download_data.py` | Download precomputed archives from Dropbox (exact_distributions, cdf_exact, cvm_beta, cvm_gamma) |
| `fit_all.sh` | Fit distributions: beta+gamma by default, `--all` for all 29 |

**Pipeline modes:**
- `./pipeline.sh --plots-only` — regenerate plots from existing data only
- `./pipeline.sh --from-scratch` — full pipeline: compute exact distributions (Python), CDFs, fit beta+gamma, run experiments, generate plots
- `./pipeline.sh --from-scratch --use-c` — same but use compiled C binary for exact distributions (much faster)
- `./pipeline.sh --from-scratch -a` — download archives instead of computing, then run experiments and plots

**Pipeline steps:** 0: download archives (`-a`), 0a: exact DTV distributions (`--from-scratch`), 0b: exact CDFs, 0c: fit distributions, 1-7: figures, 8: ADC n=256 experiment, 9: beta approximation quality stats

## Core Computation

### Exact DTV Distributions

| Script | Description |
|--------|-------------|
| `ct.py` | Python DP computation. `-N 500 -n 500 -u` computes all pairs, skipping existing |
| `ct_save.c` | C implementation with checkpoint/resume, GMP, OpenMP. See [README_ct_save.md](README_ct_save.md) |
| `ct.c` | Basic C implementation (no checkpoint) |
| `Makefile` | `make ct_save` or `make ct_c` (uses GMP, OpenMP, optional jemalloc) |

### CDF and Distribution Fitting

| Script | Description |
|--------|-------------|
| `calculate_cdf_values.py` | Convert exact distributions to CDFs. No arguments; processes all files |
| `fit.py` | CvM-fit a distribution: `--exact-dir ... --cdf-dir ... --distribution beta -o ... -v` |
| `fit_all.sh` | Fit beta+gamma (default) or all 29 distributions (`--all`) |
| `fit_*.sh` | One-liner wrappers for `fit.py` per distribution (29 scripts) |

### Comb Test Implementation

| Script | Description |
|--------|-------------|
| `comb_test.py` | Core CT module. `from comb_test import comb_test` for exact p-values, `comb_test_gamma` for MC approximation. Also runs as CLI demo |

## Experiments

| Script | Description | Output |
|--------|-------------|--------|
| `experiment_adc_dnl_data.py` | ADC alternating DNL power analysis (n=10) | `../data/adc_dnl_results.json` |
| `experiment_adc_dnl_n256.py` | Reproduce all ADC paper numbers (n=10 + n=256) | `../data/adc_dnl_all_results.json` |
| `experiment_mc_convergence_n10.py` | MC convergence vs K and N | `../data/mc_convergence_results/` |
| `generate_mean_p_values.py` | Banker's rounding mean p-values | `../data/mean_p_values_d_*_step_*.txt` |

## Plotting

| Script | Description | Output |
|--------|-------------|--------|
| `plot_dtv_distribution_histogram.py` | DTV histogram for given N, n | PNG (`-o`) |
| `plot_mc_convergence.py` | MC convergence two-panel figure | PNG (`--output`) |
| `plot_mean_p_values.py` | Mean p-value curves | PNG (`-o`) |
| `plot_adc_dnl.py` | ADC DNL power comparison | PNG (`-o`) |
| `generate_beta_figures.py` | Beta figures (overlay, heatmap, diagonal KS) | `../paper/img/` |

## Analysis

| Script | Description |
|--------|-------------|
| `compute_beta_match_stats.py` | Critical-value match rates: exact vs beta at alpha=0.05 |
