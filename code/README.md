# Code Directory

All scripts for the Comb Test project. Run scripts from this directory (`cd code/`); data paths are relative (`../data/`), paper output goes to `../paper/img/`.

## Pipeline

| Script | Description |
|--------|-------------|
| `pipeline.sh` | Master pipeline (see modes below) |
| `create_plots.sh` | Regenerate all paper figures from existing data |
| `download_data.py` | Download precomputed archives from Dropbox (exact_distributions, cdf_exact, cvm_beta, cvm_gamma) |
| `fit_all.sh` | Fit distributions: beta+gamma by default, `--all` for all 29 |

**Pipeline modes:**
- `./pipeline.sh --plots-only` — regenerate plots from existing data only
- `./pipeline.sh --from-scratch` — full pipeline: compute exact distributions (Python), CDFs, fit beta+gamma, run experiments, generate plots
- `./pipeline.sh --from-scratch --use-c` — same but use compiled C binary for exact distributions (much faster)
- `./pipeline.sh --from-scratch -a` — download archives instead of computing, then run experiments and plots

**Pipeline steps:** 0: download archives (`-a`), 0a: exact DTV distributions (`--from-scratch`), 0b: exact CDFs, 0c: fit distributions, 1-7: figures, 8: ADC n=256 experiment, 9: beta approximation quality stats, 10: Table I, 11: real image data, 12: paired route comparison, 13: MC convergence across n

## Core Computation

### Exact DTV Distributions

| Script | Description |
|--------|-------------|
| `ct.py` | Python DP computation. `-N 500 -n 500 -u` computes all pairs, skipping existing |
| `ct_save.c` | C implementation with checkpoint/resume, GMP, OpenMP. See [README_ct_save.md](README_ct_save.md) |
| `ct.c` | Basic C implementation (no checkpoint) |
| `faster_ct.c` | Optimized C (range tracking, flat work list) |
| `even_faster_ct.c` | Further optimized C (per-thread accumulators, parallel clear) |
| `ct_c.sh` | Wrapper for ct_c binary; `-t <threads>` sets OpenMP threads |
| `keep.sh` | Watchdog: restarts ct_save tmux session if it dies |
| `Makefile` | `make ct_save` or `make ct_c` (uses GMP, OpenMP, optional jemalloc) |

### CDF and Distribution Fitting

| Script | Description |
|--------|-------------|
| `calculate_cdf_values.py` | Convert exact distributions to CDFs. No arguments; processes all files |
| `fit.py` | CvM-fit a distribution: `--exact-dir ... --cdf-dir ... --distribution beta -o ... -v` |
| `fit_all.sh` | Fit beta+gamma (default) or all 29 distributions (`--all`) |
| `mle_fit.py` | MLE-fit a distribution: `--exact-dir ... --distribution beta -o ... -v` |
| `fit_beta_missing.py` | Bulk-fit missing beta files using multiprocessing |
| `fit_*.sh` | One-liner wrappers for `fit.py` per distribution (29 scripts) |
| `mle_fit_*.sh` | One-liner wrappers for `mle_fit.py` per distribution (29 scripts) |
| `start_fits.sh` / `stop_fits.sh` | Launch/kill tmux sessions for all CvM fits in parallel |
| `start_mle_fits.sh` / `stop_mle_fits.sh` | Launch/kill tmux sessions for all MLE fits in parallel |

### Comb Test Implementation

| Script | Description |
|--------|-------------|
| `comb_test.py` | Core CT module. `from comb_test import comb_test` for exact p-values, `comb_test_gamma` for MC approximation. Also runs as CLI demo |

## Experiments

| Script | Description | Output |
|--------|-------------|--------|
| `experiment_adc_dnl_data.py` | ADC alternating DNL power analysis (n=10) | `../data/adc_dnl_results.json` |
| `experiment_adc_dnl.py` | Same experiment + immediate plot (no JSON). Superseded by `experiment_adc_dnl_data.py` + `plot_adc_dnl.py`; it uses seed 42 and overwrites the paper figure, so do not run it in a checkout you want to keep | `../paper/img/adc_dnl_power_comparison.png` |
| `experiment_adc_dnl_n256.py` | Reproduce all ADC paper numbers (n=10 + n=256) | `../data/adc_dnl_all_results.json` |
| `experiment_l1_matched_power.py` | Table I: power over ten shapes of departure at fixed L1 distance, plus the uniform null, against chi-square, G, alternating contrast, von Neumann, runs and coincidence tests | `../data/l1_matched_power_N*_n*.txt` |
| `experiment_real_images.py` | Section IV-C3: size and power at n=256 for N up to 5e4 on nine scikit-image test images | `../data/real_images_results.txt` |
| `experiment_mc_convergence_n10.py` | MC convergence vs K and N | `../data/mc_convergence_results/` |
| `experiment_mc_convergence.py` | MC convergence (stdout only, simpler version) | stdout |
| `experiment_mc_beta.py` | Validate MC+beta against exact distributions | stdout |
| `experiment_real_data.py` | CT on real data (Python round, /dev/urandom, NumPy) | stdout |
| `generate_mean_p_values.py` | Banker's rounding mean p-values | `../data/mean_p_values_d_*_step_*.txt` |
| `run_mean_p_values.sh` | Generate + plot mean p-values (both d=0 and d=1) | text files + PNGs |
| `simulate_rounding.py` | Rounding method power comparison | optional JSON |
| `simulate_additional_tests.py` | CT vs non-uniform patterns (sinusoidal, block, spike) | optional JSON |
| `sp_power_comparison.py` | Signal processing scenario comparison | stdout |
| `mc_beta_validation.py` | MC+beta validation summary | stdout |

## Plotting

| Script | Description | Output |
|--------|-------------|--------|
| `plot_dtv_distribution_histogram.py` | DTV histogram for given N, n | PNG (`-o`) |
| `plot_heatmap.py` | KS heatmap from .npy data | PNG (`-o`) |
| `plot_mc_convergence.py` | MC convergence two-panel figure | PNG (`--output`) |
| `plot_mean_p_values.py` | Mean p-value curves | PNG (`-o`) |
| `plot_adc_dnl.py` | ADC DNL power comparison | PNG (`-o`) |
| `generate_beta_figures.py` | All beta figures (overlay, heatmap, comparison) | `../paper/img/` |
| `generate_plots.py` | Plots from simulation JSON results | `report_plots/` |
| `prepare_heatmap_data.py` | Compute KS heatmap .npy (incremental) | `../data/heatmap_ks_*.npy` |

## Analysis and Validation

| Script | Description |
|--------|-------------|
| `compute_beta_match_stats.py` | Critical-value match rates: exact vs beta at alpha=0.05 |
| `compare_routes_n256.py` | Section III-A: exact, CvM-beta, CvM-gamma and MC-beta p-values evaluated on the same histograms (critical values and power) |
| `compare.py` | Plot exact vs fitted distribution. CLI: `-N 50 -n 10 -d beta -d gamma` |
| `compare.sh` / `compare_mle.sh` | Shell wrappers for compare.py (CvM / MLE fits) |
| `check.sh` / `check_mle.sh` | Query fitted parameters for one (N,n) pair: `-N 50 -n 10` |
| `analyze.py` | Full model selection (9 metrics + Vuong tests). `--all --csv out.csv` |
| `analyze_dtv_distribution_fit.py` | Sample-based distribution ranking |
| `analyze_pvalue_accuracy.py` | Stratified p-value error (gamma vs beta vs nbinom) |
| `analyze_pvalue_accuracy_all.py` | Full p-value accuracy across all 440K+ pairs |
| `analyze_pvalue_accuracy_full.py` | Optimized single-thread variant of the above |
| `analyze_pvalue_all_dists.py` | P-value accuracy across all 22+ distributions |
| `model_selection.py` | General model selection tool with Vuong tests |

## Legacy

| Script | Description |
|--------|-------------|
| `dtv.py` | Original monolithic module (~2700 lines). Contains DP implementation, plotting, and analysis functions. Uses hardcoded `/mnt/data/tvor/` paths. Superseded by the individual scripts above |
