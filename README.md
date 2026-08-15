# Comb Test

A statistical test for detecting alternating (comb-like) deviations in discrete histograms, based on the Discrete Total Variation (DTV) statistic.

The DTV is defined as the sum of absolute differences between adjacent bins: DTV = sum |h_{i+1} - h_i|. Under the uniform null hypothesis, DTV follows a known distribution that can be computed exactly via dynamic programming or approximated using a four-parameter beta distribution.

## Project Structure

```
ct/
├── paper/
│   └── img/            # Figures written here by pipeline.sh (not tracked)
├── code/               # All scripts (Python, C, shell)
│   ├── README.md       # Detailed script reference
│   ├── pipeline.sh     # Master pipeline
│   ├── comb_test.py    # Core CT implementation
│   ├── ct.py           # Exact DTV (Python)
│   ├── ct_save.c       # Exact DTV (C, with checkpoint)
│   ├── download_data.py # Download precomputed data from Dropbox
│   └── ...             # 40 .py, 72 .sh, 4 .c files
├── data/               # Result files (tracked) + precomputed grids (not tracked)
│   ├── *.json, *.txt         # Experiment outputs the paper's numbers come from
│   ├── mc_convergence_results/  # Per-N MC convergence output (Fig. 2)
│   ├── exact_distributions/  # ~450K files, ~87 GB  (download or recompute)
│   ├── cdf_exact/            # ~450K files, ~3.6 GB (download or recompute)
│   └── fitted/               # Distribution fits, 29 CvM-fitted subdirectories
│       ├── cvm_beta/         # Primary approximation      (downloadable)
│       ├── cvm_gamma/        # Used for all paper p-values (downloadable)
│       └── cvm_.../          # + 27 others, ~54 GB (must be refitted locally)
├── Dockerfile          # Docker image for full pipeline
├── requirements.txt    # Python dependencies
├── build.sh            # Build Docker image
├── pipeline_docker.sh  # Run pipeline in Docker
├── data.md             # Data format documentation
├── convergence.md      # MC convergence experiment details
└── README.md           # This file
```

## Quick Start

### Using the Comb Test

```python
from comb_test import comb_test, comb_test_gamma

histogram = [12, 8, 13, 7, 10]  # observed bin counts

p_value = comb_test(histogram)   # exact p-value from precomputed data
p_value = comb_test_gamma(histogram, seed=13)  # gamma approximation, no data needed
```

`comb_test` reads the exact CDF from `data/cdf_exact/` and returns `None` if the
file for that (N, n) has not been downloaded or computed. `comb_test_gamma`
needs no precomputed data and is the route used for every approximate p-value in
the paper.

### Running the Pipeline

```bash
cd code/

# Regenerate plots from existing data
./pipeline.sh --plots-only

# Full pipeline (exact distributions + fitting + experiments + plots)
./pipeline.sh --from-scratch

# Use compiled C binary for exact distributions (much faster)
./pipeline.sh --from-scratch --use-c

# Download precomputed archives from Dropbox, then run experiments + plots
./pipeline.sh --from-scratch -a
```

### Docker

```bash
# Build image
./build.sh

# Run full pipeline in container (results persist on host via bind mounts)
./pipeline_docker.sh --from-scratch --use-c
```

## Reproducing the paper

Every number and figure in the paper is produced by `code/pipeline.sh`. The
result files themselves are tracked in `data/`, so the printed values can be
checked immediately; recomputing them needs the precomputed grids described
under *Data prerequisites* below.

| Paper item | Script (pipeline step) | Needs |
|------------|------------------------|-------|
| Fig. 1a, DTV distribution (N=20, n=4) | `plot_dtv_distribution_histogram.py` (1) | one file from `exact_distributions/` |
| Fig. 1b, beta/gamma overlay (N=50, n=10) | `generate_beta_figures.py` (2) | one file each from `exact_distributions/`, `cvm_beta/`, `cvm_gamma/` |
| Fig. 2, MC convergence | `experiment_mc_convergence_n10.py`, `plot_mc_convergence.py` (4) | `cdf_exact/` for n=10, N<=600; output is tracked, so `--plots-only` works as is |
| Fig. 3, KS heatmaps and diagonal | `generate_beta_figures.py` (3) | the full `exact_distributions/`, `cdf_exact/`, `cvm_beta/`, `cvm_gamma/` grid over N,n in {2..500} |
| Fig. 4, rounding bias | `generate_mean_p_values.py`, `plot_mean_p_values.py` (5, 6) | `cdf_exact/`, `cvm_beta/`, `cvm_gamma/`; regeneration takes hours (d=1 runs to N=160,000) |
| Fig. 5, ADC differential nonlinearity | `experiment_adc_dnl_data.py`, `plot_adc_dnl.py` (7) | `cdf_exact/`, `cvm_beta/`, `cvm_gamma/` |
| Section III-A, exact vs. beta vs. gamma routes at N=500, n=256 | `compare_routes_n256.py` (12) | `cdf_exact/`, `cvm_beta/`, `cvm_gamma/` at that pair |
| Section III-B, convergence depends on K/N but not on n | `experiment_mc_convergence.py` (13) | `cdf_exact/` for n in {5, 10, 20, 30} |
| Section IV-A, exact-vs-beta critical value and p-value agreement | `compute_beta_match_stats.py --N-max 500` (9) | the full `cdf_exact/` and `cvm_beta/` grid |
| Section IV-A, beta ranked first of 29 candidates by median W^2 | `./fit_all.sh --all`, then `analyze.py --all --csv out.csv` | **all 29** `fitted/cvm_*` directories (see below) |
| Table I, L1-matched power | `experiment_l1_matched_power.py` (10) | nothing; self-contained |
| Section IV-C2, ADC at n=256 | `experiment_adc_dnl_n256.py` (8) | `cdf_exact/`, `cvm_beta/`, `cvm_gamma/` |
| Section IV-C3, real image data | `experiment_real_images.py` (11) | nothing; self-contained (needs scikit-image) |

### Data prerequisites

`code/download_data.py` (or `./pipeline.sh --from-scratch -a`) fetches four
archives: `exact_distributions` (~87 GB), `cdf_exact` (~3.6 GB), and the fitted
`cvm_beta` and `cvm_gamma` parameters. That covers every row of the table above
except one.

The remaining **27 fitted distributions (~54 GB) are not distributed**. They are
needed only to reproduce the ranking of the twenty-nine candidate distributions
in Section IV-A, and can be regenerated with:

```bash
cd code/
./fit_all.sh --all       # fits all 29 distributions; expect hours to days
python3 analyze.py --all --csv out.csv
```

All other results depend on beta and gamma only.

## Key Results

- The **beta distribution** is the best continuous approximation for DTV (median p-value error 0.000116 across 249,001 (N,n) pairs)
- **MC+beta** approximation: K=50,000 Monte Carlo samples achieves error < 0.001 for N <= 500
- Among ten L1-matched departure shapes, the **period-2 comb** is the one direction where CT beats chi-squared and the G-test, at both operating points (0.623 vs 0.449 and 0.461 at n=10, N=200; 0.376 vs 0.229 at n=256, N=500)
- CT has **no advantage** for smooth or monotonic deviations, nor for single jumps or depleted blocks (by design)

## Documentation

| File | Contents |
|------|----------|
| [code/README.md](code/README.md) | Complete script reference (all Python, shell, C files) |
| [code/README_ct_save.md](code/README_ct_save.md) | C implementation: compilation, flags, checkpoint/resume |
| [data.md](data.md) | Data directory formats and creation scripts |
| [convergence.md](convergence.md) | MC convergence experiment methodology and results |

## Dependencies

**Python:** numpy, scipy, matplotlib, tqdm, opencv-python-headless, scikit-image (see `requirements.txt`)

**C compilation:** GCC, GMP (libgmp-dev), OpenMP (libomp-dev). Optional: jemalloc for faster multi-threaded runs.

```bash
# Ubuntu/Debian
sudo apt install gcc libgmp-dev libomp-dev

# Compile
cd code && make ct_save
```
