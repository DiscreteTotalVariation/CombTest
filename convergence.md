# MC Convergence Experiment

## Overview

Tests how many Monte Carlo samples K are needed for the beta approximation of the DTV distribution to converge to the best achievable fit (obtained from the exact distribution). For each N value, the experiment:

1. Loads the exact DTV CDF
2. Fits a beta distribution to the exact CDF (ground truth baseline)
3. For each K value, repeats the MC-based beta fitting multiple times
4. Compares MC-fitted p-values against exact p-values at α=0.05

## Scripts

### `experiment_mc_convergence_n10.py`

Runs the experiment and saves per-N results as JSON files.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n` | 10 | Number of histogram bins (ignored if `--all-n` or `--n-list`) |
| `--all-n` | off | Auto-discover all n values from `cdf_exact/` and run for each |
| `--n-list` | — | Run for specific n values (e.g., `--n-list 4 10 100`) |
| `--N-min` | 1 | Minimum sample size N |
| `--N-max` | 600 | Maximum sample size N |
| `--K-values` | 100 500 1000 2000 5000 10000 20000 50000 100000 | MC sample sizes to test |
| `--n-repeats` | 100 | Number of independent repeats per K value |
| `--seed` | 13 | Base random seed (repeat i uses seed + i) |
| `--alpha` | 0.05 | Significance level for p-value error computation |
| `--output-dir` | ../data/mc_convergence_results | Directory for output JSON files |
| `--workers` | cpu_count - 2 | Number of parallel workers |

**Usage:**

```bash
# Run with defaults (n=10, N=1..600, K up to 100000, 100 repeats)
python3 experiment_mc_convergence_n10.py

# Custom single n value
python3 experiment_mc_convergence_n10.py --n 20 --N-max 300

# Multiple specific n values
python3 experiment_mc_convergence_n10.py --n-list 4 10 50 100

# Auto-discover and run for ALL available n values in cdf_exact/
python3 experiment_mc_convergence_n10.py --all-n
```

**Output:**
- Single n mode: `{output-dir}/N_{N}.json` files + `config.json`
- Multi-n mode (`--all-n` or `--n-list`): `{output-dir}/n_{n}/N_{N}.json` files per n value

Each JSON file contains:
- `exact_beta_params`: beta parameters fitted to the exact CDF
- `exact_error`: p-value error of the exact-beta fit
- `K_values`: for each K, the list of errors across repeats plus summary statistics (mean, std, median, max)

### `plot_mc_convergence.py`

Generates a two-panel figure from saved results.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | ../data/mc_convergence_results | Directory with per-N JSON files |
| `--output` | ../paper/img/mc_convergence.png | Output figure path |

**Usage:**

```bash
# Plot from default directory
python3 plot_mc_convergence.py

# Custom paths
python3 plot_mc_convergence.py --input-dir my_results --output my_plot.png
```

**Output figure:**
- Left panel: aggregate mean, median, and 95th percentile of p-value error vs K, with exact-beta baselines
- Right panel: mean p-value error vs N for selected K values

## Results (n=10, N=1..600, 100 repeats)

| K | Mean error | Median error | 95th percentile |
|---|-----------|-------------|-----------------|
| 100 | 0.018071 | 0.016578 | 0.018681 |
| 500 | 0.009106 | 0.007383 | 0.009007 |
| 1,000 | 0.007173 | 0.005236 | 0.007600 |
| 2,000 | 0.005908 | 0.003815 | 0.007233 |
| 5,000 | 0.005368 | 0.003055 | 0.007969 |
| 10,000 | 0.004982 | 0.002575 | 0.007723 |
| 20,000 | 0.004908 | 0.002392 | 0.007879 |
| 50,000 | 0.004848 | 0.002354 | 0.007950 |
| 100,000 | 0.004828 | 0.002356 | 0.007933 |
| Exact beta | 0.004798 | 0.002393 | — |

## Key findings

- **Convergence is governed by K/N:** At K/N ≥ 100, the mean p-value error is within 2% of the exact-beta baseline. At K/N ≥ 50, within 3%.
- **No dependence on n:** Convergence rate is the same for n ∈ {4, 5, 10, 20, 50, 100} at matched K/N ratios.
- **Intuition:** Larger N widens the DTV support, requiring proportionally more MC samples to resolve the tail. The number of bins n affects the distribution shape but not the sampling difficulty.
- **Practical rule:** Set K = 100·N for full convergence, or use K = 50,000 for N ≤ 500.
