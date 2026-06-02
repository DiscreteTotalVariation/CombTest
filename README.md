# Comb Test

[![arXiv](https://img.shields.io/badge/arXiv-2606.01465-b31b1b.svg)](https://arxiv.org/abs/2606.01465)

A statistical test for detecting alternating (comb-like) deviations in discrete histograms, based on the Discrete Total Variation (DTV) statistic.

Paper: [*Comb Test: Histogram Uniformity Testing Based on Discrete Total Variation*](https://arxiv.org/abs/2606.01465) (arXiv:2606.01465).

The DTV is defined as the sum of absolute differences between adjacent bins: DTV = Σ|h(i+1) − h(i)|. Under the uniform null hypothesis, DTV follows a known distribution that can be computed exactly via dynamic programming or approximated using a gamma distribution with Monte Carlo parameter estimation.

## Project Structure

```
CombTest/
├── code/               # All scripts (Python, C, shell)
│   ├── pipeline.sh     # Master pipeline
│   ├── comb_test.py    # Core CT implementation
│   ├── ct.py           # Exact DTV (Python)
│   ├── ct_save.c       # Exact DTV (C, with checkpoint/resume)
│   ├── download_data.py # Download precomputed data from Dropbox
│   ├── fit.py          # Distribution fitting (29 distributions)
│   └── ...             # Experiments, plotting, fitting scripts
├── data/               # Precomputed data (populated via download or computation)
│   ├── exact_distributions/  # Exact DTV PMFs
│   ├── cdf_exact/            # Exact DTV CDFs
│   ├── fitted/               # Distribution fits (29 CvM-fitted subdirectories)
│   │   ├── cvm_beta/         # Primary approximation
│   │   ├── cvm_gamma/        # Secondary (used for power comparisons)
│   │   └── cvm_.../          # + 27 other distributions
│   └── ...
├── Dockerfile          # Docker image for full pipeline
├── requirements.txt    # Python dependencies
├── build.sh            # Build Docker image
├── pipeline_docker.sh  # Run pipeline in Docker
├── data.md             # Data format documentation
└── convergence.md      # MC convergence experiment details
```

## Quick Start

### Using the Comb Test

```python
from comb_test import comb_test, comb_test_gamma

histogram = [12, 8, 13, 7, 10]  # observed bin counts

# Exact p-value (requires precomputed data)
p_exact = comb_test(histogram)

# Gamma approximation (works for any N, n)
p_gamma = comb_test_gamma(histogram)
```

### Running the Pipeline

```bash
cd code/

# Download precomputed data, then run experiments + plots
./pipeline.sh --from-scratch -a

# Full pipeline from scratch (exact distributions + fitting + experiments + plots)
./pipeline.sh --from-scratch

# Use compiled C binary for exact distributions (much faster)
./pipeline.sh --from-scratch --use-c

# Regenerate plots from existing data only
./pipeline.sh --plots-only
```

### Docker

```bash
# Build image
./build.sh

# Run full pipeline in container (results persist on host via bind mounts)
./pipeline_docker.sh --from-scratch --use-c
```

## Key Results

- The **beta distribution** is the best continuous approximation for DTV (median p-value error 0.000116 across 249,001 (N,n) pairs)
- **MC+gamma** approximation with K=50,000 Monte Carlo samples provides conservative p-values suitable for hypothesis testing
- CT **outperforms** chi-squared and G-test for **comb-like (alternating) deviations** (up to 67% higher power at n=256)
- CT has **no advantage** for smooth or monotonic deviations (by design)

## Documentation

| File | Contents |
|------|----------|
| [code/README.md](code/README.md) | Script reference |
| [code/README_ct_save.md](code/README_ct_save.md) | C implementation: compilation, flags, checkpoint/resume |
| [code/fitting.md](code/fitting.md) | Distribution fitting methodology |
| [data.md](data.md) | Data directory formats and creation scripts |
| [convergence.md](convergence.md) | MC convergence experiment methodology and results |

## Dependencies

**Python:** numpy, scipy, matplotlib, tqdm, opencv-python-headless (see `requirements.txt`)

**C compilation:** GCC, GMP (libgmp-dev), OpenMP (libomp-dev). Optional: jemalloc for faster multi-threaded runs.

```bash
# Ubuntu/Debian
sudo apt install gcc libgmp-dev libomp-dev

# Compile
cd code && make ct_save
```

## Citation

If you use this code, please cite:

> N. Banić and N. Elezović, "Comb Test: Histogram Uniformity Testing Based on Discrete Total Variation," arXiv:2606.01465, 2026. https://arxiv.org/abs/2606.01465

```bibtex
@article{banic2026combtest,
  title   = {Comb Test: Histogram Uniformity Testing Based on Discrete Total Variation},
  author  = {Bani\'c, Nikola and Elezovi\'c, Neven},
  journal = {arXiv preprint arXiv:2606.01465},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.01465}
}
```

## License

MIT
