# ct_save.c - Fast Exact DTV Distribution Computation

C implementation of the exact DTV distribution computation with checkpoint/resume support. This is the fast alternative to `ct.py` and is recommended for large-scale computation.

## Features

- **Arbitrary-precision integers** via GMP (GNU Multiple Precision Arithmetic Library)
- **Parallel computation** via OpenMP
- **Checkpoint/resume** — saves state on SIGINT/SIGTERM or at regular intervals
- **LPT work ordering** — largest-first scheduling for better load balance
- **Range tracking** — skips zero-valued regions of the DP table for efficiency
- **Skip existing** — with `-u`, skips (N,n) pairs that already have output files

## Compilation

```bash
gcc -O3 -fopenmp -o ct_save ct_save.c -lgmp -lm
```

Or using make (if Makefile is available):
```bash
make ct_save
```

## Usage

### Fresh run
```bash
./ct_save -N <N_max> -n <n_max> [options]
```

Computes exact DTV distributions for all (N, n) pairs from (2, 2) up to (N_max, n_max).

### Resume from checkpoint
```bash
./ct_save -c <checkpoint_file> [options]
```

Resumes computation from a saved checkpoint. The `-N`, `-n`, `-o`, `-u`, and `-l` values are loaded from the checkpoint file.

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-N <int>` | Maximum sample size N | (required) |
| `-n <int>` | Maximum number of bins n | (required) |
| `-o <dir>` | Output directory for distribution files | `../data/exact_distributions/` |
| `-u` | Skip (N,n) pairs that already have non-empty output files | off |
| `-l <int>` | Start computation from n = l_start (skip smaller n values) | 2 |
| `-t <int>` | Number of OpenMP threads | all available |
| `-s <path>` | Checkpoint file path for periodic saving | (none) |
| `-i <int>` | Save checkpoint every N iterations | signal-only |
| `-c <path>` | Resume from this checkpoint file | (none) |

## Examples

```bash
# Compute all (N,n) for N,n in [2,500], skip existing files
./ct_save -N 500 -n 500 -u -o ../data/exact_distributions

# Same but with checkpoint every 20 iterations
./ct_save -N 500 -n 500 -u -s ckpt.bin -i 20

# Resume from checkpoint
./ct_save -c ckpt.bin -s ckpt.bin -i 20

# Use 8 threads
./ct_save -N 500 -n 500 -u -t 8
```

## Output

Each computed pair produces a file `N_{N}_n_{n}.txt` in the output directory with two columns per line:
- Column 1: DTV value (integer, 0 to 2N)
- Column 2: Unnormalized count (arbitrary-precision integer)

## Signal Handling

- **First SIGINT/SIGTERM**: Saves checkpoint (if `-s` is set) and exits gracefully after the current (N,n) pair completes.
- **Second signal**: Terminates immediately.

## Memory

Approximately `2 * (N+1)^2 * (2N+2) * sizeof(mpz_t)` plus GMP's internal allocations. For N=500, this is roughly 2-4 GB.

## Other C Implementations

- `ct.c` — Basic version without checkpoint support
- `faster_ct.c` — Optimized version without checkpoint
- `even_faster_ct.c` — Further optimized version without checkpoint

All produce identical output. `ct_save` is recommended as it includes all optimizations plus checkpoint/resume.
