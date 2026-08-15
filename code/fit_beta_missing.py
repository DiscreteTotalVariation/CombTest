#!/usr/bin/env python3
"""Fit beta distribution for all (N,n) pairs missing from fitted_beta/, using multiprocessing."""

import os
import sys
import multiprocessing as mp
from fit import load_distribution, load_cdf, fit_distribution


def fit_one(fname):
    exact_path = os.path.join('../data/exact_distributions', fname)
    cdf_path = os.path.join('../data/cdf_exact', fname)
    out_path = os.path.join('../data/fitted/cvm_beta', fname)

    try:
        values, counts = load_distribution(exact_path)
        if not values:
            return fname, 'empty'
        cdf_values, cdf_probs = load_cdf(cdf_path)
        params, cvm = fit_distribution('beta', values, counts, cdf_values, cdf_probs)
        with open(out_path, 'w') as f:
            f.write(' '.join(map(str, params)) + '\n')
            f.write(str(cvm) + '\n')
        return fname, 'ok'
    except Exception as e:
        return fname, str(e)


def main():
    cdf_files = set(os.listdir('../data/cdf_exact'))
    beta_files = set(os.listdir('../data/fitted/cvm_beta'))
    missing = sorted(cdf_files - beta_files)
    # Only keep those that also have exact_distributions
    missing = [f for f in missing if os.path.isfile(os.path.join('../data/exact_distributions', f))]
    print(f"Missing beta fits: {len(missing)}")

    if not missing:
        print("Nothing to do.")
        return

    n_workers = max(1, mp.cpu_count() - 2)
    print(f"Using {n_workers} workers")

    done = 0
    errors = 0
    with mp.Pool(n_workers) as pool:
        for fname, status in pool.imap_unordered(fit_one, missing, chunksize=64):
            done += 1
            if status != 'ok':
                errors += 1
            if done % 5000 == 0:
                print(f"  {done}/{len(missing)} done ({errors} errors)")

    print(f"Finished: {done} total, {errors} errors")


if __name__ == '__main__':
    main()
