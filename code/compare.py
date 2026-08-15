import argparse
import os
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from fit import load_distribution, load_cdf, fit_distribution, SUPPORTED_DISTRIBUTIONS


DISCRETE_DISTS = {'poisson', 'binomial', 'nbinom'}

DIST_MAP = {
    'normal':     lambda p: stats.norm(loc=p[0], scale=p[1]),
    'lognormal':  lambda p: stats.lognorm(s=p[0], loc=p[1], scale=p[2]),
    'gamma':      lambda p: stats.gamma(a=p[0], loc=p[1], scale=p[2]),
    'beta':       lambda p: stats.beta(a=p[0], b=p[1], loc=p[2], scale=p[3]),
    'poisson':    lambda p: stats.poisson(mu=p[0]),
    'binomial':   lambda p: stats.binom(n=int(round(p[0])), p=p[1]),
    'exponential':lambda p: stats.expon(loc=p[0], scale=p[1]),
    'weibull':    lambda p: stats.weibull_min(c=p[0], loc=p[1], scale=p[2]),
    'chi2':       lambda p: stats.chi2(df=p[0], loc=p[1], scale=p[2]),
    'nbinom':     lambda p: stats.nbinom(n=p[0], p=p[1]),
    'uniform':    lambda p: stats.uniform(loc=p[0], scale=p[1]),
    'invgauss':   lambda p: stats.invgauss(mu=p[0], loc=p[1], scale=p[2]),
    'gengamma':   lambda p: stats.gengamma(a=p[0], c=p[1], loc=p[2], scale=p[3]),
    'nakagami':   lambda p: stats.nakagami(nu=p[0], loc=p[1], scale=p[2]),
    'triang':     lambda p: stats.triang(c=p[0], loc=p[1], scale=p[2]),
    'betaprime':  lambda p: stats.betaprime(a=p[0], b=p[1], loc=p[2], scale=p[3]),
    'johnsonsb':  lambda p: stats.johnsonsb(a=p[0], b=p[1], loc=p[2], scale=p[3]),
    'arcsine':    lambda p: stats.arcsine(loc=p[0], scale=p[1]),
    'powerlaw':   lambda p: stats.powerlaw(a=p[0], loc=p[1], scale=p[2]),
    'bradford':   lambda p: stats.bradford(c=p[0], loc=p[1], scale=p[2]),
    'burr':       lambda p: stats.burr(c=p[0], d=p[1], loc=p[2], scale=p[3]),
    'burr12':     lambda p: stats.burr12(c=p[0], d=p[1], loc=p[2], scale=p[3]),
    'fisk':       lambda p: stats.fisk(c=p[0], loc=p[1], scale=p[2]),
    'genpareto':  lambda p: stats.genpareto(c=p[0], loc=p[1], scale=p[2]),
    'genextreme': lambda p: stats.genextreme(c=p[0], loc=p[1], scale=p[2]),
    'rayleigh':   lambda p: stats.rayleigh(loc=p[0], scale=p[1]),
    'rice':       lambda p: stats.rice(b=p[0], loc=p[1], scale=p[2]),
    'pareto':     lambda p: stats.pareto(b=p[0], loc=p[1], scale=p[2]),
    'f':          lambda p: stats.f(dfn=p[0], dfd=p[1], loc=p[2], scale=p[3]),
}


def load_fitted_params(path):
    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
    params = tuple(map(float, lines[0].strip().split()))
    cvm = float(lines[1].strip())
    return params, cvm


def get_params(dist_name, N, n, force_fit, mle=False):
    fname = f"N_{N}_n_{n}.txt"
    prefix = "../data/fitted/mle_" if mle else "../data/fitted/cvm_"
    fitted_dir = f"{prefix}{dist_name}"
    fitted_path = os.path.join(fitted_dir, fname)

    if not force_fit and os.path.isdir(fitted_dir) and os.path.isfile(fitted_path):
        params, cvm = load_fitted_params(fitted_path)
        return params, cvm

    exact_path = os.path.join("../data/exact_distributions", fname)
    cdf_path = os.path.join("../data/cdf_exact", fname)
    if not os.path.isfile(exact_path):
        print(f"Error: exact distribution file not found: {exact_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(cdf_path):
        print(f"Error: CDF file not found: {cdf_path}", file=sys.stderr)
        sys.exit(1)

    values, counts = load_distribution(exact_path)
    cdf_values, cdf_probs = load_cdf(cdf_path)
    params, cvm = fit_distribution(dist_name, values, counts, cdf_values, cdf_probs)
    return params, cvm


def main():
    parser = argparse.ArgumentParser(
        description="Compare exact DTV distribution with fitted approximation(s)")
    parser.add_argument('-N', type=int, required=True, help="N parameter")
    parser.add_argument('-n', type=int, required=True, help="n parameter")
    parser.add_argument('-d', '--distribution', action='append', required=True,
                        choices=SUPPORTED_DISTRIBUTIONS,
                        help="Distribution(s) to compare (can be specified multiple times)")
    parser.add_argument('-f', '--fit', action='store_true',
                        help="Force re-fitting even if fitted parameters exist")
    parser.add_argument('-o', '--output', default=None,
                        help="Output image path")
    parser.add_argument('--mle', action='store_true',
                        help="Use MLE-fitted parameters (fitted/mle_* dirs)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--cdf', action='store_true', help="Plot CDF (default is PDF/PMF)")
    mode.add_argument('--pdf', action='store_true', help="Plot PDF/PMF (default)")
    args = parser.parse_args()

    plot_cdf = args.cdf

    fname = f"N_{args.N}_n_{args.n}.txt"
    exact_path = os.path.join("../data/exact_distributions", fname)
    if not os.path.isfile(exact_path):
        print(f"Error: exact distribution file not found: {exact_path}", file=sys.stderr)
        sys.exit(1)

    values, counts = load_distribution(exact_path)
    values = np.array(values, dtype=float)
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    pmf = counts / total

    fig, ax = plt.subplots(figsize=(12, 7))

    x_min, x_max = values.min(), values.max()

    if plot_cdf:
        cdf = np.cumsum(pmf)
        ax.step(values, cdf, where='post', linewidth=2, color='steelblue',
                label='Exact DTV')

        x_smooth = np.linspace(max(x_min - 1, 0), x_max + 1, 1000)
        for dist_name in args.distribution:
            params, cvm = get_params(dist_name, args.N, args.n, args.fit, args.mle)
            dist_obj = DIST_MAP[dist_name](params)
            theo_cdf = dist_obj.cdf(x_smooth)
            ax.plot(x_smooth, theo_cdf, linewidth=2,
                    label=f'{dist_name} (CvM={cvm:.6g})')

        ax.set_ylabel('CDF')
    else:
        ax.bar(values, pmf, width=0.8, alpha=0.5, color='steelblue', label='Exact DTV')

        x_smooth = np.linspace(max(x_min - 1, 0), x_max + 1, 1000)
        for dist_name in args.distribution:
            params, cvm = get_params(dist_name, args.N, args.n, args.fit, args.mle)
            dist_obj = DIST_MAP[dist_name](params)

            if dist_name in DISCRETE_DISTS:
                x_int = np.arange(int(x_min), int(x_max) + 1)
                theo_pmf = dist_obj.pmf(x_int)
                ax.plot(x_int, theo_pmf, 'o-', markersize=4,
                        label=f'{dist_name} (CvM={cvm:.6g})')
            else:
                theo_pdf = dist_obj.pdf(x_smooth)
                ax.plot(x_smooth, theo_pdf, linewidth=2,
                        label=f'{dist_name} (CvM={cvm:.6g})')

        ax.set_ylabel('Probability')

    ax.set_xlabel('DTV value')
    ax.set_title(f'DTV Distribution: N={args.N}, n={args.n}')
    ax.legend()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.output}")

    plt.show()


if __name__ == "__main__":
    main()
