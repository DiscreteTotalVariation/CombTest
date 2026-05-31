import argparse
import os
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm

SUPPORTED_DISTRIBUTIONS = [
    'normal', 'lognormal', 'gamma', 'beta', 'poisson',
    'binomial', 'exponential', 'weibull', 'chi2',
    'nbinom', 'uniform', 'invgauss', 'gengamma', 'nakagami',
    'triang', 'betaprime', 'johnsonsb', 'arcsine', 'powerlaw',
    'bradford', 'burr', 'burr12', 'fisk', 'genpareto',
    'genextreme', 'rayleigh', 'rice', 'pareto', 'f'
]


def load_distribution(path):
    values, counts = [], []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                values.append(int(parts[0]))
                counts.append(int(parts[1]))
    return values, counts


def load_cdf(path):
    values, cdfs = [], []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                values.append(int(parts[0]))
                cdfs.append(float(parts[1]))
    return np.array(values), np.array(cdfs)


def compute_moments(values, counts):
    from fractions import Fraction
    total = sum(counts)
    if total == 0:
        return 0.0, 0.0
    mean_frac = Fraction(sum(v * c for v, c in zip(values, counts)), total)
    var_frac = Fraction(sum(c * (v - mean_frac) ** 2 for v, c in zip(values, counts)), total)
    return float(mean_frac), float(var_frac)


def compute_cvm(empirical_cdf, theoretical_cdf):
    return np.sum((empirical_cdf - theoretical_cdf) ** 2)


def _multi_start_beta(eval_points, cdf_probs, init_a, init_b, max_val, n_restarts=5):
    """Fit beta distribution with multiple starting points, return best result.

    Uses the moment-based initialization plus random perturbations to avoid
    local minima in the CvM objective.
    """
    eps = 1e-10

    def cvm_objective(log_params):
        a = np.exp(log_params[0])
        b = np.exp(log_params[1])
        sc = np.exp(log_params[2])
        theo = stats.beta.cdf(eval_points, a=a, b=b, loc=0, scale=sc)
        return np.sum((cdf_probs - theo) ** 2)

    x0_primary = [np.log(init_a), np.log(init_b), np.log(max_val)]

    # Collect all starting points: primary + random perturbations
    starts = [x0_primary]
    rng = np.random.RandomState(int(max_val * 1000 + init_a * 100) % (2**31))
    for _ in range(n_restarts):
        perturbation = rng.normal(0, 0.5, size=3)
        starts.append([x0_primary[j] + perturbation[j] for j in range(3)])

    best_result = None
    best_cvm = np.inf
    for x0 in starts:
        result = minimize(cvm_objective, x0=x0, method='Nelder-Mead',
                          options={'maxiter': 10000})
        if result.fun < best_cvm:
            best_cvm = result.fun
            best_result = result

    a = np.exp(best_result.x[0])
    b = np.exp(best_result.x[1])
    scale = np.exp(best_result.x[2])
    params = (a, b, 0.0, scale)
    theo_cdf = stats.beta.cdf(eval_points, a=a, b=b, loc=0, scale=scale)
    return params, theo_cdf, best_cvm


def fit_distribution(dist_name, values, counts, cdf_values, cdf_probs,
                     n_restarts=5):
    mean, var = compute_moments(values, counts)
    std = np.sqrt(var) if var > 0 else 1e-10
    eps = 1e-10

    eval_points = cdf_values

    if dist_name == 'normal':
        def cvm_objective(opt_params):
            loc = opt_params[0]
            sc = np.exp(opt_params[1])
            theo = stats.norm.cdf(eval_points, loc=loc, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective, x0=[mean, np.log(max(std, eps))],
                          method='Nelder-Mead')
        loc = result.x[0]
        scale = np.exp(result.x[1])
        params = (loc, scale)
        theo_cdf = stats.norm.cdf(eval_points, loc=loc, scale=scale)

    elif dist_name == 'lognormal':
        m = max(mean, eps)
        sigma_sq = np.log(1 + var / m ** 2)
        sigma = np.sqrt(sigma_sq)
        mu = np.log(m) - sigma_sq / 2

        def cvm_objective(opt_params):
            s = np.exp(opt_params[0])
            sc = np.exp(opt_params[1])
            theo = stats.lognorm.cdf(eval_points, s=s, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective,
                          x0=[np.log(max(sigma, eps)), mu],
                          method='Nelder-Mead')
        sigma_opt = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (sigma_opt, 0.0, scale)
        theo_cdf = stats.lognorm.cdf(eval_points, s=sigma_opt, loc=0, scale=scale)

    elif dist_name == 'gamma':
        m = max(mean, eps)
        v = max(var, eps)
        init_scale = v / m
        init_shape = m / init_scale

        def cvm_objective(log_params):
            a = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            theo = stats.gamma.cdf(eval_points, a=a, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        x0_primary = [np.log(max(init_shape, eps)), np.log(max(init_scale, eps))]
        starts = [x0_primary]
        rng = np.random.RandomState(int(m * 1000 + v * 100) % (2**31))
        for _ in range(n_restarts):
            perturbation = rng.normal(0, 0.5, size=2)
            starts.append([x0_primary[j] + perturbation[j] for j in range(2)])

        best_result = None
        best_cvm = np.inf
        for x0 in starts:
            result = minimize(cvm_objective, x0=x0, method='Nelder-Mead',
                              options={'maxiter': 10000})
            if result.fun < best_cvm:
                best_cvm = result.fun
                best_result = result

        shape = np.exp(best_result.x[0])
        scale = np.exp(best_result.x[1])
        params = (shape, 0.0, scale)
        theo_cdf = stats.gamma.cdf(eval_points, a=shape, loc=0, scale=scale)

    elif dist_name == 'beta':
        max_val = float(max(values))
        if max_val <= 0:
            max_val = 1.0
        m = np.clip(mean / max_val, eps, 1 - eps)
        v = var / max_val ** 2
        if v <= 0 or v >= m * (1 - m):
            v = m * (1 - m) / 2
        sum_ab = m * (1 - m) / v - 1
        sum_ab = max(sum_ab, eps)
        init_a = max(m * sum_ab, eps)
        init_b = max((1 - m) * sum_ab, eps)

        params, theo_cdf, _ = _multi_start_beta(
            eval_points, cdf_probs, init_a, init_b, max_val,
            n_restarts=n_restarts)

    elif dist_name == 'poisson':
        init_mu = max(mean, eps)

        def cvm_objective(log_mu):
            mu = np.exp(log_mu[0])
            theo = stats.poisson.cdf(eval_points, mu=mu)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective, x0=[np.log(init_mu)],
                          method='Nelder-Mead')
        mu = np.exp(result.x[0])
        params = (mu,)
        theo_cdf = stats.poisson.cdf(eval_points, mu=mu)

    elif dist_name == 'binomial':
        max_val = max(values)
        if mean <= 0:
            n = max(max_val, 1)
            p = 0.5
        else:
            if var >= mean:
                p = max(1 - eps, mean / (mean + 1))
            else:
                p = np.clip(1 - var / mean, eps, 1 - eps)
            n = max(int(round(mean / p)), max_val, 1)

        def cvm_objective(opt_params):
            p_opt = 1.0 / (1.0 + np.exp(-opt_params[0]))
            theo = stats.binom.cdf(eval_points, n=n, p=p_opt)
            return np.sum((cdf_probs - theo) ** 2)

        init_logit_p = np.log(max(p, eps) / max(1 - p, eps))
        result = minimize(cvm_objective, x0=[init_logit_p],
                          method='Nelder-Mead')
        p = 1.0 / (1.0 + np.exp(-result.x[0]))
        params = (n, p)
        theo_cdf = stats.binom.cdf(eval_points, n=n, p=p)

    elif dist_name == 'exponential':
        def cvm_objective(log_scale):
            sc = np.exp(log_scale[0])
            theo = stats.expon.cdf(eval_points, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective, x0=[np.log(max(mean, eps))],
                          method='Nelder-Mead')
        scale = np.exp(result.x[0])
        params = (0.0, scale)
        theo_cdf = stats.expon.cdf(eval_points, loc=0, scale=scale)

    elif dist_name == 'weibull':
        def cvm_objective(log_params):
            c = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            theo = stats.weibull_min.cdf(eval_points, c=c, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        init_scale = max(mean, eps)
        result = minimize(cvm_objective, x0=[0.0, np.log(init_scale)],
                          method='Nelder-Mead')
        c = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (c, 0.0, scale)
        theo_cdf = stats.weibull_min.cdf(eval_points, c=c, loc=0, scale=scale)

    elif dist_name == 'chi2':
        init_df = max(mean, eps)

        def cvm_objective(log_params):
            df = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            theo = stats.chi2.cdf(eval_points, df=df, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective,
                          x0=[np.log(init_df), 0.0],
                          method='Nelder-Mead')
        df = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (df, 0.0, scale)
        theo_cdf = stats.chi2.cdf(eval_points, df=df, loc=0, scale=scale)

    elif dist_name == 'nbinom':
        if var <= mean:
            init_p = 1 - eps
            init_n = max(mean * init_p / (1 - init_p), eps)
        else:
            init_p = np.clip(mean / var, eps, 1 - eps)
            init_n = max(mean * init_p / (1 - init_p), eps)

        def cvm_objective(opt_params):
            n_param = np.exp(opt_params[0])
            p_param = 1.0 / (1.0 + np.exp(-opt_params[1]))
            theo = stats.nbinom.cdf(eval_points, n=n_param, p=p_param)
            return np.sum((cdf_probs - theo) ** 2)

        init_logit_p = np.log(max(init_p, eps) / max(1 - init_p, eps))
        result = minimize(cvm_objective,
                          x0=[np.log(max(init_n, eps)), init_logit_p],
                          method='Nelder-Mead')
        n_param = np.exp(result.x[0])
        p_param = 1.0 / (1.0 + np.exp(-result.x[1]))
        params = (n_param, p_param)
        theo_cdf = stats.nbinom.cdf(eval_points, n=n_param, p=p_param)

    elif dist_name == 'uniform':
        init_a = float(min(values))
        init_width = max(float(max(values)) - init_a, eps)

        def cvm_objective(opt_params):
            a = opt_params[0]
            w = np.exp(opt_params[1])
            theo = stats.uniform.cdf(eval_points, loc=a, scale=w)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective,
                          x0=[init_a, np.log(init_width)],
                          method='Nelder-Mead')
        a = result.x[0]
        width = np.exp(result.x[1])
        params = (a, width)
        theo_cdf = stats.uniform.cdf(eval_points, loc=a, scale=width)

    elif dist_name == 'invgauss':
        m = max(mean, eps)
        v = max(var, eps)
        init_scale = m ** 3 / v
        init_mu = m / init_scale

        def cvm_objective(log_params):
            mu_p = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            theo = stats.invgauss.cdf(eval_points, mu=mu_p, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective,
                          x0=[np.log(max(init_mu, eps)), np.log(max(init_scale, eps))],
                          method='Nelder-Mead')
        mu_param = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (mu_param, 0.0, scale)
        theo_cdf = stats.invgauss.cdf(eval_points, mu=mu_param, loc=0, scale=scale)

    elif dist_name == 'gengamma':
        def cvm_objective(log_params):
            a_p = np.exp(log_params[0])
            c_p = np.exp(log_params[1])
            sc = np.exp(log_params[2])
            theo = stats.gengamma.cdf(eval_points, a=a_p, c=c_p, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        m = max(mean, eps)
        v = max(var, eps)
        init_scale = v / m
        init_shape = m / init_scale
        result = minimize(cvm_objective,
                          x0=[np.log(max(init_shape, eps)), 0.0,
                              np.log(max(init_scale, eps))],
                          method='Nelder-Mead')
        a_p = np.exp(result.x[0])
        c_p = np.exp(result.x[1])
        scale = np.exp(result.x[2])
        params = (a_p, c_p, 0.0, scale)
        theo_cdf = stats.gengamma.cdf(eval_points, a=a_p, c=c_p, loc=0, scale=scale)

    elif dist_name == 'nakagami':
        m = max(mean, eps)
        v = max(var, eps)
        omega = m ** 2 + v

        def cvm_objective(log_nu):
            nu = np.exp(log_nu[0])
            sc = np.sqrt(omega)
            theo = stats.nakagami.cdf(eval_points, nu=nu, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        init_nu = max(m ** 2 / v, eps)
        result = minimize(cvm_objective, x0=[np.log(init_nu)],
                          method='Nelder-Mead')
        nu = np.exp(result.x[0])
        scale = np.sqrt(omega)
        params = (nu, 0.0, scale)
        theo_cdf = stats.nakagami.cdf(eval_points, nu=nu, loc=0, scale=scale)

    elif dist_name == 'triang':
        min_val = float(min(values))
        max_val = float(max(values))
        width = max(max_val - min_val, eps)

        def cvm_objective(params_opt):
            c = 1.0 / (1.0 + np.exp(-params_opt[0]))
            theo = stats.triang.cdf(eval_points, c=c, loc=min_val, scale=width)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective, x0=[0.0], method='Nelder-Mead')
        c = 1.0 / (1.0 + np.exp(-result.x[0]))
        params = (c, min_val, width)
        theo_cdf = stats.triang.cdf(eval_points, c=c, loc=min_val, scale=width)

    elif dist_name == 'betaprime':
        def cvm_objective(log_params):
            a_p = np.exp(log_params[0])
            b_p = np.exp(log_params[1])
            sc = np.exp(log_params[2])
            theo = stats.betaprime.cdf(eval_points, a=a_p, b=b_p, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        m = max(mean, eps)
        result = minimize(cvm_objective,
                          x0=[0.0, np.log(2.0), np.log(max(m, eps))],
                          method='Nelder-Mead')
        a_p = np.exp(result.x[0])
        b_p = np.exp(result.x[1])
        scale = np.exp(result.x[2])
        params = (a_p, b_p, 0.0, scale)
        theo_cdf = stats.betaprime.cdf(eval_points, a=a_p, b=b_p, loc=0, scale=scale)

    elif dist_name == 'johnsonsb':
        def cvm_objective(opt_params):
            a_p = opt_params[0]
            b_p = np.exp(opt_params[1])
            loc_p = opt_params[2]
            sc = np.exp(opt_params[3])
            theo = stats.johnsonsb.cdf(eval_points, a=a_p, b=b_p, loc=loc_p, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        min_val = float(min(values))
        max_val = float(max(values))
        width = max(max_val - min_val, eps)
        result = minimize(cvm_objective,
                          x0=[0.0, 0.0, min_val, np.log(width)],
                          method='Nelder-Mead')
        a_p = result.x[0]
        b_p = np.exp(result.x[1])
        loc_p = result.x[2]
        scale = np.exp(result.x[3])
        params = (a_p, b_p, loc_p, scale)
        theo_cdf = stats.johnsonsb.cdf(eval_points, a=a_p, b=b_p, loc=loc_p, scale=scale)

    elif dist_name == 'arcsine':
        init_loc = float(min(values))
        init_width = max(float(max(values)) - init_loc, eps)

        def cvm_objective(opt_params):
            loc = opt_params[0]
            sc = np.exp(opt_params[1])
            theo = stats.arcsine.cdf(eval_points, loc=loc, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective,
                          x0=[init_loc, np.log(init_width)],
                          method='Nelder-Mead')
        loc = result.x[0]
        width = np.exp(result.x[1])
        params = (loc, width)
        theo_cdf = stats.arcsine.cdf(eval_points, loc=loc, scale=width)

    elif dist_name == 'powerlaw':
        min_val = float(min(values))
        max_val = float(max(values))
        width = max(max_val - min_val, eps)

        def cvm_objective(log_a):
            a_p = np.exp(log_a[0])
            theo = stats.powerlaw.cdf(eval_points, a=a_p, loc=min_val, scale=width)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective, x0=[0.0], method='Nelder-Mead')
        a_p = np.exp(result.x[0])
        params = (a_p, min_val, width)
        theo_cdf = stats.powerlaw.cdf(eval_points, a=a_p, loc=min_val, scale=width)

    elif dist_name == 'bradford':
        min_val = float(min(values))
        max_val = float(max(values))
        width = max(max_val - min_val, eps)

        def cvm_objective(log_c):
            c_p = np.exp(log_c[0])
            theo = stats.bradford.cdf(eval_points, c=c_p, loc=min_val, scale=width)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective, x0=[0.0], method='Nelder-Mead')
        c_p = np.exp(result.x[0])
        params = (c_p, min_val, width)
        theo_cdf = stats.bradford.cdf(eval_points, c=c_p, loc=min_val, scale=width)

    elif dist_name == 'burr':
        def cvm_objective(log_params):
            c_p = np.exp(log_params[0])
            d_p = np.exp(log_params[1])
            sc = np.exp(log_params[2])
            theo = stats.burr.cdf(eval_points, c=c_p, d=d_p, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        m = max(mean, eps)
        result = minimize(cvm_objective,
                          x0=[0.0, 0.0, np.log(m)],
                          method='Nelder-Mead')
        c_p = np.exp(result.x[0])
        d_p = np.exp(result.x[1])
        scale = np.exp(result.x[2])
        params = (c_p, d_p, 0.0, scale)
        theo_cdf = stats.burr.cdf(eval_points, c=c_p, d=d_p, loc=0, scale=scale)

    elif dist_name == 'burr12':
        def cvm_objective(log_params):
            c_p = np.exp(log_params[0])
            d_p = np.exp(log_params[1])
            sc = np.exp(log_params[2])
            theo = stats.burr12.cdf(eval_points, c=c_p, d=d_p, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        m = max(mean, eps)
        result = minimize(cvm_objective,
                          x0=[0.0, 0.0, np.log(m)],
                          method='Nelder-Mead')
        c_p = np.exp(result.x[0])
        d_p = np.exp(result.x[1])
        scale = np.exp(result.x[2])
        params = (c_p, d_p, 0.0, scale)
        theo_cdf = stats.burr12.cdf(eval_points, c=c_p, d=d_p, loc=0, scale=scale)

    elif dist_name == 'fisk':
        def cvm_objective(log_params):
            c_p = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            theo = stats.fisk.cdf(eval_points, c=c_p, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        m = max(mean, eps)
        result = minimize(cvm_objective,
                          x0=[0.0, np.log(m)],
                          method='Nelder-Mead')
        c_p = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (c_p, 0.0, scale)
        theo_cdf = stats.fisk.cdf(eval_points, c=c_p, loc=0, scale=scale)

    elif dist_name == 'genpareto':
        def cvm_objective(opt_params):
            c_p = opt_params[0]
            sc = np.exp(opt_params[1])
            theo = stats.genpareto.cdf(eval_points, c=c_p, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        m = max(mean, eps)
        result = minimize(cvm_objective,
                          x0=[0.0, np.log(m)],
                          method='Nelder-Mead')
        c_p = result.x[0]
        scale = np.exp(result.x[1])
        params = (c_p, 0.0, scale)
        theo_cdf = stats.genpareto.cdf(eval_points, c=c_p, loc=0, scale=scale)

    elif dist_name == 'genextreme':
        def cvm_objective(opt_params):
            c_p = opt_params[0]
            loc_p = opt_params[1]
            sc = np.exp(opt_params[2])
            theo = stats.genextreme.cdf(eval_points, c=c_p, loc=loc_p, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        m = max(mean, eps)
        s = max(std, eps)
        result = minimize(cvm_objective,
                          x0=[0.0, m, np.log(s)],
                          method='Nelder-Mead')
        c_p = result.x[0]
        loc_p = result.x[1]
        scale = np.exp(result.x[2])
        params = (c_p, loc_p, scale)
        theo_cdf = stats.genextreme.cdf(eval_points, c=c_p, loc=loc_p, scale=scale)

    elif dist_name == 'rayleigh':
        m = max(mean, eps)
        init_scale = m / np.sqrt(np.pi / 2)

        def cvm_objective(log_scale):
            sc = np.exp(log_scale[0])
            theo = stats.rayleigh.cdf(eval_points, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        result = minimize(cvm_objective, x0=[np.log(max(init_scale, eps))],
                          method='Nelder-Mead')
        scale = np.exp(result.x[0])
        params = (0.0, scale)
        theo_cdf = stats.rayleigh.cdf(eval_points, loc=0, scale=scale)

    elif dist_name == 'rice':
        def cvm_objective(log_params):
            b_p = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            theo = stats.rice.cdf(eval_points, b=b_p, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        m = max(mean, eps)
        s = max(std, eps)
        result = minimize(cvm_objective,
                          x0=[0.0, np.log(s)],
                          method='Nelder-Mead')
        b_p = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (b_p, 0.0, scale)
        theo_cdf = stats.rice.cdf(eval_points, b=b_p, loc=0, scale=scale)

    elif dist_name == 'pareto':
        def cvm_objective(log_params):
            b_p = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            theo = stats.pareto.cdf(eval_points, b=b_p, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        m = max(mean, eps)
        result = minimize(cvm_objective,
                          x0=[0.0, np.log(m)],
                          method='Nelder-Mead')
        b_p = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (b_p, 0.0, scale)
        theo_cdf = stats.pareto.cdf(eval_points, b=b_p, loc=0, scale=scale)

    elif dist_name == 'f':
        def cvm_objective(log_params):
            dfn = np.exp(log_params[0])
            dfd = np.exp(log_params[1])
            sc = np.exp(log_params[2])
            theo = stats.f.cdf(eval_points, dfn=dfn, dfd=dfd, loc=0, scale=sc)
            return np.sum((cdf_probs - theo) ** 2)

        m = max(mean, eps)
        result = minimize(cvm_objective,
                          x0=[0.0, np.log(2.0), np.log(m)],
                          method='Nelder-Mead')
        dfn = np.exp(result.x[0])
        dfd = np.exp(result.x[1])
        scale = np.exp(result.x[2])
        params = (dfn, dfd, 0.0, scale)
        theo_cdf = stats.f.cdf(eval_points, dfn=dfn, dfd=dfd, loc=0, scale=scale)

    else:
        raise ValueError("Unsupported distribution: " + dist_name)

    cvm_stat = compute_cvm(cdf_probs, theo_cdf)
    return params, cvm_stat


def _fit_and_save(task):
    """Fit a single file and save results. Module-level for pickling."""
    name, exact_dir, cdf_dir, output_dir, dist_name, n_restarts = task
    exact_path = os.path.join(exact_dir, name)
    cdf_path = os.path.join(cdf_dir, name)
    values, counts = load_distribution(exact_path)
    cdf_values, cdf_probs = load_cdf(cdf_path)
    params, cvm_stat = fit_distribution(
        dist_name, values, counts, cdf_values, cdf_probs,
        n_restarts=n_restarts)
    output_path = os.path.join(output_dir, name)
    with open(output_path, "w", encoding="utf8") as fo:
        fo.write(" ".join(map(str, params)) + "\n")
        fo.write(str(cvm_stat) + "\n")
    return name


def main():

    parser = argparse.ArgumentParser(
        description="Fit distributions to exact distribution data and compute CvM statistic")
    parser.add_argument("--exact-dir", required=True,
                        help="Directory with exact distribution files (value count per line)")
    parser.add_argument("--cdf-dir", required=True,
                        help="Directory with CDF value files (value cdf per line)")
    parser.add_argument("--distribution", required=True, choices=SUPPORTED_DISTRIBUTIONS,
                        help="Distribution to fit")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory for fitted parameters and CvM statistics")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show progress with tqdm")
    parser.add_argument("-w", "--overwrite", action="store_true",
                        help="Overwrite existing output files (default: skip them)")
    parser.add_argument("--restarts", type=int, default=5,
                        help="Number of random restarts for multi-start optimization (default: 5)")
    parser.add_argument("-p", "--processes", type=int, default=1,
                        help="Number of parallel processes to use (default: 1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    names = sorted(os.listdir(args.exact_dir))
    names = [n for n in names
             if os.path.isfile(os.path.join(args.exact_dir, n))
             and os.path.isfile(os.path.join(args.cdf_dir, n))]

    if not args.overwrite:
        before = len(names)
        names = [n for n in names
                 if not os.path.isfile(os.path.join(args.output_dir, n))]
        skipped = before - len(names)
        if skipped > 0:
            print("Skipping %d already fitted files (use -w to overwrite)" % skipped)

    tasks = [(name, args.exact_dir, args.cdf_dir, args.output_dir,
              args.distribution, args.restarts) for name in names]

    if args.processes == 1:
        iterator = tqdm(tasks) if args.verbose else tasks
        for task in iterator:
            _fit_and_save(task)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        pbar = tqdm(total=len(tasks)) if args.verbose else None
        with ProcessPoolExecutor(max_workers=args.processes) as executor:
            futures = {executor.submit(_fit_and_save, task): task[0] for task in tasks}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Error fitting {name}: {exc}")
                if pbar:
                    pbar.update(1)
        if pbar:
            pbar.close()


if __name__ == "__main__":
    main()
