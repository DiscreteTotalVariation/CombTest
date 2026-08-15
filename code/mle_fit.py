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

DISCRETE_DISTS = {'poisson', 'binomial', 'nbinom'}


def load_distribution(path):
    values, counts = [], []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                values.append(int(parts[0]))
                counts.append(int(parts[1]))
    return values, counts


def compute_moments(values, counts):
    from fractions import Fraction
    total = sum(counts)
    if total == 0:
        return 0.0, 0.0
    mean_frac = Fraction(sum(v * c for v, c in zip(values, counts)), total)
    var_frac = Fraction(sum(c * (v - mean_frac) ** 2 for v, c in zip(values, counts)), total)
    return float(mean_frac), float(var_frac)


def safe_nll(log_prob_values, probs):
    """Cross-entropy H(p, q) = -sum p_i * log q_i, using pre-computed log q_i."""
    log_prob_values = np.where(np.isfinite(log_prob_values), log_prob_values, -700)
    return -np.sum(probs * log_prob_values)


def mle_fit_distribution(dist_name, values, counts):
    mean, var = compute_moments(values, counts)
    std = np.sqrt(var) if var > 0 else 1e-10
    eps = 1e-10

    # Counts can be huge arbitrary-precision ints; normalize via Fraction
    from fractions import Fraction
    total_int = sum(counts)
    ev_list, ep_list = [], []
    for v, c in zip(values, counts):
        if c > 0:
            ev_list.append(float(v))
            ep_list.append(float(Fraction(c, total_int)))
    ev = np.array(ev_list)
    ep = np.array(ep_list)
    ep = ep / ep.sum()

    if dist_name == 'normal':
        def nll_obj(opt_params):
            loc = opt_params[0]
            sc = np.exp(opt_params[1])
            return safe_nll(stats.norm.logpdf(ev, loc=loc, scale=sc), ep)

        result = minimize(nll_obj, x0=[mean, np.log(max(std, eps))],
                          method='Nelder-Mead')
        loc = result.x[0]
        scale = np.exp(result.x[1])
        params = (loc, scale)
        nll = safe_nll(stats.norm.logpdf(ev, loc=loc, scale=scale), ep)

    elif dist_name == 'lognormal':
        m = max(mean, eps)
        sigma_sq = np.log(1 + var / m ** 2)
        sigma = np.sqrt(sigma_sq)
        mu = np.log(m) - sigma_sq / 2

        def nll_obj(opt_params):
            s = np.exp(opt_params[0])
            sc = np.exp(opt_params[1])
            return safe_nll(stats.lognorm.logpdf(ev, s=s, loc=0, scale=sc), ep)

        result = minimize(nll_obj,
                          x0=[np.log(max(sigma, eps)), mu],
                          method='Nelder-Mead')
        sigma_opt = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (sigma_opt, 0.0, scale)
        nll = safe_nll(stats.lognorm.logpdf(ev, s=sigma_opt, loc=0, scale=scale), ep)

    elif dist_name == 'gamma':
        m = max(mean, eps)
        v = max(var, eps)
        init_scale = v / m
        init_shape = m / init_scale

        def nll_obj(log_params):
            a = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            return safe_nll(stats.gamma.logpdf(ev, a=a, loc=0, scale=sc), ep)

        result = minimize(nll_obj,
                          x0=[np.log(max(init_shape, eps)), np.log(max(init_scale, eps))],
                          method='Nelder-Mead')
        shape = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (shape, 0.0, scale)
        nll = safe_nll(stats.gamma.logpdf(ev, a=shape, loc=0, scale=scale), ep)

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

        def nll_obj(log_params):
            a = np.exp(log_params[0])
            b = np.exp(log_params[1])
            sc = np.exp(log_params[2])
            return safe_nll(stats.beta.logpdf(ev, a=a, b=b, loc=0, scale=sc), ep)

        result = minimize(nll_obj,
                          x0=[np.log(init_a), np.log(init_b), np.log(max_val)],
                          method='Nelder-Mead')
        a = np.exp(result.x[0])
        b = np.exp(result.x[1])
        scale = np.exp(result.x[2])
        params = (a, b, 0.0, scale)
        nll = safe_nll(stats.beta.logpdf(ev, a=a, b=b, loc=0, scale=scale), ep)

    elif dist_name == 'poisson':
        init_mu = max(mean, eps)

        def nll_obj(log_mu):
            mu = np.exp(log_mu[0])
            return safe_nll(stats.poisson.logpmf(ev.astype(int), mu=mu), ep)

        result = minimize(nll_obj, x0=[np.log(init_mu)],
                          method='Nelder-Mead')
        mu = np.exp(result.x[0])
        params = (mu,)
        nll = safe_nll(stats.poisson.logpmf(ev.astype(int), mu=mu), ep)

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

        def nll_obj(opt_params):
            p_opt = 1.0 / (1.0 + np.exp(-opt_params[0]))
            return safe_nll(stats.binom.logpmf(ev.astype(int), n=n, p=p_opt), ep)

        init_logit_p = np.log(max(p, eps) / max(1 - p, eps))
        result = minimize(nll_obj, x0=[init_logit_p],
                          method='Nelder-Mead')
        p = 1.0 / (1.0 + np.exp(-result.x[0]))
        params = (n, p)
        nll = safe_nll(stats.binom.logpmf(ev.astype(int), n=n, p=p), ep)

    elif dist_name == 'exponential':
        def nll_obj(log_scale):
            sc = np.exp(log_scale[0])
            return safe_nll(stats.expon.logpdf(ev, loc=0, scale=sc), ep)

        result = minimize(nll_obj, x0=[np.log(max(mean, eps))],
                          method='Nelder-Mead')
        scale = np.exp(result.x[0])
        params = (0.0, scale)
        nll = safe_nll(stats.expon.logpdf(ev, loc=0, scale=scale), ep)

    elif dist_name == 'weibull':
        def nll_obj(log_params):
            c = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            return safe_nll(stats.weibull_min.logpdf(ev, c=c, loc=0, scale=sc), ep)

        init_scale = max(mean, eps)
        result = minimize(nll_obj, x0=[0.0, np.log(init_scale)],
                          method='Nelder-Mead')
        c = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (c, 0.0, scale)
        nll = safe_nll(stats.weibull_min.logpdf(ev, c=c, loc=0, scale=scale), ep)

    elif dist_name == 'chi2':
        init_df = max(mean, eps)

        def nll_obj(log_params):
            df = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            return safe_nll(stats.chi2.logpdf(ev, df=df, loc=0, scale=sc), ep)

        result = minimize(nll_obj,
                          x0=[np.log(init_df), 0.0],
                          method='Nelder-Mead')
        df = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (df, 0.0, scale)
        nll = safe_nll(stats.chi2.logpdf(ev, df=df, loc=0, scale=scale), ep)

    elif dist_name == 'nbinom':
        if var <= mean:
            init_p = 1 - eps
            init_n = max(mean * init_p / (1 - init_p), eps)
        else:
            init_p = np.clip(mean / var, eps, 1 - eps)
            init_n = max(mean * init_p / (1 - init_p), eps)

        def nll_obj(opt_params):
            n_param = np.exp(opt_params[0])
            p_param = 1.0 / (1.0 + np.exp(-opt_params[1]))
            return safe_nll(stats.nbinom.logpmf(ev.astype(int), n=n_param, p=p_param), ep)

        init_logit_p = np.log(max(init_p, eps) / max(1 - init_p, eps))
        result = minimize(nll_obj,
                          x0=[np.log(max(init_n, eps)), init_logit_p],
                          method='Nelder-Mead')
        n_param = np.exp(result.x[0])
        p_param = 1.0 / (1.0 + np.exp(-result.x[1]))
        params = (n_param, p_param)
        nll = safe_nll(stats.nbinom.logpmf(ev.astype(int), n=n_param, p=p_param), ep)

    elif dist_name == 'uniform':
        init_a = float(min(values))
        init_width = max(float(max(values)) - init_a, eps)

        def nll_obj(opt_params):
            a = opt_params[0]
            w = np.exp(opt_params[1])
            return safe_nll(stats.uniform.logpdf(ev, loc=a, scale=w), ep)

        result = minimize(nll_obj,
                          x0=[init_a, np.log(init_width)],
                          method='Nelder-Mead')
        a = result.x[0]
        width = np.exp(result.x[1])
        params = (a, width)
        nll = safe_nll(stats.uniform.logpdf(ev, loc=a, scale=width), ep)

    elif dist_name == 'invgauss':
        m = max(mean, eps)
        v = max(var, eps)
        init_scale = m ** 3 / v
        init_mu = m / init_scale

        def nll_obj(log_params):
            mu_p = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            return safe_nll(stats.invgauss.logpdf(ev, mu=mu_p, loc=0, scale=sc), ep)

        result = minimize(nll_obj,
                          x0=[np.log(max(init_mu, eps)), np.log(max(init_scale, eps))],
                          method='Nelder-Mead')
        mu_param = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (mu_param, 0.0, scale)
        nll = safe_nll(stats.invgauss.logpdf(ev, mu=mu_param, loc=0, scale=scale), ep)

    elif dist_name == 'gengamma':
        def nll_obj(log_params):
            a_p = np.exp(log_params[0])
            c_p = np.exp(log_params[1])
            sc = np.exp(log_params[2])
            return safe_nll(stats.gengamma.logpdf(ev, a=a_p, c=c_p, loc=0, scale=sc), ep)

        m = max(mean, eps)
        v = max(var, eps)
        init_scale = v / m
        init_shape = m / init_scale
        result = minimize(nll_obj,
                          x0=[np.log(max(init_shape, eps)), 0.0,
                              np.log(max(init_scale, eps))],
                          method='Nelder-Mead')
        a_p = np.exp(result.x[0])
        c_p = np.exp(result.x[1])
        scale = np.exp(result.x[2])
        params = (a_p, c_p, 0.0, scale)
        nll = safe_nll(stats.gengamma.logpdf(ev, a=a_p, c=c_p, loc=0, scale=scale), ep)

    elif dist_name == 'nakagami':
        m = max(mean, eps)
        v = max(var, eps)
        omega = m ** 2 + v

        def nll_obj(log_nu):
            nu = np.exp(log_nu[0])
            sc = np.sqrt(omega)
            return safe_nll(stats.nakagami.logpdf(ev, nu=nu, loc=0, scale=sc), ep)

        init_nu = max(m ** 2 / v, eps)
        result = minimize(nll_obj, x0=[np.log(init_nu)],
                          method='Nelder-Mead')
        nu = np.exp(result.x[0])
        scale = np.sqrt(omega)
        params = (nu, 0.0, scale)
        nll = safe_nll(stats.nakagami.logpdf(ev, nu=nu, loc=0, scale=scale), ep)

    elif dist_name == 'triang':
        min_val = float(min(values))
        max_val = float(max(values))
        width = max(max_val - min_val, eps)

        def nll_obj(params_opt):
            c = 1.0 / (1.0 + np.exp(-params_opt[0]))
            return safe_nll(stats.triang.logpdf(ev, c=c, loc=min_val, scale=width), ep)

        result = minimize(nll_obj, x0=[0.0], method='Nelder-Mead')
        c = 1.0 / (1.0 + np.exp(-result.x[0]))
        params = (c, min_val, width)
        nll = safe_nll(stats.triang.logpdf(ev, c=c, loc=min_val, scale=width), ep)

    elif dist_name == 'betaprime':
        def nll_obj(log_params):
            a_p = np.exp(log_params[0])
            b_p = np.exp(log_params[1])
            sc = np.exp(log_params[2])
            return safe_nll(stats.betaprime.logpdf(ev, a=a_p, b=b_p, loc=0, scale=sc), ep)

        m = max(mean, eps)
        result = minimize(nll_obj,
                          x0=[0.0, np.log(2.0), np.log(max(m, eps))],
                          method='Nelder-Mead')
        a_p = np.exp(result.x[0])
        b_p = np.exp(result.x[1])
        scale = np.exp(result.x[2])
        params = (a_p, b_p, 0.0, scale)
        nll = safe_nll(stats.betaprime.logpdf(ev, a=a_p, b=b_p, loc=0, scale=scale), ep)

    elif dist_name == 'johnsonsb':
        def nll_obj(opt_params):
            a_p = opt_params[0]
            b_p = np.exp(opt_params[1])
            loc_p = opt_params[2]
            sc = np.exp(opt_params[3])
            return safe_nll(stats.johnsonsb.logpdf(ev, a=a_p, b=b_p, loc=loc_p, scale=sc), ep)

        min_val = float(min(values))
        max_val = float(max(values))
        width = max(max_val - min_val, eps)
        result = minimize(nll_obj,
                          x0=[0.0, 0.0, min_val, np.log(width)],
                          method='Nelder-Mead')
        a_p = result.x[0]
        b_p = np.exp(result.x[1])
        loc_p = result.x[2]
        scale = np.exp(result.x[3])
        params = (a_p, b_p, loc_p, scale)
        nll = safe_nll(stats.johnsonsb.logpdf(ev, a=a_p, b=b_p, loc=loc_p, scale=scale), ep)

    elif dist_name == 'arcsine':
        init_loc = float(min(values))
        init_width = max(float(max(values)) - init_loc, eps)

        def nll_obj(opt_params):
            loc = opt_params[0]
            sc = np.exp(opt_params[1])
            return safe_nll(stats.arcsine.logpdf(ev, loc=loc, scale=sc), ep)

        result = minimize(nll_obj,
                          x0=[init_loc, np.log(init_width)],
                          method='Nelder-Mead')
        loc = result.x[0]
        width = np.exp(result.x[1])
        params = (loc, width)
        nll = safe_nll(stats.arcsine.logpdf(ev, loc=loc, scale=width), ep)

    elif dist_name == 'powerlaw':
        min_val = float(min(values))
        max_val = float(max(values))
        width = max(max_val - min_val, eps)

        def nll_obj(log_a):
            a_p = np.exp(log_a[0])
            return safe_nll(stats.powerlaw.logpdf(ev, a=a_p, loc=min_val, scale=width), ep)

        result = minimize(nll_obj, x0=[0.0], method='Nelder-Mead')
        a_p = np.exp(result.x[0])
        params = (a_p, min_val, width)
        nll = safe_nll(stats.powerlaw.logpdf(ev, a=a_p, loc=min_val, scale=width), ep)

    elif dist_name == 'bradford':
        min_val = float(min(values))
        max_val = float(max(values))
        width = max(max_val - min_val, eps)

        def nll_obj(log_c):
            c_p = np.exp(log_c[0])
            return safe_nll(stats.bradford.logpdf(ev, c=c_p, loc=min_val, scale=width), ep)

        result = minimize(nll_obj, x0=[0.0], method='Nelder-Mead')
        c_p = np.exp(result.x[0])
        params = (c_p, min_val, width)
        nll = safe_nll(stats.bradford.logpdf(ev, c=c_p, loc=min_val, scale=width), ep)

    elif dist_name == 'burr':
        def nll_obj(log_params):
            c_p = np.exp(log_params[0])
            d_p = np.exp(log_params[1])
            sc = np.exp(log_params[2])
            return safe_nll(stats.burr.logpdf(ev, c=c_p, d=d_p, loc=0, scale=sc), ep)

        m = max(mean, eps)
        result = minimize(nll_obj,
                          x0=[0.0, 0.0, np.log(m)],
                          method='Nelder-Mead')
        c_p = np.exp(result.x[0])
        d_p = np.exp(result.x[1])
        scale = np.exp(result.x[2])
        params = (c_p, d_p, 0.0, scale)
        nll = safe_nll(stats.burr.logpdf(ev, c=c_p, d=d_p, loc=0, scale=scale), ep)

    elif dist_name == 'burr12':
        def nll_obj(log_params):
            c_p = np.exp(log_params[0])
            d_p = np.exp(log_params[1])
            sc = np.exp(log_params[2])
            return safe_nll(stats.burr12.logpdf(ev, c=c_p, d=d_p, loc=0, scale=sc), ep)

        m = max(mean, eps)
        result = minimize(nll_obj,
                          x0=[0.0, 0.0, np.log(m)],
                          method='Nelder-Mead')
        c_p = np.exp(result.x[0])
        d_p = np.exp(result.x[1])
        scale = np.exp(result.x[2])
        params = (c_p, d_p, 0.0, scale)
        nll = safe_nll(stats.burr12.logpdf(ev, c=c_p, d=d_p, loc=0, scale=scale), ep)

    elif dist_name == 'fisk':
        def nll_obj(log_params):
            c_p = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            return safe_nll(stats.fisk.logpdf(ev, c=c_p, loc=0, scale=sc), ep)

        m = max(mean, eps)
        result = minimize(nll_obj,
                          x0=[0.0, np.log(m)],
                          method='Nelder-Mead')
        c_p = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (c_p, 0.0, scale)
        nll = safe_nll(stats.fisk.logpdf(ev, c=c_p, loc=0, scale=scale), ep)

    elif dist_name == 'genpareto':
        def nll_obj(opt_params):
            c_p = opt_params[0]
            sc = np.exp(opt_params[1])
            return safe_nll(stats.genpareto.logpdf(ev, c=c_p, loc=0, scale=sc), ep)

        m = max(mean, eps)
        result = minimize(nll_obj,
                          x0=[0.0, np.log(m)],
                          method='Nelder-Mead')
        c_p = result.x[0]
        scale = np.exp(result.x[1])
        params = (c_p, 0.0, scale)
        nll = safe_nll(stats.genpareto.logpdf(ev, c=c_p, loc=0, scale=scale), ep)

    elif dist_name == 'genextreme':
        def nll_obj(opt_params):
            c_p = opt_params[0]
            loc_p = opt_params[1]
            sc = np.exp(opt_params[2])
            return safe_nll(stats.genextreme.logpdf(ev, c=c_p, loc=loc_p, scale=sc), ep)

        m = max(mean, eps)
        s = max(std, eps)
        result = minimize(nll_obj,
                          x0=[0.0, m, np.log(s)],
                          method='Nelder-Mead')
        c_p = result.x[0]
        loc_p = result.x[1]
        scale = np.exp(result.x[2])
        params = (c_p, loc_p, scale)
        nll = safe_nll(stats.genextreme.logpdf(ev, c=c_p, loc=loc_p, scale=scale), ep)

    elif dist_name == 'rayleigh':
        m = max(mean, eps)
        init_scale = m / np.sqrt(np.pi / 2)

        def nll_obj(log_scale):
            sc = np.exp(log_scale[0])
            return safe_nll(stats.rayleigh.logpdf(ev, loc=0, scale=sc), ep)

        result = minimize(nll_obj, x0=[np.log(max(init_scale, eps))],
                          method='Nelder-Mead')
        scale = np.exp(result.x[0])
        params = (0.0, scale)
        nll = safe_nll(stats.rayleigh.logpdf(ev, loc=0, scale=scale), ep)

    elif dist_name == 'rice':
        def nll_obj(log_params):
            b_p = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            return safe_nll(stats.rice.logpdf(ev, b=b_p, loc=0, scale=sc), ep)

        m = max(mean, eps)
        s = max(std, eps)
        result = minimize(nll_obj,
                          x0=[0.0, np.log(s)],
                          method='Nelder-Mead')
        b_p = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (b_p, 0.0, scale)
        nll = safe_nll(stats.rice.logpdf(ev, b=b_p, loc=0, scale=scale), ep)

    elif dist_name == 'pareto':
        def nll_obj(log_params):
            b_p = np.exp(log_params[0])
            sc = np.exp(log_params[1])
            return safe_nll(stats.pareto.logpdf(ev, b=b_p, loc=0, scale=sc), ep)

        m = max(mean, eps)
        result = minimize(nll_obj,
                          x0=[0.0, np.log(m)],
                          method='Nelder-Mead')
        b_p = np.exp(result.x[0])
        scale = np.exp(result.x[1])
        params = (b_p, 0.0, scale)
        nll = safe_nll(stats.pareto.logpdf(ev, b=b_p, loc=0, scale=scale), ep)

    elif dist_name == 'f':
        def nll_obj(log_params):
            dfn = np.exp(log_params[0])
            dfd = np.exp(log_params[1])
            sc = np.exp(log_params[2])
            return safe_nll(stats.f.logpdf(ev, dfn=dfn, dfd=dfd, loc=0, scale=sc), ep)

        m = max(mean, eps)
        result = minimize(nll_obj,
                          x0=[0.0, np.log(2.0), np.log(m)],
                          method='Nelder-Mead')
        dfn = np.exp(result.x[0])
        dfd = np.exp(result.x[1])
        scale = np.exp(result.x[2])
        params = (dfn, dfd, 0.0, scale)
        nll = safe_nll(stats.f.logpdf(ev, dfn=dfn, dfd=dfd, loc=0, scale=scale), ep)

    else:
        raise ValueError("Unsupported distribution: " + dist_name)

    return params, nll


def main():
    parser = argparse.ArgumentParser(
        description="MLE fit of distributions to exact distribution data")
    parser.add_argument("--exact-dir", required=True,
                        help="Directory with exact distribution files (value count per line)")
    parser.add_argument("--distribution", required=True, choices=SUPPORTED_DISTRIBUTIONS,
                        help="Distribution to fit")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory for fitted parameters and NLL values")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show progress with tqdm")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    names = sorted(os.listdir(args.exact_dir))
    names = [n for n in names
             if os.path.isfile(os.path.join(args.exact_dir, n))]

    iterator = tqdm(names) if args.verbose else names

    for name in iterator:
        exact_path = os.path.join(args.exact_dir, name)

        values, counts = load_distribution(exact_path)
        params, nll = mle_fit_distribution(args.distribution, values, counts)

        output_path = os.path.join(args.output_dir, name)
        with open(output_path, "w", encoding="utf8") as fo:
            fo.write(" ".join(map(str, params)) + "\n")
            fo.write(str(nll) + "\n")


if __name__ == "__main__":
    main()
