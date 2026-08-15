#!/usr/bin/env python3
"""Comprehensive distribution selection analysis.

For each (N, n) pair this script:
  1. Loads the exact discrete distribution and all available fitted
     continuous approximations (both MLE and CvM fits).
  2. Computes 9 comparison metrics using the right fit type for each:
       - MLE fits  -> AIC, BIC, KL divergence, cross-entropy  (info-theoretic)
       - CvM fits  -> CvM, KS, Anderson-Darling               (CDF-based)
       - MLE fits  -> L1 (total variation), Chi-squared         (PMF-based)
  3. Ranks distributions by AIC and interprets delta-AIC via the
     Burnham-Anderson scale (dAIC<2 = substantial support, 2-7 = less
     support, >10 = essentially none).
  4. Cross-checks whether BIC, KL, CvM, and KS agree on the top pick.
  5. Runs Vuong's closeness test between AIC-competitive models (dAIC<2)
     to see whether differences are statistically distinguishable.
  6. Flags degenerate metrics (AD / Chi-squared overflow from tail
     mismatch) so the user knows which numbers to ignore.
  7. Emits a clear, reasoned recommendation.

Usage:
  python analyze.py -N 100 -n 50          # verbose single-pair analysis
  python analyze.py --all --csv out.csv   # batch over all pairs
"""

import argparse
import csv
import os
import sys
import numpy as np
from scipy import stats as sp_stats
from fractions import Fraction

from fit import load_distribution, load_cdf, SUPPORTED_DISTRIBUTIONS
from compare import DIST_MAP, DISCRETE_DISTS, load_fitted_params
from model_selection import (
    NUM_PARAMS, compute_log_probs, compute_metrics, vuong_test,
    get_available_distributions, parse_Nn,
    METRIC_NAMES, METRIC_SHORT,
)

# Thresholds
DELTA_AIC_TIER1 = 2.0     # substantial support
DELTA_AIC_TIER2 = 7.0     # some support
VUONG_ALPHA = 0.05        # significance level for Vuong
AD_OVERFLOW = 1e5          # flag AD values above this
CHI2_OVERFLOW = 1e10       # flag Chi2 values above this
MIN_SUPPORT_FOR_VUONG = 5  # need at least this many support points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg="", **kw):
    print(msg, flush=True, **kw)


def fmt(v, width=11):
    """Format a float for the table."""
    if not np.isfinite(v):
        return f"{'inf':>{width}}"
    if abs(v) < 0.001 or abs(v) > 99999:
        return f"{v:>{width}.3e}"
    return f"{v:>{width}.6f}"


def load_true_distribution(N, n):
    """Return (values, true_probs, cdf_values, cdf_probs, entropy) or None."""
    fname = f"N_{N}_n_{n}.txt"
    exact_path = os.path.join("../data/exact_distributions", fname)
    cdf_path = os.path.join("../data/cdf_exact", fname)
    if not os.path.isfile(exact_path) or not os.path.isfile(cdf_path):
        return None

    values, counts = load_distribution(exact_path)
    cdf_values, cdf_probs = load_cdf(cdf_path)

    total = sum(counts)
    true_probs = np.array([float(Fraction(c, total)) for c in counts])
    values_arr = np.array(values, dtype=float)
    cdf_values_f = cdf_values.astype(float)

    log_p = np.log(np.maximum(true_probs, 1e-300))
    entropy = -np.sum(true_probs * log_p)

    return values_arr, true_probs, cdf_values_f, cdf_probs, entropy


def winner_by(results, metric):
    """Return the dist_name with the lowest value for *metric*."""
    best = min(results, key=lambda r: r["combined"][metric])
    return best["dist_name"]


# ---------------------------------------------------------------------------
# Single-pair analysis (verbose)
# ---------------------------------------------------------------------------

def analyze_single(N, n, verbose=True):
    """Full analysis for one (N, n) pair.

    Returns a summary dict (or None if data missing):
        winner, runner_up, tier1_names, all_rows (for CSV)
    """
    fname = f"N_{N}_n_{n}.txt"

    # ---- Step 1 ----
    if verbose:
        log(f"\n{'=' * 72}")
        log(f"  Analysis for N={N}, n={n}")
        log(f"{'=' * 72}")
        log(f"\n[Step 1] Loading exact distribution ...")

    td = load_true_distribution(N, n)
    if td is None:
        if verbose:
            log("  SKIPPED — exact distribution or CDF file not found.")
        return None
    values, true_probs, cdf_values, cdf_probs, entropy = td
    K = len(values)
    val_min, val_max = int(values.min()), int(values.max())

    if verbose:
        log(f"  Support size K = {K},  range [{val_min}, {val_max}]")
        log(f"  Entropy H(p)   = {entropy:.6f} nats")

    # ---- Step 2 ----
    if verbose:
        log(f"\n[Step 2] Discovering fitted distributions ...")

    mle_set = set(get_available_distributions(N, n, mle=True))
    cvm_set = set(get_available_distributions(N, n, mle=False))
    all_dists = sorted(mle_set | cvm_set)

    if verbose:
        log(f"  MLE fits : {len(mle_set)}")
        log(f"  CvM fits : {len(cvm_set)}")
        log(f"  Total    : {len(all_dists)} distinct distributions")

    if not all_dists:
        if verbose:
            log("  SKIPPED — no fitted distributions found.")
        return None

    # ---- Step 3 ----
    if verbose:
        log(f"\n[Step 3] Computing metrics ...")
        log(f"  Strategy:")
        log(f"    - AIC, BIC, KL, H(p,q)  from MLE fits  (optimised for likelihood)")
        log(f"    - CvM, KS, AD           from CvM fits  (optimised for CDF distance)")
        log(f"    - L1, Chi2              from MLE fits  (PMF-level comparison)")
        log(f"  If only one fit type exists for a distribution, it is used for all.")

    results = []
    for dist_name in all_dists:
        entry = {"dist_name": dist_name, "n_params": NUM_PARAMS[dist_name]}

        m_mle = m_cvm = None
        if dist_name in mle_set:
            p_mle, _ = load_fitted_params(
                os.path.join(f"../data/fitted/mle_{dist_name}", fname))
            entry["params_mle"] = p_mle
            m_mle = compute_metrics(dist_name, p_mle, values, true_probs,
                                    cdf_values, cdf_probs)
            entry["metrics_mle"] = m_mle

        if dist_name in cvm_set:
            p_cvm, _ = load_fitted_params(
                os.path.join(f"../data/fitted/cvm_{dist_name}", fname))
            entry["params_cvm"] = p_cvm
            m_cvm = compute_metrics(dist_name, p_cvm, values, true_probs,
                                    cdf_values, cdf_probs)
            entry["metrics_cvm"] = m_cvm

        # Build the combined row (best-source-per-metric)
        src_info = m_mle if m_mle else m_cvm
        src_cdf  = m_cvm if m_cvm else m_mle
        combined = {}
        for m in ("cross_entropy", "kl_div", "aic", "bic"):
            combined[m] = src_info[m]
        for m in ("cvm", "ks", "ad"):
            combined[m] = src_cdf[m]
        for m in ("l1", "chi2"):
            combined[m] = src_info[m]
        combined["n_params"] = NUM_PARAMS[dist_name]
        entry["combined"] = combined
        results.append(entry)

    # ---- Step 4 ----
    if verbose:
        log(f"\n[Step 4] Ranking by AIC  (AIC = 2k + 2·H(p,q)) ...")
        log(f"  AIC balances fit quality against model complexity.")
        log(f"  Lower is better.  dAIC = AIC - AIC_best.\n")

    results.sort(key=lambda r: r["combined"]["aic"])
    best_aic = results[0]["combined"]["aic"]
    for r in results:
        r["delta_aic"] = r["combined"]["aic"] - best_aic

    if verbose:
        hdr = (f"  {'#':>3}  {'Distribution':<14} {'k':>2}  "
               f"{'AIC':>11}  {'dAIC':>6}  {'BIC':>11}  "
               f"{'KL':>11}  {'CvM':>11}  {'KS':>11}  {'L1':>11}")
        log(hdr)
        log("  " + "-" * (len(hdr) - 2))
        for i, r in enumerate(results, 1):
            c = r["combined"]
            log(f"  {i:>3}  {r['dist_name']:<14} {c['n_params']:>2}  "
                f"{fmt(c['aic'])}  {r['delta_aic']:>6.2f}  "
                f"{fmt(c['bic'])}  {fmt(c['kl_div'])}  "
                f"{fmt(c['cvm'])}  {fmt(c['ks'])}  {fmt(c['l1'])}")

    tier1 = [r for r in results if r["delta_aic"] < DELTA_AIC_TIER1]
    tier2 = [r for r in results
             if DELTA_AIC_TIER1 <= r["delta_aic"] < DELTA_AIC_TIER2]

    if verbose:
        log(f"\n  Burnham-Anderson interpretation:")
        log(f"    dAIC < 2  (substantial support)  : "
            f"{', '.join(r['dist_name'] for r in tier1)}")
        if tier2:
            log(f"    dAIC 2-7  (some support)         : "
                f"{', '.join(r['dist_name'] for r in tier2)}")
        n_no = sum(1 for r in results if r["delta_aic"] >= 10)
        if n_no:
            log(f"    dAIC >= 10 (no support)          : {n_no} distributions")

    # ---- Step 5 ----
    if verbose:
        log(f"\n[Step 5] Cross-checking metric agreement ...")

    aic_winner = results[0]["dist_name"]
    bic_winner = winner_by(results, "bic")
    kl_winner  = winner_by(results, "kl_div")
    cvm_winner = winner_by(results, "cvm")
    ks_winner  = winner_by(results, "ks")
    l1_winner  = winner_by(results, "l1")

    metric_winners = {
        "AIC": aic_winner, "BIC": bic_winner, "KL": kl_winner,
        "CvM": cvm_winner, "KS": ks_winner, "L1": l1_winner,
    }

    if verbose:
        for label, w in metric_winners.items():
            tag = "" if w == aic_winner else "  <-- differs from AIC"
            log(f"    {label:<4} winner: {w}{tag}")

    agrees = {k for k, v in metric_winners.items() if v == aic_winner}
    disagrees = {k: v for k, v in metric_winners.items() if v != aic_winner}

    if verbose:
        if not disagrees:
            log(f"\n  All six key metrics agree on '{aic_winner}'.  Strong consensus.")
        else:
            log(f"\n  {len(agrees)}/6 metrics agree on '{aic_winner}'.")
            if "CvM" in disagrees or "KS" in disagrees:
                cdf_alt = disagrees.get("CvM", disagrees.get("KS"))
                log(f"  CDF-based metrics favour '{cdf_alt}' — it matches the "
                    f"empirical CDF more closely,")
                log(f"  but AIC/BIC penalise its extra complexity or it has worse "
                    f"likelihood fit.")
            if "BIC" in disagrees:
                log(f"  BIC (stronger complexity penalty) favours '{bic_winner}' "
                    f"(fewer parameters).")

    # ---- Step 6 ----
    run_vuong = (K >= MIN_SUPPORT_FOR_VUONG and len(tier1) > 1)

    if verbose:
        log(f"\n[Step 6] Vuong's closeness test ...")
        if K < MIN_SUPPORT_FOR_VUONG:
            log(f"  SKIPPED — support size K={K} is too small for a meaningful test.")
        elif len(tier1) <= 1:
            log(f"  SKIPPED — only one Tier-1 distribution; no close rival to test.")
        else:
            log(f"  Testing AIC winner '{aic_winner}' against each Tier-1 rival.")
            log(f"  Effective sample size = K = {K}.  Significance level alpha = {VUONG_ALPHA}.")
            log(f"  Positive Vuong stat => '{aic_winner}' is closer to the truth.\n")

    vuong_results = {}
    if run_vuong:
        # Use MLE params for Vuong (likelihood-based test)
        winner_entry = results[0]
        winner_params = winner_entry.get("params_mle", winner_entry.get("params_cvm"))

        for rival in tier1[1:]:
            rival_params = rival.get("params_mle", rival.get("params_cvm"))
            vr = vuong_test(aic_winner, winner_params,
                            rival["dist_name"], rival_params,
                            values, true_probs)
            vuong_results[rival["dist_name"]] = vr

            if verbose:
                rname = rival["dist_name"]
                if vr["p_value"] < VUONG_ALPHA:
                    if vr["vuong_stat"] > 0:
                        verdict = f"'{aic_winner}' is significantly closer"
                    else:
                        verdict = f"'{rname}' is significantly closer"
                else:
                    verdict = "indistinguishable"

                log(f"  {aic_winner} vs {rname}:")
                log(f"    E[log-ratio] = {vr['mean_d']:+.4e},  "
                    f"StdDev = {vr['std_d']:.4e}")
                log(f"    Vuong stat   = {vr['vuong_stat']:+.3f},  "
                    f"p = {vr['p_value']:.4e}  =>  {verdict}")

                if vr["corrected_p"] < VUONG_ALPHA:
                    if vr["corrected_stat"] > 0:
                        cv = aic_winner
                    else:
                        cv = rname
                    log(f"    AIC-corrected: {cv} significantly closer "
                        f"(p = {vr['corrected_p']:.4e})")
                else:
                    log(f"    AIC-corrected: still indistinguishable "
                        f"(p = {vr['corrected_p']:.4e})")
                log()

        # Also test against CvM winner if it's not in Tier 1
        if cvm_winner != aic_winner and cvm_winner not in [r["dist_name"] for r in tier1]:
            cvm_entry = next(r for r in results if r["dist_name"] == cvm_winner)
            cvm_params = cvm_entry.get("params_mle", cvm_entry.get("params_cvm"))
            vr = vuong_test(aic_winner, winner_params,
                            cvm_winner, cvm_params, values, true_probs)
            vuong_results[cvm_winner] = vr
            if verbose:
                if vr["p_value"] < VUONG_ALPHA:
                    if vr["vuong_stat"] > 0:
                        verdict = f"'{aic_winner}' is significantly closer"
                    else:
                        verdict = f"'{cvm_winner}' is significantly closer"
                else:
                    verdict = "indistinguishable"
                log(f"  (Extra: AIC winner vs CvM winner)")
                log(f"  {aic_winner} vs {cvm_winner}:")
                log(f"    Vuong stat = {vr['vuong_stat']:+.3f},  "
                    f"p = {vr['p_value']:.4e}  =>  {verdict}\n")

    # ---- Step 7 ----
    if verbose:
        log(f"[Step 7] Checking for degenerate metrics ...")

    n_ad_bad = sum(1 for r in results if r["combined"]["ad"] > AD_OVERFLOW)
    n_chi2_bad = sum(1 for r in results if r["combined"]["chi2"] > CHI2_OVERFLOW)

    if verbose:
        if n_ad_bad:
            log(f"  AD overflow (>{AD_OVERFLOW:.0e}) in {n_ad_bad}/{len(results)} "
                f"distributions.")
            log(f"    This happens when the fitted CDF is near 0 or 1 at points "
                f"where the exact CDF is not.")
            log(f"    AD is unreliable for those distributions — rely on CvM/KS "
                f"instead.")
        if n_chi2_bad:
            log(f"  Chi2 overflow (>{CHI2_OVERFLOW:.0e}) in {n_chi2_bad}/{len(results)} "
                f"distributions.")
            log(f"    This happens when a fitted distribution assigns near-zero "
                f"probability mass to")
            log(f"    a support point with positive exact probability.  "
                f"Rely on L1/KL instead.")
        if not n_ad_bad and not n_chi2_bad:
            log(f"  No degenerate metrics detected.  All values are usable.")

    # ---- Step 8 ----
    if verbose:
        log(f"\n[Step 8] Recommendation")
        log(f"{'─' * 72}")

    winner = results[0]
    runner_up = results[1] if len(results) > 1 else None
    wc = winner["combined"]

    # Determine confidence level
    if len(tier1) == 1 and not disagrees:
        confidence = "HIGH"
        reason = "all metrics agree and no close rival exists"
    elif len(tier1) == 1:
        confidence = "MODERATE"
        reason = "clear AIC winner but some metrics disagree"
    elif all(vr["p_value"] >= VUONG_ALPHA for vr in vuong_results.values()):
        confidence = "LOW"
        reason = ("multiple Tier-1 models and Vuong tests cannot distinguish "
                  "them — consider the simplest")
    else:
        confidence = "MODERATE"
        reason = "close competition but Vuong tests show some differentiation"

    if verbose:
        log(f"\n  PRIMARY recommendation:  {winner['dist_name']}")
        log(f"    Parameters     : {wc['n_params']} free")
        log(f"    AIC            : {wc['aic']:.6f}")
        log(f"    KL divergence  : {wc['kl_div']:.6e}")
        log(f"    CvM statistic  : {wc['cvm']:.6e}")
        log(f"    L1 distance    : {wc['l1']:.6e}")
        log(f"    Confidence     : {confidence} ({reason})")

        if runner_up:
            rc = runner_up["combined"]
            log(f"\n  Runner-up:  {runner_up['dist_name']}")
            log(f"    dAIC = {runner_up['delta_aic']:.2f},  "
                f"KL = {rc['kl_div']:.6e},  CvM = {rc['cvm']:.6e}")

        # Note CvM alternative if different from AIC winner
        if cvm_winner != aic_winner:
            cvm_entry = next(r for r in results if r["dist_name"] == cvm_winner)
            log(f"\n  CDF-accuracy alternative:  {cvm_winner}")
            log(f"    Best CvM = {cvm_entry['combined']['cvm']:.6e}  "
                f"(vs winner's {wc['cvm']:.6e})")
            log(f"    dAIC = {cvm_entry['delta_aic']:.2f}  "
                f"(complexity cost for better CDF fit)")

        # Parsimony tie-break advice
        if len(tier1) > 1:
            simplest = min(tier1, key=lambda r: r["combined"]["n_params"])
            if simplest["dist_name"] != aic_winner:
                log(f"\n  Parsimony note: '{simplest['dist_name']}' "
                    f"({simplest['combined']['n_params']} params) is the simplest "
                    f"Tier-1 model.")
                log(f"    If Vuong cannot distinguish it from the AIC winner, "
                    f"prefer it for interpretability.")

        log()

    # ---- Build CSV rows ----
    csv_rows = []
    for r in results:
        row = {"N": N, "n": n, "distribution": r["dist_name"],
               "n_params": r["combined"]["n_params"],
               "delta_aic": r["delta_aic"]}
        for m in METRIC_NAMES:
            row[m] = r["combined"][m]
        csv_rows.append(row)

    return {
        "N": N, "n": n,
        "winner": winner["dist_name"],
        "winner_aic": wc["aic"],
        "winner_kl": wc["kl_div"],
        "winner_cvm": wc["cvm"],
        "winner_k": wc["n_params"],
        "runner_up": runner_up["dist_name"] if runner_up else None,
        "runner_up_daic": runner_up["delta_aic"] if runner_up else None,
        "n_tier1": len(tier1),
        "confidence": confidence,
        "tier1_names": [r["dist_name"] for r in tier1],
        "csv_rows": csv_rows,
    }


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def analyze_all(csv_path=None):
    """Run analysis over every (N, n) pair; print progress and aggregate."""
    fnames = sorted(os.listdir("../data/exact_distributions"))
    pairs = []
    for f in fnames:
        if not f.endswith(".txt"):
            continue
        try:
            N, n = parse_Nn(f)
            pairs.append((N, n))
        except (ValueError, IndexError):
            continue

    total = len(pairs)
    log(f"Found {total} (N, n) pairs in ../data/exact_distributions/.\n")

    all_csv_rows = []
    winner_counts = {}
    confidence_counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
    tier1_size_hist = {}
    processed = 0
    skipped = 0

    for i, (N, n) in enumerate(pairs):
        result = analyze_single(N, n, verbose=False)

        if result is None:
            skipped += 1
        else:
            processed += 1
            w = result["winner"]
            winner_counts[w] = winner_counts.get(w, 0) + 1
            conf = result["confidence"]
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
            nt = result["n_tier1"]
            tier1_size_hist[nt] = tier1_size_hist.get(nt, 0) + 1
            all_csv_rows.extend(result["csv_rows"])

        # Progress every 5000 pairs
        done = i + 1
        if done % 5000 == 0 or done == total:
            log(f"  [{done:>{len(str(total))}}/{total}]  "
                f"processed={processed}  skipped={skipped}  "
                f"current_leader={max(winner_counts, key=winner_counts.get) if winner_counts else '?'}")

    # ---- Aggregate summary ----
    log(f"\n{'=' * 72}")
    log(f"  AGGREGATE SUMMARY  ({processed} pairs analysed, {skipped} skipped)")
    log(f"{'=' * 72}")

    log(f"\n  AIC winner counts (how often each distribution is #1):\n")
    for dist, cnt in sorted(winner_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / processed
        bar = "#" * int(pct / 2)
        log(f"    {dist:<14}  {cnt:>6}  ({pct:5.1f}%)  {bar}")

    log(f"\n  Confidence distribution:")
    for level in ("HIGH", "MODERATE", "LOW"):
        cnt = confidence_counts.get(level, 0)
        pct = 100 * cnt / processed if processed else 0
        log(f"    {level:<10}  {cnt:>6}  ({pct:5.1f}%)")

    log(f"\n  Tier-1 size (how many models are within dAIC < 2 of the best):")
    for sz in sorted(tier1_size_hist):
        cnt = tier1_size_hist[sz]
        pct = 100 * cnt / processed if processed else 0
        log(f"    {sz:>2} model(s)  {cnt:>6}  ({pct:5.1f}%)")

    if csv_path and all_csv_rows:
        fieldnames = (["N", "n", "distribution", "n_params", "delta_aic"]
                      + METRIC_NAMES)
        with open(csv_path, "w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        log(f"\n  Wrote {len(all_csv_rows)} rows to {csv_path}")

    log()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive distribution selection analysis")
    parser.add_argument("-N", type=int, help="N parameter")
    parser.add_argument("-n", type=int, help="n parameter")
    parser.add_argument("--all", action="store_true",
                        help="Analyse every (N,n) pair (batch mode)")
    parser.add_argument("--csv", default=None, metavar="PATH",
                        help="Write per-distribution metrics to CSV")
    args = parser.parse_args()

    if args.all:
        analyze_all(csv_path=args.csv)
    elif args.N is not None and args.n is not None:
        result = analyze_single(args.N, args.n, verbose=True)
        if result and args.csv:
            fieldnames = (["N", "n", "distribution", "n_params", "delta_aic"]
                          + METRIC_NAMES)
            with open(args.csv, "w", newline="", encoding="utf8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(result["csv_rows"])
            log(f"Wrote {len(result['csv_rows'])} rows to {args.csv}")
    else:
        parser.error("Provide -N and -n, or use --all")


if __name__ == "__main__":
    main()
