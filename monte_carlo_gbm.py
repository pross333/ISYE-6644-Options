
import argparse
import os
from math import exp, log, sqrt, erf, ceil
import numpy as np
import pandas as pd

# ---- Matplotlib backend: force a headless-safe backend before importing pyplot ----
# Prefer Agg (file-only). If a display is available, you can remove these lines.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
try:
    matplotlib.use(os.environ["MPLBACKEND"])
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------ Normal CDF & Black–Scholes ------------------------------
def std_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def black_scholes_price(S0, K, r, q, sigma, T, option_type='call'):
    if T <= 0 or sigma <= 0:
        forward = S0 * exp((r - q) * T)
        df = exp(-r * T)
        return df * (max(forward - K, 0) if option_type == 'call' else max(K - forward, 0))
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    df = exp(-r * T)
    dq = exp(-q * T)
    if option_type.lower() == 'call':
        return dq * S0 * std_norm_cdf(d1) - df * K * std_norm_cdf(d2)
    else:
        return df * K * std_norm_cdf(-d2) - dq * S0 * std_norm_cdf(-d1)

# ------------------------------ Implied volatility via bisection ------------------------------
def implied_vol_bisection(S0, K, r, q, T, market_price, option_type='call', tol=1e-6, max_iter=200):
    low, high = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price = black_scholes_price(S0, K, r, q, mid, T, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)

# ------------------------------ Monte Carlo: exact GBM to S_T ------------------------------
def mc_european_option_gbm(S0, K, r, q, sigma, T, n_paths=200_000, option_type='call',
                           antithetic=True, control_variate=True, seed=42):
    rng = np.random.default_rng(seed)
    m = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal(m)

    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * sqrt(T)

    ST_pos = S0 * np.exp(drift + vol * Z)
    if antithetic:
        ST_neg = S0 * np.exp(drift - vol * Z)
        ST_all = np.concatenate([ST_pos, ST_neg])
    else:
        ST_all = ST_pos

    df = exp(-r * T)

    if option_type.lower() == 'call':
        payoffs = np.maximum(ST_all - K, 0.0)
    else:
        payoffs = np.maximum(K - ST_all, 0.0)

    disc_payoffs = df * payoffs

    if control_variate:
        X = df * ST_all
        EX = S0 * exp(-q * T)
        cov_YX = np.cov(disc_payoffs, X, ddof=1)[0, 1]
        var_X  = np.var(X, ddof=1)
        c_opt  = cov_YX / var_X if var_X > 0 else 0.0
        adj = disc_payoffs - c_opt * (X - EX)
        est = np.mean(adj)
        sample_std = np.std(adj, ddof=1)
    else:
        est = np.mean(disc_payoffs)
        sample_std = np.std(disc_payoffs, ddof=1)

    n = len(ST_all)
    stderr = sample_std / sqrt(n)
    ci95 = (est - 1.96 * stderr, est + 1.96 * stderr)
    bs_price = black_scholes_price(S0, K, r, q, sigma, T, option_type)
    return {"price": est, "stderr": stderr, "ci95": ci95, "bs_price": bs_price,
            "n_paths": n, "used_antithetic": antithetic, "used_control_variate": control_variate}

# ------------------------------ Evaluation helpers ------------------------------
def evaluate_single_run(result):
    price = result['price']
    bs = result['bs_price']
    stderr = result['stderr']
    ci_low, ci_high = result['ci95']
    abs_err = abs(price - bs)
    rel_err = abs_err / max(bs, 1e-12)
    bs_in_ci = (ci_low <= bs <= ci_high)
    z = (price - bs) / max(stderr, 1e-12)
    return {"abs_error": abs_err, "rel_error": rel_err, "bs_in_ci": bs_in_ci,
            "z_score": z, "ci_low": ci_low, "ci_high": ci_high}

def compare_to_market(market_df: pd.DataFrame, default_n_paths=150_000, seed_base=500):
    rows = []
    for i, row in market_df.iterrows():
        S0 = float(row['S0']); K = float(row['K']); r = float(row['r']); q = float(row['q']); T = float(row['T'])
        typ = str(row['option_type']).lower()
        market_mid = float(row['market_mid'])
        bid = float(row['market_bid']) if 'market_bid' in row and pd.notna(row['market_bid']) else None
        ask = float(row['market_ask']) if 'market_ask' in row and pd.notna(row['market_ask']) else None
        n_paths = int(row['n_paths']) if 'n_paths' in row and pd.notna(row['n_paths']) else default_n_paths
        sigma = float(row['sigma']) if 'sigma' in row and pd.notna(row['sigma']) else implied_vol_bisection(S0, K, r, q, T, market_mid, option_type=typ)

        res = mc_european_option_gbm(S0=S0, K=K, r=r, q=q, sigma=sigma, T=T,
                                     n_paths=n_paths, option_type=typ,
                                     antithetic=True, control_variate=True, seed=seed_base + i)
        price = res['price']; stderr = res['stderr']; ci_low, ci_high = res['ci95']; bs = res['bs_price']
        abs_err = abs(price - market_mid); rel_err = abs_err / max(market_mid, 1e-12)
        in_ci = (ci_low <= market_mid <= ci_high)
        in_ba = None
        if bid is not None and ask is not None and bid <= ask:
            in_ba = (bid <= price <= ask)

        rows.append({"S0": S0, "K": K, "r": r, "q": q, "T": T, "type": typ,
                    "market_mid": market_mid, "market_bid": bid, "market_ask": ask,
                    "sigma": sigma, "n_paths": n_paths, "mc_price": price, "stderr": stderr,
                    "ci_low": ci_low, "ci_high": ci_high, "bs_price": bs,
                    "abs_error": abs_err, "rel_error": rel_err,
                    "market_in_mc_ci": in_ci, "price_in_bid_ask": in_ba})

    out_df = pd.DataFrame(rows)
    rmse = float(np.sqrt(np.mean(out_df['abs_error'] ** 2))) if len(out_df) else float('nan')
    coverage_ci = float(np.mean(out_df['market_in_mc_ci'])) if len(out_df) else float('nan')
    coverage_ba = None
    if 'price_in_bid_ask' in out_df.columns and out_df['price_in_bid_ask'].notna().any():
        coverage_ba = float(np.mean(out_df['price_in_bid_ask'].dropna()))
    return out_df, rmse, coverage_ci, coverage_ba

# ------------------------------ Main execution ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Monte Carlo GBM pricing + baseline comparison")
    parser.add_argument('--market_csv', type=str, default=None,
                        help='Optional CSV with columns: S0,K,r,q,T,option_type,market_mid,[market_bid],[market_ask],[sigma],[n_paths]')
    parser.add_argument('--seed', type=int, default=7, help='Base RNG seed')
    parser.add_argument('--no_plot', action='store_true', help='Skip generating the convergence plot')
    args = parser.parse_args()

    # 1) Single-run baseline (European call)
    S0, K, r, q, sigma, T = 100, 100, 0.03, 0.01, 0.20, 1.0
    single = mc_european_option_gbm(S0, K, r, q, sigma, T,
                                    n_paths=200_000, option_type='call',
                                    antithetic=True, control_variate=True, seed=args.seed)
    single_eval = evaluate_single_run(single)
    single_df = pd.DataFrame([{**single, **single_eval}]).round(6)
    single_df.to_csv('single_run_results.csv', index=False)
    print("\n=== Single-run (European call) ===")
    print(single_df.to_string(index=False))

    # 2) Convergence study across path counts
    N_list = [20_000, 50_000, 100_000, 200_000, 400_000]
    rows = []
    for i, n in enumerate(N_list):
        res = mc_european_option_gbm(S0, K, r, q, sigma, T,
                                     n_paths=n, option_type='call',
                                     antithetic=True, control_variate=True,
                                     seed=args.seed + 100 + i)
        evalr = evaluate_single_run(res)
        rows.append({
            "n_paths": res['n_paths'], "price": res['price'], "stderr": res['stderr'],
            "ci_low": res['ci95'][0], "ci_high": res['ci95'][1],
            "bs_price": res['bs_price'], "bs_in_ci": evalr['bs_in_ci']
        })
    conv_df = pd.DataFrame(rows).round(6)
    conv_df.to_csv('convergence_results.csv', index=False)
    print("\n=== Convergence across N paths ===")
    print(conv_df.to_string(index=False))

    # Plot convergence (skip if --no_plot)
    if not args.no_plot:
        plt.figure(figsize=(7, 4.5))
        plt.plot(conv_df['n_paths'], conv_df['price'], marker='o', label='MC price')
        plt.fill_between(conv_df['n_paths'], conv_df['ci_low'], conv_df['ci_high'],
                         color='orange', alpha=0.2, label='95% CI')
        plt.axhline(conv_df['bs_price'].iloc[0], color='red', linestyle='--', label='Black–Scholes')
        plt.xlabel('Number of paths (N)')
        plt.ylabel('Price')
        plt.title('Monte Carlo price convergence (European call)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('convergence_plot.png', dpi=160)
        plt.close()
        print("\nSaved plot: convergence_plot.png")

    # 3) Cross-section vs Black–Scholes (calls & puts)
    cross_params = [
        {"K": 90,  "T": 0.5,  "option_type": 'call', "sigma": 0.22},
        {"K": 100, "T": 1.0,  "option_type": 'put',  "sigma": 0.20},
        {"K": 110, "T": 0.25, "option_type": 'call', "sigma": 0.18},
    ]
    rows = []
    for j, p in enumerate(cross_params):
        res = mc_european_option_gbm(S0, p['K'], r, q, p['sigma'], p['T'],
                                     n_paths=150_000, option_type=p['option_type'],
                                     antithetic=True, control_variate=True, seed=args.seed + 300 + j)
        evalr = evaluate_single_run(res)
        rows.append({
            "S0": S0, "K": p['K'], "T": p['T'], "type": p['option_type'], "sigma": p['sigma'],
            "mc_price": res['price'], "stderr": res['stderr'],
            "ci_low": res['ci95'][0], "ci_high": res['ci95'][1],
            "bs_price": res['bs_price'], "abs_error_vs_bs": evalr['abs_error'],
            "rel_error_vs_bs": evalr['rel_error']
        })
    cs_df = pd.DataFrame(rows).round(6)
    cs_df.to_csv('cross_section_vs_bs.csv', index=False)
    print("\n=== Cross-section vs Black–Scholes ===")
    print(cs_df.to_string(index=False))

    # 4) Optional: compare to market CSV
    if args.market_csv and os.path.isfile(args.market_csv):
        market_df = pd.read_csv(args.market_csv)
        out_df, rmse, coverage_ci, coverage_ba = compare_to_market(market_df)
        out_df = out_df.round(6)
        out_df.to_csv('market_comparison.csv', index=False)
        print("\n=== Market comparison (" + args.market_csv + ") ===")
        print(out_df.to_string(index=False))
        print(f"\nSummary: RMSE={rmse:.6f}, Market mid in MC 95% CI={coverage_ci:.0%}, "
              f"Price within bid-ask={coverage_ba if coverage_ba is not None else 'N/A'}")
    else:
        print("\n(No market CSV provided; skipped market comparison.)")

    print("\nFiles saved: single_run_results.csv, convergence_results.csv, cross_section_vs_bs.csv"
          + (", convergence_plot.png" if not args.no_plot else "")
          + (", market_comparison.csv" if args.market_csv else ""))

if __name__ == '__main__':
    main()