#Import relevant libraries
import argparse
import csv
import sys
from math import exp, log, sqrt, erf, ceil
from typing import List, Dict, Optional

import numpy as np



#Stuff for printing tables
def format_number(x, digits=6):
    """Consistent numeric formatting."""
    try:
        return f"{x:.{digits}f}"
    except Exception:
        return str(x)


def print_table(headers: List[str], rows: List[List], digits=6):
    """Print a simple aligned ASCII table without external libraries."""
    cols = len(headers)
    #Prep string rows
    str_rows = []
    for row in rows:
        str_row = []
        for v in row:
            if isinstance(v, float):
                str_row.append(format_number(v, digits))
            else:
                str_row.append(str(v))
        str_rows.append(str_row)
    #Column widths
    widths = [len(h) for h in headers]
    for r in str_rows:
        for j in range(cols):
            widths[j] = max(widths[j], len(r[j]))
    #Print the header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(cols))
    print(header_line)
    print(sep_line)
    #Print all rows
    for r in str_rows:
        print(" | ".join(r[i].ljust(widths[i]) for i in range(cols)))
    print()



# Modeleling-> Normal CDF + Black–Scholes
def std_norm_cdf(x: float) -> float:
    """
    Standard normal CDF via error function
    """
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_price(S0, K, r, q, sigma, T, option_type='call'):
    """
    Black-Scholes/Merton price for European call/put with continuous dividend yield q.
    """
    #Edge handling: T ~ 0 or sigma ~ 0 → discounted forward intrinsic
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


# Get implied volatility via bisection
def implied_vol_bisection(S0, K, r, q, T, market_price, option_type='call',
                          tol=1e-6, max_iter=200, vol_low=1e-6, vol_high=5.0):
    """Back out implied vol so Black–Scholes price equals market price."""
    low, high = vol_low, vol_high
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


# Monte Carlo -> the exact GBM to S_T
def mc_european_option_gbm(
    S0, K, r, q, sigma, T,
    n_paths=200_000,
    option_type='call',
    antithetic=True,
    control_variate=True,
    seed=7
):
    """
    Monte Carlo price for European call/put under risk-neutral GBM.
    Uses exact one-step to S_T (no time discretization bias for Europeans).
    """
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

    #Calculate payoffs
    if option_type.lower() == 'call':
        payoffs = np.maximum(ST_all - K, 0.0)
    else:
        payoffs = np.maximum(K - ST_all, 0.0)

    disc_payoffs = df * payoffs

    #control variate
    if control_variate:
        X = df * ST_all
        EX = S0 * exp(-q * T)
        cov_YX = np.cov(disc_payoffs, X, ddof=1)[0, 1]
        var_X  = np.var(X, ddof=1)
        c_opt  = cov_YX / var_X if var_X > 0 else 0.0
        adj = disc_payoffs - c_opt * (X - EX)
        est = float(np.mean(adj))
        sample_std = float(np.std(adj, ddof=1))
    else:
        est = float(np.mean(disc_payoffs))
        sample_std = float(np.std(disc_payoffs, ddof=1))

    n = len(ST_all)
    stderr = sample_std / sqrt(n)
    ci95 = (est - 1.96 * stderr, est + 1.96 * stderr)
    bs_price = black_scholes_price(S0, K, r, q, sigma, T, option_type)

    return {
        "price": est,
        "stderr": stderr,
        "ci95": ci95,
        "bs_price": bs_price,
        "n_paths": n,
        "used_antithetic": antithetic,
        "used_control_variate": control_variate
    }


#Helpers for evaluation
def evaluate_single_run(result: Dict[str, float]) -> Dict[str, float]:
    """Diagnostics vs Black–Scholes for one MC run."""
    price = result['price']
    bs = result['bs_price']
    stderr = result['stderr']
    ci_low, ci_high = result['ci95']
    abs_err = abs(price - bs)
    rel_err = abs_err / max(bs, 1e-12)
    bs_in_ci = (ci_low <= bs <= ci_high)
    z = (price - bs) / max(stderr, 1e-12)
    return {
        "abs_error": abs_err,
        "rel_error": rel_err,
        "bs_in_ci": bs_in_ci,
        "z_score": z,
        "ci_low": ci_low,
        "ci_high": ci_high
    }


#Helpers for CSV files
def save_csv(path: str, headers: List[str], rows: List[List]):
    """Write a CSV file using only the standard library."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def read_market_csv(path: str) -> List[Dict[str, str]]:
    """Read market CSV into list-of-dicts using only the standard library."""
    out = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out


# ------------------------------
# Main execution
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Minimal Monte Carlo GBM pricing + baseline comparison")
    parser.add_argument("--market_csv", type=str, default=None,
                        help="Optional CSV: S0,K,r,q,T,option_type,market_mid,[market_bid],[market_ask],[sigma],[n_paths]")
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed")
    args = parser.parse_args()

    # 1) Single-run baseline scenario (European call)
    S0, K, r, q, sigma, T = 100.0, 100.0, 0.03, 0.01, 0.20, 1.0
    single = mc_european_option_gbm(S0, K, r, q, sigma, T,
                                    n_paths=200_000, option_type='call',
                                    antithetic=True, control_variate=True, seed=args.seed)
    single_eval = evaluate_single_run(single)

    print("\n=== Single-run (European call) ===")
    headers = ["price", "stderr", "ci_low", "ci_high", "bs_price",
               "abs_error", "rel_error", "bs_in_ci", "z_score", "n_paths"]
    row = [
        single["price"], single["stderr"], single_eval["ci_low"], single_eval["ci_high"],
        single["bs_price"], single_eval["abs_error"], single_eval["rel_error"],
        single_eval["bs_in_ci"], single_eval["z_score"], single["n_paths"]
    ]
    print_table(headers, [row])

    save_csv("single_run_results.csv", headers, [row])

    # 2) Convergence study across path counts
    N_list = [20_000, 50_000, 100_000, 200_000, 400_000]
    conv_headers = ["n_paths", "price", "stderr", "ci_low", "ci_high", "bs_price", "bs_in_ci"]
    conv_rows = []
    print("=== Convergence across N paths ===")
    for i, n in enumerate(N_list):
        res = mc_european_option_gbm(S0, K, r, q, sigma, T,
                                     n_paths=n, option_type='call',
                                     antithetic=True, control_variate=True, seed=args.seed + 100 + i)
        evalr = evaluate_single_run(res)
        conv_rows.append([
            res["n_paths"], res["price"], res["stderr"],
            evalr["ci_low"], evalr["ci_high"], res["bs_price"], evalr["bs_in_ci"]
        ])
    print_table(conv_headers, conv_rows)
    save_csv("convergence_results.csv", conv_headers, conv_rows)

    # 3) Cross-section vs Black–Scholes (calls & puts)
    cross_params = [
        {"K": 90.0,  "T": 0.5,  "option_type": "call", "sigma": 0.22},
        {"K": 100.0, "T": 1.0,  "option_type": "put",  "sigma": 0.20},
        {"K": 110.0, "T": 0.25, "option_type": "call", "sigma": 0.18},
    ]
    cs_headers = ["S0", "K", "T", "type", "sigma", "mc_price", "stderr",
                  "ci_low", "ci_high", "bs_price", "abs_error_vs_bs", "rel_error_vs_bs"]
    cs_rows = []
    print("=== Cross-section vs Black–Scholes ===")
    for j, p in enumerate(cross_params):
        res = mc_european_option_gbm(S0, p["K"], r, q, p["sigma"], p["T"],
                                     n_paths=150_000, option_type=p["option_type"],
                                     antithetic=True, control_variate=True, seed=args.seed + 300 + j)
        evalr = evaluate_single_run(res)
        cs_rows.append([
            S0, p["K"], p["T"], p["option_type"], p["sigma"],
            res["price"], res["stderr"], evalr["ci_low"], evalr["ci_high"],
            res["bs_price"], evalr["abs_error"], evalr["rel_error"]
        ])
    print_table(cs_headers, cs_rows)
    save_csv("cross_section_vs_bs.csv", cs_headers, cs_rows)

    # 4) Optional: compare to actual market CSV
    if args.market_csv:
        try:
            market_rows = read_market_csv(args.market_csv)
        except Exception as e:
            print(f"\n[WARN] Could not read market CSV '{args.market_csv}': {e}")
            market_rows = []

        if market_rows:
            out_rows, rmse, coverage_ci, coverage_ba = compare_to_market_rows(market_rows)
            mkt_headers = ["S0", "K", "r", "q", "T", "type", "market_mid", "market_bid", "market_ask",
                           "sigma", "n_paths", "mc_price", "stderr", "ci_low", "ci_high", "bs_price",
                           "abs_error", "rel_error", "market_in_mc_ci", "price_in_bid_ask"]
            mkt_data = [[
                r["S0"], r["K"], r["r"], r["q"], r["T"], r["type"],
                r["market_mid"], r["market_bid"], r["market_ask"], r["sigma"], r["n_paths"],
                r["mc_price"], r["stderr"], r["ci_low"], r["ci_high"], r["bs_price"],
                r["abs_error"], r["rel_error"], r["market_in_mc_ci"], r["price_in_bid_ask"]
            ] for r in out_rows]

            print("=== Market comparison ===")
            print_table(mkt_headers, mkt_data)
            print(f"Summary: RMSE={format_number(rmse)}, Market mid in MC 95% CI={coverage_ci:.0%}, "
                  f"Price within bid-ask={('N/A' if coverage_ba is None else f'{coverage_ba:.0%}')}")

            save_csv("market_comparison.csv", mkt_headers, mkt_data)
        else:
            print("\n[INFO] No market rows found; skipped market comparison.")
    else:
        print("\n(No market CSV provided; skipped market comparison.)")

    print("\nFiles saved: single_run_results.csv, convergence_results.csv, cross_section_vs_bs.csv"
          + (", market_comparison.csv" if args.market_csv else ""))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
