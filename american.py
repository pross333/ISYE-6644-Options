# Switch to 'european' to match Black–Scholes when div_yield=0.
# Greeks here are computed with finite differences (bump-and-revalue) so they’re stable and work for both American and European cases.
# vega() returns sensitivity per 1.00 change in vol; I divide by 100 in the print so you get per 1% vol.
# theta() is per year (default bump is 1 day). Adjust abs_bump if you want a different step.


import math
from dataclasses import dataclass

@dataclass
class BinomialOption:
    stock_price: float
    strike: float
    ttm: float                 # years
    rfr: float                 # risk-free rate (cont. comp.)
    sigma: float               # annualized vol
    option_type: str = 'call'  # 'call' or 'put'
    exercise: str = 'american' # 'american' or 'european'
    steps: int = 500           # tree depth
    div_yield: float = 0.0     # continuous dividend yield

    def __post_init__(self):
        self.option_type = self.option_type.lower()
        self.exercise = self.exercise.lower()
        if self.option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.exercise not in ('american', 'european'):
            raise ValueError("exercise must be 'american' or 'european'")
        if min(self.stock_price, self.strike, self.ttm, self.sigma, self.steps) <= 0:
            raise ValueError("S, K, T, sigma, steps must be > 0")

    # --- core pricing via Cox–Ross–Rubinstein binomial tree ---
    def _price_core(self, S, K, T, r, sigma, steps, q):
        dt = T / steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-r * dt)
        p = (math.exp((r - q) * dt) - d) / (u - d)
        # numeric guard for extreme params
        p = min(1.0, max(0.0, p))

        # terminal payoffs
        vals = [0.0] * (steps + 1)
        for j in range(steps + 1):
            S_T = S * (u ** j) * (d ** (steps - j))
            if self.option_type == 'call':
                vals[j] = max(S_T - K, 0.0)
            else:
                vals[j] = max(K - S_T, 0.0)

        amer = (self.exercise == 'american')
        # backward induction
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                cont = disc * (p * vals[j + 1] + (1 - p) * vals[j])
                if amer:
                    S_ij = S * (u ** j) * (d ** (i - j))
                    intrinsic = (S_ij - K) if self.option_type == 'call' else (K - S_ij)
                    intrinsic = max(intrinsic, 0.0)
                    vals[j] = max(cont, intrinsic)
                else:
                    vals[j] = cont
        return vals[0]

    def price(self):
        return self._price_core(self.stock_price, self.strike, self.ttm,
                                self.rfr, self.sigma, self.steps, self.div_yield)

    # --- Greeks via bump-and-revalue finite differences ---
    def delta(self, rel_bump=1e-4):
        S = self.stock_price
        h = S * rel_bump
        up = self._price_core(S + h, self.strike, self.ttm, self.rfr, self.sigma, self.steps, self.div_yield)
        dn = self._price_core(S - h, self.strike, self.ttm, self.rfr, self.sigma, self.steps, self.div_yield)
        return (up - dn) / (2.0 * h)

    def gamma(self, rel_bump=1e-4):
        S = self.stock_price
        h = S * rel_bump
        up = self._price_core(S + h, self.strike, self.ttm, self.rfr, self.sigma, self.steps, self.div_yield)
        mid = self._price_core(S,     self.strike, self.ttm, self.rfr, self.sigma, self.steps, self.div_yield)
        dn = self._price_core(S - h, self.strike, self.ttm, self.rfr, self.sigma, self.steps, self.div_yield)
        return (up - 2.0 * mid + dn) / (h ** 2)

    def vega(self, abs_bump=1e-4):
        # vega per 1.00 change in sigma; divide by 100 for per 1% vol
        s_up = self._price_core(self.stock_price, self.strike, self.ttm, self.rfr, self.sigma + abs_bump, self.steps, self.div_yield)
        s_dn = self._price_core(self.stock_price, self.strike, self.ttm, self.rfr, self.sigma - abs_bump, self.steps, self.div_yield)
        return (s_up - s_dn) / (2.0 * abs_bump)

    def theta(self, abs_bump=1/365):  # approx per year; default 1 day step
        # Use backward difference: (V(T) - V(T - dT)) / dT
        T = self.ttm
        dT = min(abs_bump, 0.5 * T)  # keep positive and not too large
        now = self._price_core(self.stock_price, self.strike, T, self.rfr, self.sigma, self.steps, self.div_yield)
        earlier = self._price_core(self.stock_price, self.strike, T - dT, self.rfr, self.sigma, max(2, int(self.steps * (T - dT) / T)), self.div_yield)
        return (earlier - now) / dT  # (≈ ∂V/∂t, typically negative)

    def rho(self, abs_bump=1e-4):
        r_up = self._price_core(self.stock_price, self.strike, self.ttm, self.rfr + abs_bump, self.sigma, self.steps, self.div_yield)
        r_dn = self._price_core(self.stock_price, self.strike, self.ttm, self.rfr - abs_bump, self.sigma, self.steps, self.div_yield)
        return (r_up - r_dn) / (2.0 * abs_bump)



if __name__ == "__main__":
    stock_price = 100.0
    strike      = 100.0
    ttm         = 1.0      # years
    rfr         = 0.05     # risk-free rate
    sigma       = 0.20     # volatility

    call_am = BinomialOption(stock_price, strike, ttm, rfr, sigma,
                             option_type='call', exercise='american', steps=1000, div_yield=0.0)
    put_am  = BinomialOption(stock_price, strike, ttm, rfr, sigma,
                             option_type='put',  exercise='american', steps=1000, div_yield=0.0)
    call_eu = BinomialOption(stock_price, strike, ttm, rfr, sigma,
                             option_type='call', exercise='european', steps=1000, div_yield=0.0)

    print("American Call:")
    print(f"  Price : {call_am.price():.6f}")
    print(f"  Delta : {call_am.delta():.6f}")
    print(f"  Gamma : {call_am.gamma():.6f}")
    print(f"  Vega  : {call_am.vega()/100:.6f} per 1% vol")
    print(f"  Theta : {call_am.theta():.6f} per year")
    print(f"  Rho   : {call_am.rho():.6f}")

    print("\nAmerican Put:")
    print(f"  Price : {put_am.price():.6f}")
    print(f"  Delta : {put_am.delta():.6f}")
    print(f"  Gamma : {put_am.gamma():.6f}")
    print(f"  Vega  : {put_am.vega()/100:.6f} per 1% vol")
    print(f"  Theta : {put_am.theta():.6f} per year")
    print(f"  Rho   : {put_am.rho():.6f}")

    print("\nEuropean Call (benchmark ~ Black–Scholes for q=0):")
    print(f"  Price : {call_eu.price():.6f}")
