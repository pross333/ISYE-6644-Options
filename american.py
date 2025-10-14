# Switch to 'european' to match Black–Scholes when div_yield=0.
# Greeks here are computed with finite differences (bump-and-revalue) so they’re stable and work for both American and European cases.
# vega() returns sensitivity per 1.00 change in vol; I divide by 100 in the print so you get per 1% vol.
# theta() is per year (default bump is 1 day). Adjust abs_bump if you want a different step.


import math
from dataclasses import dataclass
from typing import Iterable, Optional, Literal, Dict, Any

BarrierType = Optional[Dict[str, Any]]  # {"type": "up-and-out"|"down-and-out", "level": float, "rebate": float}

@dataclass
class BinomialOption:
    stock_price: float
    strike: float
    ttm: float                 # years
    rfr: float                 # risk-free rate (cont. comp.)
    sigma: float               # annualized vol
    option_type: str = 'call'  # 'call' or 'put'

    style: Literal['european','american','bermudan'] = 'american'
    bermudan_steps: Optional[Iterable[int]] = None   # which step indices allow exercise (0..steps); None for Euro/Amer
    steps: int = 500
    div_yield: float = 0.0
    # ---- NEW: barriers (knock-out) ----
    barrier: BarrierType = None  # dict like {"type":"up-and-out","level":120.0,"rebate":0.0}

    def __post_init__(self):
        self.option_type = self.option_type.lower()
        self.style = self.style.lower()
        if self.option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.style not in ('american', 'european', 'bermudan'):
            raise ValueError("style must be 'european', 'american', or 'bermudan'")
        if min(self.stock_price, self.strike, self.ttm, self.sigma, self.steps) <= 0:
            raise ValueError("S, K, T, sigma, steps must be > 0")
        # normalize bermudan steps
        if self.style == 'bermudan':
            if self.bermudan_steps is None:
                raise ValueError("Provide bermudan_steps when style='bermudan'")
            self._bermudan_set = set(int(i) for i in self.bermudan_steps)
        else:
            self._bermudan_set = set()

        # barrier normalization
        if self.barrier is not None:
            btype = self.barrier.get("type", "").lower()
            if btype not in ("up-and-out", "down-and-out"):
                raise ValueError("barrier.type must be 'up-and-out' or 'down-and-out' (knock-out only)")
            if "level" not in self.barrier or self.barrier["level"] <= 0:
                raise ValueError("barrier.level must be a positive number")
            self.barrier.setdefault("rebate", 0.0)

    # ------------ helpers ------------
    def _knocked_out(self, S_node: float) -> bool:
        if self.barrier is None:
            return False
        btype = self.barrier["type"]
        H = self.barrier["level"]
        if btype == "up-and-out":
            return S_node >= H
        else:  # down-and-out
            return S_node <= H

    def _rebate_disc(self, t_idx: int, dt: float) -> float:
        # rebate paid immediately upon knock-out, discounted from T: rebate * exp(-r*(T - t))
        # at terminal (t_idx==steps), this equals rebate * exp(0) = rebate
        return float(self.barrier["rebate"]) * math.exp(-self.rfr * (self.ttm - t_idx * dt))

    # --- core pricing via Cox–Ross–Rubinstein binomial tree ---
    def _price_core(self, S, K, T, r, sigma, steps, q):
        dt = T / steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-r * dt)
        p = (math.exp((r - q) * dt) - d) / (u - d)
        p = min(1.0, max(0.0, p))  # numeric guard

        is_call = (self.option_type == 'call')
        # terminal payoffs (or rebates if knocked out)
        vals = [0.0] * (steps + 1)
        for j in range(steps + 1):
            S_T = S * (u ** j) * (d ** (steps - j))
            if self._knocked_out(S_T):
                vals[j] = self._rebate_disc(steps, dt)
            else:
                vals[j] = max(S_T - K, 0.0) if is_call else max(K - S_T, 0.0)

        # backward induction
        for i in range(steps - 1, -1, -1):
            allow_exercise = (
                (self.style == 'american') or
                (self.style == 'bermudan' and i in self._bermudan_set)
            )
            for j in range(i + 1):
                S_ij = S * (u ** j) * (d ** (i - j))

                # barrier knock-out at this node → rebate (discounted from maturity)
                if self._knocked_out(S_ij):
                    vals[j] = self._rebate_disc(i, dt)
                    continue

                cont = disc * (p * vals[j + 1] + (1 - p) * vals[j])

                if allow_exercise:
                    intrinsic = (S_ij - K) if is_call else (K - S_ij)
                    intrinsic = max(intrinsic, 0.0)
                    vals[j] = max(cont, intrinsic)
                else:
                    vals[j] = cont
        return vals[0]

    # ---------- public API ----------
    def price(self):
        return self._price_core(self.stock_price, self.strike, self.ttm,
                                self.rfr, self.sigma, self.steps, self.div_yield)

    # Greeks via bump-and-revalue (still works for Bermudan & barriers)
    def delta(self, rel_bump=1e-4):
        S, h = self.stock_price, self.stock_price * rel_bump
        up = self._price_core(S + h, self.strike, self.ttm, self.rfr, self.sigma, self.steps, self.div_yield)
        dn = self._price_core(S - h, self.strike, self.ttm, self.rfr, self.sigma, self.steps, self.div_yield)
        return (up - dn) / (2.0 * h)

    def gamma(self, rel_bump=1e-4):
        S, h = self.stock_price, self.stock_price * rel_bump
        up = self._price_core(S + h, self.strike, self.ttm, self.rfr, self.sigma, self.steps, self.div_yield)
        mid = self._price_core(S,     self.strike, self.ttm, self.rfr, self.sigma, self.steps, self.div_yield)
        dn = self._price_core(S - h,  self.strike, self.ttm, self.rfr, self.sigma, self.steps, self.div_yield)
        return (up - 2.0 * mid + dn) / (h ** 2)

    def vega(self, abs_bump=1e-4):
        up = self._price_core(self.stock_price, self.strike, self.ttm, self.rfr, self.sigma + abs_bump, self.steps, self.div_yield)
        dn = self._price_core(self.stock_price, self.strike, self.ttm, self.rfr, self.sigma - abs_bump, self.steps, self.div_yield)
        return (up - dn) / (2.0 * abs_bump)

    def theta(self, day_bump=1/365):
        T = self.ttm
        dT = min(day_bump, 0.5 * T)
        now = self.price()
        steps2 = max(2, int(self.steps * (T - dT) / T))
        earlier = self._price_core(self.stock_price, self.strike, T - dT, self.rfr, self.sigma, steps2, self.div_yield)
        return (earlier - now) / dT

    def rho(self, abs_bump=1e-4):
        up = self._price_core(self.stock_price, self.strike, self.ttm, self.rfr + abs_bump, self.sigma, self.steps, self.div_yield)
        dn = self._price_core(self.stock_price, self.strike, self.ttm, self.rfr - abs_bump, self.sigma, self.steps, self.div_yield)
        return (up - dn) / (2.0 * abs_bump)

    # --------- convenience: knock-in via parity (same params) ---------
    def knock_in_price(self):
        """
        For simple parity: knock-in = vanilla - knock-out.
        Requires self.barrier to be 'up-and-in' or 'down-and-in' conceptually.
        Here we compute vanilla (no barrier) - this knock-out price.
        """
        if self.barrier is None:
            raise ValueError("Set a knock-out barrier, then use parity: knock-in = vanilla - knock-out")
        # price vanilla (no barrier)
        vanilla = BinomialOption(
            self.stock_price, self.strike, self.ttm, self.rfr, self.sigma,
            option_type=self.option_type, style=self.style, bermudan_steps=self.bermudan_steps,
            steps=self.steps, div_yield=self.div_yield, barrier=None
        ).price()
        knockout = self.price()
        return vanilla - knockout


