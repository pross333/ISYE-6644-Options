import math
import numpy as np
from scipy.stats import norm

class BlackScholes:
    start_price: float
    strike_price: float
    rfr: float
    time: float
    sigma: float
    option: str = 'call'
    d1: float
    d2: float

    def __init__(self, start_price, strike_price, rfr, time, sigma, option):
        self.start_price = start_price
        self.strike_price = strike_price
        self.rfr = rfr
        self.time = time
        self.sigma = sigma
        self.option = option.lower()
        if self.option not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
        if min(self.start_price, self.strike_price, self.time, self.sigma) <= 0:
            raise ValueError("Prices, risk free interest rate, time until expiration, and volatility must be > 0")
        self.d1 = self._calc_d1()
        self.d2 = self._calc_d2()

    def _calc_d1(self):
        return (
            (math.log(self.start_price / self.strike_price) + (self.rfr + 0.5 * self.sigma**2) * self.time) /
            (self.sigma * math.sqrt(self.time))
        )

    def _calc_d2(self):
        return (self.d1 - self.sigma * math.sqrt(self.time))

    def get_premium(self):
        if self.option == 'call':
            return self.start_price * norm.cdf(self.d1) - self.strike_price * np.exp(-self.rfr * self.time) * norm.cdf(self.d2)
        else:
            return self.strike_price * np.exp(-self.rfr * self.time) * norm.cdf(-self.d2) - self.start_price * norm.cdf(-self.d1)

    def get_delta(self):
        if self.option == 'call':
            return norm.cdf(self.d1)
        else:
            return norm.cdf(self.d1) - 1

    def get_gamma(self):
        return norm.pdf(self.d1) / (self.start_price * self.sigma * math.sqrt(self.time))

    def get_theta(self):
        if self.option == 'call':
            return (
                (self.start_price * self.sigma * norm.pdf(self.d1)) / (2 * math.sqrt(self.time)) -
                self.rfr * self.strike_price * math.exp(-self.rfr * self.time) * norm.cdf(self.d2)
            )
        else:
            return (
                    (self.start_price * self.sigma * norm.pdf(self.d1)) / (2 * math.sqrt(self.time)) +
                    self.rfr * self.strike_price * math.exp(-self.rfr * self.time) * norm.cdf(-self.d2)
            )

    def get_vega(self):
        return self.start_price * math.sqrt(self.time) * norm.pdf(self.d1)

    def get_rho(self):
        if self.option == 'call':
            return self.strike_price * self.time * np.exp(-self.rfr * self.time) * norm.cdf(self.d2)
        else:
            return -self.strike_price * self.time * np.exp(-self.rfr * self.time) * norm.cdf(-self.d2)
