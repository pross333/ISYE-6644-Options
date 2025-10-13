'''
Notes:

Delta: Measures how much an options price is 
expected to change based on a $1 move in the underlying asset.

Gamma: Measures the rate of change in Delta as the underlying 
price changes.

Theta: Represents the effect of time decay on an options 
value.

Vega: Gauges how sensitive an options price is to 
changes in implied volatility.

Rho: Measures how much an options price is expected to 
change based on a 1% change in the risk-free interest rate.'''

import math
from scipy.stats import norm

class BlackScholesOption:
    def __init__(self, stock_price, strike, ttm, rfr, sigma, option_type='call'):
        self.stock_price = stock_price       
        self.strike = strike
        self.ttm = ttm    
        self.rfr = rfr   
        self.sigma = sigma  
        self.option_type = option_type.lower()
        self.d1 = self._calculate_d1()
        self.d2 = self._calculate_d2()

    def _calculate_d1(self):
        return (math.log(self.stock_price / self.strike) + (self.rfr + 0.5 * self.sigma ** 2) * self.ttm) / (self.sigma * math.sqrt(self.ttm))

    def _calculate_d2(self):
        return self.d1 - self.sigma * math.sqrt(self.ttm)

    def price(self):
        if self.option_type == 'call':
            return self.stock_price * norm.cdf(self.d1) - self.strike * math.exp(-self.rfr * self.ttm) * norm.cdf(self.d2)
        elif self.option_type == 'put':
            return self.strike * math.exp(-self.rfr * self.ttm) * norm.cdf(-self.d2) - self.stock_price * norm.cdf(-self.d1)

    def delta(self):
        return norm.cdf(self.d1) if self.option_type == 'call' else norm.cdf(self.d1) - 1

    def gamma(self):
        return norm.pdf(self.d1) / (self.stock_price * self.sigma * math.sqrt(self.ttm))

    def vega(self):
        return self.stock_price * norm.pdf(self.d1) * math.sqrt(self.ttm)

    def theta(self):
        term1 = - (self.stock_price * norm.pdf(self.d1) * self.sigma) / (2 * math.sqrt(self.ttm))
        if self.option_type == 'call':
            term2 = self.rfr * self.strike * math.exp(-self.rfr * self.T) * norm.cdf(self.d2)
            return term1 - term2
        else:
            term2 = self.rfr * self.strike * math.exp(-self.rfr * self.ttm) * norm.cdf(-self.d2)
            return term1 + term2

    def rho(self):
        if self.option_type == 'call':
            return self.strike * self.ttm * math.exp(-self.rfr * self.ttm) * norm.cdf(self.d2)
        else:
            return -self.strike * self.ttm * math.exp(-self.rfr * self.ttm) * norm.cdf(-self.d2)
        
#testing
if __name__ == "__main__":
    stock_price = 100
    strike = 100 
    ttm = 1        
    rfr = 0.05    
    sigma = 0.2 

    call = BlackScholesOption(stock_price, strike, ttm, rfr, sigma, 'call')
    put = BlackScholesOption(stock_price, strike, ttm, rfr, sigma, 'put')

    print("Call Option:")
    print("Price: " +call.price())
    print("Delta: " +call.delta())
    print("Gamma: " +call.gamma())
    print("Vega: " +call.vega())
    print("Theta: " +call.theta())
    print("Rho: " +call.rho())

    print("Put Option:")
    print("Price: " +put.price())
    print("Delta: " +put.delta())
    print("Gamma: " +put.gamma())
    print("Vega: " +put.vega())
    print("Theta: " +put.theta())
    print("Rho: " +put.rho())