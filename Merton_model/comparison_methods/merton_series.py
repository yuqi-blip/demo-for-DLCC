import numpy as np
from scipy.stats import norm
from math import factorial, exp

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def merton_series(S, K, T, r, sigma, lam, muJ, sigmaJ, N=50):
    kappa = exp(muJ + 0.5*sigmaJ**2) - 1
    lam_p = lam * (1 + kappa)
    price = 0.0
    for n in range(N):
        r_n = r - lam*kappa + n*(muJ + 0.5*sigmaJ**2)/T
        sigma_n = np.sqrt(sigma**2 + n*sigmaJ**2/T)
        bs = bs_call(S, K, T, r_n, sigma_n)
        prob = exp(-lam_p*T) * (lam_p*T)**n / factorial(n)
        price += prob * bs
    return price


lamb_list = [0.8, 1.0, 1.3]
s_list = [0.8, 0.9, 1.0, 1.1, 1.2]
# example
K, T, r = 1.0, 1.0, 0.05
sigma, muJ, sigmaJ = 0.1, -0.1, 0.4
for lamb in lamb_list:
    print(lamb)
    for s in s_list:
        print("Merton series price:", merton_series(s, K, T, r, sigma, lamb, muJ, sigmaJ))
    print('\n')