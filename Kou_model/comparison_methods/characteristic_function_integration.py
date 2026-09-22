import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from base_case.config import *

# compute kappa
kappa = (params['p'] * params['eta_1'] / (params['eta_1'] - 1) +
         (1 - params['p']) * params['eta_2'] / (params['eta_2'] + 1) - 1)


def kou_char_func(u, S0, T, r, sigma, lam, p, eta1, eta2):
    """Kou model characteristic function (log price)"""
    drift = r - lam * kappa - 0.5 * sigma ** 2
    # characteristic function of the jump part
    jump_cf = p * eta1 / (eta1 - 1j * u) + (1 - p) * eta2 / (eta2 + 1j * u)
    cf = np.exp(1j * u * (np.log(S0) + drift * T)
                - 0.5 * sigma ** 2 * u ** 2 * T
                + lam * T * (jump_cf - 1))
    return cf


def kou_call_integral(S0, K, T, r, sigma, lam, p, eta1, eta2):
    """European call price by numerical quadrature (Carr-Madan formula)"""
    alpha = 1.5  # damping factor
    logK = np.log(K)
    integrand = lambda u: np.real(
        np.exp(-1j * u * logK) * kou_char_func(u - (alpha + 1) * 1j, S0, T, r, sigma, lam, p, eta1, eta2)
        / (alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u)
    ) * np.exp(-r * T) / np.pi

    integral, _ = quad(integrand, 0, 200, limit=200)
    return np.exp(-alpha * logK) * integral

lamb_list = [0.3, 0.4, 0.5]
s_list = [0.8, 0.9, 1.0, 1.1, 1.2]
for lamb in lamb_list:
    print(lamb)
    for s in s_list:
        price_integral = kou_call_integral(s, params['E'], params['T'], params['r'],
                                           params['sigma'], lamb, params['p'],
                                           params['eta_1'], params['eta_2'])
        print(f"characteristic function quadrature price: {price_integral:.6f}")