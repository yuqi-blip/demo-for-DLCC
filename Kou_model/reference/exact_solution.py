import numpy as np
from scipy.integrate import quad
from typing import Callable
from base_case.config import *

def call_price_gilpelaez(S0: float, K: float, T: float, r: float, q: float,
                         char_func: Callable[[complex], complex]) -> float:
    """
    Compute the European call price via the Gil-Pelaez inversion formula.

    Parameters:
        S0       : current underlying price
        K        : strike price
        T        : time to maturity in years
        r        : risk-free rate
        q        : dividend yield
        char_func: characteristic function of ln(S_T), phi(u) = E[exp(i*u*ln(S_T))]

    Returns:
        European call price
    """
    # characteristic function value phi(-i) used for normalisation; equals S0 * exp((r-q)*T)
    phi_minus_i = char_func(-1j)

    # the theoretical value could be used directly, but char_func is kept for consistency

    # integrand 1: used to compute Pi1
    def integrand1(u: float) -> float:
        val = np.exp(-1j * u * np.log(K)) * char_func(u - 1j) / (1j * u * phi_minus_i)
        return np.real(val)

    # integrand 2: used to compute Pi2
    def integrand2(u: float) -> float:
        val = np.exp(-1j * u * np.log(K)) * char_func(u) / (1j * u)
        return np.real(val)

    # numerical integration from 0 to infinity
    # a large limit handles the oscillation; epsabs controls the accuracy
    int1, _ = quad(integrand1, 0, np.inf, limit=2000, epsabs=1e-12)
    int2, _ = quad(integrand2, 0, np.inf, limit=2000, epsabs=1e-12)

    Pi1 = 0.5 + int1 / np.pi
    Pi2 = 0.5 + int2 / np.pi

    call = S0 * np.exp(-q * T) * Pi1 - K * np.exp(-r * T) * Pi2
    return call


# ==========================================
# Merton jump-diffusion model (1976)
# ==========================================
def merton_char_func(u: complex, S0: float, T: float, r: float, q: float,
                     sigma: float, lam: float, mu_j: float, sigma_j: float) -> complex:
    """
    Characteristic function of ln(S_T) under the Merton jump-diffusion model (risk-neutral measure).

    Parameters:
        sigma   : diffusion volatility
        lam     : jump intensity (Poisson arrival rate)
        mu_j    : mean of the log jump size
        sigma_j : standard deviation of the log jump size
    """
    kappa = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    drift = r - q - lam * kappa - 0.5 * sigma ** 2

    # characteristic function phi(u) = exp( i*u*(ln S0 + drift*T) - 0.5*sigma^2*u^2*T + lam*T*(exp(i*u*mu_j - 0.5*sigma_j^2*u^2)-1) )
    phi = np.exp(
        1j * u * (np.log(S0) + drift * T)
        - 0.5 * sigma ** 2 * u ** 2 * T
        + lam * T * (np.exp(1j * u * mu_j - 0.5 * sigma_j ** 2 * u ** 2) - 1.0)
    )
    return phi


def merton_call_price(S0: float, K: float, T: float, r: float, q: float,
                      sigma: float, lam: float, mu_j: float, sigma_j: float) -> float:
    """Convenience wrapper for Merton model European call pricing"""
    char_func = lambda u: merton_char_func(u, S0, T, r, q, sigma, lam, mu_j, sigma_j)
    return call_price_gilpelaez(S0, K, T, r, q, char_func)


# ==========================================
# Kou double-exponential jump-diffusion model (2002)
# ==========================================
def kou_char_func(u: complex, S0: float, T: float, r: float, q: float,
                  sigma: float, lam: float, p: float, eta1: float, eta2: float) -> complex:
    """
    Characteristic function of ln(S_T) under the Kou double-exponential jump-diffusion model (risk-neutral measure).

    Parameters:
        sigma : diffusion volatility
        lam   : jump intensity
        p     : probability of an upward jump (0 <= p <= 1)
        eta1  : exponential decay rate of upward jumps (> 0, usually > 1 so the first moment is finite)
        eta2  : exponential decay rate of downward jumps (> 0)
    """
    # characteristic function phi_Y(u) of the jump size Y
    phi_Y = p * eta1 / (eta1 - 1j * u) + (1 - p) * eta2 / (eta2 + 1j * u)
    # first-moment correction kappa = E[e^Y] - 1
    kappa = p * eta1 / (eta1 - 1.0) + (1 - p) * eta2 / (eta2 + 1.0) - 1.0

    drift = r - q - lam * kappa - 0.5 * sigma ** 2

    phi = np.exp(
        1j * u * (np.log(S0) + drift * T)
        - 0.5 * sigma ** 2 * u ** 2 * T
        + lam * T * (phi_Y - 1.0)
    )
    return phi


def kou_call_price(S0: float, K: float, T: float, r: float, q: float,
                   sigma: float, lam: float, p: float, eta1: float, eta2: float) -> float:
    """Convenience wrapper for Kou model European call pricing"""
    char_func = lambda u: kou_char_func(u, S0, T, r, q, sigma, lam, p, eta1, eta2)
    return call_price_gilpelaez(S0, K, T, r, q, char_func)


# ==========================================
# Example
# ==========================================
if __name__ == "__main__":
    # market parameters
    K = params['E']
    T = params['T']
    r = params['r']
    q = 0.
    lamb_list = [0.3, 0.4, 0.5]
    s_list = [0.8, 0.9, 1.0, 1.1, 1.2]
    for lamb in lamb_list:
        print(lamb)
        for s in s_list:
            call_kou = kou_call_price(s, K, T, r, q, params['sigma'], lamb,
                                      params['p'], params['eta_1'], params['eta_2'])
            print(call_kou)

