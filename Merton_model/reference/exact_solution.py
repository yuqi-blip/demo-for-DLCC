import numpy as np
from scipy.integrate import quad
from typing import Callable


def call_price_gilpelaez(S0: float, K: float, T: float, r: float, q: float,
                         char_func: Callable[[complex], complex]) -> float:
    """
    Compute the European call price via the Gil-Pelaez inversion formula.

    Parameters:
        S0       : current price of the underlying asset
        K        : strike price
        T        : time to maturity (years)
        r        : risk-free rate
        q        : dividend yield
        char_func: characteristic function of ln(S_T), phi(u) = E[exp(i*u*ln(S_T))]

    Returns:
        European call price
    """
    # characteristic function value phi(-i) used for normalization, should equal S0 * exp((r-q)*T)
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
    # a large limit copes with the oscillation, epsabs controls the accuracy
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
        mu_j    : mean of the logarithmic jump size
        sigma_j : standard deviation of the logarithmic jump size
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
    """Convenience function for pricing a European call under the Merton model"""
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
        eta1  : exponential parameter of upward jumps (> 0, usually > 1 for a finite first moment)
        eta2  : exponential parameter of downward jumps (> 0)
    """
    # characteristic function phi_Y(u) of the jump size Y
    phi_Y = p * eta1 / (eta1 - 1j * u) + (1 - p) * eta2 / (eta2 + 1j * u)
    # first-moment adjustment term kappa = E[e^Y] - 1
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
    """Convenience function for pricing a European call under the Kou model"""
    char_func = lambda u: kou_char_func(u, S0, T, r, q, sigma, lam, p, eta1, eta2)
    return call_price_gilpelaez(S0, K, T, r, q, char_func)


# ==========================================
# Example
# ==========================================
if __name__ == "__main__":
    # market parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02

    # Merton model parameters
    sigma_m = 0.2  # diffusion volatility
    lam_m = 1.0  # jump intensity
    mu_j = -0.1  # mean logarithmic jump size
    sigma_j = 0.2  # standard deviation of the logarithmic jump size

    # Kou model parameters
    sigma_k = 0.2
    lam_k = 1.0
    p_k = 0.3  # 30% upward jumps, 70% downward jumps
    eta1_k = 3.0  # exponential parameter of upward jumps (>1)
    eta2_k = 2.0  # exponential parameter of downward jumps

    # compute option prices
    call_merton = merton_call_price(S0, K, T, r, q, sigma_m, lam_m, mu_j, sigma_j)
    call_kou = kou_call_price(S0, K, T, r, q, sigma_k, lam_k, p_k, eta1_k, eta2_k)

    # BSM price for comparison
    from scipy.stats import norm

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma_m ** 2) * T) / (sigma_m * np.sqrt(T))
    d2 = d1 - sigma_m * np.sqrt(T)
    call_bsm = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    print(f"Black-Scholes (no jumps) : {call_bsm:.6f}")
    print(f"Merton jump-diffusion    : {call_merton:.6f}")
    print(f"Kou double-exponential   : {call_kou:.6f}")