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

    phi = np.exp(
        1j * u * (np.log(S0) + drift * T)
        - 0.5 * sigma ** 2 * u ** 2 * T
        + lam * T * (np.exp(1j * u * mu_j - 0.5 * sigma_j ** 2 * u ** 2) - 1.0)
    )
    return phi


def merton_call_price(S0, K, T, r, q,
                      sigma, lam, mu_j, sigma_j):
    """
    Merton model European call pricing.

    S0 and T may be scalars or arrays of the same shape; the remaining parameters are scalars.
    """
    S0_arr = np.asarray(S0, dtype=float)
    T_arr = np.asarray(T, dtype=float)

    if S0_arr.shape != T_arr.shape:
        raise ValueError("S0 and T must have the same shape")

    scalar_input = (S0_arr.ndim == 0 and T_arr.ndim == 0)

    S0_flat = S0_arr.ravel()
    T_flat = T_arr.ravel()

    prices = np.empty(S0_flat.shape, dtype=float)

    for i, (s0_i, t_i) in enumerate(zip(S0_flat, T_flat)):
        # bind the current s0_i and t_i as default arguments to avoid late-binding closure issues
        char_func = lambda u, s0=s0_i, t=t_i: merton_char_func(
            u, s0, t, r, q, sigma, lam, mu_j, sigma_j
        )
        prices[i] = call_price_gilpelaez(s0_i, K, t_i, r, q, char_func)

    prices = prices.reshape(S0_arr.shape)
    return prices.item() if scalar_input else prices


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


def kou_call_price(S0, K, T, r, q,
                   sigma, lam, p, eta1, eta2):
    """
    Kou model European call pricing.

    S0 and T may be scalars or arrays of the same shape; the remaining parameters are scalars.
    """
    S0_arr = np.asarray(S0, dtype=float)
    T_arr = np.asarray(T, dtype=float)

    if S0_arr.shape != T_arr.shape:
        raise ValueError("S0 and T must have the same shape")

    scalar_input = (S0_arr.ndim == 0 and T_arr.ndim == 0)

    S0_flat = S0_arr.ravel()
    T_flat = T_arr.ravel()

    prices = np.empty(S0_flat.shape, dtype=float)

    for i, (s0_i, t_i) in enumerate(zip(S0_flat, T_flat)):
        # bind the current s0_i and t_i as default arguments to avoid late-binding closure issues
        char_func = lambda u, s0=s0_i, t=t_i: kou_char_func(
            u, s0, t, r, q, sigma, lam, p, eta1, eta2
        )
        prices[i] = call_price_gilpelaez(s0_i, K, t_i, r, q, char_func)

    prices = prices.reshape(S0_arr.shape)
    return prices.item() if scalar_input else prices


# ==========================================
# Example
# ==========================================
if __name__ == "__main__":
    # market parameters
    S0 = np.array([90.0, 100.0, 110.0])  # vector
    T = np.array([0.5, 1.0, 1.5])        # same shape as S0
    K = 100.0
    r = 0.05
    q = 0.02

    # Merton model parameters
    sigma_m = 0.2
    lam_m = 1.0
    mu_j = -0.1
    sigma_j = 0.2

    # Kou model parameters
    sigma_k = 0.2
    lam_k = 1.0
    p_k = 0.3
    eta1_k = 3.0
    eta2_k = 2.0

    # compute option prices
    call_merton = merton_call_price(S0, K, T, r, q, sigma_m, lam_m, mu_j, sigma_j)
    call_kou = kou_call_price(S0, K, T, r, q, sigma_k, lam_k, p_k, eta1_k, eta2_k)

    # BSM price for comparison (supports vectors directly)
    from scipy.stats import norm

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma_m ** 2) * T) / (sigma_m * np.sqrt(T))
    d2 = d1 - sigma_m * np.sqrt(T)
    call_bsm = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    print("S0 =", S0)
    print("T  =", T)
    print(f"Black-Scholes (no jumps) : {call_bsm}")
    print(f"Merton jump-diffusion    : {call_merton}")
    print(f"Kou double-exponential   : {call_kou}")