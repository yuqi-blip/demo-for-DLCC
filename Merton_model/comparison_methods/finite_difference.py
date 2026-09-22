import numpy as np
from scipy.stats import norm
import math

def merton_pde_fd(S0, K, T, r, sigma, lam, muJ, sigmaJ,
                  M=200, N=2000, S_mult=4.0):
    """
    Price a European call under the Merton jump-diffusion model with the finite-difference method.

    Parameters:
    S0: current price
    K: strike price
    T: time to maturity (years)
    r: risk-free rate
    sigma: diffusion volatility
    lam: jump intensity
    muJ: mean logarithmic jump size
    sigmaJ: standard deviation of the logarithmic jump size
    M: number of log-price grid points (default 200)
    N: number of time steps (default 2000)
    S_mult: upper-bound multiple of the grid (S_max = S_mult * K), default 4

    Returns:
    option_price: option price at S0
    """
    # jump parameters
    kappa = np.exp(muJ + 0.5 * sigmaJ ** 2) - 1  # E[Y-1]
    drift = r - lam * kappa - 0.5 * sigma ** 2  # log-price drift (excluding the jump compensator)

    # grid setup: x = ln(S)
    S_min = 1e-4 * K  # lower bound, approximately 0
    S_max = S_mult * K  # upper bound
    x_min = np.log(S_min)
    x_max = np.log(S_max)
    dx = (x_max - x_min) / M
    x = np.linspace(x_min, x_max, M + 1)
    S = np.exp(x)

    # time step
    dt = T / N

    # terminal condition: V(T, S) = max(S-K, 0)
    V = np.maximum(S - K, 0.0)

    # jump density kernel: in the x coordinate the jump increment is y = ln(Y) ~ N(muJ, sigmaJ^2)
    # we need the integral ∫ V(x+y) f(y) dy, discretized as sum_j V_j * f(x_j - x_i) * dx
    # precompute the jump density at the grid points (the kernel); the kernel is sampled in x with step dx
    # the normal density decays with distance, so truncate to the range muJ ± 10*sigmaJ
    n_std = 10
    y_min = muJ - n_std * sigmaJ
    y_max = muJ + n_std * sigmaJ
    # kernel index range: from floor((y_min)/dx) to ceil((y_max)/dx)
    idx_min = int(np.floor(y_min / dx))
    idx_max = int(np.ceil(y_max / dx))
    kernel_indices = np.arange(idx_min, idx_max + 1)
    kernel_values = norm.pdf(kernel_indices * dx, loc=muJ, scale=sigmaJ) * dx

    # backward time recursion (explicit Euler)
    for n in range(N):
        V_old = V.copy()

        # first- and second-order differences (central differences, interior points)
        # for interior points i=1..M-1
        dV_dx = (V_old[2:] - V_old[:-2]) / (2 * dx)
        d2V_dx2 = (V_old[2:] - 2 * V_old[1:-1] + V_old[:-2]) / dx ** 2

        # jump integral term
        integral = np.zeros_like(V_old)
        # for each grid point i, integral = sum_j V_old[j] * kernel[j-i]
        # implemented with a loop (simple and fast enough)
        for i in range(M + 1):
            # kernel indices relative to i
            j_indices = i + kernel_indices
            # keep only the valid range
            valid = (j_indices >= 0) & (j_indices <= M)
            j_valid = j_indices[valid]
            if len(j_valid) > 0:
                integral[i] = np.sum(V_old[j_valid] * kernel_values[valid])

        # update V (explicit scheme)
        # interior points: i=1..M-1
        V[1:-1] = V_old[1:-1] + dt * (
                0.5 * sigma ** 2 * d2V_dx2 +
                drift * dV_dx -
                (r + lam) * V_old[1:-1] +
                lam * integral[1:-1]
        )

        # boundary conditions
        # left boundary (S→0, i.e. x→-infinity): V=0
        V[0] = 0.0
        # right boundary (S→S_max): V ≈ S - K*exp(-r*(T - n*dt))
        tau = T - n * dt
        V[-1] = S_max - K * np.exp(-r * tau)

    # linear interpolation at S0
    price = np.interp(np.log(S0), x, V)
    return price


lamb_list = [0.8, 1.0, 1.3]
s_list = [0.8, 0.9, 1.0, 1.1, 1.2]
# example usage
if __name__ == "__main__":
    K, T, r =  1.0, 1.0, 0.05
    sigma, muJ, sigmaJ = 0.1, -0.1, 0.4
    for lamb in lamb_list:
        print(lamb)
        for s in s_list:
            # finite-difference price
            price_fd = merton_pde_fd(s, K, T, r, sigma, lamb, muJ, sigmaJ, M=300, N=3000)
            print(s, price_fd)
        print('\n')