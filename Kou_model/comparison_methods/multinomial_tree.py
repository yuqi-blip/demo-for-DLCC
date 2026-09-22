import numpy as np
from base_case.config import *

import numpy as np
from numpy.polynomial.laguerre import laggauss


def kou_trinomial_tree(S0, K, T, r, sigma, lam, p, eta1, eta2,
                       N=200, n_jump_nodes=3):
    """
    Trinomial lattice for the Kou model European call price (the paper's multinomial-tree, MT, row).

    Each node has up, middle and down diffusion branches (Boyle parameters); the jump part is
    discretised with Gauss-Laguerre nodes and added as extra branches.

    Parameters:
    S0, K, T, r: standard option parameters
    sigma: diffusion volatility
    lam: jump intensity
    p: probability of an upward jump
    eta1, eta2: double-exponential decay rates (eta1 > 1, eta2 > 0)
    N: number of time steps (>= 200 recommended)
    n_jump_nodes: number of jump discretisation nodes per direction (default 3)

    Returns:
    option_price: option price
    """
    # jump compensation kappa = E[e^Y - 1]
    kappa = p * eta1 / (eta1 - 1.0) + (1.0 - p) * eta2 / (eta2 + 1.0) - 1.0
    dt = T / N

    # ensure the jump probability does not exceed 1
    if lam * dt >= 1.0:
        raise ValueError("time step too large; increase N so that lam*dt < 1")

    # log-price drift without jumps
    drift = r - lam * kappa - 0.5 * sigma ** 2

    # trinomial parameters: up/middle/down branches
    dx = sigma * np.sqrt(3.0 * dt)
    p_u = 1.0 / 6.0 + drift * np.sqrt(dt / (12.0 * sigma ** 2))
    p_m = 2.0 / 3.0
    p_d = 1.0 / 6.0 - drift * np.sqrt(dt / (12.0 * sigma ** 2))
    if p_u < 0 or p_d < 0:
        raise ValueError("negative trinomial branch probability; increase N")

    # grid setup (log-price coordinates)
    x_min = np.log(S0) - 5 * sigma * np.sqrt(T) - 5.0 / eta2
    x_max = np.log(S0) + 5 * sigma * np.sqrt(T) + 5.0 / eta1
    M = int(np.ceil((x_max - x_min) / dx))
    x = np.linspace(x_min, x_min + M * dx, M + 1)  # grid points
    S = np.exp(x)

    # discretise the double-exponential jump density (Gauss-Laguerre quadrature)
    # upward jumps: density p * eta1 * exp(-eta1*y), y>0
    lag_nodes, lag_weights = laggauss(n_jump_nodes)  # nodes z_i and weights w_i for e^{-z}
    y_up = lag_nodes / eta1  # upward jump sizes
    w_up = p * lag_weights  # weights (they sum to p)

    # downward jumps: density (1-p) * eta2 * exp(eta2*y), y<0
    y_down = -lag_nodes / eta2  # downward jump sizes
    w_down = (1.0 - p) * lag_weights  # weights (they sum to 1-p)

    # round the jump sizes to the nearest integer grid offsets
    jump_up_offsets = np.round(y_up / dx).astype(int)
    jump_down_offsets = np.round(y_down / dx).astype(int)
    # combine the offsets and probabilities
    all_offsets = np.concatenate([jump_up_offsets, jump_down_offsets])
    all_weights = np.concatenate([w_up, w_down])
    # normalise the weights to sum to 1 (discretisation introduces a slight bias)
    all_weights = all_weights / np.sum(all_weights)

    # terminal condition
    V = np.maximum(S - K, 0.0)

    # backward induction
    no_jump_prob = 1.0 - lam * dt
    discount = np.exp(-r * dt)

    for step in range(N):
        V_old = V.copy()
        # expectation at every grid point
        for i in range(M + 1):
            # the three diffusion branches
            idx_u = min(i + 1, M)
            idx_d = max(i - 1, 0)
            val = p_u * V_old[idx_u] + p_m * V_old[i] + p_d * V_old[idx_d]
            val *= no_jump_prob

            # jump branches
            if lam * dt > 0:
                for off, w in zip(all_offsets, all_weights):
                    j = i + off
                    # boundary handling: clip to [0, M]
                    j_clipped = np.clip(j, 0, M)
                    val += lam * dt * w * V_old[j_clipped]

            V[i] = discount * val

    # linear interpolation at S0
    price = np.interp(np.log(S0), x, V)
    return price


# ============ example ============
if __name__ == "__main__":

    lamb_list = [0.3, 0.4, 0.5]
    s_list = [0.8, 0.9, 1.0, 1.1, 1.2]
    for lamb in lamb_list:
        print(lamb)
        for s in s_list:
            price_pde = kou_trinomial_tree(s, params['E'], params['T'], params['r'],
                                           params['sigma'], lamb, params['p'],
                                           params['eta_1'], params['eta_2'])
            print(f"trinomial lattice price: {price_pde:.6f}")
