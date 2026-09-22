import numpy as np
from base_case.config import *

def kou_mc(S0, K, T, r, sigma, lam, p, eta1, eta2, n_paths=200000, seed=42):
    np.random.seed(seed)
    kappa = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1
    drift = r - lam * kappa - 0.5 * sigma ** 2

    # number of Poisson jumps
    N = np.random.poisson(lam * T, n_paths)

    # total log jump size of each path
    log_jump = np.zeros(n_paths)
    for i in range(n_paths):
        if N[i] > 0:
            # draw N[i] jump directions
            directions = np.random.rand(N[i]) < p  # True: upward
            n_up = np.sum(directions)
            n_down = N[i] - n_up
            # upward jumps follow Exp(eta1) (positive), downward jumps follow Exp(eta2) (negative)
            jumps_up = np.random.exponential(1 / eta1, n_up)
            jumps_down = -np.random.exponential(1 / eta2, n_down)
            log_jump[i] = np.sum(jumps_up) + np.sum(jumps_down)

    # continuous diffusion part
    Z = np.random.normal(size=n_paths)
    ST = S0 * np.exp(drift * T + sigma * np.sqrt(T) * Z + log_jump)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    return price

lamb_list = [0.3, 0.4, 0.5]
s_list = [0.8, 0.9, 1.0, 1.1, 1.2]
for lamb in lamb_list:
    print(lamb)
    for s in s_list:
        price_mc = kou_mc(s, params['E'], params['T'], params['r'],
                          params['sigma'], lamb, params['p'],
                          params['eta_1'], params['eta_2'], n_paths=200000)
        print(f"Monte Carlo price: {price_mc:.6f}")

