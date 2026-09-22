import numpy as np

def merton_mc(S, K, T, r, sigma, lam, muJ, sigmaJ, n_paths=100000, seed=42):
    np.random.seed(seed)
    dt = T  # simulate the terminal price directly, ignoring intermediate paths
    kappa = np.exp(muJ + 0.5*sigmaJ**2) - 1
    # draw the number of Poisson jumps
    N = np.random.poisson(lam*T, n_paths)
    # generate the jump sizes (log-normal)
    J = np.exp(muJ + sigmaJ*np.random.normal(size=n_paths) * np.sqrt(1))  # each jump is independent, but here all jumps are assumed identical for simplicity? they should be generated separately
    # more accurately: for each path the jump product is exp( sum_{i=1}^{N} (muJ + sigmaJ*Z_i) )
    # first generate the total logarithmic jump size
    log_jump = np.zeros(n_paths)
    for i in range(n_paths):
        if N[i] > 0:
            log_jump[i] = np.sum(np.random.normal(muJ, sigmaJ, N[i]))
    # continuous diffusion of the no-jump part
    Z = np.random.normal(size=n_paths)
    # drift adjustment
    drift = (r - 0.5*sigma**2 - lam*kappa)*T
    ST = S * np.exp(drift + sigma*np.sqrt(T)*Z + log_jump)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r*T) * np.mean(payoff)
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
            price_mc = merton_mc(s, K, T, r, sigma, lamb, muJ, sigmaJ, n_paths=200000)
            print(s, price_mc)
        print('\n')

