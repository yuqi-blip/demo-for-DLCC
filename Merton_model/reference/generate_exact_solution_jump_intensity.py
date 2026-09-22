from exact_solution import *
from base_case.config import *

x = np.linspace(np.log(0.01), np.log(5), 100)
t = np.linspace(0, 1, 101)[1:]
X, T = np.meshgrid(x, t)

xx = X.reshape(-1, 1)
tt = T.reshape(-1, 1)
xt = np.hstack([xx, tt])



lamb_list = [0.8, 1.0, 1.3]
s_list = [0.8, 0.9, 1.0, 1.1, 1.2]
for lamb in lamb_list:
    print(lamb)
    for s in s_list:
        exact = merton_call_price(S0=s, K=params['E'], T=1.0, r=params['r'], q=0., sigma=params['sigma'],
                              mu_j=params['m'], sigma_j=params['v'], lam=lamb)
        print(s, exact)
    print('\n')
