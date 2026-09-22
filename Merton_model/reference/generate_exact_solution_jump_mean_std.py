from exact_solution import *
from base_case.config import *
import time


x = np.linspace(np.log(0.01), np.log(5), 100)
t = np.linspace(0, 1, 101)[1:]
X, T = np.meshgrid(x, t)

xx = X.reshape(-1, 1)
tt = T.reshape(-1, 1)
xt = np.hstack([xx, tt])


m_list = [-0.1, 0., 0.1]
v_list = [0.3, 0.4, 0.5, 0.6, 0.7]
for m in m_list:
    for v in v_list:
        ex = []
        for i in xt:
            S = np.exp(i[0])
            exact = merton_call_price(S0=S, K=params['E'], T=i[1], r=params['r'], q=0., sigma=params['sigma'],
                                  mu_j=m, sigma_j=v, lam=params['lamb'])
            ex.append(exact)
        ex = np.array(ex)
        np.save(f'exact_solution_{m}_{v}.npy', ex)
