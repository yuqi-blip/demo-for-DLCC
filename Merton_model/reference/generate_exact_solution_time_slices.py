from exact_solution import *
from base_case.config import *
import time

x = np.linspace(np.log(0.01), np.log(5), 100)
t = np.array([0.3, 0.6, 0.9])


ex = np.array([])

for time in t:
    ex = []
    for xx in x:
        S = np.exp(xx)
        exact = merton_call_price(S0=S, K=params['E'], T=time, r=params['r'], q=0., sigma=params['sigma'],
                              mu_j=params['m'], sigma_j=params['v'], lam=params['lamb'])
        ex.append(exact)
    ex = np.array(ex)
    np.save(f'exact_solution-t{time}.npy', ex)




