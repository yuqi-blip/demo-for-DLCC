from exact_solution import *
from base_case.config import *

x = np.linspace(np.log(0.01), np.log(5), 100)
t = np.array([0.3, 0.6, 0.9])


ex = np.array([])

for time in t:
    ex = []
    for i in x:
        S = np.exp(i)
        exact = kou_call_price(S0=S, K=params['E'], T=time, r=params['r'], q=0.,
                               sigma=params['sigma'], eta1=params['eta_1'], eta2=params['eta_2'],
                               p=params['p'], lam=params['lamb'])
        ex.append(exact)
    ex = np.array(ex)
    np.save(f'exact_solution-t{time}.npy', ex)




