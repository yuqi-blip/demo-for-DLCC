from exact_solution import *
from base_case.config import *
import time

x = np.linspace(np.log(0.01), np.log(5), 100)
t = np.linspace(0, 1, 101)[1:]
X, T = np.meshgrid(x, t)

xx = X.reshape(-1, 1)
tt = T.reshape(-1, 1)
xt = np.hstack([xx, tt])
ex = []

start = time.time()
for i in xt:
    S = np.exp(i[0])
    exact = merton_call_price(S0=S, K=params['E'], T=i[1], r=params['r'], q=0., sigma=params['sigma'],
                          mu_j=params['m'], sigma_j=params['v'], lam=params['lamb'])
    ex.append(exact)
ex = np.array(ex)
end = time.time()

np.save('exact_solution.npy', ex)
print(end - start, '\n', ex.shape)

