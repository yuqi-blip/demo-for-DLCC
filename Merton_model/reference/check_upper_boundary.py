import matplotlib.pyplot as plt

from exact_solution import *
from base_case.config import *
import time


t = np.linspace(0, 1, 101)[1:]
t = t.reshape(-1, 1)
x = np.log(7) * np.ones_like(t)
ex = []
xt = np.hstack([x, t])

for i in xt:
    S = np.exp(i[0])
    exact = merton_call_price(S0=S, K=params['E'], T=i[1], r=params['r'], q=0., sigma=params['sigma'],
                          mu_j=params['m'], sigma_j=params['v'], lam=params['lamb'])
    ex.append(exact)
ex = np.array(ex).reshape(-1, 1)
pred = np.exp(x) - np.exp(-params['r'] * t)

plt.figure()
plt.plot(t, ex, label='Exact')
plt.plot(t, pred, label='Predict')
plt.legend()
plt.show()

plt.figure()
plt.plot(t, ex - pred, label='Exact')
plt.legend()
plt.show()


