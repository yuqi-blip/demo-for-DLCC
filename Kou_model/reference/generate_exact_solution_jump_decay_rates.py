from exact_solution import *
from base_case.config import *



x = np.linspace(np.log(0.01), np.log(5), 100)
t = np.linspace(0, 1, 101)[1:]
X, T = np.meshgrid(x, t)

xx = X.reshape(-1, 1)
tt = T.reshape(-1, 1)
xt = np.hstack([xx, tt])


eta_1_list = [2.5, 3.0, 3.5]
eta_2_list = [1.5, 2.0, 2.5]
for eta_1 in eta_1_list:
    for eta_2 in eta_2_list:
        ex = []
        for i in xt:
            S = np.exp(i[0])
            exact = kou_call_price(S0=S, K=params['E'], T=i[1], r=params['r'], q=0., sigma=params['sigma'],
                                  p=params['p'], eta1=eta_1, eta2=eta_2, lam=params['lamb'])
            ex.append(exact)
        ex = np.array(ex)
        np.save(f'exact_solution_{eta_1}_{eta_2}.npy', ex)
