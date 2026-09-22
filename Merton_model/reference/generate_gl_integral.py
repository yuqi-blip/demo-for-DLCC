from vectorized_exact_solution import *
from base_case.config import params
import math


m = params['m']
v = params['v']

def prob_df(x):
    p = 1 / ((2. * np.pi) ** (1 / 2) * v) * np.exp(- (x - m) ** 2 / (2 * v ** 2))
    return p

def exact(x, t):
    ex = merton_call_price(S0=np.exp(x), K=params['E'], T=t, r=params['r'], q=0., sigma=params['sigma'],
                          mu_j=params['m'], sigma_j=params['v'], lam=params['lamb'])
    return ex

# integration interval
y_min = math.log(0.01)
y_max = math.log(5)

# use the same Gauss nodes and weights as the first part (read from params)
root_std = np.array(params['root'])          # standard nodes on [-1, 1]
weights_std = np.array(params['weights'])    # standard weights

# map to the actual integration interval [y_min, y_max]
y_nodes = (y_max - y_min) / 2 * root_std + (y_max + y_min) / 2
weights = (y_max - y_min) / 2 * weights_std

# independent-variable grid
x = np.linspace(np.log(0.01), np.log(5), 100)
t = np.linspace(0, 1, 101)[1:]

# 1. evaluate exact on (y_nodes, t), shape (M, len(t))
#    use broadcasting to build the grid directly, avoiding meshgrid indexing ambiguity
Y_nodes, T_grid = np.meshgrid(y_nodes, t, indexing='ij')   # (M, len(t))
exact_values = exact(Y_nodes, T_grid)                      # (M, len(t))

# 2. build the probability matrix P(y_nodes - x) and multiply it by the Gauss weights
#    target shape (len(x), M), where prob_matrix[i,j] = prob_df(y_nodes[j] - x[i]) * weights[j]
X_col = x.reshape(-1, 1)          # (len(x), 1)
Y_row = y_nodes.reshape(1, -1)    # (1, M)
diff = Y_row - X_col              # (len(x), M)
prob_matrix = prob_df(diff) * weights.reshape(1, -1)   # (len(x), M)

# 3. matrix multiplication gives the integral, shape (len(x), len(t))
integral_GL = prob_matrix @ exact_values   # (len(x), M) @ (M, len(t)) -> (len(x), len(t))

# transpose to (len(t), len(x)) to match the original output
integral_GL = integral_GL.T

np.save('gl_integral.npy', integral_GL)