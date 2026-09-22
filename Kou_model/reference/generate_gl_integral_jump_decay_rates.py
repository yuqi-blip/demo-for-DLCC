import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from vectorized_exact_solution import *          # assumes it provides kou_call_price
from base_case.config import params


def prob_df(x, eta1, eta2):
    """
    Kou double-exponential jump density; discontinuous at x = 0.
    x: independent variable (may be an array)
    """
    return np.where(x >= 0,
                    params['p'] * eta1 * np.exp(-eta1 * x),
                    (1 - params['p']) * eta2 * np.exp(eta2 * x))


def exact(x, t, eta_1, eta_2):
    """
    Option price part of the integrand; assumes kou_call_price is defined.
    x and t may be arrays; returns an array of the same shape.
    """
    ex = kou_call_price(S0=np.exp(x), K=params['E'], T=t, r=params['r'], q=0.,
                        sigma=params['sigma'], p=params['p'], eta1=eta_1,
                        eta2=eta_2, lam=params['lamb'])
    return ex


# ---------- parameter setup ----------
y_min = math.log(0.01)
y_max = math.log(5)

# standard Gauss nodes and weights (from params)
root_std = np.array(params['root'])      # shape (M,)
weights_std = np.array(params['weights'])  # shape (M,)

# independent-variable grids
x = np.linspace(y_min, y_max, 100)       # the y range matches log(0.01) to log(5)
t = np.linspace(0, 1, 101)[1:]           # drop t=0, leaving 100 points

# eta combinations to process
eta_1_list = [2.5, 3.0, 3.5]
eta_2_list = [1.5, 2.0, 2.5]

# ---------- grid density used for the precomputed option price table ----------
# a denser grid lowers the interpolation error but increases the precomputation time; 300-500 points are usually enough
N_y_grid = 400  # number of grid points in y

# y grid for the interpolator (covers the whole integration interval and extends slightly beyond it to avoid boundary error)
y_grid = np.linspace(y_min, y_max, N_y_grid)

# the time grid is the same as t above
t_grid = t

# ---------- compute the integrals ----------
for eta_1 in eta_1_list:
    print(f"Processing eta1={eta_1}")
    for eta_2 in eta_2_list:
        print(f"  eta2={eta_2}")

        # 1. precompute the option price table: a single call to exact gives a matrix of shape (len(y_grid), len(t_grid))
        Y_mesh, T_mesh = np.meshgrid(y_grid, t_grid, indexing='ij')
        price_table = exact(Y_mesh, T_mesh, eta_1, eta_2)   # (N_y_grid, len(t))

        # 2. create the linear interpolator
        #    note: the first argument of exact is y (log price), the second is t
        interp = RegularGridInterpolator(
            (y_grid, t_grid),          # the two coordinate axes
            price_table,               # the corresponding function values
            method='linear',
            bounds_error=False,        # out-of-bounds is not allowed (the nodes all lie inside)
            fill_value=None
        )

        # 3. result array: shape (len(t), len(x)), matching the original output
        integral_GL = np.zeros((len(t), len(x)))

        # handle each x separately (the jump discontinuity is at y = x)
        for i, xi in enumerate(x):
            # ---- left sub-interval: [y_min, xi] ----
            if xi > y_min:
                a_left, b_left = y_min, xi
                # map the Gauss nodes onto [a_left, b_left]
                y_left = (b_left - a_left) / 2.0 * root_std + (a_left + b_left) / 2.0
                w_left = (b_left - a_left) / 2.0 * weights_std

                # the interpolator needs both y and t
                # build input points of shape (M, len(t))
                Y_left_mesh, T_left_mesh = np.meshgrid(y_left, t, indexing='ij')
                points_left = np.stack([Y_left_mesh.ravel(), T_left_mesh.ravel()], axis=1)
                # evaluate the interpolator, then restore the (M, len(t)) shape
                exact_left = interp(points_left).reshape(Y_left_mesh.shape)  # (M, len(t))

                # prob_df depends only on y - xi and has shape (M,)
                prob_left = prob_df(y_left - xi, eta_1, eta_2)              # (M,)

                # weighted sum over the node axis gives the integral as a function of t
                integral_left = np.sum(prob_left[:, None] * w_left[:, None] * exact_left, axis=0)  # (len(t),)
            else:
                integral_left = np.zeros(len(t))

            # ---- right sub-interval: [xi, y_max] ----
            if xi < y_max:
                a_right, b_right = xi, y_max
                y_right = (b_right - a_right) / 2.0 * root_std + (a_right + b_right) / 2.0
                w_right = (b_right - a_right) / 2.0 * weights_std

                Y_right_mesh, T_right_mesh = np.meshgrid(y_right, t, indexing='ij')
                points_right = np.stack([Y_right_mesh.ravel(), T_right_mesh.ravel()], axis=1)
                exact_right = interp(points_right).reshape(Y_right_mesh.shape)  # (M, len(t))

                prob_right = prob_df(y_right - xi, eta_1, eta_2)               # (M,)

                integral_right = np.sum(prob_right[:, None] * w_right[:, None] * exact_right, axis=0)  # (len(t),)
            else:
                integral_right = np.zeros(len(t))

            # total integral = left sub-interval + right sub-interval
            integral_GL[:, i] = integral_left + integral_right

        # save the result (shape (len(t), len(x)))
        np.save(f'gl_integral_{eta_1}_{eta_2}.npy', integral_GL)
        print(f"  Saved gl_integral_{eta_1}_{eta_2}.npy")

print("All computations finished.")