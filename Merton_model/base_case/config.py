import math
import numpy as np

x, w = np.polynomial.legendre.leggauss(51)

layers_ing= [3, 100, 100, 100, 100, 1]
layers_pred = [2, 100, 100, 100, 100, 1]

params = {
    'epochs' : 30000,
    'r'     :   0.05,
    'sigma' :   0.1,
    'T'     :   1.,
    'E'     :   1.,
    'lamb'  :   1.,
    'm'     :   -0.1,
    'v'     :   0.4,
    'x_min' : math.log(0.01),
    'x_max' : math.log(5),
    'root'  : x,
    'weights' : w
}