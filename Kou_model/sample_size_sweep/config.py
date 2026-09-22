import math
from numpy.polynomial.legendre import leggauss

layers_ing_1 = [3, 100, 100, 100, 100, 1]
layers_ing_2 = [3, 100, 100, 100, 100, 1]
layers_pred = [2, 100, 100, 100, 100, 1]
root, weights = leggauss(26)
params = {
    'epochs' : 30000,
    'r'     :   0.02,
    'sigma' :   0.15,
    'T'     :   1.,
    'E'     :   1.,
    'lamb'  :   0.2,
    'eta_1' :   3,
    'eta_2' :   2,
    'p'     :   0.5,
    'x_min' : math.log(0.01),
    'x_max' : math.log(5),
    'root' : root,
    'weights' : weights,
}