import math
from networks import NeuralNetwork
from collocation_points import CreatePoint
import copy
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import torch.nn as nn

from common.training import KouTrainBase



class Train(KouTrainBase):
    def __init__(self, layers_ing_1, layers_ing_2, layers_pred, params, eta1, eta2):
        self.params = params
        self.eta_1 = eta1
        self.eta_2 = eta2
        self.lamb = self.params['lamb']

        self._init_train(layers_ing_1, layers_ing_2, layers_pred, params,
                         n_inner_default=10000, n_cons_default=10000)


    def train(self):
        self._train_loop()
        self._save_best(f'integral_network_1_{self.eta_1}_{self.eta_2}.pth',
                        f'integral_network_2_{self.eta_1}_{self.eta_2}.pth',
                        f'solution_network_{self.eta_1}_{self.eta_2}.pth')
