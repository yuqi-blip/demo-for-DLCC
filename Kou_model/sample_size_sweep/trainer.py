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
    def __init__(self, layers_ing_1, layers_ing_2, layers_pred, params, N_h, N_f):
        self.params = params
        self.N_h = N_h
        self.N_f = N_f
        self.eta_1 = self.params['eta_1']
        self.eta_2 = self.params['eta_2']
        self.lamb = self.params['lamb']

        self._init_train(layers_ing_1, layers_ing_2, layers_pred, params,
                         n_inner_default=self.N_h, n_cons_default=self.N_f)


    def train(self):
        self._train_loop()
        self._save_best(f'integral_network_1_{self.N_h}_{self.N_f}.pth',
                        f'integral_network_2_{self.N_h}_{self.N_f}.pth',
                        f'solution_network_{self.N_h}_{self.N_f}.pth')
