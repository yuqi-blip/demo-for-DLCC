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

from common.training import MertonTrainBase



class Train(MertonTrainBase):
    def __init__(self, layers_ing, layers_pred, params, N_h, N_f):
        self.params = params
        self.N_h = N_h
        self.N_f = N_f
        self.m = self.params['m']
        self.v = self.params['v']
        self.lamb = self.params['lamb']

        self._init_train(layers_ing, layers_pred, params,
                         n_inner_default=self.N_h, n_cons_default=self.N_f)


    def train(self):
        self._train_loop()
        self._save_best(f'integral_network_{self.N_h}_{self.N_f}.pth',
                        f'solution_network_{self.N_h}_{self.N_f}.pth')
