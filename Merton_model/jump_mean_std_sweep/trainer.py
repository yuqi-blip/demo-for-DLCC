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
    def __init__(self, layers_ing, layers_pred, params,m , v):
        self.params = params
        self.m = m
        self.v = v
        self.lamb = self.params['lamb']

        self._init_train(layers_ing, layers_pred, params,
                         n_inner_default=10000, n_cons_default=10000)


    def train(self):
        self._train_loop()
        self._save_best(f'integral_network_{self.m}_{self.v}.pth',
                        f'solution_network_{self.m}_{self.v}.pth')
