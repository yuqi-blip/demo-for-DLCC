from trainer import Train
from config import *
import torch

eta_1_list = [2.5, 3.0, 3.5]
eta_2_list = [1.5, 2.0, 2.5]
for eta_1 in eta_1_list:
    for eta_2 in eta_2_list:
        train = Train(layers_ing_1=layers_ing_1, layers_ing_2=layers_ing_2, layers_pred=layers_pred, params=params, eta1=eta_1, eta2=eta_2)
        if __name__ == '__main__':
            torch.cuda.init()
            train.train()