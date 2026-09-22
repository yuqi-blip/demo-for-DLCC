from trainer import Train
from config import *
import torch


lamb_list = [0.3, 0.4, 0.5]
for lamb in lamb_list:
    train = Train(layers_ing_1 = layers_ing_1, layers_ing_2=layers_ing_2
                  , layers_pred=layers_pred, params=params, lamb=lamb)
    if __name__ == '__main__':
        torch.cuda.init()
        train.train()