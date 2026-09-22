from trainer import Train
from config import *
import torch


lamb_list = [0.8, 1.0, 1.3]
for lamb in lamb_list:
    train = Train(layers_ing=layers_ing, layers_pred=layers_pred, params=params, lamb=lamb)
    if __name__ == '__main__':
        torch.cuda.init()
        train.train()