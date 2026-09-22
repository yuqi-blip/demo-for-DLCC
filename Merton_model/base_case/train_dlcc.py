from trainer import Train
from config import *
import torch


train = Train(layers_ing=layers_ing, layers_pred=layers_pred, params=params)
if __name__ == '__main__':
    torch.cuda.init()
    train.train()