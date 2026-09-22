from trainer import Train
from config import *
import torch


train = Train(layers_ing_1=layers_ing_1, layers_ing_2=layers_ing_2, layers_pred=layers_pred, params=params)
if __name__ == '__main__':
    torch.cuda.init()
    train.train()