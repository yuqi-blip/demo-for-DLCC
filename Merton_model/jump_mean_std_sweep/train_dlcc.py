from trainer import Train
from config import *
import torch

m_list = [-0.1, 0., 0.1]
v_list = [0.3, 0.4, 0.5, 0.6]
for m in m_list:
    for v in v_list:
        train = Train(layers_ing=layers_ing, layers_pred=layers_pred, params=params, m=m, v=v)
        if __name__ == '__main__':
            torch.cuda.init()
            train.train()