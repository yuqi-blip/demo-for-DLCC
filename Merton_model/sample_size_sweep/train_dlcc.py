from trainer import Train
from config import *
import torch

N_hh = [5000, 10000]
N_ff = [2000, 5000, 10000]


if __name__ == '__main__':
    for N_h in N_hh:
        for N_f in N_ff:
            train = Train(layers_ing=layers_ing, layers_pred=layers_pred, params=params, N_h=N_h, N_f=N_f)
            if __name__ == '__main__':
                torch.cuda.init()
                train.train()