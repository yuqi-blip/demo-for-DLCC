from torch import nn as nn
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super(NeuralNetwork, self).__init__()

        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        u = x
        for linear in self.layers[:-1]:
            u = self.activation(linear(u))
        u = self.layers[-1](u)
        return u


def CreatePoint(lb, ub, n_points):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    lb = torch.tensor(lb, device=device, dtype=torch.float32)
    ub = torch.tensor(ub, device=device, dtype=torch.float32)

    # draw uniform random numbers between 0 and 1
    random =torch.rand(n_points, 1, device=device, dtype=torch.float32)
    return lb + (ub - lb) * random
