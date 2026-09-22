import matplotlib.pyplot as plt
import torch
from reference.exact_solution import *
import train_dlcc

from common.figures import plot_exact_vs_pred


model = train_dlcc.train
state_dict = torch.load(r'solution_network.pth', map_location='cpu')
model.NN_pred.load_state_dict(state_dict)

plot_exact_vs_pred(model, 'solution_surfaces.png',
                   title_hat=r'$\hat{U}_\theta(x,\tau)$')
