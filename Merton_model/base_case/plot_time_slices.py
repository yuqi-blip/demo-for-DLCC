import matplotlib.pyplot as plt
import torch
import numpy as np
from reference.exact_solution import *
import train_dlcc

from common.figures import plot_solution_and_error

# ================= global font settings (recommended for paper figures) =================
plt.rcParams.update({
    'font.size': 16,          # global default font size
    'axes.labelsize': 20,     # axis label font size
    'axes.titlesize': 20,     # subplot title font size
    'xtick.labelsize': 20,    # x tick label font size
    'ytick.labelsize': 16,    # y tick label font size
    'legend.fontsize': 16,    # legend font size
    'figure.dpi': 110,        # figure resolution
    'savefig.dpi': 300,       # saved-figure resolution (300 dpi is common for papers)
    'axes.linewidth': 1.2,    # axis line width
})

# ---------- model loading ----------
model = train_dlcc.train
state_dict = torch.load(r'solution_network.pth', map_location='cpu')
model.NN_pred.load_state_dict(state_dict)

plot_solution_and_error(model, 'solution_time_slices.png')
