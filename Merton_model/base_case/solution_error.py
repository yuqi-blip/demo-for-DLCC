import numpy as np
import torch
from reference.exact_solution import *
import train_dlcc


model = train_dlcc.train
state_dict = torch.load(r'solution_network.pth', map_location='cpu')
model.NN_pred.load_state_dict(state_dict)
pred_sol = model.u_pred

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# generate the grid
x = np.linspace(np.log(0.01), np.log(5), 100)
t = np.linspace(0, 1, 101)[1:]
X, T = np.meshgrid(x, t)
dx = x[1] - x[0]
dt = t[1] - t[0]


XX = torch.from_numpy(X).float().reshape(-1, 1).to(device)
TT = torch.from_numpy(T).float().reshape(-1, 1).to(device)
with torch.no_grad():
    pred = pred_sol(XX, TT)
pred = pred.detach().cpu().numpy().reshape(X.shape)


# reference solution
exact = np.load('../reference/exact_solution.npy').reshape(X.shape)
# custom error value
error = np.mean(np.square(pred - exact))
print(error)
print(np.max(np.abs(pred - exact)))