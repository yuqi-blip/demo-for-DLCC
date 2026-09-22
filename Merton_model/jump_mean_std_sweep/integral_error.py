import torch
from reference.exact_solution import *
import math
from trainer import Train
from config import *

m_list = [-0.1, 0., 0.1]
v_list = [0.3, 0.4, 0.5, 0.6]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
y_min = math.log(0.01)
y_max = math.log(5)

x = np.linspace(np.log(0.01), np.log(5), 100)
t = np.linspace(0, 1, 101)[1:]
X, T = np.meshgrid(x, t)
XX = torch.from_numpy(X).float().reshape(-1, 1).to(device)
TT = torch.from_numpy(T).float().reshape(-1, 1).to(device)
Y_min = y_min * torch.ones_like(XX)
Y_max = y_max * torch.ones_like(XX)

for m in m_list:
    for v in v_list:
        model = Train(layers_ing=layers_ing, layers_pred=layers_pred, params=params, m=m, v=v)
        state_dict_1 = torch.load(rf'solution_network_{m}_{v}.pth', map_location='cpu')
        state_dict_2 = torch.load(rf'integral_network_{m}_{v}.pth', map_location='cpu')
        model.NN_pred.load_state_dict(state_dict_1)
        model.NN_ing.load_state_dict(state_dict_2)
        integral = model.integrate_fn
        u_pred = model.u_pred
        norm_pdf = model.prob_df
        pred_cons = integral(XX, Y_max, TT) - integral(XX, Y_min, TT)
        pred_cons = pred_cons.reshape(-1, ).detach().cpu().numpy()
        exact_ing = np.load(f'../reference/gl_integral_{m}_{v}.npy').reshape(pred_cons.shape)
        error = pred_cons - exact_ing
        print(m, v, max(error), np.mean(np.square(error)), end='\n')