import torch
from reference.exact_solution import *
import math
from trainer import Train
from config import *


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


eta_1_list = [2.5, 3.0, 3.5]
eta_2_list = [1.5, 2.0, 2.5]
for eta_1 in eta_1_list:
    for eta_2 in eta_2_list:
        model = Train(layers_ing_1=layers_ing_1, layers_ing_2=layers_ing_2, layers_pred=layers_pred, params=params,
                      eta1=eta_1, eta2=eta_2)
        state_dict_1 = torch.load(rf'solution_network_{eta_1}_{eta_2}.pth', map_location='cpu')
        state_dict_2 = torch.load(rf'integral_network_1_{eta_1}_{eta_2}.pth', map_location='cpu')
        state_dict_3 = torch.load(rf'integral_network_2_{eta_1}_{eta_2}.pth', map_location='cpu')
        model.NN_pred.load_state_dict(state_dict_1)
        model.NN_ing_1.load_state_dict(state_dict_2)
        model.NN_ing_2.load_state_dict(state_dict_3)
        integral_1 = model.integrate_fn_1
        integral_2 = model.integrate_fn_2
        u_pred = model.u_pred
        pred_cons = integral_1(XX, Y_max, TT) - integral_1(XX, XX, TT) + integral_2(XX, XX, TT) - integral_2(XX, Y_min, TT)
        pred_cons = pred_cons.reshape(-1, ).detach().cpu().numpy()
        exact_ing = np.load(f'../reference/gl_integral_{eta_1}_{eta_2}.npy').reshape(pred_cons.shape)
        error = pred_cons - exact_ing
        print(eta_2, eta_1, np.max(exact_ing) ,max(error), np.mean(np.square(error)), end='\n')