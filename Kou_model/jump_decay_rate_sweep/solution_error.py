import torch
from reference.exact_solution import *
from trainer import Train
from config import *




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# generate the grid
x = np.linspace(np.log(0.01), np.log(5), 100)
t = np.linspace(0, 1, 101)[1:]
X, T = np.meshgrid(x, t)
dx = x[1] - x[0]
dt = t[1] - t[0]


XX = torch.from_numpy(X).float().reshape(-1, 1).to(device)
TT = torch.from_numpy(T).float().reshape(-1, 1).to(device)

eta_1_list = [2.5, 3.0, 3.5]
eta_2_list = [1.5, 2.0, 2.5]
for eta_1 in eta_1_list:
    for eta_2 in eta_2_list:
        state_dict = torch.load(rf'solution_network_{eta_1}_{eta_2}.pth', map_location='cpu')
        model = Train(layers_ing_1=layers_ing_1, layers_ing_2=layers_ing_2, layers_pred=layers_pred, params=params,
                      eta1=eta_1, eta2=eta_2)
        model.NN_pred.load_state_dict(state_dict)
        pred_sol = model.u_pred
        with torch.no_grad():
            pred = pred_sol(XX, TT)
        pred = pred.detach().cpu().numpy().reshape(X.shape)
        # reference solution
        exact = np.load(f'../reference/exact_solution_{eta_1}_{eta_2}.npy').reshape(X.shape)
        # custom error value
        error = np.mean(np.square(pred - exact))
        print(rf'$\eta_1$', eta_1, fr'$\eta_2$:', eta_2, 'Error:', error)