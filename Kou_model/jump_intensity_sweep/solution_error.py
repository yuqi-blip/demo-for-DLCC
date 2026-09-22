import torch
from reference.exact_solution import *
import train_dlcc
import math

model = train_dlcc.train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





lamb_list = [0.3, 0.4, 0.5]
s_list = [0.8, 0.9, 1.0, 1.1, 1.2]
for lamb in lamb_list:
    print(lamb)
    for s in s_list:
        x = torch.tensor([[math.log(s)]]).float().to(device)
        t = torch.ones_like(x).float().to(device)

        state_dict = torch.load(rf'solution_network_{lamb}.pth', map_location='cpu')
        model.NN_pred.load_state_dict(state_dict)
        pred_sol = model.u_pred
        with torch.no_grad():
            pred = pred_sol(x, t).cpu().numpy().item()
            print(s, pred)
        print('\n')