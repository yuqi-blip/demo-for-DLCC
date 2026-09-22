import torch
from reference.exact_solution import *
from trainer import Train
from config import *
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_hh = [5000, 10000]
N_ff = [2000, 5000, 10000]
y_min = math.log(0.01)
y_max = math.log(5)
x = np.linspace(np.log(0.01), np.log(5), 100)
t = np.linspace(0, 1, 101)[1:]
X, T = np.meshgrid(x, t)
XX = torch.from_numpy(X).float().reshape(-1, 1).to(device)
TT = torch.from_numpy(T).float().reshape(-1, 1).to(device)
Y_min = y_min * torch.ones_like(XX)
Y_max = y_max * torch.ones_like(XX)
exact_ing = np.load('../reference/gl_integral.npy')

max_gl_er = []
mse_gl_er = []
for N_h in N_hh:
    for N_f in N_ff:
        model = Train(layers_ing=layers_ing, layers_pred=layers_pred, params=params, N_h=N_h, N_f=N_f)
        state_dict_1 = torch.load(rf'solution_network_{N_h}_{N_f}.pth', map_location='cpu')
        state_dict_2 = torch.load(rf'integral_network_{N_h}_{N_f}.pth', map_location='cpu')
        model.NN_pred.load_state_dict(state_dict_1)
        model.NN_ing.load_state_dict(state_dict_2)
        integral = model.integrate_fn
        u_pred = model.u_pred
        norm_pdf = model.prob_df
        pred_cons = integral(XX, Y_max, TT) - integral(XX, Y_min, TT)
        pred_cons = pred_cons.reshape(-1, ).detach().cpu().numpy()
        exact_ing = exact_ing.reshape(pred_cons.shape)
        error = pred_cons - exact_ing
        max_gl_er.append(np.max(error))
        mse_gl_er.append(np.mean(np.square(error)))



# ---------- global font settings ----------
plt.rcParams.update({
    'font.size': 14,          # base font size
    'axes.titlesize': 16,     # subplot title
    'axes.labelsize': 15,     # axis labels
    'xtick.labelsize': 13,    # x ticks
    'ytick.labelsize': 13,    # y ticks
    'legend.fontsize': 13,
    'figure.dpi': 100,
})

# ---------- create the figure and the subplots ----------
fig = plt.figure(figsize=(18, 8))   # enlarge the canvas
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# ---------- prepare the bar positions ----------
x_centers = np.arange(len(N_ff))
y_centers = np.arange(len(N_hh))
xpos, ypos = np.meshgrid(x_centers, y_centers)
xpos = xpos.flatten() - 0.3   # positions are given directly after the offset
ypos = ypos.flatten() - 0.3
zpos = np.zeros_like(xpos)
dx = dy = 0.6

# ---------- first subplot: Max Error ----------
dz1 = np.array(max_gl_er)
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz1, shade=True,
          color='#4FC3F7', edgecolor='black', linewidth=0.8)
ax1.set_xticks(x_centers)
ax1.set_xticklabels(N_ff, fontsize=13)
ax1.set_yticks(y_centers)
ax1.set_yticklabels(N_hh, fontsize=13)
ax1.set_xlabel(r'$N^1$', fontsize=20, labelpad=12)
ax1.set_ylabel(r'$N_h$', fontsize=20, labelpad=12)
ax1.set_title(r'(a) $\mathcal{E}^c_{L^\infty}$', fontsize=23, pad=15)
ax1.view_init(elev=25, azim=-60)
ax1.grid(False)
# adjust the z-axis tick label size (3D plots sometimes need this set separately)
ax1.tick_params(axis='z', labelsize=13)
# cap the number of z-axis ticks on the first subplot ax1 (5, for example)
ax1.zaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

# ---------- second subplot: MSE Error ----------
dz2 = np.array(mse_gl_er)
ax2.bar3d(xpos, ypos, zpos, dx, dy, dz2, shade=True,
          color='#4FC3F7', edgecolor='black', linewidth=0.8)
ax2.set_xticks(x_centers)
ax2.set_xticklabels(N_ff, fontsize=13)
ax2.set_yticks(y_centers)
ax2.set_yticklabels(N_hh, fontsize=13)
ax2.set_xlabel(r'$N^1$', fontsize=20, labelpad=12)
ax2.set_ylabel(r'$N_h$', fontsize=20, labelpad=12)
ax2.set_title(r'(b) $\mathcal{E}^c_{L^2}$', fontsize=23, pad=15)
ax2.view_init(elev=25, azim=-60)
ax2.grid(False)
ax2.tick_params(axis='z', labelsize=13)
# cap the number of z-axis ticks on the second subplot ax2
ax2.zaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
# ---------- layout and save ----------
plt.tight_layout()

plt.savefig('integral_error_bars.png', dpi=300, bbox_inches='tight')
plt.show()