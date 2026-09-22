import numpy as np
import matplotlib.pyplot as plt



# generate data (same as the original code, but flattened into 1-D arrays for scatter)
t = np.linspace(0, 1, 11)
S1 = np.linspace(0, 5, 26)[1:]          # 30 points, from 5/31 to 5
t_grid, S1_grid = np.meshgrid(t, S1)    # shape (30,30)
x1 = np.log(S1_grid)                    # log(S1)

t_ones = np.zeros_like(t_grid)



plt.rcParams.update({
    'font.size': 16,          # global default font size
    'axes.labelsize': 16,     # axis label font size
    'axes.titlesize': 16,     # subplot title font size
    'xtick.labelsize': 16,    # x tick label font size
    'ytick.labelsize': 16,    # y tick label font size
    'legend.fontsize': 16,    # legend font size
    'figure.dpi': 110,        # figure resolution
    'savefig.dpi': 300,       # saved-figure resolution (300 dpi is common for papers)
    'axes.linewidth': 1.2,    # axis line width
})
# create 2x2 subplots
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
# shared colormap and normalization (graded by the t value from 0 to 1)
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=0, vmax=1)


# --- subplot 1: x1 vs t (log(S) vs t) ---
# --- subplot 2: S1 vs t ---
ax = axes[0]
sc2 = ax.scatter(S1_grid.ravel(), t_grid.ravel(),
                 c=t_ones.ravel(), cmap=cmap, norm=norm,
                 s=20)

ax.set_xticks([0, 2.5, 5])
ax.set_ylabel(r'$\tau$')
ax.set_yticks([])
# if the labels should keep two decimal places
ax.set_xticklabels([r'$e^{x_{min}}$', r'$S$', r'$e^{x_{max}}$'])
ax.set_title(r'(a) Equidistant grid in $S$')
ax.grid(True, linestyle='--', alpha=0.3)


ax = axes[1]

sc1 = ax.scatter(x1.ravel(), t_grid.ravel(),
                 c=t_grid.T.ravel(), cmap=cmap, norm=norm,
                 s=20)

ax.set_xticks([-1.6, 0, 1.6])
ax.set_xticklabels([r'$x_{min}$', r'$x=\ln S$', r'$x_{max}$'])
ax.set_ylabel(r'$\tau$')
ax.set_yticks([])
ax.set_title(r'(b) Equidistant grid in $x=\ln S$')
ax.grid(True, linestyle='--', alpha=0.3)

# adjust the layout to avoid overlap
plt.tight_layout()
plt.savefig('sampling_grids.png')
plt.show()