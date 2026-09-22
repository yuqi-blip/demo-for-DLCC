import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Figure 3 / Figure 6
def plot_exact_vs_pred(model, filename, title_hat=r'$\hat{U}_{\theta}(x,\tau)$'):
    """Predicted solution / reference solution / absolute error, three panels."""
    pred_sol = model.u_pred
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # generate the grid
    x = np.linspace(np.log(0.01), np.log(5), 100)
    t = np.linspace(0, 1, 101)[1:]
    X, T = np.meshgrid(x, t)

    XX = torch.from_numpy(X).float().reshape(-1, 1).to(device)
    TT = torch.from_numpy(T).float().reshape(-1, 1).to(device)
    with torch.no_grad():
        pred = pred_sol(XX, TT)
    pred = pred.detach().cpu().numpy().reshape(X.shape)

    # reference solution
    exact = np.load('../reference/exact_solution.npy').reshape(X.shape)
    # absolute error
    error = np.abs(pred - exact)

    print('Error:', np.mean(np.square(pred - exact)))

    # the first two panels share a color range for direct comparison
    vmin = min(exact.min(), pred.min())
    vmax = max(exact.max(), pred.max())

    # ====== shared configuration ======
    FIG_SIZE = (6.0, 5.0)      # shared size for the three panels
    SAVE_DPI = 300              # save resolution
    CBAR_SIZE = "3.5%"          # colorbar width: 3.5% of the main panel width, narrow
    CBAR_PAD = 0.08             # gap between the colorbar and the main panel

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=110)

    # 1. predicted solution
    im1 = axes[0].imshow(
        pred,
        extent=[x.min(), x.max(), 0., t.max()],
        origin='lower',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax
    )
    axes[0].set_title(title_hat, fontsize=20, fontweight='bold')
    axes[0].set_xlabel('x', fontsize=20)
    axes[0].set_ylabel(rf'$\tau$', fontsize=20)
    axes[0].tick_params(axis='both',          # modify both the x and y axes
                   labelsize=14,         # tick label font size
                   colors='black',       # tick text color
                   length=6,             # tick length
                   width=1.5,            # tick width
                   direction='in')       # ticks point inward

    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. reference solution
    im2 = axes[1].imshow(
        exact,
        extent=[x.min(), x.max(), 0., t.max()],
        origin='lower',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax
    )
    axes[1].set_title(r'$U(x,\tau)$', fontsize=20, fontweight='bold')
    axes[1].set_xlabel('x', fontsize=20)
    axes[1].set_ylabel(rf'$\tau$', fontsize=20)
    axes[1].tick_params(axis='both',          # modify both the x and y axes
                   labelsize=14,         # tick label font size
                   colors='black',       # tick text color
                   length=6,             # tick length
                   width=1.5,            # tick width
                   direction='in')       # ticks point inward
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. absolute error
    im3 = axes[2].imshow(
        error,
        extent=[x.min(), x.max(), 0., t.max()],
        origin='lower',
        cmap='viridis'
    )
    axes[2].set_title(r'$|\hat{U}_\theta(x,\tau)-U(x,\tau)|$', fontsize=20, fontweight='bold')
    axes[2].set_xlabel('x', fontsize=20)
    axes[2].set_ylabel(rf'$\tau$', fontsize=20)
    axes[2].tick_params(axis='both',          # modify both the x and y axes
                   labelsize=14,         # tick label font size
                   colors='black',       # tick text color
                   length=6,             # tick length
                   width=1.5,            # tick width
                   direction='in')       # ticks point inward
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # show the maximum error value on the error panel
    axes[2].text(
        # position of the error value
        0.03, 0.97,
        f'Max error: {error.max():.3f}',
        transform=axes[2].transAxes,
        va='top', ha='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )


    for ax in axes:
        ax.set_aspect('auto')

    fig.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f'Maximum absolute error: {error.max():.4f}')


# Figure 4 / Figure 7
def plot_solution_and_error(model, filename, verbose=False):
    """Solution comparison / absolute error at three times, 2x3 subplots."""
    pred_sol = model.u_pred
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- input points ----------
    x = np.linspace(np.log(0.01), np.log(5), 100)
    t_list = [0.3, 0.6, 0.9]

    xx = torch.from_numpy(x).float().reshape(-1, 1).to(device)

    # ---------- predict at the three times ----------
    preds = []
    for time in t_list:
        tt = torch.ones_like(xx) * time
        pred = pred_sol(xx, tt)
        pred = pred.detach().cpu().numpy().reshape(-1, )
        preds.append(pred)

    # ---------- load the reference solutions ----------
    base = '../reference'
    exact_sols = {
        0.3: np.load(base + '/exact_solution-t0.3.npy').reshape(-1),
        0.6: np.load(base + '/exact_solution-t0.6.npy').reshape(-1),
        0.9: np.load(base + '/exact_solution-t0.9.npy').reshape(-1),
    }
    if verbose:
        print(exact_sols)

    # ---------- 2x3 subplots with larger horizontal and vertical spacing ----------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=110)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.10,
                        wspace=0.28, hspace=0.35)

    for j, time in enumerate(t_list):
        # ========== first row: solution comparison ==========
        ax_sol = axes[0, j]

        ax_sol.plot(
            x, preds[j],
            color='red', linewidth=6,
            label=r'Predicted Solution',
        )
        ax_sol.plot(
            x, exact_sols[time],
            color='#1f77b4', linewidth=6, linestyle='--',
            label=r'Exact Solution',
        )

        ax_sol.set_xlim(x.min(), x.max())
        ax_sol.set_ylim(-0.5, 4.5)
        ax_sol.set_xticks(np.arange(-3, 2, 1))
        ax_sol.set_yticks(np.arange(0, 5, 1))

        ax_sol.set_xlabel('x', fontsize=16)
        ax_sol.set_ylabel(r'$Solution$', fontsize=16)
        ax_sol.set_title(rf'$\tau={time}$: $\hat{{U}}_\theta$ vs $U$',
                         fontsize=16, fontweight='bold')

        ax_sol.grid(True, linestyle='--', alpha=0.4)
        ax_sol.legend(loc='upper left', fontsize=16, framealpha=0.9, edgecolor='gray')

        # ========== second row: absolute error ==========
        ax_err = axes[1, j]

        abs_e = np.abs(preds[j] - exact_sols[time])
        ax_err.plot(x, abs_e, color='#1f77b4', linewidth=4)

        # mark the peak
        y_peak = abs_e.max()
        x_peak = x[abs_e.argmax()]
        '''ax_err.plot(x_peak, y_peak, 'o', color='black', markersize=6)
        ax_err.annotate(
            f'Peak: {y_peak:.5f}',
            xy=(x_peak, y_peak),
            xytext=(x_peak - 0.6, y_peak * 1.2),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=16
        )
    '''
        # the y axis of the error panel follows the error at the current time
        y_margin = max(y_peak * 0.3, 0.001)
        ax_err.set_xlim(x.min(), x.max())
        ax_err.set_ylim(-0.001, 0.005)  # key: the upper limit must exceed the peak
        ax_err.set_xticks(np.arange(-3, 2, 1))
        ax_err.set_yticks(np.arange(0, 0.005, 0.002))

        ax_err.set_xlabel('x', fontsize=16)
        ax_err.set_ylabel(r'$Absolute\ Error$', fontsize=16)
        ax_err.set_title(rf'$\tau={time}$: Absolute Error', fontsize=16, fontweight='bold')

        ax_err.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
