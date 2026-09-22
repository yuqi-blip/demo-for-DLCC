import torch
from reference.exact_solution import *
import train_dlcc
import math

model = train_dlcc.train
state_dict_1 = torch.load(r'solution_network.pth', map_location='cpu')
state_dict_2 = torch.load(r'integral_network.pth', map_location='cpu')
model.NN_pred.load_state_dict(state_dict_1)
model.NN_ing.load_state_dict(state_dict_2)
integral = model.integrate_fn
u_pred = model.u_pred
norm_pdf = model.prob_df
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


pred_cons = integral(XX, Y_max, TT) - integral(XX, Y_min, TT)
pred_cons = pred_cons.reshape(-1, ).detach().cpu().numpy()
exact_ing = np.load('../reference/gl_integral.npy').reshape(pred_cons.shape)
error = pred_cons - exact_ing

print(max(error))
print(np.mean(np.square(error)))


'''#
# create 3D axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


# prepare grid data
x = np.arange(3)          # x direction: 0,1,2
y = np.arange(3)          # y direction: 0,1,2
X, Y = np.meshgrid(x, y)  # generate the grid

# base height of the bars (usually 0)
Z = np.zeros_like(X)

# width and depth of the bars
dx = dy = 0.6

# height of each bar (at the corresponding grid position)
dz = np.array([
    [3, 5, 2],
    [4, 6, 1],
    [2, 7, 3]
])

# color varies with height
norm = plt.Normalize(dz.min(), dz.max())
colors = plt.cm.viridis(norm(dz))

# draw the 3D bar chart (all arrays must be flattened to 1-D)
ax.bar3d(
    X.ravel(), Y.ravel(), Z.ravel(),   # lower-left corner coordinates of the bar
    dx, dy, dz.ravel(),                # width, depth, height
    color=colors.reshape(-1, 4),       # color
    shade=True                          # show shading
)

# set labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Matplotlib 3D bar chart')

plt.show()'''