import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from networks import NeuralNetwork
from collocation_points import CreatePoint


class MertonTrainBase:
    """PIDE / consistency-constraint training logic shared by the four Merton experiment directories.

    __init__ and train() stay in each experiment directory's own trainer.py, because the
    default numbers of sampling points and the checkpoint file names they pass in differ.
    """

    def _init_train(self, layers_ing, layers_pred, params,
                    n_inner_default, n_cons_default):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_ing = layers_ing
        self.layers_pred = layers_pred
        self.params = params

        self.epochs = params['epochs']
        # normal distribution, used later for the CDF
        self.dist = torch.distributions.Normal(0, 1)

        self.r = self.params['r']
        self.sigma = self.params['sigma']
        self.E = self.params['E']
        self.T  = self.params['T']
        self.k = math.exp(self.m + self.v ** 2 / 2) - 1

        self.S_min = 0.01
        self.S_max = 5.0
        self.x_min = params['x_min']
        self.x_max = params['x_max']
        self.y_min = self.x_min
        self.y_max = self.x_max

        self.mse = nn.MSELoss()

        self.loss_history = []
        self.NN_ing = NeuralNetwork(self.layers_ing)
        self.NN_pred = NeuralNetwork(self.layers_pred)

        # load the models onto the GPU
        self.NN_ing.to(self.device)
        self.NN_pred.to(self.device)

        # optimizer configuration (tunable)
        self.adam_lr = params.get('adam_lr', 1e-3)
        self.scheduler_step = params.get('scheduler_step', 5000)
        self.scheduler_gamma = params.get('scheduler_gamma', 0.8)

        # two independent optimizers
        self.optimizer_ing = optim.Adam(self.NN_ing.parameters(), lr=self.adam_lr)
        self.optimizer_pred = optim.Adam(self.NN_pred.parameters(), lr=self.adam_lr)

        # the learning-rate schedulers are split accordingly
        self.scheduler_ing = StepLR(self.optimizer_ing, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        self.scheduler_pred = StepLR(self.optimizer_pred, step_size=self.scheduler_step, gamma=self.scheduler_gamma)


        # early stopping and best-model saving
        self.early_stopping = {
            'best_step': 0,
            'ing_parameters': copy.deepcopy(self.NN_ing.state_dict()),
            'pred_parameters': copy.deepcopy(self.NN_pred.state_dict()),
            'min_loss': float('inf'),
        }
        # numbers of collocation points (configurable)
        self.n_inner = params.get('n_inner', n_inner_default)
        self.n_init = params.get('n_init', 2000)
        self.n_bound = params.get('n_bound', 1000)
        self.n_cons = params.get('n_cons', n_cons_default)


        self.inner_x = CreatePoint(lb = self.S_min, ub = self.S_max, n_points = self.n_inner).detach().to(self.device)
        self.inner_x = torch.log(self.inner_x)
        self.inner_t = CreatePoint(lb = 0, ub = self.T, n_points = self.n_inner).detach().to(self.device)

        self.init_x  = CreatePoint(lb = self.S_min, ub = self.S_max, n_points = self.n_init).detach().to(self.device)
        self.init_x = torch.log(self.init_x)
        self.init_t = CreatePoint(lb=0., ub=0., n_points= self.n_init).detach().to(self.device)

        self.lb_x = CreatePoint(lb = self.x_min, ub = self.x_min, n_points = self.n_bound).detach().to(self.device)
        self.lb_t = CreatePoint(lb=0., ub=self.T, n_points=self.n_bound).detach().to(self.device)

        self.ub_x = CreatePoint(lb = self.x_max, ub = self.x_max, n_points = self.n_bound).detach().to(self.device)
        self.ub_t = CreatePoint(lb=0., ub=self.T, n_points=self.n_bound).detach().to(self.device)

        self.cons_x = CreatePoint(lb = self.S_min, ub = self.S_max, n_points = self.n_cons).detach().to(self.device)
        self.cons_x = torch.log(self.cons_x)
        self.cons_y = CreatePoint(lb = self.y_min, ub = self.y_max, n_points = self.n_cons).detach().to(self.device)
        self.cons_t = CreatePoint(lb = 0., ub = self.T, n_points = self.n_cons).detach().to(self.device)


    def _rar_sample_cons(self, cand_x, cand_y, cand_t, n_add):
        residual = self.cons_residual(cand_x, cand_y, cand_t).detach().abs().squeeze()
        prob = residual ** 2
        prob = prob / (prob.sum() + 1e-12)

        idx = torch.multinomial(prob, n_add, replacement=False)
        return cand_x[idx], cand_y[idx], cand_t[idx]


    def _rar_sample(self, cand_x, cand_t, n_add):
        residual = self.pide_residual(cand_x, cand_t).detach().abs().squeeze()
        prob = residual ** 2
        prob = prob / (prob.sum() + 1e-12)  # avoid division by zero when all entries are 0

        idx = torch.multinomial(prob, n_add, replacement=False)
        return cand_x[idx], cand_t[idx]


    def prob_df(self, x):
        p = 1 / ((2. * np.pi) ** (1 / 2) * self.v) * torch.exp(- (x - self.m) ** 2 / (2 * self.v ** 2))
        # add a constant truncation to prevent gradient explosion
        return p

    def u_pred(self, x, t):
        pred = self.NN_pred(torch.cat((x, t), dim=1))
        return pred


    def integrate_fn(self, x, y, t):
        pred = self.NN_ing(torch.cat((x, y, t), dim=1))
        return pred


    def cons_residual(self, x, y, t):
        x = x.detach()
        y = y.detach().requires_grad_(True)
        t = t.detach()

        ff_pred = self.integrate_fn(x, y, t)
        f_y = torch.autograd.grad(ff_pred, y, grad_outputs=torch.ones_like(ff_pred), create_graph=True)[0]
        targ = (self.u_pred(y, t) * self.prob_df(y - x))  # .detach() treats the NN_pred parameters as constants
        return f_y - targ


    # consistency loss
    def cons_loss(self, x, y, t):

        cons_resi = self.cons_residual(x, y, t)

        return torch.mean(torch.square(cons_resi))


    def pide_residual(self, x, t):
        # y does not need gradient information
        y_min = torch.ones_like(x).detach() * self.y_min
        y_max = torch.ones_like(x).detach() * self.y_max

        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)


        u = self.u_pred(x, t)
        u_x = torch.autograd.grad(u, x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        u_t = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]

        integral_term = (self.integrate_fn(x, y_max, t) -
                         self.integrate_fn(x, y_min, t) -
                         self.E * torch.exp(-self.r * t) * self.dist.cdf((self.m + x - self.y_max) / self.v) +
                         torch.exp(self.m + x + self.v ** 2 / 2) *
                         self.dist.cdf((self.m + x + self.v ** 2 - self.y_max) / self.v)
                         )         # treat the NN_ing parameters as constants

        pide = (u_t - 0.5 * self.sigma ** 2 * u_xx - (self.r - 0.5 * self.sigma ** 2 - self.lamb * self.k) * u_x +
                (self.r + self.lamb) * u - self.lamb * integral_term)
        return pide


    # PIDE loss
    def pde_loss(self, x, t):
        pide = self.pide_residual(x, t)
        targ = torch.zeros_like(pide)
        return self.mse(pide, targ)


    def init_loss(self, x, t):
        x = x.detach()
        t = t.detach()
        u = self.u_pred(x, t)
        targ = torch.maximum(torch.exp(x) - self.E, torch.zeros_like(x))
        return self.mse(u, targ)


    def lb_loss(self, x, t):
        x = x.detach()
        t = t.detach()
        u = self.u_pred(x, t)
        return self.mse(u, torch.zeros_like(x))


    def ub_loss(self, x, t):
        x = x.detach()
        t = t.detach()
        u = self.u_pred(x, t)
        targ = torch.exp(x) - self.E * torch.exp(-self.r * t)
        return self.mse(u, targ)


    def loss_fn(self):

        cons_l = self.cons_loss(self.cons_x, self.cons_y, self.cons_t)
        pde_l = (self.pde_loss(self.inner_x, self.inner_t)
                 + self.init_loss(self.init_x, self.init_t)
                 + self.lb_loss(self.lb_x, self.lb_t)
                 + self.ub_loss(self.ub_x, self.ub_t))
        tol_loss = cons_l + pde_l

        return tol_loss


    def _train_loop(self):
        """Main training loop: update on the training loss each epoch and keep the best weights."""
        p_bar = tqdm(range(self.epochs), desc="Training Epochs")
        for epoch in p_bar:
            self.optimizer_ing.zero_grad()
            self.optimizer_pred.zero_grad()
            loss = self.loss_fn()
            loss.backward()

            self.optimizer_ing.step()
            self.optimizer_pred.step()

            self.scheduler_ing.step()
            self.scheduler_pred.step()

            if loss.item() < self.early_stopping['min_loss']:
                self.early_stopping['min_loss'] = loss.item()
                self.early_stopping['ing_parameters'] = copy.deepcopy(self.NN_ing.state_dict())
                self.early_stopping['pred_parameters'] = copy.deepcopy(self.NN_pred.state_dict())
                self.early_stopping["best_step"] = epoch

            # set the tqdm postfix
            p_bar.set_postfix({'Error'  :  f'{loss.item():.6f}'})

        return p_bar


    def _save_best(self, ing_name, pred_name):
        # load the model parameters with the lowest loss
        self.NN_ing.load_state_dict(self.early_stopping['ing_parameters'])
        self.NN_pred.load_state_dict(self.early_stopping['pred_parameters'])
        # save the model parameters with the lowest loss to disk
        torch.save(self.NN_ing.state_dict(), ing_name)
        torch.save(self.NN_pred.state_dict(), pred_name)
