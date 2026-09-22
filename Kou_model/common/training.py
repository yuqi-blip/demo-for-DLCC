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


class KouTrainBase:
    """Double-exponential jump-diffusion training logic shared by the four Kou experiment directories.

    __init__ and train() stay in each experiment directory's own trainer.py, because the
    eta/lambda values, the default sample counts and the checkpoint file names differ.
    """

    def _init_train(self, layers_ing_1, layers_ing_2, layers_pred, params,
                    n_inner_default, n_cons_default):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_ing_1 = layers_ing_1
        self.layers_ing_2 = layers_ing_2
        self.layers_pred = layers_pred
        self.params = params

        self.epochs = params['epochs']
        self.p = params['p']
        self.q = 1 - self.p

        self.r = self.params['r']
        self.sigma = self.params['sigma']
        self.E = self.params['E']
        self.T  = self.params['T']
        self.k = self.p * self.eta_1 / (self.eta_1 - 1) + self.q * self.eta_2 / (self.eta_2 + 1) - 1

        self.S_min = 0.01
        self.S_max = 5.0
        self.x_min = params['x_min']
        self.x_max = params['x_max']
        self.y_min = self.x_min
        self.y_max = self.x_max

        self.mse = nn.MSELoss()
        self.loss_history = []
        self.NN_ing_1 = NeuralNetwork(self.layers_ing_1)
        self.NN_ing_2 = NeuralNetwork(self.layers_ing_2)
        self.NN_pred = NeuralNetwork(self.layers_pred)

        # move the models to the GPU
        self.NN_ing_1.to(self.device)
        self.NN_ing_2.to(self.device)
        self.NN_pred.to(self.device)

        # optimizer configuration (tunable)
        self.adam_lr = params.get('adam_lr', 1e-3)
        self.scheduler_step = params.get('scheduler_step', 5000)
        self.scheduler_gamma = params.get('scheduler_gamma', 0.8)

        # three independent optimizers
        self.optimizer_ing_1 = optim.Adam(self.NN_ing_1.parameters(), lr=self.adam_lr)
        self.optimizer_ing_2 = optim.Adam(self.NN_ing_2.parameters(), lr=self.adam_lr)
        self.optimizer_pred = optim.Adam(self.NN_pred.parameters(), lr=self.adam_lr)

        # the learning-rate schedulers are kept separate as well
        self.scheduler_ing_1 = StepLR(self.optimizer_ing_1, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        self.scheduler_ing_2 = StepLR(self.optimizer_ing_2, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        self.scheduler_pred = StepLR(self.optimizer_pred, step_size=self.scheduler_step, gamma=self.scheduler_gamma)


        # early stopping and best-model checkpointing
        self.early_stopping = {
            'best_step': 0,
            'ing_parameters_1': copy.deepcopy(self.NN_ing_1.state_dict()),
            'ing_parameters_2': copy.deepcopy(self.NN_ing_2.state_dict()),
            'pred_parameters': copy.deepcopy(self.NN_pred.state_dict()),
            'min_loss': float('inf'),
        }

        # number of sample points (configurable)
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

        (self.cons_x_1, self.cons_y_1, self.cons_t_1, self.cons_x_2, self.cons_y_2,
         self.cons_t_2) = self.create_cons()

    # split the training set
    def create_cons(self):
        # indices with y > x
        mask_gt = self.cons_x < self.cons_y
        x_1 = self.cons_x[mask_gt]
        y_1 = self.cons_y[mask_gt]
        t_1 = self.cons_t[mask_gt]
        # indices with y < x
        mask_lt = self.cons_y < self.cons_x
        x_2 = self.cons_x[mask_lt]
        y_2 = self.cons_y[mask_lt]
        t_2 = self.cons_t[mask_lt]
        return x_1.view(-1, 1), y_1.view(-1, 1), t_1.view(-1, 1), x_2.view(-1, 1), y_2.view(-1, 1), t_2.view(-1, 1)

    def u_pred(self, x, t):
        pred = self.NN_pred(torch.cat((x, t), dim=1))
        return pred


    def integrate_fn_1(self, x, y, t):
        pred = self.NN_ing_1(torch.cat((x, y, t), dim=1))
        return pred


    def integrate_fn_2(self, x, y, t):
        pred = self.NN_ing_2(torch.cat((x, y, t), dim=1))
        return pred


    def cons_residual_1(self, x, y, t):
        x = x.detach()
        y = y.detach().requires_grad_(True)
        t = t.detach()
        ff_pred = self.integrate_fn_1(x, y, t)
        f_y = torch.autograd.grad(ff_pred, y, grad_outputs=torch.ones_like(ff_pred), create_graph=True)[0]
        targ = self.u_pred(y, t) * self.p * self.eta_1 * torch.exp(self.eta_1 * (x - y))
        return f_y - targ


    def cons_residual_2(self, x, y, t):
        x = x.detach()
        y = y.detach().requires_grad_(True)
        t = t.detach()
        ff_pred = self.integrate_fn_2(x, y, t)
        f_y = torch.autograd.grad(ff_pred, y, grad_outputs=torch.ones_like(ff_pred), create_graph=True)[0]
        targ = self.u_pred(y, t) * self.q * self.eta_2 * torch.exp(self.eta_2 * (y - x))
        return f_y - targ


    # consistency loss
    def cons_loss_1(self, x, y, t):
        cons_resi = self.cons_residual_1(x, y, t)
        return torch.mean(torch.square(cons_resi))


    def cons_loss_2(self, x, y, t):
        cons_resi= self.cons_residual_2(x, y, t)
        return torch.mean(torch.square(cons_resi))


    def ff2(self, x, t):
        f2 = (self.p * self.eta_1 / (self.eta_1 - 1) * torch.exp(self.eta_1 * (x - self.y_max) + self.y_max) -
              self.E * self.p * torch.exp(self.eta_1 * (x - self.y_max) - self.r * t))
        return f2


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

        integral_term = (self.integrate_fn_1(x, y_max, t) - self.integrate_fn_1(x, x, t) +
                         self.integrate_fn_2(x, x, t) - self.integrate_fn_2(x, y_min, t) +
                         self.ff2(x, t))

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

        cons_l = self.cons_loss_1(self.cons_x_1, self.cons_y_1, self.cons_t_1) + self.cons_loss_2(self.cons_x_2, self.cons_y_2, self.cons_t_2)
        pde_l = (self.pde_loss(self.inner_x, self.inner_t)
                 + self.init_loss(self.init_x, self.init_t)
                 + self.lb_loss(self.lb_x, self.lb_t)
                 + self.ub_loss(self.ub_x, self.ub_t))
        tol_loss = cons_l + pde_l

        return tol_loss


    def _train_loop(self):
        """Training loop: update on the training loss each round and keep the best weights."""
        p_bar = tqdm(range(self.epochs), desc="Training Epochs")
        for epoch in p_bar:
            self.optimizer_ing_1.zero_grad()
            self.optimizer_ing_2.zero_grad()
            self.optimizer_pred.zero_grad()
            loss = self.loss_fn()
            loss.backward()

            self.optimizer_ing_1.step()
            self.optimizer_ing_2.step()
            self.optimizer_pred.step()

            self.scheduler_ing_1.step()
            self.scheduler_ing_2.step()
            self.scheduler_pred.step()


            if loss.item() < self.early_stopping['min_loss']:
                self.early_stopping['min_loss'] = loss.item()
                self.early_stopping['ing_parameters_1'] = copy.deepcopy(self.NN_ing_1.state_dict())
                self.early_stopping['ing_parameters_2'] = copy.deepcopy(self.NN_ing_2.state_dict())
                self.early_stopping['pred_parameters'] = copy.deepcopy(self.NN_pred.state_dict())
                self.early_stopping["best_step"] = epoch

            # update the tqdm progress-bar description
            p_bar.set_postfix({'Error'  :  f'{loss.item():.6f}'})

        return p_bar


    def _save_best(self, ing1_name, ing2_name, pred_name):
        # load the model parameters with the smallest loss
        self.NN_ing_1.load_state_dict(self.early_stopping['ing_parameters_1'])
        self.NN_ing_2.load_state_dict(self.early_stopping['ing_parameters_2'])
        self.NN_pred.load_state_dict(self.early_stopping['pred_parameters'])
        # save the model parameters with the smallest loss to disk
        torch.save(self.NN_ing_1.state_dict(), ing1_name)
        torch.save(self.NN_ing_2.state_dict(), ing2_name)
        torch.save(self.NN_pred.state_dict(), pred_name)
