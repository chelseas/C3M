import torch
from torch import nn
from torch.autograd import grad
import numpy as np

effective_dim_start = 2
effective_dim_end = 4

INVERSE_METRIC = False


class MixedTanh(nn.Module):
    def __init__(self):
        super(MixedTanh, self).__init__()
        self.mixing = 0.0  # start as all tanh

    def set_mixing(self, mixing):
        self.mixing = mixing

    def forward(self, x):
        tanh = torch.nn.functional.tanh(x)
        hardtanh = torch.nn.functional.hardtanh(x) 
        return (1.0 - self.mixing) * tanh + self.mixing * hardtanh


class U_FUNC(nn.Module):
    """docstring for U_FUNC."""

    def __init__(self, model_u_w1, num_dim_x, num_dim_control):
        super(U_FUNC, self).__init__()
        self.model_u_w1 = model_u_w1
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x, xe, uref):
        # x: B x n x 1
        # u: B x m x 1
        bs = x.shape[0]

        w1_xe = self.model_u_w1(torch.cat([x, xe], dim=1).squeeze(-1)).reshape(
            bs, self.num_dim_control, -1
        )
        w1_x0 = self.model_u_w1(
            torch.cat([x, torch.zeros(xe.shape).type(xe.type())], dim=1).squeeze(-1)
        ).reshape(bs, self.num_dim_control, -1)
        u = w1_xe - w1_x0 + uref
        return u

    def convert_to_hardtanh(self):
        for i, layer in enumerate(self.model_u_w1):
            if layer._get_name() == "Tanh":
                self.model_u_w1[i] = torch.nn.Hardtanh()

    def set_mixing(self, mixing):
        for i, layer in enumerate(self.model_u_w1):
            if layer._get_name() == "MixedTanh":
                self.model_u_w1[i].set_mixing(mixing)


class W_FUNC(nn.Module):
    """docstring for W_FUNC."""

    def __init__(self, model_W, model_Wbot, num_dim_x, num_dim_control, w_lb):
        super(W_FUNC, self).__init__()
        self.model_W = model_W
        self.model_Wbot = model_Wbot
        self.w_lb = w_lb
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x):
        bs = x.shape[0]
        x = x.squeeze(-1)

        W = self.model_W(x[:, effective_dim_start:effective_dim_end]).view(
            bs, self.num_dim_x, self.num_dim_x
        )
        Wbot = self.model_Wbot(torch.ones(bs, 1).type(x.type())).view(
            bs,
            self.num_dim_x - self.num_dim_control,
            self.num_dim_x - self.num_dim_control,
        )
        W[
            :,
            0 : self.num_dim_x - self.num_dim_control,
            0 : self.num_dim_x - self.num_dim_control,
        ] = Wbot
        W[
            :,
            self.num_dim_x - self.num_dim_control : :,
            0 : self.num_dim_x - self.num_dim_control,
        ] = 0

        # W = self.model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, self.num_dim_x, self.num_dim_x)

        W = W.transpose(1, 2).matmul(W)
        W = W + self.w_lb * torch.eye(self.num_dim_x).view(
            1, self.num_dim_x, self.num_dim_x
        ).type(x.type())
        return W

    def convert_to_hardtanh(self):
        for i, layer in enumerate(self.model_W):
            if layer._get_name() == "Tanh":
                self.model_W[i] = torch.nn.Hardtanh()

        for i, layer in enumerate(self.model_Wbot):
            if layer._get_name() == "Tanh":
                self.model_Wbot[i] = torch.nn.Hardtanh()

    def set_mixing(self, mixing):
        for i, layer in enumerate(self.model_W):
            if layer._get_name() == "MixedTanh":
                self.model_W[i].set_mixing(mixing)

        for i, layer in enumerate(self.model_Wbot):
            if layer._get_name() == "MixedTanh":
                self.model_Wbot[i].set_mixing(mixing)


def get_model(num_dim_x, num_dim_control, w_lb, use_cuda=False):
    model_Wbot = torch.nn.Sequential(
        torch.nn.Linear(1, 128, bias=True),
        MixedTanh(),
        torch.nn.Linear(128, (num_dim_x - num_dim_control) ** 2, bias=False),
    )

    dim = effective_dim_end - effective_dim_start
    model_W = torch.nn.Sequential(
        torch.nn.Linear(dim, 128, bias=True),
        MixedTanh(),
        torch.nn.Linear(128, num_dim_x * num_dim_x, bias=False),
    )

    model_u_w1 = torch.nn.Sequential(
        # torch.nn.Linear(2 * num_dim_x, num_dim_control, bias=True),
        torch.nn.Linear(2 * num_dim_x, 64, bias=False),
        MixedTanh(),
        torch.nn.Linear(64, 64, bias=False),
        MixedTanh(),
        torch.nn.Linear(64, num_dim_control, bias=False),
    )

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()

    u_func = U_FUNC(model_u_w1, num_dim_x, num_dim_control)
    W_func = W_FUNC(model_W, model_Wbot, num_dim_x, num_dim_control, w_lb)

    return model_W, model_Wbot, model_u_w1, W_func, u_func
