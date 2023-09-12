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
        tanh = torch.tanh(x)
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
        # print("W.shape: ", W.shape)
        W_right = W[:, :, (self.num_dim_x - self.num_dim_control):]

        Wbot = self.model_Wbot(torch.ones(bs, 1).type(x.type())).view(
            bs,
            self.num_dim_x - self.num_dim_control,
            self.num_dim_x - self.num_dim_control,
        )

        # stack to create final W matrix
        W_left = torch.concatenate([Wbot, torch.zeros(bs, self.num_dim_control, self.num_dim_x-self.num_dim_control)], dim=1)
        W_full = torch.concatenate([W_left, W_right], dim=2)

        W_final = W_full.transpose(1, 2).matmul(W_full)
        print("0. W_final.shape = :", W_final.shape)
        W_final = W_final + self.w_lb * torch.eye(self.num_dim_x).repeat(bs,1,1).type(x.type())
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


def get_model_mixed(num_dim_x, num_dim_control, w_lb, use_cuda=False, mixing=0.0):
    M_width = 32
    M_depth = 2
    u_width = 32
    u_depth = 2
    if mixing > 0.0:
        M_width = 64
        M_depth = 4
        u_width = 64
        u_depth = 4

    Wbot_layers = [torch.nn.Linear(1, M_width, bias=True), MixedTanh()]
    for i in range(M_depth - 1):
        Wbot_layers += [
            torch.nn.Linear(M_width, M_width, bias=True),
            MixedTanh(),
        ]
    Wbot_layers.append(
        torch.nn.Linear(M_width, (num_dim_x - num_dim_control) ** 2, bias=False)
    )

    model_Wbot = torch.nn.Sequential(*Wbot_layers)

    dim = effective_dim_end - effective_dim_start
    W_layers = [torch.nn.Linear(dim, M_width, bias=True), MixedTanh()]
    for i in range(M_depth - 1):
        W_layers += [torch.nn.Linear(M_width, M_width, bias=True), MixedTanh()]
    W_layers.append(torch.nn.Linear(M_width, num_dim_x * num_dim_x, bias=False))
    model_W = torch.nn.Sequential(*W_layers)

    u_layers = [torch.nn.Linear(2 * num_dim_x, u_width, bias=False), MixedTanh()]
    for i in range(u_depth - 1):
        u_layers += [torch.nn.Linear(u_width, u_width, bias=False), MixedTanh()]
    u_layers.append(torch.nn.Linear(u_width, num_dim_control, bias=False))
    model_u_w1 = torch.nn.Sequential(*u_layers)

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()

    u_func = U_FUNC(model_u_w1, num_dim_x, num_dim_control)
    W_func = W_FUNC(model_W, model_Wbot, num_dim_x, num_dim_control, w_lb)

    u_func.set_mixing(mixing)
    W_func.set_mixing(mixing)

    return model_W, model_Wbot, model_u_w1, W_func, u_func


def get_model(num_dim_x, num_dim_control, w_lb, use_cuda=False):
    model_W, model_Wbot, model_u_w1, W_func, u_func = get_model_mixed(
        num_dim_x, num_dim_control, w_lb, use_cuda=use_cuda, mixing=0.0
    )
    (
        model_W_hard,
        model_Wbot_hard,
        model_u_w1_hard,
        W_func_hard,
        u_func_hard,
    ) = get_model_mixed(num_dim_x, num_dim_control, w_lb, use_cuda=use_cuda, mixing=1.0)

    return (
        model_W,
        model_Wbot,
        model_u_w1,
        W_func,
        u_func,
        model_W_hard,
        model_Wbot_hard,
        model_u_w1_hard,
        W_func_hard,
        u_func_hard,
    )
