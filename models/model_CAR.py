import torch
from torch import nn
from torch.autograd import grad
import numpy as np

effective_dim_start = 2
effective_dim_end = 4

SINGLE_NETWORK_VANILLA = False
SINGLE_NETWORK_MULTIPLIED = True
INVERSE_METRIC = False


class U_FUNC(nn.Module):
    """docstring for U_FUNC."""

    def __init__(self, model_u_w1, model_u_w2, num_dim_x, num_dim_control):
        super(U_FUNC, self).__init__()
        self.model_u_w1 = model_u_w1
        self.model_u_w2 = model_u_w2
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x, xe, uref):
        # x: B x n x 1
        # u: B x m x 1
        bs = x.shape[0]

        if not SINGLE_NETWORK_VANILLA:
            w1 = self.model_u_w1(
                torch.cat(
                    [
                        x[:, effective_dim_start:effective_dim_end, :],
                        (x - xe)[:, effective_dim_start:effective_dim_end, :],
                    ],
                    dim=1,
                ).squeeze(-1)
            ).reshape(bs, -1, self.num_dim_x)
            w2 = self.model_u_w2(
                torch.cat(
                    [
                        x[:, effective_dim_start:effective_dim_end, :],
                        (x - xe)[:, effective_dim_start:effective_dim_end, :],
                    ],
                    dim=1,
                ).squeeze(-1)
            ).reshape(bs, self.num_dim_control, -1)
            if not SINGLE_NETWORK_MULTIPLIED:
                u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref
            else:
                u = w1.matmul(xe) + uref
        else:
            u1 = self.model_u_w1(
                torch.cat(
                    [
                        x[:, effective_dim_start:effective_dim_end, :],
                        (x - xe)[:, effective_dim_start:effective_dim_end, :],
                    ],
                    dim=1,
                ).squeeze(-1)
            ).reshape(bs, self.num_dim_control, 1)
            u2 = self.model_u_w1(
                torch.cat(
                    [
                        x[:, effective_dim_start:effective_dim_end, :],
                        0 * (x - xe)[:, effective_dim_start:effective_dim_end, :],
                    ],
                    dim=1,
                ).squeeze(-1)
            ).reshape(bs, self.num_dim_control, 1)

            u = u1 - u2 + uref

        return u


class W_FUNC(nn.Module):
    """docstring for W_FUNC."""

    def __init__(self, model_W, model_Wbot, num_dim_x, num_dim_control, w_lb):
        super(W_FUNC, self).__init__()
        self.model_W = model_W
        self.model_Wbot = model_Wbot
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control
        self.w_lb = w_lb

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
            0: self.num_dim_x - self.num_dim_control,
            0: self.num_dim_x - self.num_dim_control,
        ] = Wbot
        W[
            :,
            self.num_dim_x - self.num_dim_control : :,
            0: self.num_dim_x - self.num_dim_control,
        ] = 0

        # W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, self.num_dim_x, self.num_dim_x)

        # W = W.transpose(1, 2).matmul(W)
        W = W.transpose(1, 2) + W
        W = W + self.w_lb * torch.eye(self.num_dim_x).view(
            1, self.num_dim_x, self.num_dim_x
        ).type(x.type())
        return W


def get_model(num_dim_x, num_dim_control, w_lb, use_cuda=False):
    model_Wbot = torch.nn.Sequential(
        torch.nn.Linear(1, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, (num_dim_x - num_dim_control) ** 2, bias=False),
    )

    dim = effective_dim_end - effective_dim_start
    model_W = torch.nn.Sequential(
        torch.nn.Linear(dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_x * num_dim_x, bias=False),
    )

    c = 3 * num_dim_x
    if SINGLE_NETWORK_MULTIPLIED:
        c = num_dim_control
    model_u_w1 = torch.nn.Sequential(
        torch.nn.Linear(2 * dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, c * num_dim_x, bias=True),
    )

    model_u_w2 = torch.nn.Sequential(
        torch.nn.Linear(2 * dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_control * c, bias=True),
    )
    if SINGLE_NETWORK_VANILLA:
        model_u_w1 = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, num_dim_control, bias=True),
        )

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()
        model_u_w2 = model_u_w2.cuda()

    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control)
    W_func = W_FUNC(model_W, model_Wbot, num_dim_x, num_dim_control, w_lb)

    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func
