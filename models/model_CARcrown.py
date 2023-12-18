import torch
from torch import nn
from torch.autograd import grad
import numpy as np

effective_dim_start = 2
effective_dim_end = 4

INVERSE_METRIC = False

use_mps = False
device = 'cpu'
# if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     torch.set_default_device('mps')
#     use_mps = True
#     device = 'mps'
# whether or not to use translation invariance in controller
contr_tvdim = True
print("Using translation invariance in controller: ", contr_tvdim)


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
        if contr_tvdim:
            d1 = effective_dim_start
            d2 = effective_dim_end
        else: # use whole x
            d1 = 0
            d2 = self.num_dim_x
        # print("input shape: ", torch.cat([x[:, d1:d2], xe], dim=1).reshape(bs, -1).shape)
        w1_xe = self.model_u_w1(torch.cat([x[:, d1:d2], xe], dim=1).reshape(bs, -1)).reshape(
            bs, self.num_dim_control, -1
        )
        w1_x0 = self.model_u_w1(
            torch.cat([x[:, d1:d2], torch.zeros_like(xe)], dim=1).reshape(bs, -1)
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
        self.eye_bias = self.w_lb * torch.eye(self.num_dim_x)

    def forward(self, x):
        self.eye_bias = self.eye_bias.to(x.device)

        bs = x.shape[0]
        x = x.reshape(bs, -1)

        W = self.model_W(x[:, effective_dim_start:effective_dim_end]).view(
            bs, self.num_dim_x, self.num_dim_x
        )
        # print("W.shape: ", W.shape)
        W_right = W[:, :, (self.num_dim_x - self.num_dim_control):]

        Wbot = self.model_Wbot(
            torch.ones(bs, 1, device=x.device, dtype=x.dtype)
        ).view(
            bs,
            self.num_dim_x - self.num_dim_control,
            self.num_dim_x - self.num_dim_control,
        )

        # stack to create final W matrix
        W_left = torch.concat([
            Wbot, torch.zeros(
                bs,
                self.num_dim_control,
                self.num_dim_x - self.num_dim_control,
                device=x.device,
            )], dim=1)
        W_full = torch.concat([W_left, W_right], dim=2)

        W_final = W_full.transpose(1, 2).matmul(W_full)
        # print("0. W_final.shape = :", W_final.shape)
        W_final = W_final + self.eye_bias

        return W_final

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


def get_model_mixed(num_dim_x, num_dim_control, w_lb, use_cuda=False, mixing=0.0, M_width=128, M_depth=2, u_width=128, u_depth=2):
    M_width = M_width
    M_depth = M_depth
    u_width = u_width
    u_depth = u_depth
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

    if contr_tvdim:
        u_size = dim + num_dim_x
    else:
        u_size = num_dim_x*2
    u_layers = [torch.nn.Linear(u_size, u_width, bias=False), MixedTanh()]
    for i in range(u_depth - 1):
        u_layers += [torch.nn.Linear(u_width, u_width, bias=False), MixedTanh()]
    u_layers.append(torch.nn.Linear(u_width, num_dim_control, bias=False))
    model_u_w1 = torch.nn.Sequential(*u_layers)

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()
    elif use_mps:
        print("casting to mps")
        model_W = model_W.to("mps")
        print("check casting: ", model_W[0].weight.device)
        model_Wbot = model_Wbot.to("mps")
        model_u_w1 = model_u_w1.to("mps")

    u_func = U_FUNC(model_u_w1, num_dim_x, num_dim_control)
    W_func = W_FUNC(model_W, model_Wbot, num_dim_x, num_dim_control, w_lb)

    u_func.set_mixing(mixing)
    W_func.set_mixing(mixing)

    return model_W, model_Wbot, model_u_w1, W_func, u_func


def get_model(num_dim_x, 
              num_dim_control, 
              w_lb, 
              use_cuda=False,  
              M_width=128, M_depth=2, 
              u_width=128, u_depth=2):
    model_W, model_Wbot, model_u_w1, W_func, u_func = get_model_mixed(num_dim_x, 
                                                                      num_dim_control, 
                                                                      w_lb, 
                                                                      use_cuda=use_cuda, 
                                                                      mixing=0.0,  
                                                                      M_width=M_width, M_depth=M_depth, u_width=u_width, u_depth=u_depth)
    (model_W_hard,
     model_Wbot_hard,
     model_u_w1_hard,
     W_func_hard,
     u_func_hard) = get_model_mixed(num_dim_x,
                                    num_dim_control, 
                                    w_lb, 
                                    use_cuda=use_cuda, 
                                    mixing=1.0)

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
