import torch

num_dim_x = 3
num_dim_control = 2

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    # dx/dt = v cos(theta)
    # dy/dt = v sin(theta)
    # dtheta/dt = w
    # f is zero since there are no drift dynamics
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    return f

def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    _, _, theta = [x[:,i,0] for i in range(num_dim_x)]

    B[:, 0, 0] = torch.cos(theta)  # dx/dt = v cos(theta)
    B[:, 1, 0] = torch.sin(theta)  # dy/dt = v sin(theta)
    B[:, 2, 1] = 1  # dtheta/dt = w
    return B

def DBDx_func(x):
    raise NotImplemented('NotImplemented')
