import torch

num_dim_x = 4
num_dim_control = 2

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    x, y, theta, v = [x[:,0,0], x[:,1,0], x[:,2,0], x[:,3,0] ]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, :, 0] = torch.stack( [v * torch.cos(theta), 
                               v * torch.sin(theta),
                                torch.zeros_like(v),
                                torch.zeros_like(v),], dim=1)
    return f

def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 2, 0] = 1
    B[:, 3, 1] = 1
    return B

def DBDx_func(x):
    raise NotImplemented('NotImplemented')
