import torch

num_dim_x = 4
num_dim_control = 2

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    x, y, theta, v = [x[:,[0],[0]], x[:,[1],[0]], x[:,[2],[0]], x[:,[3],[0]] ]
    # print(f"x.shape: {x.shape}")
    f = torch.stack([v * torch.relu(theta), #torch.cos(theta), 
                     v * torch.relu(theta), #torch.sin(theta),
                     torch.zeros_like(v),
                     torch.zeros_like(v),], dim=1).type(x.type()) #.reshape(bs, num_dim_x, 1)
    # print(f"f.shape: {f.shape}")
    # print("inside car f_func")
    return f

def DfDx_func(x):
    bs = x.shape[0]
    x, y, theta, v = [x[:,0,0], x[:,1,0], x[:,2,0], x[:,3,0] ]
    row1 = torch.stack([torch.zeros(bs), torch.zeros(bs), -v*torch.sin(theta), torch.cos(theta)], dim=1).reshape(bs, 1, num_dim_x)
    row2 = torch.stack([torch.zeros(bs), torch.zeros(bs),  v*torch.cos(theta), torch.sin(theta)], dim=1).reshape(bs, 1, num_dim_x)
    row34 = torch.zeros(bs, 2, num_dim_x)
    DfDx = torch.cat([row1, row2, row34], dim=1)
    return DfDx

# def B_func(x):
#     bs = x.shape[0]
#     B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

#     B[:, 2, 0] = 1
#     B[:, 3, 1] = 1
#     return B

def B_func(x):
    bs = x.shape[0]
    B_top = torch.zeros(bs, 2, 2).type(x.type())
    B_bot = torch.eye(2).repeat(bs,1,1).type(x.type()) # I think this is giving "tile op not supported" error *facepalm of exhaustion*
    B = torch.stack([B_top, B_bot], dim=1)
    print(f"B.shape: {B.shape}")
    return B

def DBDx_func(x):
    bs = x.shape[0]
    return torch.zeros(bs,num_dim_x,num_dim_x,num_dim_control)
