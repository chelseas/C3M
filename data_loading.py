import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TrackingDataset(Dataset):
    def __init__(self, N, config, system, args, use_mps=False):
        self.X_MIN = config.X_MIN
        self.X_MAX = config.X_MAX
        self.U_MIN = config.UREF_MIN
        self.U_MAX = config.UREF_MAX
        self.XE_MIN = config.XE_MIN
        self.XE_MAX = config.XE_MAX
        self.num_dim_x = system.num_dim_x
        self.num_dim_control = system.num_dim_control
        self.args = args
        self.use_mps = use_mps
        self.num = N
        self.X = [self.sample_full() for _ in range(N)]
        #
    # constructing datasets
    def sample_xef(self):
        return (self.X_MAX - self.X_MIN) * np.random.rand(self.num_dim_x, 1) + self.X_MIN
        #
    def sample_x(self,xref):
        xe = (self.XE_MAX - self.XE_MIN) * np.random.rand(self.num_dim_x, 1) + self.XE_MIN
        x = xref + xe
        x[x > self.X_MAX] = self.X_MAX[x > self.X_MAX]
        x[x < self.X_MIN] = self.X_MIN[x < self.X_MIN]
        return x
        #
    def sample_uref(self):
        return (self.U_MAX - self.U_MIN) * np.random.rand(self.num_dim_control, 1) + self.U_MIN
        #
    def sample_full(self):
        xref = self.sample_xef()
        uref = self.sample_uref()
        x = self.sample_x(xref)
        if self.args.use_cuda:
            xref = torch.from_numpy(xref).float().cuda().detach()
            uref = torch.from_numpy(uref).float().cuda().detach()
            x = torch.from_numpy(x).float().cuda().requires_grad_()
        elif self.use_mps:
            xref = torch.from_numpy(xref).float().mps().detach()
            uref = torch.from_numpy(uref).float().mps().detach()
            x = torch.from_numpy(x).float().mps().requires_grad_()
        else:
            xref = torch.from_numpy(xref).float().detach()
            uref = torch.from_numpy(uref).float().detach()
            x = torch.from_numpy(x).float().requires_grad_()
        #
        return (x, xref, uref)
        #
    def __len__(self):
        return self.num
        #
    def __getitem__(self, idx):
        return self.X[idx], []

# def test():
#     dataset = TrackingDataset(config, system,
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#     for i_batch, sample_batched in enumerate(dataloader):
#         print(i_batch, sample_batched)