# preamble
import torch
from torch.autograd import grad
import importlib
import sys
sys.path.append("systems")
sys.path.append("configs")
sys.path.append("models")
# options
log = "saved_models/Y_eps0p0"
use_cuda = False
task = "CAR"

# load the metric and controller
filename_controller = log + "/controller_best_hardtanh.pth.tar"
filename_metric = log + "/metric_best_hardtanh.pth.tar"
map_location = "cuda" if use_cuda else "cpu"
# to remove indexing and stuff from these, have to possibly retrain models? And at least modify structure of networks
W_func = torch.load(filename_metric, map_location)
u_func = torch.load(filename_controller, map_location)

# load functions for model
system = importlib.import_module("system_" + task)
f_func = system.f_func
B_func = system.B_func
num_dim_x = system.num_dim_x
num_dim_control = system.num_dim_control
if hasattr(system, "Bbot_func"):
    Bbot_func = system.Bbot_func
    
model = importlib.import_module("model_" + task)
# get_model = model.get_model
INVERSE_METRIC = model.INVERSE_METRIC

if "Bbot_func" not in locals():
    def Bbot_func(x):  # columns of Bbot forms a basis of the null space of B^T
        bs = x.shape[0]
        Bbot = torch.cat(
            (
                torch.eye(num_dim_x - num_dim_control, num_dim_x - num_dim_control),
                torch.zeros(num_dim_control, num_dim_x - num_dim_control),
            ),
            dim=0,
        )
        if use_cuda:
            Bbot = Bbot.cuda()
        Bbot.unsqueeze(0)
        return Bbot.repeat(bs, 1, 1)

def Jacobian_Matrix(M, x):
    # NOTE that this function assume that data are independent of each other
    # along the batch dimension.
    # M: B x m x m
    # x: B x n x 1
    # ret: B x m x m x n
    bs = x.shape[0]
    m = M.size(-1)
    n = x.size(1)
    J = torch.zeros(bs, m, m, n).type(x.type())
    for i in range(m):
        for j in range(m):
            J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def Jacobian(f, x):
    # NOTE that this function assume that data are independent of each other
    f = f + 0.0 * x.sum()  # to avoid the case that f is independent of x
    # f: B x m x 1
    # x: B x n x 1
    # ret: B x m x n
    bs = x.shape[0]
    m = f.size(1)
    n = x.size(1)
    J = torch.zeros(bs, m, n).type(x.type())
    for i in range(m):
        J[:, i, :] = grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def weighted_gradients(W, v, x, detach=False):
    # v, x: bs x n x 1
    # DWDx: bs x n x n x n
    assert v.size() == x.size()
    bs = x.shape[0]
    if detach:
        return (Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(dim=3)
    else:
        return (Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

# spoof inputs
bs = 1
x = torch.rand((bs, num_dim_x, 1)).requires_grad_()
xref = torch.rand((bs, num_dim_x, 1)).requires_grad_()
uref = torch.rand((bs, num_dim_control, 1)).requires_grad_()

# get the contraction condition
assert(not INVERSE_METRIC)
M = W_func(x)
f = f_func(x)
B = B_func(x)
# The following two should be easy / already implemented somewhere in certver codebases
DfDx = Jacobian(f, x)
DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())
for i in range(num_dim_control):
    DBDx[:, :, :, i] = Jacobian(B[:, :, i].unsqueeze(-1), x)

_Bbot = Bbot_func(x)
u = u_func(x, x - xref, uref)  # u: bs x m x 1 # TODO: x - xref

K = Jacobian(u, x)
# temp_u_func = lambda x: u_func(x, x - xref, uref)
# K = zero_order_jacobian_estimate(temp_u_func, x)

A = DfDx + sum(
    [
        u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
        for i in range(num_dim_control)
    ]
)
dot_x = f + B.matmul(u)
dot_M = weighted_gradients(M, dot_x, x, detach=False)  # DMDt
# dot_M = weighted_gradients_zero_order(W_func, dot_x, x, detach=False)  # DMDt
_lambda = 0.0
Q = (
    dot_M
    + (A + B.matmul(K)).transpose(1, 2).matmul(M)
    + M.matmul(A + B.matmul(K))
    + 2 * _lambda * M
)

# table for now
# eigenvals, eigenvecs = torch.linalg.eigh(Q) # I doubt that crown is able to handle this as I believe it contains an iterative algorithm ? I'm honestly not sure if pytorch can even differentiate through it
# max_eigen = eigenvals.max() #@huan this is the value to bound. we want this value to be less than zero

# compute Gershgorin approximation of eigen values
diagonal_entries = torch.diagonal(Q, dim1=-2, dim2=-1)
off_diagonal_sum = torch.abs(Q).sum(dim=-1) - torch.abs(diagonal_entries) # row sum
# Compute upper bounds on each eigenvalue of Q
gersh_ub_eig_Q = diagonal_entries + off_diagonal_sum
gersh_ub_eig_Q_max = gersh_ub_eig_Q.max() #@huan this is an over approximation of the value to bound.