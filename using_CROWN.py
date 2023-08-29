# preamble
import torch
from torch.autograd import grad
import torchviz
import importlib
import sys
sys.path.append("systems")
sys.path.append("configs")
sys.path.append("models")
# options
log = "saved_models/Y_eps0p0"
use_cuda = False
task = "CARcrown"

# load the metric and controller
filename_controller = log + "/controller_best_hardtanh.pth.tar"
filename_metric = log + "/metric_best_hardtanh.pth.tar"
# if torch.backends.mps.is_available():
#     map_location = "mps"
#     torch.set_default_device('mps') # buggy when casting tensors to mps types
# else:
map_location = "cuda" if use_cuda else "cpu"
# maybe I don't need to restructure at all?
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

from using_crown_utils import Jacobian, Jacobian_Matrix, weighted_gradients

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
DfDx = system.DfDx_func(x)
DBDx = system.DBDx_func(x)

_Bbot = Bbot_func(x)
u = u_func(x, x - xref, uref)  # u: bs x m x 1 # TODO: x - xref
g = torchviz.make_dot(u, params={"u": u, "x": x, "xref": xref, "uref":uref})
g.view()

K = Jacobian(u, x)
# temp_u_func = lambda x: u_func(x, x - xref, uref)
# K = zero_order_jacobian_estimate(temp_u_func, x)

A = DfDx + (u.reshape(bs, 1, 1, num_dim_control)*DBDx).sum(dim=-1)
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
gQ = torchviz.make_dot(gersh_ub_eig_Q_max, params={"u": u, "x": x, "xref": xref, "uref":uref})
gQ.view()