# preamble
import torch
import torch.nn as nn
from torch.autograd import grad
import torchviz
import importlib
import sys
import colored_traceback
colored_traceback.add_hook(always=True)
sys.path.append("../Verifier_Development")
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
sys.path.append("systems")
sys.path.append("configs")
sys.path.append("models")
# options
log = "saved_models/Y_eps0p0"
use_cuda = False
task = "CARcrown"
from using_crown_utils import Jacobian, Jacobian_Matrix, weighted_gradients, clean_unsupported_ops


# load the metric and controller
use_hardtanh = False
if use_hardtanh:
    filename_controller = log + "/controller_best_hardtanh.pth.tar"
    filename_metric = log + "/metric_best_hardtanh.pth.tar"
    mixing=1.0 # all hardtanh
else:
    filename_controller = log + "/controller_best.pth.tar"
    filename_metric = log + "/metric_best.pth.tar"
    mixing=0.0 # all tanh
filename_model = log + "/model_best_hardtanh.pth.tar"
# if torch.backends.mps.is_available():
#     map_location = "mps"
#     torch.set_default_device('mps') # buggy when casting tensors to mps types
# else:
map_location = "cuda" if use_cuda else "cpu"

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
assert(not INVERSE_METRIC)

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

# spoof inputs
bs = 1
x = torch.rand((bs, num_dim_x, 1)).requires_grad_()
xref = torch.rand((bs, num_dim_x, 1)).requires_grad_()
uref = torch.rand((bs, num_dim_control, 1)).requires_grad_()
xall = torch.concatenate((x, xref, uref), dim=1)

# lirpa inputs
eps = 0.3
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, eps = eps)
x_ptb = BoundedTensor(x, ptb)
xall_ptb = BoundedTensor(xall, ptb)

def create_clean_Wu_funcs():
    # load model dict to get params 
    trained_model = torch.load(filename_model, map_location)
    w_lb = trained_model['args'].w_lb
    # load saved weights
    W_func_loaded = torch.load(filename_metric, map_location)
    u_func_loaded = torch.load(filename_controller, map_location)
    # create new version of network with modified forward function
    model_W, model_Wbot, model_u_w1, W_func, u_func =model.get_model_mixed(num_dim_x, num_dim_control, w_lb, use_cuda=False, mixing=mixing)
    # put trained weights into new function
    W_func.load_state_dict(W_func_loaded.state_dict())
    u_func.load_state_dict(u_func_loaded.state_dict())
    # return modified W_func
    W_func.model_W = clean_unsupported_ops(W_func.model_W)
    W_func.model_Wbot = clean_unsupported_ops(W_func.model_Wbot)
    u_func.model_u_w1 = clean_unsupported_ops(u_func.model_u_w1)
    return W_func, u_func

class CertVerModel(nn.Module):
    def __init__(self, x):
        super(CertVerModel, self).__init__()
        #clean upsupported ops
        self.f_func = f_func
        self.B_func_x = B_func(x) # init const with correct batch size
        # self.DBDx_x = system.DBDx_func(x)
        W_func, u_func = create_clean_Wu_funcs()
        self.u_func = u_func
        # self.W_func = W_func
        # self.DfDx = system.DfDx_func
    def forward(self, xall):
        x = xall[:,:num_dim_x]
        print("x.shape = ", x.shape)
        xref = xall[:,num_dim_x:num_dim_x*2]
        print("xref.shape = ", xref.shape)
        uref = xall[:,num_dim_x*2:]
        print("uref.shape = ", uref.shape)
        xerr = x - xref
        print("xerr.shape = ", xerr.shape)
        # return self.W_func(x) # works!!!  with IBP and CROWN
        # return self.f_func(x) # gives some error when I call lirpa_model(x_ptb)
        # return self.B_func_x.matmul(uref) # Works to build graph but not call bounds
        # return self.u_func(x, xerr, uref) # works! 
        # return self.DfDx(x) # was able to build graph, just not call bounds (with sin and cos)
        # u = self.u_func(x, xerr, uref).reshape(bs, 1, 1, num_dim_control)
        # return self.DfDx(x) + (u.reshape(bs, 1, 1, num_dim_control)*self.DBDx_x).sum(dim=-1) # works to compute bounds!!! even with cos/sin
        u = self.u_func(x, xerr, uref)
        # return self.f_func(x) + self.B_func_x.matmul(u) # able to build graph, but when computing bounds get error: assert not self.perturbed AssertionError
        return u

certvermodel = CertVerModel(xall)
out = certvermodel(xall)
print(f"out: {out}")
g = torchviz.make_dot(out, params={"x": x, "xref": xref, "uref": uref})
g.view()
lirpa_model = BoundedModule(certvermodel, torch.empty_like(xall))
print("Was able to build CROWN graph.")
lirpa_model(xall)
print("Was able to call CROWN graph.")
# lirpa_model(x_ptb) # error here when returning f_func(x)
lb, ub = lirpa_model.compute_bounds(x=(xall_ptb,), method='CROWN') #'IBP')
print("was able to compute bounds using CROWN graph.")
print(f"lb: {lb}, ub: {ub}")
assert(1==0)

# W_func = torch.load(filename_metric, map_location)
# u_func = torch.load(filename_controller, map_location)

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