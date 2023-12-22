# preamble
import torch
import torch.nn as nn
import torchviz
import importlib
import sys
import os
import colored_traceback
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.jacobian import JacobianOP, GradNorm

sys.path.append("systems")
sys.path.append("configs")
sys.path.append("models")

sys.path.append(os.path.dirname(__file__))
from using_crown_utils import Jacobian, Jacobian_Matrix, weighted_gradients, clean_unsupported_ops

colored_traceback.add_hook(always=True)


# options
log = os.path.join(os.path.dirname(__file__), "logs/1129.test_l1reg_3") # New model
# log = os.path.join(os.path.dirname(__file__), "logs/car_red_2") # New model
# log = os.path.join(os.path.dirname(__file__), "saved_models/Y_eps0p0")
# log = "../C3M/saved_models/Y_eps0p0" # old model
# log = "../C3M/logs/car_red_2"  # New model


def build_model(log, comparison=False):
    log = os.path.join(os.path.dirname(__file__), log)

    use_cuda = False
    task = "CARcrown"

    # load the metric and controller
    use_hardtanh = False
    if use_hardtanh:
        print("loading hardtanh models")
        filename_controller = log + "/controller_best_hardtanh.pth.tar"
        filename_metric = log + "/metric_best_hardtanh.pth.tar"
        mixing=1.0 # all hardtanh
        filename_model = log + "/model_best_hardtanh.pth.tar" # only used for loading w_lb
    else:
        print("loading smooth tanh models")
        filename_controller = log + "/controller_best.pth.tar"
        filename_metric = log + "/metric_best.pth.tar"
        mixing=0.0 # all tanh
        filename_model = log + "/model_best.pth.tar" # only used for loading w_lb
    # if torch.backends.mps.is_available():
    #     map_location = "mps"
    #     torch.set_default_device('mps') # buggy when casting tensors to mps types
    # else:
    map_location = "cuda" if use_cuda else "cpu"

    # load functions for model
    system = importlib.import_module("systems.system_" + task)
    f_func = system.f_func
    B_func = system.B_func
    num_dim_x = system.num_dim_x
    num_dim_control = system.num_dim_control
    if hasattr(system, "Bbot_func"):
        Bbot_func = system.Bbot_func

    model = importlib.import_module("models.model_" + task)
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
    x = torch.rand((bs, num_dim_x)).requires_grad_()
    xref = torch.rand((bs, num_dim_x)).requires_grad_()
    uref = torch.rand((bs, num_dim_control)).requires_grad_()
    xall = torch.concat((x, xref, uref), dim=1)
    # lirpa inputs
    eps = 0.3
    norm = float("inf")
    ptb = PerturbationLpNorm(norm = norm, eps = eps)
    x_ptb = BoundedTensor(x, ptb)
    xall_ptb = BoundedTensor(xall, ptb)

    xerr = x - xref
    lambda_ = 0.01

    def create_clean_Wu_funcs(replace="relu"):
        print("replacing unsupported ops with: ", replace)
        # load model dict to get params
        trained_model = torch.load(filename_model, map_location)
        w_lb = trained_model['args'].w_lb
        print("w_lb = ", w_lb)
        # load saved weights
        W_func_loaded = torch.load(filename_metric, map_location)
        u_func_loaded = torch.load(filename_controller, map_location)
        # create new version of network with modified forward function
        model_W, model_Wbot, model_u_w1, W_func, u_func =model.get_model_mixed(num_dim_x, num_dim_control, w_lb, use_cuda=False, mixing=mixing)
        # put trained weights into new function
        W_func.load_state_dict(W_func_loaded.state_dict())
        u_func.load_state_dict(u_func_loaded.state_dict())
        # return modified W_func
        W_func.model_W = clean_unsupported_ops(W_func.model_W, replace=replace)
        W_func.model_Wbot = clean_unsupported_ops(W_func.model_Wbot, replace=replace)
        u_func.model_u_w1 = clean_unsupported_ops(u_func.model_u_w1, replace=replace)
        return W_func, u_func

    class CertVerModel(nn.Module):
        def __init__(self, x, replace="relu"):
            super(CertVerModel, self).__init__()
            #clean upsupported ops
            self.f_func = f_func
            x = x.unsqueeze(-1)
            self.B = B_func(x) # init const with correct batch size
            self.DBDx_x = system.DBDx_func(x)
            W_func, u_func = create_clean_Wu_funcs(replace=replace)
            self.u_func = u_func
            self.W_func = W_func
            self.DfDx = system.DfDx_func

        def forward(self, xall):
            bsz = xall.shape[0]
            xall = xall.reshape(bsz, xall.shape[1], 1)

            self.B = self.B.to(xall.device)
            self.DBDx_x = self.DBDx_x.to(xall.device)

            # print("xall.shape = ", xall.shape)
            x = xall[:,:num_dim_x]
            xref = xall[:,num_dim_x:num_dim_x*2]
            uref = xall[:,num_dim_x*2:]
            xerr = x - xref
            u = self.u_func(x, xerr, uref)
            K = JacobianOP.apply(u, x).reshape(bsz, num_dim_control, num_dim_x)
            A = (self.DfDx(x) + (
                u.reshape(bsz, 1, 1, num_dim_control) * self.DBDx_x
            ).sum(dim=-1))
            # print("A.shape: ", A.shape)
            # print("self.f_func(x).shape =", self.f_func(x).shape)
            # print("self.B.shape =", self.B.shape)
            # print("u.shape =", u.shape)
            # print("self.B.matmul(u).shape =", self.B.matmul(u).shape)
            dxdt = self.f_func(x).reshape(bsz, num_dim_x, 1) + self.B.matmul(u)
            # print("dxdt.shape: ", dxdt.shape)
            M = self.W_func(x)
            # print("M.shape = ", M.shape)
            print("x.shape = ", x.shape)
            # dMdx = JacobianOP.apply(M.reshape(bs, -1), x.reshape(bs, num_dim_x))
            dMdx = JacobianOP.apply(M.reshape(bsz, -1), x) # creates jac of shape (1,16,4,1)
            print("dMdx.shape 0: ", dMdx.shape)
            dMdx = dMdx.reshape(bsz, -1, num_dim_x) # remove trailing 1 dimension so it's (1,16,4)
            # print("dMdx.shape 1: ", dMdx.shape)
            # print("dxdt.shape: ", dxdt.shape)
            dMdt_flat = dMdx.matmul(dxdt) # (1,16,4)x(1,4,1) should create (1,16,1)
            # print("dMdt_flat.shape: ", dMdt_flat.shape)
            # print("self.B,shape: ", self.B.shape)
            # print("K.shape: ", K.shape)
            dMdt = dMdt_flat.reshape(bsz, num_dim_x, num_dim_x)
            M_A_BK = M.matmul( A + self.B.matmul(K) )
            Q = (dMdt + M_A_BK
                    + M_A_BK.transpose(1,2)
                    + 2 * lambda_ * M
            )
            # print("Q.shape: ", Q.shape) # should be (bs, num_dim_x, num_dim_x)
            # compute Gershgorin approximation of eigen values
            diagonal_entries = torch.diagonal(Q, dim1=-2, dim2=-1)
            # print("diagonal_entries.shape: ", diagonal_entries.shape)
            off_diagonal_sum = torch.abs(Q).sum(dim=-1) - torch.abs(diagonal_entries) # row sum
            # print("off_diagonal_sum.shape: ", off_diagonal_sum.shape)
            # Compute upper bounds on each eigenvalue of Q
            gersh_ub_eig_Q = diagonal_entries + off_diagonal_sum
            # not supported: gersh_ub_eig_Q_max = gersh_ub_eig_Q.amax(dim=-1)
            return gersh_ub_eig_Q

    # May not be update-to-date
    class CertVerComparison(nn.Module):
        def __init__(self, x, replace="relu"):
            super(CertVerComparison, self).__init__()
            #clean upsupported ops
            self.f_func = f_func
            x = x.unsqueeze(-1)
            self.B = B_func(x) # init const with correct batch size
            self.DBDx_x = system.DBDx_func(x)
            W_func, u_func = create_clean_Wu_funcs(replace=replace)
            self.u_func = u_func
            self.W_func = W_func
            self.DfDx = system.DfDx_func
        def forward(self, xall):
            # print("xall.shape = ", xall.shape)
            xall = xall.unsqueeze(-1)
            x = xall[:,:num_dim_x]
            xref = xall[:,num_dim_x:num_dim_x*2]
            uref = xall[:,num_dim_x*2:]
            def get_u(x, xref, uref):
                xerr = x - xref
                u = self.u_func(x.requires_grad_(True),
                                xerr.requires_grad_(True),
                                uref.requires_grad_(True))
                return u
            K = torch.autograd.functional.jacobian(
                get_u, (x, xref, uref), create_graph=True
            )[0].reshape(bs, num_dim_control, num_dim_x)
            u = get_u(x, xref, uref)
            A = self.DfDx(x) + (u.reshape(bs, 1, 1, num_dim_control)*self.DBDx_x).sum(dim=-1)
            # print("A.shape: ", A.shape)
            # print("self.f_func(x).shape =", self.f_func(x).shape)
            # print("self.B.shape =", self.B.shape)
            # print("u.shape =", u.shape)
            # print("self.B.matmul(u).shape =", self.B.matmul(u).shape)
            dxdt = self.f_func(x).reshape(bs, num_dim_x, 1) + self.B.matmul(u)
            # print("dxdt.shape: ", dxdt.shape)
            def get_M(x):
                M = self.W_func(x.requires_grad_(True))
                return M.reshape(bs, -1)
            M = get_M(x).reshape(bs, num_dim_x, num_dim_x)
            # print("M.shape = ", M.shape)
            # print("x.shape = ", x.shape)
            # dMdx = JacobianOP.apply(M.reshape(bs, -1), x.reshape(bs, num_dim_x))
            dMdx = torch.autograd.functional.jacobian(get_M, x, create_graph=True).reshape(bs, num_dim_x**2, num_dim_x) # creates jac of shape (1,16,4)
            # print("dMdx.shape 1: ", dMdx.shape)
            # print("dxdt.shape: ", dxdt.shape)
            dMdt_flat = dMdx.matmul(dxdt) # (1,16,4)x(1,4,1) should create (1,16,1)
            # print("dMdt_flat.shape: ", dMdt_flat.shape)
            # print("self.B,shape: ", self.B.shape)
            # print("K.shape: ", K.shape)
            dMdt = dMdt_flat.reshape(bs, num_dim_x, num_dim_x)
            M_A_BK = M.matmul( A + self.B.matmul(K) )
            Q = (dMdt + M_A_BK
                    + M_A_BK.transpose(1,2)
                    + 2 * lambda_ * M
            )
            # print("Q.shape: ", Q.shape) # should be (bs, num_dim_x, num_dim_x)
            # compute Gershgorin approximation of eigen values
            diagonal_entries = torch.diagonal(Q, dim1=-2, dim2=-1)
            # print("diagonal_entries.shape: ", diagonal_entries.shape)
            off_diagonal_sum = torch.abs(Q).sum(dim=-1) - torch.abs(diagonal_entries) # row sum
            # print("off_diagonal_sum.shape: ", off_diagonal_sum.shape)
            # Compute upper bounds on each eigenvalue of Q
            gersh_ub_eig_Q = diagonal_entries + off_diagonal_sum
            # not supported: gersh_ub_eig_Q_max = gersh_ub_eig_Q.amax(dim=-1)
            return gersh_ub_eig_Q

    # Function utilized by the complete verifier to build the model.
    certvermodel = CertVerModel(xall, replace="tanh")

    if comparison:
        certvermodel_comparison = CertVerComparison(xall, replace="tanh")
        return certvermodel, certvermodel_comparison, xall, xall_ptb
    else:
        return certvermodel


if __name__ == '__main__':
    certvermodel, certvermodel_comparison, xall, xall_ptb = build_model(log, comparison=True)

    print("crown: ")
    out = certvermodel(xall)
    print("using autograd:")
    out_c = certvermodel_comparison(xall)
    print(f"out: {out}")
    print(f"out_c : {out_c}")
    # g = torchviz.make_dot(out, params={"x": x, "xref": xref, "uref": uref, "eigQ": out})
    # g.view()
    # gc = torchviz.make_dot(out_c, params={"x": x, "xref": xref, "uref": uref, "eigQ": out})
    # gc.view()
    print("trying to build CROWN graph ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
    lirpa_model = BoundedModule(certvermodel, torch.empty_like(xall))
    print("Was able to build CROWN graph.")
    print('Output', lirpa_model(xall))
    print("Was able to call CROWN graph.")
    print(f"out_c for comparison: {out_c}")
    # lirpa_model(x_ptb) # error here when returning f_func(x)
    lb, ub = lirpa_model.compute_bounds(x=(xall_ptb,), method='CROWN') #'IBP')
    print("was able to compute bounds using CROWN graph.")
    print(f"lb: {lb}, ub: {ub}")

    print('Testing batching')
    xall_ptb = xall_ptb.repeat(20, 1)
    output = lirpa_model(xall_ptb)
    print('Output', output)
    lb, ub = lirpa_model.compute_bounds(x=(xall_ptb,), method='CROWN') #'IBP')
    print(f"lb: {lb}, ub: {ub}")
