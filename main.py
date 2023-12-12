import torch
from torch.autograd import grad
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
use_mps = False
# if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     torch.set_default_device('mps')
#     use_mps = True

import importlib
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import sys

sys.path.append("systems")
sys.path.append("configs")
sys.path.append("models")
import argparse
# import pdb

Stats = namedtuple("Stats", "num_new_CEs num_lost_CEs")

EigStats = namedtuple("EigStats", "max min mean")
MeigStats = namedtuple("MeigStats", "max min cond_max")

np.random.seed(1024)

def cmdlineparse(args):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--task", type=str, default="CAR", help="Name of the model.")
    parser.add_argument(
        "--no_cuda", dest="use_cuda", action="store_false", help="Disable cuda."
    )
    parser.set_defaults(use_cuda=True)
    parser.add_argument(
        "--load_model",
        dest="load_model",
        action="store_true",
        help="Load the model instead of training it.",
    )
    parser.set_defaults(load_model=False)
    parser.add_argument("--bs", type=int, default=1024, help="Batch size.")
    parser.add_argument(
        "--num_train", type=int, default=0 * 32768 + 4 * 131072, help="Number of samples for training."
    )  # 4096 * 32
    parser.add_argument(
        "--num_test", type=int, default=32768, help="Number of samples for testing."
    )  # 1024 * 32
    parser.add_argument(
        "--lr", dest="learning_rate", type=float, default=0.001, help="Base learning rate."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr_step", type=int, default=5, help="")
    parser.add_argument(
        "--lambda", type=float, dest="_lambda", default=1.0, help="Convergence rate: lambda"
    )
    parser.add_argument(
        "--w_ub",
        type=float,
        default=10,
        help="Upper bound of the eigenvalue of the dual metric.",
    )
    parser.add_argument(
        "--w_lb",
        type=float,
        default=0.1,
        help="Lower bound of the eigenvalue of the dual metric.",
    )
    parser.add_argument("--log", type=str, help="Path to a directory for storing the log.")
    parser.add_argument("--clone", dest="clone", action="store_true", help="Flag to clone hardtanh or not.")
    parser.add_argument(
        "--robust_train",
        dest="robust_train",
        action="store_true",
        help="Do adversarial training using PGD attack.",
    )
    parser.add_argument(
        "--robust_eps",
        dest="robust_eps",
        type=float,
        default=0.5,
        help="Perturbation bound for adversarial training.",
    )
    parser.add_argument(
        "--robust_alpha",
        dest="robust_alpha",
        type=float,
        default=.05,
        help="Learning rate for adversarial training attack.",
    )
    parser.add_argument(
        "--robust_norm",
        dest="robust_norm",
        type=str,
        default="l_inf",
        help="Norm for adversarial training.",
    )
    parser.add_argument(
        "--robust_attack_iters",
        dest="robust_attack_iters",
        type=int,
        default=10,
        help="Number of iterations to create adversarial example during robust training.",
    )
    parser.add_argument(
        '--robust_restarts', dest="robust_restarts", default=1, type=int, help=" number of times that the adversarial attack can restart during adversarial training. A hyper parameter.")
    parser.add_argument(
        "--reg_coeff",
        dest="reg_coeff",
        type=float,
        default=0.0,
        help="Regularization coefficient. Reasonable value might be .0001.",
    )
    args = parser.parse_args(args)
    return args

def main(args=None):
    # pdb.set_trace()
    args = cmdlineparse(args)
    if not args.load_model:
        os.system("cp *.py " + args.log)
        os.system("cp -r models/ " + args.log)
        os.system("cp -r configs/ " + args.log)
        os.system("cp -r systems/ " + args.log)
        writer = SummaryWriter(args.log + "/tb")
    global_steps = 0
    if args.use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    epsilon = args._lambda
    print("Epsilon: ", epsilon)
    if args.reg_coeff > 0.0:
        print("Using l1 regularization with coeff: ", args.reg_coeff)

    config = importlib.import_module("config_" + args.task)
    X_MIN = config.X_MIN
    X_MAX = config.X_MAX
    U_MIN = config.UREF_MIN
    U_MAX = config.UREF_MAX
    XE_MIN = config.XE_MIN
    XE_MAX = config.XE_MAX

    system = importlib.import_module("system_" + args.task)
    f_func = system.f_func
    B_func = system.B_func
    num_dim_x = system.num_dim_x
    num_dim_control = system.num_dim_control
    if hasattr(system, "Bbot_func"):
        Bbot_func = system.Bbot_func

    model = importlib.import_module("model_" + args.task)
    get_model = model.get_model
    INVERSE_METRIC = model.INVERSE_METRIC

    (
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
    ) = get_model(num_dim_x, num_dim_control, w_lb=args.w_lb, use_cuda=args.use_cuda)

    if use_mps:
        print("casting to mps again")
        model_W = model_W.to("mps")
        model_Wbot = model_Wbot.to("mps")
        model_u_w1 = model_u_w1.to("mps")
        u_func = u_func.to("mps")
        W_func = W_func.to("mps")

    # constructing datasets
    def sample_xef():
        return (X_MAX - X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN


    def sample_x(xref):
        xe = (XE_MAX - XE_MIN) * np.random.rand(num_dim_x, 1) + XE_MIN
        x = xref + xe
        x[x > X_MAX] = X_MAX[x > X_MAX]
        x[x < X_MIN] = X_MIN[x < X_MIN]
        return x


    def sample_uref():
        return (U_MAX - U_MIN) * np.random.rand(num_dim_control, 1) + U_MIN


    def sample_full():
        xref = sample_xef()
        uref = sample_uref()
        x = sample_x(xref)
        return (x, xref, uref)


    X_tr = [sample_full() for _ in range(args.num_train)]
    X_te = [sample_full() for _ in range(args.num_test)]
    print(type(X_tr))

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
            if args.use_cuda:
                Bbot = Bbot.cuda()
            Bbot.unsqueeze(0)
            return Bbot.repeat(bs, 1, 1)


    gradient_bundle_size = 32
    gradient_bundle_variance = 0.1


    def zero_order_jacobian_estimate(f, x):
        """
        Compute the zero-order estimate of the gradient of f w.r.t. x.

        args:
            f: callable
            x: bs x n x 1 tensor
        """
        n = x.shape[1]

        # Get the function value at x
        f_x = f(x)

        # Expand the size of x to match the size of the bundle
        x = torch.repeat_interleave(x, gradient_bundle_size, dim=-1)

        # Make somewhere to store the Jacobian
        J = torch.zeros(*f_x.squeeze().shape, n).type(x.dtype)

        # Estimate the gradient in each direction of x
        for i in range(n):
            # Get the perturbations in this dimension of x
            dx_i = gradient_bundle_variance * torch.randn(gradient_bundle_size).type(
                x.dtype
            )
            x_plus_dx_i = x.clone()
            x_plus_dx_i[:, i, :] += dx_i

            # Get the function value at x + dx (iterate through each sample)
            for j in range(gradient_bundle_size):
                f_x_plus_dx_i = f(x_plus_dx_i[:, :, j].unsqueeze(-1))

                # Accumulate it into a Jacobian estimator
                J[:, :, i] += (f_x_plus_dx_i - f_x).squeeze(-1) / (
                    dx_i[j] * gradient_bundle_size
                )

        return J


    def Jacobian_Matrix(M, x):
        # NOTE that this function assume that data are independent of each other
        # along the batch dimension.
        # M: B x m x m
        # x: B x n x 1
        # ret: B x m x m x n
        bs = x.shape[0]
        m = M.size(-1)
        n = x.size(1)
        J = torch.zeros(bs, m, m, n).type(x.dtype)
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
        J = torch.zeros(bs, m, n).type(x.dtype)
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


    def weighted_gradients_zero_order(f, v, x, detach=False):
        # v, x: bs x n x 1
        # DfDx: bs x n x n x n
        assert v.size() == x.size()
        bs = x.shape[0]
        if detach:
            return (zero_order_jacobian_estimate(f, x).detach() * v.view(bs, 1, 1, -1)).sum(
                dim=3
            )
        else:
            return (zero_order_jacobian_estimate(f, x) * v.view(bs, 1, 1, -1)).sum(dim=3)


    K = 1024


    # def loss_pos_matrix_random_sampling(A):
    #     # A: bs x d x d
    #     # z: K x d
    #     z = torch.randn(K, A.size(-1))
    #     if args.use_cuda:
    #         z = z.cuda()
    #     z = z / z.norm(dim=1, keepdim=True)
    #     zTAz = (z.matmul(A) * z.view(1, K, -1)).sum(dim=2).view(-1)
    #     negative_index = zTAz.detach().cpu().numpy() < 0
    #     if negative_index.sum() > 0:
    #         negative_zTAz = zTAz[negative_index]
    #         return -1.0 * (negative_zTAz.mean())
    #     else:
    #         return torch.tensor(0.0).type(z.type()).requires_grad_()
        
    def loss_pos_matrix_random_sampling(A, reduce=True):
        # A: bs x d x d
        # z: K x d
        z = torch.randn(K, A.size(-1))
        if args.use_cuda:
            z = z.cuda()
        z = z / z.norm(dim=1, keepdim=True)
        if reduce:
            zTAz = (z.matmul(A) * z.view(1, K, -1)).sum(dim=2).view(-1) # squeeze: bs*K
            negative_index = zTAz.detach().cpu().numpy() < 0
            if negative_index.sum() > 0:
                negative_zTAz = zTAz[negative_index]
                return -1.0 * (negative_zTAz.mean())
            else:
                return torch.tensor(0.0).type(z.dtype).requires_grad_()
        else: # no reduce
            zTAz = (z.matmul(A) * z.view(1, K, -1)).sum(dim=2) # bs x K
            # compute avg violation of PD condition for each sample in batch
            positive_index = zTAz.detach().cpu().numpy() > 0
            negative_index = zTAz < 0
            # no loss contrib from pos so zero them
            zTAz[positive_index] = 0.
            # next average neg values
            num_neg_each_row = negative_index.sum(-1) # if this is zero for any data point, don't want to divide by zero
            num_neg_each_row[num_neg_each_row == 0.0] = 1.0 # 0/1 = 0
            neg_mean = zTAz.sum(dim=-1) / num_neg_each_row 
            # print("loss dim: ", neg_mean.shape)
            return -1.0 * neg_mean


    def loss_pos_matrix_eigen_values(A):
        # A: bs x d x d
        eigv = torch.linalg.eigh(A, eigenvectors=True)[0].view(-1)
        negative_index = eigv.detach().cpu().numpy() < 0
        negative_eigv = eigv[negative_index]
        return negative_eigv.norm()

    def forward(
        x,
        xref,
        uref,
        _lambda,
        verbose=False,
        acc=False,
        detach=False,
        clone=False,
        zero_order=False,
        reduce=True,
        debug=False
    ):
        # x: bs x n x 1
        bs = x.shape[0]
        if INVERSE_METRIC:
            W = W_func(x)
            M = torch.inverse(W)
        else:
            M = W_func(x)
            W = torch.inverse(M)
        f = f_func(x)
        B = B_func(x)
        DfDx = Jacobian(f, x)
        DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.dtype)
        for i in range(num_dim_control):
            DBDx[:, :, :, i] = Jacobian(B[:, :, i].unsqueeze(-1), x)

        _Bbot = Bbot_func(x)
        u = u_func(x, x - xref, uref)  # u: bs x m x 1 # TODO: x - xref

        # If clone is set, train the hardtanh networks to match the tanh networks
        if clone:
            # Clone the control
            u_hard = u_func_hard(x, x - xref, uref)
            u_error = (u - u_hard) ** 2

            # Clone the metric
            if INVERSE_METRIC:
                W_hard = W_func_hard(x)
                M_hard = torch.inverse(W_hard)
            else:
                M_hard = W_func_hard(x)
                W_hard = torch.inverse(M_hard)

            M_error = (M - M_hard) ** 2
            W_error = (W - W_hard) ** 2

            # Replace the M, W, and u with the hardtanh versions
            u, M, W = u_hard, M_hard, W_hard

        if zero_order:
            temp_u_func = lambda x: u_func(x, x - xref, uref)
            K = zero_order_jacobian_estimate(temp_u_func, x)
        else:
            K = Jacobian(u, x)

        A = DfDx + sum(
            [
                u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(num_dim_control)
            ]
        )
        dot_x = f + B.matmul(u)
        if zero_order:
            if INVERSE_METRIC:
                temp_M_func = lambda x: torch.inverse(W_func(x))
                dot_M = weighted_gradients_zero_order(
                    temp_M_func, dot_x, x, detach=detach
                )  # DMDt
                dot_W = weighted_gradients_zero_order(
                    W_func, dot_x, x, detach=detach
                )  # DWDt
            else:
                temp_W_func = lambda x: torch.inverse(W_func(x))
                dot_M = weighted_gradients_zero_order(
                    W_func, dot_x, x, detach=detach
                )  # DMDt
                dot_W = weighted_gradients_zero_order(
                    temp_W_func, dot_x, x, detach=detach
                )  # DWDt
        else:
            dot_M = weighted_gradients(M, dot_x, x, detach=detach)  # DMDt
            dot_W = weighted_gradients(W, dot_x, x, detach=detach)  # DWDt
        if detach:
            Contraction = (
                dot_M
                + (A + B.matmul(K)).transpose(1, 2).matmul(M.detach())
                + M.detach().matmul(A + B.matmul(K))
                + 2 * _lambda * M.detach()
            )
        else:
            Contraction = (
                dot_M
                + (A + B.matmul(K)).transpose(1, 2).matmul(M)
                + M.matmul(A + B.matmul(K))
                + 2 * _lambda * M
            )

        # C1
        C1_inner = (
            -weighted_gradients(W, f, x)
            + DfDx.matmul(W)
            + W.matmul(DfDx.transpose(1, 2))
            + 2 * _lambda * W
        )
        C1_LHS_1 = (
            _Bbot.transpose(1, 2).matmul(C1_inner).matmul(_Bbot)
        )  # this has to be a negative definite matrix

        # C2
        C2_inners = []
        C2s = []
        for j in range(num_dim_control):
            C2_inner = weighted_gradients(W, B[:, :, j].unsqueeze(-1), x) - (
                DBDx[:, :, :, j].matmul(W) + W.matmul(DBDx[:, :, :, j].transpose(1, 2))
            )
            C2 = _Bbot.transpose(1, 2).matmul(C2_inner).matmul(_Bbot)
            C2_inners.append(C2_inner)
            C2s.append(C2)

        # if reduce:
        #     # compute a scalar loss
        #     loss = 0
        # else:
        #     # compute a loss value for every data point in batch
        #     loss = torch.zeros(x.shape[0])
        loss_Q = loss_pos_matrix_random_sampling(
            -Contraction
            - epsilon * torch.eye(Contraction.shape[-1]).unsqueeze(0).type(x.dtype),
            reduce=reduce
        )
        loss_C1 = loss_pos_matrix_random_sampling(
            -C1_LHS_1 - epsilon * torch.eye(C1_LHS_1.shape[-1]).unsqueeze(0).type(x.dtype),
            reduce=reduce
        )
        loss_M = loss_pos_matrix_random_sampling(
            args.w_ub * torch.eye(M.shape[-1]).unsqueeze(0).type(x.dtype) - M,
            reduce=reduce
        )
        # loss += loss_pos_matrix_random_sampling(W - args.w_lb * torch.eye(W.shape[-1]).unsqueeze(0).type(x.dtype))  # Make sure W is positive definite
        if reduce:
            loss_C2 = 1.0 * sum([1.0 * (C2**2).reshape(bs, -1).sum(dim=1).mean() for C2 in C2s])
        else:
            assert(len(sum([1.0 * (C2**2).reshape(bs, -1).sum(dim=1) for C2 in C2s])) == bs)
            loss_C2 = 1.0 * sum([1.0 * (C2**2).reshape(bs, -1).sum(dim=1) for C2 in C2s])

        if clone:
            overall_clone_loss = u_error.mean() + M_error.mean() + W_error.mean()
            loss = [overall_clone_loss]
        else:
            loss = [loss_Q, loss_C1, loss_M, loss_C2]

        if verbose:
            print("true eigval min,max,mean: ",
                torch.linalg.eigh(Contraction)[0].min(dim=1)[0].min().item(),
                torch.linalg.eigh(Contraction)[0].max(dim=1)[0].max().item(),
                torch.linalg.eigh(Contraction)[0].mean().item(),
            )
        if acc:
            # Contract_easier = (dot_M
                # + (A + B.matmul(K)).transpose(1, 2).matmul(M.detach())
                # + M.detach().matmul(A + B.matmul(K))
                # + 2 * _lambda/2 * M.detach())
            true_eig_min = torch.linalg.eigh(Contraction)[0].min(dim=1)[0].min().item()
            true_eig_max = torch.linalg.eigh(Contraction)[0].max(dim=1)[0].max().item()
            true_eig_mean = torch.linalg.eigh(Contraction)[0].mean().item()

            M_eigs = torch.linalg.eigh(M)[0]
            M_eigs_min = M_eigs.min(dim=1)[0] # max eig for each datapoint in batch
            M_eigs_max = M_eigs.max(dim=1)[0]
            M_cond_max = (M_eigs_max / M_eigs_min).max().item() # cond number for each point in batch then take max
            M_eig_max = M_eigs_max.max().item()
            M_eig_min = M_eigs_min.min().item()
            # p1 is the number of datapoints where all the eigenvalues are < 0 
            return (
                loss,
                ((torch.linalg.eigh(Contraction)[0] >= 0).sum(dim=1) == 0)
                .cpu()
                .detach()
                .numpy(),
                ((torch.linalg.eigh(C1_LHS_1)[0] >= 0).sum(dim=1) == 0).cpu().detach().numpy(),
                sum(
                    [1.0 * (C2**2).reshape(bs, -1).sum(dim=1).mean() for C2 in C2s]
                ).item(),
                (EigStats(true_eig_max, true_eig_min, true_eig_mean),
                 MeigStats(M_eig_max, M_eig_min, M_cond_max)),
            )
        else:
            return loss, None, None, None, None


    optimizer = torch.optim.Adam(
        list(model_W.parameters())
        + list(model_Wbot.parameters())
        + list(model_u_w1.parameters()),
        lr=args.learning_rate,
    )

    hardtanh_optimizer = torch.optim.Adam(
        list(model_W_hard.parameters())
        + list(model_Wbot_hard.parameters())
        + list(model_u_w1_hard.parameters()),
        lr=args.learning_rate,
    )

    def regularize(epoch, reg_coeff):
        loss = 0.0
        params = list(model_W.parameters()) + list(model_Wbot.parameters()) + list(model_u_w1.parameters())
        for p in params:
            if p.grad is not None and p.requires_grad:
                loss += p.norm(1)
        coeff = reg_coeff #*epoch
        # print("regularization weight is: ", coeff)
        return loss*coeff
    def setup_delta(Xshape, Xdtype, robust_eps):
        # sample init delta values
        # delta is x, xerr, uref
        delta_low = torch.tensor(np.concatenate([X_MIN, XE_MIN, U_MIN]).reshape(1,-1)).type(Xdtype)
        delta_high = torch.tensor(np.concatenate([X_MAX, XE_MAX, U_MAX]).reshape(1,-1)).type(Xdtype)
        # print("delta_low: ", delta_low)
        # print("delta_high: ", delta_high)
        delta_range = delta_high - delta_low
        # print("delta_range: ", delta_range)
        delta_min = -robust_eps*delta_range # delta should be zero centered
        delta_max = robust_eps*delta_range 
        def sample_delta():
            delta = torch.zeros(Xshape).type(Xdtype)
            delta.uniform_(-robust_eps, robust_eps)
            delta = (delta*(delta_range)).type(Xdtype)
            assert((delta <= delta_max).all() and (delta >= delta_min).all()) # sanity check
            return delta
        return sample_delta, delta_min, delta_max

    def reformat_limits(xtype, X_MIN_, X_MAX_, XE_MIN_, XE_MAX_, U_MIN_, U_MAX_):
        X_MIN_ =  torch.tensor(X_MIN_.reshape(1,-1)).type(xtype)
        X_MAX_ =  torch.tensor(X_MAX_.reshape(1,-1)).type(xtype)
        XE_MIN_ =torch.tensor(XE_MIN_.reshape(1,-1)).type(xtype)
        XE_MAX_ =torch.tensor(XE_MAX_.reshape(1,-1)).type(xtype)
        U_MIN_ =  torch.tensor(U_MIN_.reshape(1,-1)).type(xtype)
        U_MAX_ =  torch.tensor(U_MAX_.reshape(1,-1)).type(xtype)
        return X_MIN_, X_MAX_, XE_MIN_, XE_MAX_, U_MIN_, U_MAX_

    def attack_pgd(X,
                   robust_eps,
                   alpha, 
                   attack_iters,
                   restarts,
                   norm,
                   train,
                   acc,
                   detach,
                   clone,
                   verbose=False):
        """
        This function calculates one batch of adversarial inputs using a PGD attack.
        HOWEVER, the attack is done in terms of delta = {delta_x, delta_xerr, delta_uref} not interms of x_ref.
        """
        X = X.squeeze(-1)
        xerr = X[:,0:num_dim_x] - X[:, num_dim_x:(2*num_dim_x)] # xerr = x - xref
        #  keep track of most adversarial input found so far for each training example
        max_loss = torch.zeros(X.shape[0])
        max_delta = torch.zeros_like(X)
        if args.use_cuda:
            max_loss.cuda()
            max_delta.cuda()
        sample_delta, delta_min, delta_max = setup_delta(X.shape, X.dtype, robust_eps)
        # print("delta_min: ", delta_min, ", delta_max: ", delta_max)
        # the largest possible xerr delta is a function of the data points in X (xerr)
        # XE_MIN <= xerr + delta_xerr <= XE_MAX
        # print("X_MIN: ", X_MIN, ", X_MAX: ", X_MAX)
        x_min, x_max, xe_min, xe_max, u_min, u_max = reformat_limits(X.dtype, X_MIN, X_MAX, XE_MIN, XE_MAX, U_MIN, U_MAX)
        delta_min_xerr = torch.maximum(torch.tensor(XE_MIN.reshape(1,-1)) - xerr, delta_min[:,num_dim_x:(2*num_dim_x)].reshape(1,-1)).detach() # contains batch dim
        # print("most conservative min bounds: ", delta_min_xerr.max(dim=0))
        # print("least conservative min bounds: ", delta_min_xerr.min(dim=0))
        delta_max_xerr = torch.minimum(torch.tensor(XE_MAX.reshape(1,-1)) - xerr, delta_max[:,num_dim_x:(2*num_dim_x)].reshape(1,-1)).detach()
        # print("most conservative max bounds: ", delta_max_xerr.min(dim=0))
        # print("least conservative max bounds: ", delta_max_xerr.max(dim=0))
        # X is x, xref, uref
        X_low = torch.tensor(np.concatenate([X_MIN, X_MIN, U_MIN]).reshape(1,-1)).type(X.dtype)
        X_high = torch.tensor(np.concatenate([X_MAX, X_MAX, U_MAX]).reshape(1,-1)).type(X.dtype)
        # print("X_low: ", X_low)
        ############## Sanity check: loss before perturbations
        batch_losses_uptbd, p1_uptbd, p2, l3, _ = forward( 
            X[:, 0:num_dim_x].unsqueeze(-1),
            X[:, num_dim_x:(2*num_dim_x)].unsqueeze(-1),
            X[:, (2*num_dim_x):].unsqueeze(-1),
            _lambda=args._lambda,
            verbose=False if not train else False,
            acc=True,
            detach=detach,
            clone=clone,
            zero_order=False,
            reduce=False # key to computing loss value for every example in the batch
        )
        batch_loss_uptbd = batch_losses_uptbd[0]
        ##############
        for _ in range(restarts):
            # Initialize deltas to uniform random in 2*eps*range of each variable centered at center of range
            delta = sample_delta()
            # print("delta.shape: ", delta.shape)
            delta.requires_grad = True
            delta.retain_grad()
            # optimize the deltas
            for _ in range(attack_iters):
                # compute the perturbed inputs
                delta_x = delta[:,:num_dim_x]
                delta_xerr = delta[:, num_dim_x:2*num_dim_x]
                delta_uref = delta[:, 2*num_dim_x:]
                # clamp to x_err range because it won't be clamped in ptbd_X
                delta_xerr = torch.clamp(delta_xerr, min=delta_min_xerr, max=delta_max_xerr) # this uses batch dim bounds
                # clamp to x, xref and uref ranges
                delta_forX = torch.concatenate([delta_x,
                                                delta_x - delta_xerr,
                                                delta_uref], dim=1)
                ptbd_X = torch.clamp(X + delta_forX, min=X_low, max=X_high).type(X.dtype)
                # print("ptbd_X.dtype: ", ptbd_X.dtype)
                x_ptb = ptbd_X[:,0:num_dim_x].unsqueeze(-1)
                xref_ptb = ptbd_X[:, num_dim_x:(2*num_dim_x)].unsqueeze(-1)
                uref_ptb = ptbd_X[:, (2*num_dim_x):].unsqueeze(-1)
                
                # ### Sanity check: What is range of xerr?
                # x_err_ptb = x_ptb - xref_ptb
                # print("xerr_ptb.max(): ", x_err_ptb.max(), ", xerr_ptb.min(): ", x_err_ptb.min())
                
                # compute the loss
                losses, p1, p2, l3, _ = forward( #  this computes a scaler loss
                    x_ptb,
                    xref_ptb,
                    uref_ptb,
                    _lambda=args._lambda,
                    verbose=False if not train else False,
                    acc=acc,
                    detach=detach,
                    clone=clone,
                    zero_order=False,
                )
                ##################################
                loss = losses[0] # only use Q loss
                ##################################
                # compute gradients
                loss.backward()
                grad = delta.grad.detach()
                d = delta
                g = grad
                # print("delta.grad:",delta.grad)
                if norm == "l_inf":
                    # Do gradient ascent on the disturbance delta (d)
                    d = d + alpha * torch.sign(g)
                    # clamp to limited disturbance range
                    d = torch.clamp(d, min=delta_min, max=delta_max)
                elif norm == "l_2":
                    raise NotImplementedError
                    # g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                    # scaled_g = g/(g_norm + 1e-10)
                    # d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
                # d = torch.clamp(d, delta_low, delta_high) # probably unecessary?
                delta.data = d 
                # reset gradient before next iter
                delta.grad.zero_()
                # print("loss during attack: ", loss)
            #  here we compute a loss value for each example in the batch
            # compute the perturbed inputs
            delta_x = delta[:,:num_dim_x]
            delta_xerr = delta[:, num_dim_x:2*num_dim_x]
            delta_uref = delta[:, 2*num_dim_x:]
            # clamp to x_err range because it won't be clamped in ptbd_X
            delta_xerr = torch.clamp(delta_xerr, min=delta_min_xerr, max=delta_max_xerr)
            # clamp to x, xref and uref ranges. xerr = x - xref ==> xref = x - xerr
            delta_forX = torch.concatenate([delta_x,
                                            delta_x - delta_xerr,
                                            delta_uref], dim=1)
            ptbd_X = torch.clamp(X + delta_forX, min=X_low, max=X_high).type(X.dtype)
            # print("ptbd_X.dtype: ", ptbd_X.dtype)
            x_ptb = ptbd_X[:,0:num_dim_x].unsqueeze(-1)
            xref_ptb = ptbd_X[:, num_dim_x:(2*num_dim_x)].unsqueeze(-1)
            uref_ptb = ptbd_X[:, (2*num_dim_x):].unsqueeze(-1)
            # compute the loss
            batch_losses, p1, p2, l3, _ = forward(
                x_ptb,
                xref_ptb,
                uref_ptb,
                _lambda=args._lambda,
                verbose=False if not train else False,
                acc=acc,
                detach=detach,
                clone=clone,
                zero_order=False,
                reduce=False # key to computing loss value for every example in the batch
            )
            batch_loss = batch_losses[0] # only use Q loss
            # store largest delta values
            max_delta[batch_loss >= max_loss] = delta.detach()[batch_loss >= max_loss]
            max_loss = torch.max(max_loss, batch_loss)

            # ### Sanity check: check which data points for which the loss increased
            num_succ_att = (max_loss > batch_loss_uptbd).sum()
            # print("Number of successfully increased loss points: ", num_succ_att, " out of batch: ", X.shape[0])
            # ###

            # ### Sanity check: What is range of all vars?
            xerr_ptb = x_ptb - xref_ptb
            x_ptb =x_ptb.squeeze(-1)
            xref_ptb = xref_ptb.squeeze(-1)
            xerr_ptb = xerr_ptb.squeeze(-1)
            def print_failing_exs(limL, limU, val, name):
                Ltruth = torch.logical_or(limL <= val, torch.isclose(limL, val)).all()
                if not Ltruth:
                    print(f"L {name}: ", limL)
                    print(f"failing exs {name} L: ", val[(limL <= val).sum(dim=1) < val.shape[1], :])
                assert(Ltruth) # throw error
                Utruth = torch.logical_or(limU >= val, torch.isclose(limU, val)).all()
                if not Utruth:
                    print(f"U {name}: ", limU)
                    print(f"failing exs {name} U: ", val[(val <= limU).sum(dim=1) < val.shape[1], :])
                assert(Utruth) # throw error
            print_failing_exs(x_min, x_max, x_ptb, "x_ptb")
            print_failing_exs(x_min, x_max, xref_ptb, "xref_ptb")
            print_failing_exs(xe_min, xe_max, xerr_ptb, "xerr_ptb")
            print_failing_exs(u_min, u_max, uref_ptb, "uref_ptb")
            
            # Are we finding CEs? Calculate eigenvalues
            # for debug
            x_ptb =x_ptb.unsqueeze(-1)
            xref_ptb = xref_ptb.unsqueeze(-1)
            xerr_ptb = xerr_ptb.unsqueeze(-1)
            _, p1, p2, l3, _ = forward(
                x_ptb,
                xref_ptb,
                uref_ptb,
                _lambda=args._lambda,
                verbose=False,
                acc=True,
                detach=detach,
                clone=clone,
                zero_order=False,
                reduce=False # key to computing loss value for every example in the batch
            )
            if verbose:
                print(f"Number of points with eigval  < 0 (passing) before attack {p1_uptbd.sum()} vs after: {p1.sum()} out of {X.shape[0]}")
            num_new_CE = np.logical_and(p1_uptbd == True, p1 == False).sum()
            num_lost_CE = np.logical_and(p1_uptbd == False, p1 == True).sum()
            stats= Stats(num_new_CE, num_lost_CE)
            
            # print("after attack: xerr_ptb.max(): ", x_err_ptb.max(), ", xerr_ptb.min(): ", x_err_ptb.min())
        return max_delta, stats

    def robust_trainval(
        X,
        epoch,
        bs=args.bs,
        train=True,
        _lambda=args._lambda,
        acc=False,
        detach=False,
        clone=False,
        ptb_sched=None,
        reg_coeff=0.0
    ):  
        """
        This function implements 1 epoch of training with PGD attack
        """

        if train:
            indices = np.random.permutation(len(X))
        else:
            indices = np.array(list(range(len(X))))

        total_loss = 0
        total_p1 = 0
        total_p2 = 0
        total_l3 = 0
        total_num_new_CEs = 0
        total_num_lost_CEs = 0
        eigmin = 0.0
        eigmax = 0.0
        eigmean = 0.0

        num_train_batches = len(X) // bs
        if train:
            # print("len(X):", len(X), ", bs: ", bs)
            _iter = tqdm(range(num_train_batches))
        else:
            _iter = range(num_train_batches)
        for b in _iter:
            # for each batch
            start = time.time()
            x = []
            xref = []
            uref = []
            for id in indices[b * bs : (b + 1) * bs]:
                if args.use_cuda:
                    x.append(torch.from_numpy(X[id][0]).float().cuda())
                    xref.append(torch.from_numpy(X[id][1]).float().cuda())
                    uref.append(torch.from_numpy(X[id][2]).float().cuda())
                elif use_mps:
                    x.append(torch.from_numpy(X[id][0]).float().to("mps"))
                    xref.append(torch.from_numpy(X[id][1]).float().to("mps"))
                    uref.append(torch.from_numpy(X[id][2]).float().to("mps"))
                else:
                    x.append(torch.from_numpy(X[id][0]).float())
                    xref.append(torch.from_numpy(X[id][1]).float())
                    uref.append(torch.from_numpy(X[id][2]).float())

            x, xref, uref = (torch.stack(d).detach() for d in (x, xref, uref))
            x = x.requires_grad_()

            start = time.time()
            # print("x.shape: ", x.shape)
            # print("X.shape, ", torch.concatenate([x, xref, uref], dim=1).shape)
            robust_eps_i = ptb_sched(epoch + (b + 1)/num_train_batches)
            if robust_eps_i > 0:
                delta, stats = attack_pgd(torch.concatenate([x, xref, uref], dim=1),
                                robust_eps_i,
                                args.robust_alpha, 
                                args.robust_attack_iters,
                                args.robust_restarts,
                                args.robust_norm,
                                    train,
                                    acc,
                                    detach,
                                    clone)
                total_num_new_CEs += stats.num_new_CEs 
                total_num_lost_CEs += stats.num_lost_CEs
            else:
                delta = torch.zeros_like(torch.concatenate([x, xref, uref], dim=1).squeeze(-1))
            
            x_ptb = x + delta[:,0:num_dim_x].unsqueeze(-1)
            xref_ptb = xref + delta[:, num_dim_x:(2*num_dim_x)].unsqueeze(-1)
            uref_ptb = uref + delta[:, (2*num_dim_x):].unsqueeze(-1)

            losses, p1, p2, l3, _ = forward(
                x_ptb,
                xref_ptb,
                uref_ptb,
                _lambda=_lambda,
                verbose=False if not train else False,
                acc=acc,
                detach=detach,
                clone=clone,
                zero_order=False,
            )

            if reg_coeff > 0.0:
                losses.append(regularize(epoch, reg_coeff))

            loss = sum(losses)

            start = time.time()
            if train and not clone:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print('backwad(): %.3f s'%(time.time() - start))
            elif train and clone:
                hardtanh_optimizer.zero_grad()
                loss.backward()
                hardtanh_optimizer.step()

            total_loss += loss.item() * x.shape[0]
            if acc:
                total_p1 += p1.sum()
                total_p2 += p2.sum()
                total_l3 += l3 * x.shape[0]
                eigmin  = np.minimum(stats.min, eigmin)
                eigmax = np.maximum(stats.max, eigmax)
                eigmean = (b*eigmean + stats.mean)/(b+1)
        print("robust_eps_i at epoch end was: ", robust_eps_i)
        # extra: max/min eig vals? inc/dec in # CEs
        CEstats = Stats(total_num_new_CEs, total_num_lost_CEs)
        eigstats = EigStats(eigmax, eigmin, eigmean)
        return total_loss / len(X), total_p1 / len(X), total_p2 / len(X), total_l3 / len(X), (CEstats, eigstats)


    def trainval(
        X,
        bs=args.bs,
        train=True,
        _lambda=args._lambda,
        acc=False,
        detach=False,
        clone=False,
        epoch=1.,
        reg_coeff=args.reg_coeff,
        ptb_sched=None,
        robust=False,
    ):  
        """
        This function implements 1 epoch of training with or without PGD attack.
        """
        # trainval a set of x
        # torch.autograd.set_detect_anomaly(True)

        if train:
            indices = np.random.permutation(len(X))
        else:
            indices = np.array(list(range(len(X))))

        total_num_new_CEs = 0
        total_num_lost_CEs = 0
        total_loss = 0
        total_p1 = 0
        total_p2 = 0
        total_l3 = 0
        eigmin = 1e6
        eigmax = 0.0
        eigmean = 0.0
        Meigmin = 1e6 
        Meigmax = 0.0 
        Meigcondmax = 0.0

        num_train_batches = len(X) // bs

        if train:
            # print("len(X):", len(X), ", bs: ", bs)
            _iter = tqdm(range(num_train_batches))
        else:
            _iter = range(num_train_batches)
        for b in _iter:
            start = time.time()
            x = []
            xref = []
            uref = []
            for id in indices[b * bs : (b + 1) * bs]:
                if args.use_cuda:
                    x.append(torch.from_numpy(X[id][0]).float().cuda())
                    xref.append(torch.from_numpy(X[id][1]).float().cuda())
                    uref.append(torch.from_numpy(X[id][2]).float().cuda())
                elif use_mps:
                    x.append(torch.from_numpy(X[id][0]).float().to("mps"))
                    xref.append(torch.from_numpy(X[id][1]).float().to("mps"))
                    uref.append(torch.from_numpy(X[id][2]).float().to("mps"))
                else:
                    x.append(torch.from_numpy(X[id][0]).float())
                    xref.append(torch.from_numpy(X[id][1]).float())
                    uref.append(torch.from_numpy(X[id][2]).float())

            x, xref, uref = (torch.stack(d).detach() for d in (x, xref, uref))
            x = x.requires_grad_()

            start = time.time()
            if robust:
                robust_eps_i = ptb_sched(epoch + (b + 1)/num_train_batches)
                if robust_eps_i > 0:
                    delta, stats = attack_pgd(torch.concatenate([x, xref, uref], dim=1),
                                    robust_eps_i,
                                    args.robust_alpha, 
                                    args.robust_attack_iters,
                                    args.robust_restarts,
                                    args.robust_norm,
                                        train,
                                        acc,
                                        detach,
                                        clone)
                    total_num_new_CEs += stats.num_new_CEs 
                    total_num_lost_CEs += stats.num_lost_CEs
                else:
                    delta = torch.zeros_like(torch.concatenate([x, xref, uref], dim=1).squeeze(-1))
                    
                x = x + delta[:,0:num_dim_x].unsqueeze(-1)
                xref = xref + delta[:, num_dim_x:(2*num_dim_x)].unsqueeze(-1)
                uref = uref + delta[:, (2*num_dim_x):].unsqueeze(-1)

            losses, p1, p2, l3, stats = forward(
                x,
                xref,
                uref,
                _lambda=_lambda,
                verbose=False if not train else False,
                acc=acc,
                detach=detach,
                clone=clone,
                zero_order=False,      
            )

            if reg_coeff > 0.0:
                losses.append(regularize(epoch, reg_coeff))

            sum_loss = sum(losses)

            start = time.time()
            if train and not clone:
                optimizer.zero_grad()
                sum_loss.backward()
                optimizer.step()
                # print('backwad(): %.3f s'%(time.time() - start))
            elif train and clone:
                hardtanh_optimizer.zero_grad()
                sum_loss.backward()
                hardtanh_optimizer.step()

            # print("sum_loss.item(): ", sum_loss.item())
            total_loss += sum_loss.item() * x.shape[0]
            # print("total_loss:", total_loss)
            if acc:
                total_p1 += p1.sum()
                total_p2 += p2.sum()
                total_l3 += l3 * x.shape[0]
                eigstats, Meigstats = stats
                eigmin  = np.minimum(eigstats.min, eigmin)
                eigmax = np.maximum(eigstats.max, eigmax)
                eigmean = (b*eigmean + eigstats.mean)/(b+1)
                Meigmin = np.minimum(Meigstats.min, Meigmin) 
                Meigmax = np.maximum(Meigstats.max, Meigmax)
                Meigcondmax = np.maximum(Meigstats.cond_max, Meigcondmax)
        if robust:
            print("robust_eps_i at epoch end was: ", robust_eps_i)
        eigstats = EigStats(eigmax, eigmin, eigmean)
        meigstats = MeigStats(Meigmax, Meigmin, Meigcondmax)
        CE_stats = Stats(total_num_new_CEs, total_num_lost_CEs) 
        return total_loss / len(X), total_p1 / len(X), total_p2 / len(X), total_l3 / len(X), (eigstats, meigstats, CE_stats, losses)


    best_acc = 0


    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by every args.lr_step epochs"""
        lr = args.learning_rate * (0.1 ** (epoch // args.lr_step))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


    if args.load_model:
        # load the metric and controller
        filename_controller = args.log + "/controller_best.pth.tar"
        filename_metric = args.log + "/metric_best.pth.tar"
        map_location = "cuda" if args.use_cuda else "cpu"
        W_func = torch.load(filename_metric, map_location)
        u_func = torch.load(filename_controller, map_location)

        # Unpack the test data
        x = []
        xref = []
        uref = []
        N = 30000
        for id in range(len(X_te[:N])):
            if args.use_cuda:
                x.append(torch.from_numpy(X_te[id][0]).float().cuda())
                xref.append(torch.from_numpy(X_te[id][1]).float().cuda())
                uref.append(torch.from_numpy(X_te[id][2]).float().cuda())
            else:
                x.append(torch.from_numpy(X_te[id][0]).float())
                xref.append(torch.from_numpy(X_te[id][1]).float())
                uref.append(torch.from_numpy(X_te[id][2]).float())

        x, xref, uref = (torch.stack(d).detach() for d in (x, xref, uref))
        x = x.requires_grad_()

        # get the contraction condition
        bs = x.shape[0]
        if INVERSE_METRIC:
            W = W_func(x)
            M = torch.inverse(W)
        else:
            M = W_func(x)
            W = torch.inverse(M)
        f = f_func(x)
        B = B_func(x)
        DfDx = Jacobian(f, x)
        DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.dtype)
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
        Contraction = (
            dot_M
            + (A + B.matmul(K)).transpose(1, 2).matmul(M)
            + M.matmul(A + B.matmul(K))
            + 2 * _lambda * M
        )

        # Plot the maximum eigenvalue of the contraction condition
        max_eig_Q = torch.linalg.eigh(Contraction)[0].max(dim=1)[0].detach()
        violation = max_eig_Q > 0.0
        print("-------------")
        print(f"Total # violations: {violation.sum()} ({violation.sum() * 100 / N} %)")
        print("violation: ", violation)
        if violation.any():
            print(f"mean violation: {max_eig_Q[violation].mean()} ({max_eig_Q[violation].max()} max)")
        print("-------------")
        fig, axs = plt.subplots(4, 4)
        for i in range(4):
            for j in range(4):
                axs[i, j].scatter(
                    x[violation, j].detach().cpu(),
                    x[violation, i].detach().cpu(),
                    c=max_eig_Q[violation].cpu(),
                    s=0.1,
                )
                axs[i, j].set_xlabel(f"x_{j}")
                axs[i, j].set_ylabel(f"x_{i}")
                axs[i, j].set_xlim(
                    [
                        x[:, j].detach().min().cpu() - 0.1,
                        x[:, j].detach().max().cpu() + 0.1,
                    ]
                )
                axs[i, j].set_ylim(
                    [
                        x[:, i].detach().min().cpu() - 0.1,
                        x[:, i].detach().max().cpu() + 0.1,
                    ]
                )

        # fig = plt.figure()
        # # ax = fig.add_subplot(projection="3d")
        # ax = fig.add_subplot()
        # ax.scatter(
        #     x[violation, 2].detach(),
        #     x[violation, 3].detach(),
        #     c=max_eig_Q[violation],
        # )
        # # plt.colorbar()
        # ax.set_xlabel("theta")
        # ax.set_ylabel("velocity")
        # ax.set_title("Maximum eigenvalue of Q (when > 0)")
        # plt.show()

        # import seaborn as sns

        # max_eig_M_dot = torch.linalg.eigh(dot_M)[0].view(-1).detach()
        # MABK = (A + B.matmul(K)).transpose(1, 2).matmul(M) + M.matmul(A + B.matmul(K))
        # max_eig_MABK = torch.linalg.eigh(MABK)[0].view(-1).detach()
        # max_eig_M = torch.linalg.eigh(2 * _lambda * M)[0].view(-1).detach()
        # max_eig_Q = torch.linalg.eigh(Contraction)[0].view(-1).detach()

        # _, axs = plt.subplots(1, 3)
        # sns.histplot(x=max_eig_M_dot, ax=axs[0])
        # sns.histplot(x=max_eig_MABK, ax=axs[1])
        # sns.histplot(x=max_eig_Q, ax=axs[2])

        # axs[0].set_title("Mdot")
        # axs[1].set_title("MABK")
        # axs[2].set_title("Q")

        plt.show()

        sys.exit()

    layout = {
        "Eigenvalues":{
            "Eigenvalues": ["Multiline", ["eig/max", "eig/min", "eig/mea"]],
        },
    }
    layout2 = {
        "M_Eigen": {
            "M_Eigen": ["multiline", ["Meig/max", "Meig/min", "Meig/condmax"]],
        },
    }
    layout3 = {
        "Loss_Terms": {
            "Loss_Terms": ["multiline", ["loss/Q", "loss/C1", "loss/W"]],
        },
    }

    writer.add_custom_scalars(layout)
    writer.add_custom_scalars(layout2)
    writer.add_custom_scalars(layout3)

    # perturbation schedule for robust training
    ptb_schedule = lambda t: np.interp([t], [0, 1.5, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [0.0, 0.0, args.robust_eps/2., args.robust_eps/1.2, args.robust_eps])[0]
    # Start training a tanh network
    print("Using robust training: ", args.robust_train)
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # if args.robust_train:
        #     print("Using adversarial training.")
        #     loss, p1_train, _, _, extra_train = robust_trainval(X_tr,
        #                                     epoch,
        #                     train=True,
        #                     _lambda=args._lambda,
        #                     acc=False,
        #                     detach=True if epoch < args.lr_step else False,
        #                     ptb_sched = ptb_schedule,
        #                 )
        #     CE_train, eig_train = extra_train
        #     writer.add_scalar("number new CE from training", CE_train.num_new_CEs, epoch)
        #     writer.add_scalar("number lost CE from training", CE_train.num_lost_CEs, epoch)
        # else:
            # print("Using vanilla training.")
        loss, _, _, _, train_stats = trainval(X_tr,
                                    train=True,
                                    _lambda=args._lambda,
                                    acc=False,
                                    detach=True if epoch < args.lr_step else False,
                                    epoch=epoch,
                                    ptb_sched = ptb_schedule,
                                    robust=args.robust_train
                                )
        _, _, CE_stats, _ = train_stats
        if args.robust_train:
            writer.add_scalar("number new CE from training", CE_stats.num_new_CEs, epoch)
            writer.add_scalar("number lost CE from training", CE_stats.num_lost_CEs, epoch)
        print("Training loss: ", loss)
        # now test
        loss, p1, p2, l3, test_stats = trainval(X_te, train=False, _lambda=0.0, acc=True, detach=False)
        print("Epoch %d: Testing loss/p1/p2/l3: " % epoch, loss, p1, p2, l3)
        eig_stats, meig_stats, CE_stats, losses = test_stats
        if len(losses) == 4:
            loss_Q, loss_C1, loss_W, loss_C2 = losses
        elif len(losses) == 5:
            loss_Q, loss_C1, loss_W, loss_C2, loss_reg = losses
            writer.add_scalar("Regularization loss", loss_reg, epoch)

        writer.add_scalar("Loss", loss, epoch)
        writer.add_scalar("% of pts with max eig Q < 0", p1, epoch)
        writer.add_scalar("% of pts with max eig C1_LHS < 0", p2, epoch)
        writer.add_scalar("eig/max", eig_stats.max, epoch)
        writer.add_scalar("eig/min", eig_stats.min, epoch)
        writer.add_scalar("eig/mean", eig_stats.mean, epoch) 
        writer.add_scalar("Meig/max", meig_stats.max, epoch)
        writer.add_scalar("Meig/min", meig_stats.min, epoch)
        writer.add_scalar("Meig/condmax", meig_stats.cond_max, epoch)
        writer.add_scalar("mean C2^2", l3, epoch)
        writer.add_scalar("loss/Q", loss_Q, epoch)
        writer.add_scalar("loss/C1", loss_C1, epoch)
        writer.add_scalar("loss/W", loss_W, epoch)

        if p1 + p2 >= best_acc:
            best_acc = p1 + p2
            filename = args.log + "/model_best.pth.tar"
            filename_controller = args.log + "/controller_best.pth.tar"
            filename_metric = args.log + "/metric_best.pth.tar"
            torch.save(
                {
                    "args": args,
                    "precs": (loss, p1, p2),
                    "model_W": model_W.state_dict(),
                    "model_Wbot": model_Wbot.state_dict(),
                    "model_u_w1": model_u_w1.state_dict(),
                    "model_W_hard": model_W_hard.state_dict(),
                    "model_Wbot_hard": model_Wbot_hard.state_dict(),
                    "model_u_w1_hard": model_u_w1_hard.state_dict(),
                },
                filename,
            )
            torch.save(u_func, filename_controller)
            torch.save(W_func, filename_metric)

        writer.close()

    if args.clone:
        # Once that's trained, train a hardtanh network to imitate it
        print("------------ Initial training done; transferring to hardtanh ------------")
        best_acc = 0
        args.learning_rate = 0.01
        for epoch in range(args.epochs):
            adjust_learning_rate(hardtanh_optimizer, epoch)
            loss, _, _, _ = trainval(
                X_tr,
                train=True,
                _lambda=args._lambda,
                acc=False,
                detach=False,
                clone=True,
            )
            print("Training loss: ", loss)
            loss, p1, p2, l3 = trainval(
                X_te, train=False, _lambda=0.0, acc=True, detach=False, clone=True
            )
            print("Epoch %d: Testing loss/p1/p2/l3: " % epoch, loss, p1, p2, l3)

            writer.add_scalar("Loss", loss, epoch + args.epochs)
            writer.add_scalar("% of pts with max eig Q < 0", p1, epoch + args.epochs)
            writer.add_scalar("% of pts with max eig C1_LHS < 0", p2, epoch + args.epochs)
            writer.add_scalar("mean C2^2", l3, epoch + args.epochs)

            if p1 + p2 >= best_acc:
                best_acc = p1 + p2
                filename = args.log + "/model_best_hardtanh.pth.tar"
                filename_controller = args.log + "/controller_best_hardtanh.pth.tar"
                filename_metric = args.log + "/metric_best_hardtanh.pth.tar"
                torch.save(
                    {
                        "args": args,
                        "precs": (loss, p1, p2),
                        "model_W": model_W.state_dict(),
                        "model_Wbot": model_Wbot.state_dict(),
                        "model_u_w1": model_u_w1.state_dict(),
                        "model_W_hard": model_W_hard.state_dict(),
                        "model_Wbot_hard": model_Wbot_hard.state_dict(),
                        "model_u_w1_hard": model_u_w1_hard.state_dict(),
                    },
                    filename,
                )
                torch.save(u_func_hard, filename_controller)
                torch.save(W_func_hard, filename_metric)

            writer.close()

if __name__ == '__main__':
    main()

"""
to use from another script:
from main import main

main(["--log", "saved_models/V_2x16_to_3x64", "--load_model"])
"""