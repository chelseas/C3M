import torch
from torch.autograd import grad
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
use_mps = False
import pdb
# if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     torch.set_default_device('mps')
#     use_mps = True

from torch.utils.data import Dataset, DataLoader
from data_loading import TrackingDataset
import importlib
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import sys

import argparse
# import pdb

Stats = namedtuple("Stats", "num_new_CEs num_lost_CEs")

EigStats = namedtuple("EigStats", "max min mean")
MeigStats = namedtuple("MeigStats", "max min cond_max")
CounterExamples = namedtuple("CounterExamples", "x xref uref val")

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
        "--lambda", type=float, dest="_lambda", default=1.0, help="training convergence rate: lambda"
    )
    parser.add_argument(
        "--lambda_test", type=float, dest="test_lambda", default=0.1, help="test convergence rate: lambda"
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
        default=0.1,
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
        "--robust_attack_iters_train",
        dest="robust_attack_iters_train",
        type=int,
        default=2,
        help="Number of iterations to create adversarial example during robust training.",
    )
    parser.add_argument(
        "--robust_attack_iters_test",
        dest="robust_attack_iters_test",
        type=int,
        default=5,
        help="Number of iterations to create adversarial example during robust testing.",
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
    parser.add_argument(
        "--M_width",
        dest="M_width",
        type=int,
        default=128,
        help="Width of the metric network.",
    )
    parser.add_argument(
        "--M_depth",
        dest="M_depth",
        type=int,
        default=2,
        help="Depth of the metric network.",
    )
    parser.add_argument(
        "--u_width",
        dest="u_width",
        type=int,
        default=128,
        help="Width of the control network.",
    )
    parser.add_argument(
        "--u_depth",
        dest="u_depth",
        type=int,
        default=2,
        help="Depth of the control network.",
    )
    parser.add_argument(
        "--gersh_spread",
        dest="useopt",
        action="store_false",
        help="An attempt at a gershgorin loss that has better gradient sharing. ",
    )
    args = parser.parse_args(args)
    return args

def main(args=None):
    # pdb.set_trace()
    args = cmdlineparse(args)
    print("cmd line args: ", args)
    if not os.path.isdir(args.log):
        os.mkdir(args.log)
    if not args.load_model:
        os.system("cp *.py " + args.log)
        os.system("cp -r models/ " + args.log + "/models")
        os.system("cp -r configs/ " + args.log + "/configs")
        os.system("cp -r systems/ " + args.log + "/systems")
        writer = SummaryWriter(args.log + "/tb")

    sys.path.append(args.log)
    sys.path.append(args.log + "/systems")
    sys.path.append(args.log + "/configs")
    sys.path.append(args.log + "/models")
    global_steps = 0
    if args.use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    epsilon = args._lambda
    print("Epsilon/Lambda: ", epsilon, " test eps/lamb: ", args.test_lambda)
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


    #########
    # load data
    if args.use_cuda:
        generator=torch.Generator(device='cuda')
    else:
        generator=torch.Generator(device='cpu')    
    X_tr = TrackingDataset(args.num_train, config, system, args, use_mps=use_mps)
    X_tr_DL = DataLoader(X_tr, batch_size=args.bs, shuffle=True, generator=generator)
    X_te = TrackingDataset(args.num_test, config, system, args, use_mps=use_mps)
    X_te_DL = DataLoader(X_te, batch_size=args.bs, shuffle=True, generator=generator)
    print(type(X_tr_DL))
    # for i_batch, sample_batched in enumerate(dataloader):
    #     if i_batch == 0:
    #         print(i_batch, sample_batched[0][0].shape,
    #                        sample_batched[0][1].shape,
    #                        sample_batched[0][2].shape)
    #     assert 1==0
    #########

    model = importlib.import_module("model_" + args.task)
    get_model = model.get_model
    INVERSE_METRIC = model.INVERSE_METRIC

    if args.load_model:
        filename_model = args.log + "/model_best.pth.tar"
        filename_metric = args.log + "/metric_best.pth.tar"
        filename_controller = args.log + "/controller_best.pth.tar"
        map_location = "cuda" if args.use_cuda else "cpu"
        trained_model = torch.load(filename_model, map_location=map_location)
        model_W = trained_model['model_W']
        model_Wbot = trained_model['model_Wbot']
        model_u_w1 = trained_model['model_u_w1']
        W_func = torch.load(filename_metric, map_location=map_location)
        u_func = torch.load(filename_controller, map_location=map_location)
        # model_W_hard = trained_model['model_W_hard']
        # model_Wbot_hard = trained_model['model_Wbot_hard']
        # model_u_w1_hard = trained_model['model_u_w1_hard']
        loaded_args = trained_model['args']
    else:
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
        ) = get_model(num_dim_x, num_dim_control, w_lb=args.w_lb, use_cuda=args.use_cuda,
                    M_width=args.M_width, M_depth=args.M_depth, u_width=args.u_width, u_depth=args.u_depth)

    if use_mps:
        print("casting to mps again")
        model_W = model_W.to("mps")
        model_Wbot = model_Wbot.to("mps")
        model_u_w1 = model_u_w1.to("mps")
        u_func = u_func.to("mps")
        W_func = W_func.to("mps")

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

    def gersh_criterion(A, useopt=True):
        A = A.squeeze(-1) # remove trailing 1 dim
        diagonal_entries = torch.diagonal(A, dim1=-2, dim2=-1)
        off_diagonal_sum = torch.abs(A).sum(dim=-1) - torch.abs(diagonal_entries)
        gersh_LBs = diagonal_entries - off_diagonal_sum # want this to be >0
        if useopt:
            gersh_LB = gersh_LBs.amin(dim=-1)
        else:
            gersh_LB = gersh_LBs
        return gersh_LB
        
    
    def gershgorin_loss(A, reduce=True, useopt=True):
        gersh_LB = gersh_criterion(A, useopt=useopt)
        # print("gersh_LB.shape: ", gersh_LB.shape)
        negative_index = gersh_LB < 0
        if reduce:
            if negative_index.sum() > 0:
                return -1 * gersh_LB[negative_index].mean()
            else:
                return 0.0
        else:
            gersh_LB[torch.logical_not(negative_index)] = 0.0
            if useopt:
                return -1 * gersh_LB
            else:
                return -1 * gersh_LB.sum(dim=-1)
            

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
        debug=False,
        CEs=False,
        useopt=True
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
        # print("u: ", u)

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

        ts = time.time()
        loss_Q1 = loss_pos_matrix_random_sampling(
            -Contraction
            - epsilon * torch.eye(Contraction.shape[-1]).unsqueeze(0).type(x.dtype),
            reduce=reduce
        )
        # # tf = time.time()
        # loss_Q2 = gershgorin_loss(-Contraction,  
            # reduce=reduce, useopt=useopt)
        # loss_Q = loss_Q1 + loss_Q2
        # trueigs = torch.linalg.eigvalsh(Contraction).max(dim=1)[0]
        # # print("true eigs: ", trueigs.shape, trueigs)
        # loss_Q3 = (trueigs[trueigs > 0]).mean()
        # print("loss_Q: ", loss_Q.shape, loss_Q)
        loss_Q = loss_Q1 # + loss_Q2 #+ loss_Q3
        tf = time.time()

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
            Q_eigs = torch.linalg.eigvalsh(Contraction)
            Q_eigs_max = Q_eigs.max(dim=1)[0]
            true_eig_min = Q_eigs.min(dim=1)[0].min().item()
            true_eig_max =           Q_eigs_max.max().item()
            true_eig_mean = Q_eigs.mean().item()
            if CEs:
                Qvi = Q_eigs_max.detach() > 0.0 # Qvi = Q violation indices
                QCEs = CounterExamples(x[Qvi, :], xref[Qvi, :], uref[Qvi, :], Q_eigs_max[Qvi])
            else:
                QCEs = CounterExamples(torch.zeros((1,1)), torch.zeros((1,1)), torch.zeros((1,1)), torch.zeros((1,1)))

            Q_eigbounds = -gersh_criterion(-Contraction) # upperbound with the negations like that
            Q_eigbounds_min = Q_eigbounds.min().item()
            Q_eigbounds_max = Q_eigbounds.max().item()
            Q_eigbounds_mean = Q_eigbounds.mean().item()
            Qb_p1 = Q_eigbounds.cpu().detach().numpy() < 0.0 # passing indices
            if CEs:
                Qbvi = np.logical_not(Qb_p1) # Qbvi = Q bound violation indices
                print("---------------------------------type of x: ", type(x))
                QbCEs = CounterExamples(x[Qbvi, :], xref[Qbvi, :], uref[Qbvi, :], Q_eigbounds[Qbvi])
            else:
                QbCEs = CounterExamples(torch.zeros((1,1)), torch.zeros((1,1)), torch.zeros((1,1)), torch.zeros((1,1)))

            M_eigs = torch.linalg.eigh(M)[0]
            M_eigs_min = M_eigs.min(dim=1)[0] # max eig for each datapoint in batch
            M_eigs_max = M_eigs.max(dim=1)[0]
            M_cond_max = (M_eigs_max / M_eigs_min).max().item() # cond number for each point in batch then take max
            M_eig_max = M_eigs_max.max().item()
            M_eig_min = M_eigs_min.min().item()
            # p1 is the number of datapoints where all the eigenvalues are < 0 
            return (
                loss,
                ((Q_eigs >= 0).sum(dim=1) == 0).cpu().detach().numpy(),
                ((torch.linalg.eigh(C1_LHS_1)[0] >= 0).sum(dim=1) == 0).cpu().detach().numpy(),
                sum([1.0 * (C2**2).reshape(bs, -1).sum(dim=1).mean() for C2 in C2s]).item(),
                (EigStats(true_eig_max, true_eig_min, true_eig_mean),
                 MeigStats(M_eig_max, M_eig_min, M_cond_max),
                 EigStats(Q_eigbounds_max, Q_eigbounds_min, Q_eigbounds_mean),
                 QCEs, 
                 QbCEs,
                 Qb_p1,
                 (tf-ts)),
            )
        else:
            return loss, None, None, None, tf-ts


    if not args.load_model:
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
        def sample_delta(restarts):
            delta_shape = (Xshape[0]*restarts, *Xshape[1:])
            delta = torch.zeros(delta_shape).type(Xdtype)
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
                   useopt,
                   verbose=False):
        """
        This function calculates one batch of adversarial inputs using a PGD attack.
        HOWEVER, the attack is done in terms of delta = {delta_x, delta_xerr, delta_uref} not interms of x_ref.
        """
        total_Q_time = 0.0
        X = X.squeeze(-1)
        xerr = X[:,0:num_dim_x] - X[:, num_dim_x:(2*num_dim_x)] # xerr = x - xref
        sample_delta, delta_min, delta_max = setup_delta(X.shape, X.dtype, robust_eps)
        # print("delta_min: ", delta_min, ", delta_max: ", delta_max)
        # the largest possible xerr delta is a function of the data points in X (xerr)
        # XE_MIN <= xerr + delta_xerr <= XE_MAX
        # print("X_MIN: ", X_MIN, ", X_MAX: ", X_MAX)
        x_min, x_max, xe_min, xe_max, u_min, u_max = reformat_limits(X.dtype, X_MIN, X_MAX, XE_MIN, XE_MAX, U_MIN, U_MAX)
        delta_min_xerr = torch.maximum(torch.tensor(XE_MIN.reshape(1,-1)) - xerr, delta_min[:,num_dim_x:(2*num_dim_x)].reshape(1,-1)).detach().unsqueeze(0) # contains batch dim
        # print("most conservative min bounds: ", delta_min_xerr.max(dim=1))
        # print("least conservative min bounds: ", delta_min_xerr.min(dim=1))
        delta_max_xerr = torch.minimum(torch.tensor(XE_MAX.reshape(1,-1)) - xerr, delta_max[:,num_dim_x:(2*num_dim_x)].reshape(1,-1)).detach().unsqueeze(0)
        # print("most conservative max bounds: ", delta_max_xerr.min(dim=1))
        # print("least conservative max bounds: ", delta_max_xerr.max(dim=1))
        # X is x, xref, uref
        X_low = torch.tensor(np.concatenate([X_MIN, X_MIN, U_MIN]).reshape(1,-1)).type(X.dtype)
        X_high = torch.tensor(np.concatenate([X_MAX, X_MAX, U_MAX]).reshape(1,-1)).type(X.dtype)
        # print("X_low: ", X_low)
        ############## Sanity check: loss before perturbations
        # batch_losses_uptbd, p1_uptbd, p2, l3, _ = forward( 
        batch_losses_uptbd, _, _, _, stats = forward( 
            X[:, 0:num_dim_x].unsqueeze(-1),
            X[:, num_dim_x:(2*num_dim_x)].unsqueeze(-1),
            X[:, (2*num_dim_x):].unsqueeze(-1),
            _lambda=args._lambda,
            verbose=False if not train else False,
            # acc=True,
            acc=acc,
            detach=detach,
            clone=clone,
            zero_order=False,
            reduce=False, # key to computing loss value for every example in the batch
            useopt=useopt
        )
        batch_loss_uptbd = batch_losses_uptbd[0]
        #  keep track of most adversarial input found so far for each training example
        max_loss = batch_loss_uptbd
        max_delta = torch.zeros_like(X)
        if args.use_cuda:
            max_loss.cuda()
            max_delta.cuda()
        if acc:
            _, _, _, _, _, _, times = stats
        else:
            times =stats
        total_Q_time += times
        ##############
        # Initialize deltas to uniform random in 2*eps*range of each variable centered at zero
        delta = sample_delta(restarts).reshape(restarts, *X.shape)  # shape (restarts, bs, state_dim)
        # print("delta is leaf? ", delta.is_leaf)
        # print("delta.shape: ", delta.shape)
        delta.requires_grad = True
        delta.retain_grad()
        # optimize the deltas
        for _ in range(attack_iters):
            # delta = delta.reshape(restarts, *X.shape)
            # compute the perturbed inputs
            delta_x = delta[:,:,:num_dim_x]
            delta_xerr = delta[:,:, num_dim_x:2*num_dim_x] # shape (restarts, bs, x_state_dim)
            delta_uref = delta[:,:, 2*num_dim_x:]
            # clamp to x_err range because it won't be clamped in ptbd_X
            delta_xerr = torch.clamp(delta_xerr, min=delta_min_xerr, max=delta_max_xerr) # this uses batch dim bounds
            # clamp to x, xref and uref ranges
            delta_forX = torch.concatenate([delta_x,
                                            delta_x - delta_xerr,
                                            delta_uref], dim=-1) # should be (restarts, *X.shape)
            # in this call, we now put it into the shape (bs*restarts, state_dim)
            ptbd_X = torch.clamp((X.unsqueeze(0) + delta_forX).reshape((X.shape[0]*restarts, *X.shape[1:])), min=X_low, max=X_high).type(X.dtype)
            x_ptb = ptbd_X[:,0:num_dim_x].unsqueeze(-1)
            xref_ptb = ptbd_X[:, num_dim_x:(2*num_dim_x)].unsqueeze(-1)
            uref_ptb = ptbd_X[:, (2*num_dim_x):].unsqueeze(-1)
            
            # ### Sanity check: What is range of xerr?
            # x_err_ptb = x_ptb - xref_ptb
            # print("xerr_ptb.max(): ", x_err_ptb.max(), ", xerr_ptb.min(): ", x_err_ptb.min())
            
            # compute the loss
            losses, p1, p2, l3, stats = forward( #  this computes a scaler loss
                x_ptb,
                xref_ptb,
                uref_ptb,
                _lambda=args._lambda,
                verbose=False if not train else False,
                acc=acc,
                detach=detach,
                clone=clone,
                zero_order=False,
                useopt=useopt
            )
            if acc:
                _, _, _, _, _, _, times = stats
            else:
                times = stats
            total_Q_time += times
            ##################################
            loss = losses[0] # only use Q loss
            ##################################
            # compute gradients
            loss.backward()
            grad = delta.grad.detach()
            # print("grad: ", type(grad), ", grad.shape: ", grad.shape)
            d = delta
            g = grad
            # print("delta.grad:",delta.grad)
            if norm == "l_inf":
                # Do gradient ascent on the disturbance delta (d)
                d = d + alpha * torch.sign(g)
                # clamp to limited disturbance range
                d = torch.clamp(d, min=delta_min.unsqueeze(0), max=delta_max.unsqueeze(0))
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
        # delta = delta.reshape(restarts, *X.shape)
        delta_x = delta[:,:,:num_dim_x]
        delta_xerr = delta[:,:,num_dim_x:2*num_dim_x]
        delta_uref = delta[:,:, 2*num_dim_x:]
        # clamp to x_err range because it won't be clamped in ptbd_X
        delta_xerr = torch.clamp(delta_xerr, min=delta_min_xerr, max=delta_max_xerr)
        # clamp to x, xref and uref ranges. xerr = x - xref ==> xref = x - xerr
        delta_forX = torch.concatenate([delta_x,
                                        delta_x - delta_xerr,
                                        delta_uref], dim=-1)
        ptbd_X = torch.clamp((X.unsqueeze(0) + delta_forX).reshape((X.shape[0]*restarts, *X.shape[1:])), min=X_low, max=X_high).type(X.dtype)
        # print("ptbd_X.dtype: ", ptbd_X.dtype)
        x_ptb = ptbd_X[:,0:num_dim_x].unsqueeze(-1)
        xref_ptb = ptbd_X[:, num_dim_x:(2*num_dim_x)].unsqueeze(-1)
        uref_ptb = ptbd_X[:, (2*num_dim_x):].unsqueeze(-1)
        # compute the loss
        batch_losses, p1, p2, l3, stats = forward(
            x_ptb,
            xref_ptb,
            uref_ptb,
            _lambda=args._lambda,
            verbose=False if not train else False,
            acc=acc,
            detach=detach,
            clone=clone,
            zero_order=False,
            reduce=False, # key to computing loss value for every example in the batch
            useopt=useopt
        )
        batch_loss = batch_losses[0] # only use Q loss

        # take max over restarts (done in parallel)
        bl = batch_loss.reshape((restarts, X.shape[0]))
        dl = delta.reshape((restarts, *X.shape))
        mor_bl, mor_ind = bl.max(dim=0)
        mor_delta = dl[mor_ind, torch.arange(dl.shape[1]), :].reshape(X.shape)

        # store largest delta values
        max_delta[mor_bl >= max_loss] = mor_delta.detach()[mor_bl >= max_loss]
        max_loss = torch.max(max_loss, mor_bl)
        if acc:
            _, _, _, _, _, _,times = stats
        else:
            times =stats
        total_Q_time += times

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
        # x_ptb =x_ptb.unsqueeze(-1)
        # xref_ptb = xref_ptb.unsqueeze(-1)
        # xerr_ptb = xerr_ptb.unsqueeze(-1)
        # _, p1, p2, l3, CEstats = forward(
        #     x_ptb,
        #     xref_ptb,
        #     uref_ptb,
        #     _lambda=args._lambda,
        #     verbose=False,
        #     acc=True,
        #     detach=detach,
        #     clone=clone,
        #     zero_order=False,
        #     reduce=False # key to computing loss value for every example in the batch
        # )
        # if verbose:
        #     print(f"Number of points with eigval  < 0 (passing) before attack {p1_uptbd.sum()} vs after: {p1.sum()} out of {X.shape[0]}")
        # num_new_CE = np.logical_and(p1_uptbd == True, p1 == False).sum()
        # num_lost_CE = np.logical_and(p1_uptbd == False, p1 == True).sum()
        # Meig_stats, Qeig_stats = CEstats
        # if verbose:
        #     print("max eigvalue after attack: ", Qeig_stats.max)
        # stats= Stats(num_new_CE, num_lost_CE)
        stats = Stats(0., 0.)

            # print("after attack: xerr_ptb.max(): ", x_err_ptb.max(), ", xerr_ptb.min(): ", x_err_ptb.min())
        return max_delta, stats, total_Q_time

    def package_CEs(CE_list):
        x, xref, uref, val = zip(*CE_list) 
        CEs = CounterExamples(torch.cat(x, dim=0),
                            torch.cat(xref, dim=0),
                            torch.cat(uref, dim=0),
                            torch.cat(val, dim=0))
        return CEs

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
        verbose=False,
        CEs=False,
    ):  
        """
        This function implements 1 epoch of training with or without PGD attack.
        """
        # trainval a set of x
        # torch.autograd.set_detect_anomaly(True)

        # if train:
        #     indices = np.random.permutation(len(X))
        # else:
        #     indices = np.array(list(range(len(X))))

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
        total_Q_time = 0.0
        Geigmin = 1e6
        Geigmax = 0.0
        Geigmean = 0.0
        totalQCEs = []
        totalQbCEs = []
        total_p1b = 0

        num_train_batches = len(X.dataset) // bs
        
        for b, data in enumerate(tqdm(X)):
            x, xref, uref = data[0] # data[1] is an empty list
            start = time.time()
            if robust:
                if train:
                    attack_iters = args.robust_attack_iters_train
                else:
                    attack_iters = args.robust_attack_iters_test
                robust_eps_i = ptb_sched(epoch + (b + 1)/num_train_batches)
                if robust_eps_i > 0:
                    delta, stats, Q_time = attack_pgd(torch.concatenate([x, xref, uref], dim=1),
                                    robust_eps_i,
                                    args.robust_alpha, 
                                    attack_iters,
                                    args.robust_restarts,
                                    args.robust_norm,
                                        train,
                                        acc,
                                        detach,
                                        clone,
                                        useopt=args.useopt,
                                        verbose=verbose)
                    total_num_new_CEs += stats.num_new_CEs 
                    total_num_lost_CEs += stats.num_lost_CEs   
                    total_Q_time += Q_time                
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
                CEs=CEs,
                useopt=args.useopt      
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
                eigstats, Meigstats, Geigstats, QCEs, QbCEs, Qb_p1, times = stats
                eigmin  = np.minimum(eigstats.min, eigmin)
                eigmax = np.maximum(eigstats.max, eigmax)
                eigmean = (b*eigmean + eigstats.mean)/(b+1)
                Meigmin = np.minimum(Meigstats.min, Meigmin) 
                Meigmax = np.maximum(Meigstats.max, Meigmax)
                Meigcondmax = np.maximum(Meigstats.cond_max, Meigcondmax)
                Geigmin  = np.minimum(Geigstats.min, Geigmin)
                Geigmax = np.maximum(Geigstats.max, Geigmax)
                Geigmean = (b*Geigmean + Geigstats.mean)/(b+1)
                total_p1b += Qb_p1.sum()
                totalQCEs.append(QCEs)
                totalQbCEs.append(QbCEs)
            else:
                times = stats
            total_Q_time += times
        if robust:
            print("robust_eps_i at epoch end was: ", robust_eps_i)
        eigstats = EigStats(eigmax, eigmin, eigmean)
        Geigstats = EigStats(Geigmax, Geigmin, Geigmean)
        meigstats = MeigStats(Meigmax, Meigmin, Meigcondmax)
        CE_stats = Stats(total_num_new_CEs, total_num_lost_CEs)  # CEs found within PGD attack
        # pdb.set_trace()
        if acc:
            allQCEs = package_CEs(totalQCEs) # CEs found in forward during testing pass regardless of PGD attack or not
            allQbCEs = package_CEs(totalQbCEs) # gershbound CEs
        else:
            allQCEs = []
            allQbCEs = []

        return (total_loss / len(X.dataset),
                total_p1 / len(X.dataset), 
                total_p2 / len(X.dataset), 
                total_l3 / len(X.dataset), 
                (eigstats, meigstats, Geigstats, CE_stats, losses, allQCEs, allQbCEs, total_p1b / len(X.dataset), total_Q_time / len(X.dataset)))


    best_acc = 0


    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by every args.lr_step epochs"""
        lr = args.learning_rate * (0.1 ** (epoch // args.lr_step))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


    if args.load_model:
        # load the metric and controller # should already be loaded
        # filename_controller = args.log + "/controller_best.pth.tar"
        # filename_metric = args.log + "/metric_best.pth.tar"
        # map_location = "cuda" if args.use_cuda else "cpu"
        # W_func = torch.load(filename_metric, map_location)
        # u_func = torch.load(filename_controller, map_location)

        ptb_schedule = lambda t: args.robust_eps
        _, p1, p2, l3, test_stats = trainval(X_te_DL, 
                                                train=False, 
                                                _lambda=args.test_lambda, 
                                                acc=True, 
                                                detach=False, 
                                                ptb_sched = ptb_schedule, 
                                                robust=args.robust_train,
                                                CEs=True)

        eigstats, meigstats, Geigstats, _, losses, allQCEs, allQbCEs, p1b, Q_time = test_stats
        # Plot the maximum eigenvalue of the contraction condition
        Nte = len(X_te_DL.dataset)
        print("-------------")
        print(f"Total # true Q violations: {Nte-p1*Nte} ({(1 - p1)*100} %)")
        print(f"Total # Q bound violations: {Nte-p1b*Nte} ({(1 - p1b)*100} %)")
        if p1 < 1.0:
            print(f"mean trueig violation: {allQCEs.val.mean()} ({allQCEs.val.max()} max)")
        if p1b < 1.0:
            print(f"mean eigbound violation: {allQbCEs.val.mean()} ({allQbCEs.val.max()} max)")
        print("-------------")
        # fig, axs = plt.subplots(4, 4)
        # for i in range(4):
        #     for j in range(4):
        #         axs[i, j].scatter(
        #             x[violation, j].detach().cpu(),
        #             x[violation, i].detach().cpu(),
        #             c=max_eig_Q[violation].cpu(),
        #             s=0.1,
        #         )
        #         axs[i, j].set_xlabel(f"x_{j}")
        #         axs[i, j].set_ylabel(f"x_{i}")
        #         axs[i, j].set_xlim(
        #             [
        #                 x[:, j].detach().min().cpu() - 0.1,
        #                 x[:, j].detach().max().cpu() + 0.1,
        #             ]
        #         )
        #         axs[i, j].set_ylim(
        #             [
        #                 x[:, i].detach().min().cpu() - 0.1,
        #                 x[:, i].detach().max().cpu() + 0.1,
        #             ]
        #         )

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

        # plt.show()

        sys.exit()

    layout0 = {
        "Eigenvalues":{
            "Eigenvalues": ["Multiline", ["eig/max", "eig/min", "eig/mean"]],
        },
    }

    layout1 = {
        "GershApproxEigenvalues":{
            "GershApproxEigenvalues": ["Multiline", ["Geig/max", "Geig/min", "Geig/mean"]],
        },
    }
    # layout1 = {
    #     "TrainEigenvalues":{
    #         "TrainEigenvalues": ["Multiline", ["traineig/max", "traineig/min", "traineig/mea"]],
    #     },
    # }
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

    writer.add_custom_scalars(layout0)
    writer.add_custom_scalars(layout1)
    writer.add_custom_scalars(layout2)
    writer.add_custom_scalars(layout3)

    # perturbation schedule for robust training
    ptb_schedule = lambda t: args.robust_eps
    # np.interp([t], [0, 1.5, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [0.0, 0.0, args.robust_eps/2., args.robust_eps/2., args.robust_eps])[0]
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
        loss, _, _, _, train_stats = trainval(X_tr_DL,
                                    train=True,
                                    _lambda=args._lambda,
                                    acc=False,
                                    detach=True if epoch < args.lr_step else False,
                                    epoch=epoch,
                                    ptb_sched = ptb_schedule,
                                    robust=args.robust_train
                                )
        _, _, _, CE_stats, _, QCEs_tr, QbCEs_tr, p1b_tr, Q_time_tr = train_stats
        if args.robust_train:
            writer.add_scalar("number new CE from training", CE_stats.num_new_CEs, epoch)
            writer.add_scalar("number lost CE from training", CE_stats.num_lost_CEs, epoch)
        print("Training loss: ", loss)
        writer.add_scalar("Avg Q loss computation time (train): ", Q_time_tr, epoch)
        # writer.add_scalar("traineig/max", train_eig_stats.max, epoch)
        # writer.add_scalar("traineig/min", train_eig_stats.min, epoch)
        # writer.add_scalar("traineig/mean", train_eig_stats.mean, epoch) 

        ### now test ###################################################################
        # pdb.set_trace()
        loss, p1, p2, l3, test_stats = trainval(X_te_DL, 
                                                train=False, 
                                                _lambda=args.test_lambda, 
                                                acc=True, 
                                                detach=False, 
                                                epoch=epoch,
                                                ptb_sched = ptb_schedule, 
                                                robust=args.robust_train)
        eig_stats, meig_stats, geig_stats, CE_stats, losses, QCEs, QbCEs, p1b, Q_time = test_stats
        print("Epoch %d: Testing loss/p1/p1b/p2/l3: " % epoch, loss, p1, p1b, p2, l3)
        writer.add_scalar("Avg Q loss computation time (test): ", Q_time, epoch)
        if len(losses) == 4:
            loss_Q, loss_C1, loss_W, loss_C2 = losses
        elif len(losses) == 5:
            loss_Q, loss_C1, loss_W, loss_C2, loss_reg = losses
            writer.add_scalar("Regularization loss", loss_reg, epoch)

        writer.add_scalar("Loss", loss, epoch)
        writer.add_scalar("% of pts with max eig Q < 0", p1, epoch)
        writer.add_scalar("% of pts with max eigbound Q < 0", p1b, epoch)
        writer.add_scalar("% of pts with max eig C1_LHS < 0", p2, epoch)
        writer.add_scalar("eig/max", eig_stats.max, epoch)
        writer.add_scalar("eig/min", eig_stats.min, epoch)
        writer.add_scalar("eig/mean", eig_stats.mean, epoch)
        writer.add_scalar("Geig/max", geig_stats.max, epoch)
        writer.add_scalar("Geig/min", geig_stats.min, epoch)
        writer.add_scalar("Geig/mean", geig_stats.mean, epoch) 
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
                    "eigstats": eig_stats,
                    "M_eigstats": meig_stats,
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