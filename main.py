import torch
from torch.autograd import grad
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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
        "--num_train", type=int, default=4 * 32768 + 0 * 131072, help="Number of samples for training."
    )  # 4096 * 32
    parser.add_argument(
        "--num_test", type=int, default=32768, help="Number of samples for testing."
    )  # 1024 * 32
    parser.add_argument(
        "--lr", dest="learning_rate", type=float, default=0.001, help="Base learning rate."
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--lr_step", type=int, default=5, help="")
    parser.add_argument(
        "--lambda", type=float, dest="_lambda", default=0.5, help="Convergence rate: lambda"
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
        type=float,
        default=0.3,
        help="Perturbation bound for adversarial training.",
    )
    parser.add_argument(
        "--robust_alpha",
        type=float,
        default=2.0,
        help="Learning rate for adversarial training attack.",
    )
    parser.add_argument(
        "--robust_norm",
        type=str,
        default="l_inf",
        help="Norm for adversarial training.",
    )
    parser.add_argument(
        "--robust_attack_iters",
        type=int,
        default=10,
        help="Number of iterations to create adversarial example during robust training.",
    )
    parser.add_argument(
        '--robust_restarts',   default=1, type=int, help=" number of times that the adversarial attack can restart during adversarial training. A hyper parameter.")

    args = parser.parse_args(args)
    return args

def main(args=None):
    args = cmdlineparse(args)
    if not args.load_model:
        os.system("cp *.py " + args.log)
        os.system("cp -r models/ " + args.log)
        os.system("cp -r configs/ " + args.log)
        os.system("cp -r systems/ " + args.log)
        writer = SummaryWriter(args.log + "/tb")
    global_steps = 0

    # epsilon = args._lambda
    epsilon = 0.0
    print("Epsilon: ", epsilon)

    config = importlib.import_module("config_" + args.task)
    X_MIN = config.X_MIN
    X_MAX = config.X_MAX
    U_MIN = config.UREF_MIN
    U_MAX = config.UREF_MAX
    XE_MIN = config.XE_MIN
    XE_MAX = config.XE_MAX
    low = np.concatenate([X_MIN, X_MIN, U_MIN]).reshape(-1,1)
    high = np.concatenate([X_MAX, X_MAX, U_MAX]).reshape(-1,1)
    ranges = torch.tensor(np.hstack([low, high])).reshape(-1,2)
    print("ranges: ", ranges)

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
        zTAz = (z.matmul(A) * z.view(1, K, -1)).sum(dim=2) # bs x K
        if reduce:
            zTAz = zTAz.view(-1)
            negative_index = zTAz.detach().cpu().numpy() < 0
            if negative_index.sum() > 0:
                negative_zTAz = zTAz[negative_index]
                return -1.0 * (negative_zTAz.mean())
            else:
                return torch.tensor(0.0).type(z.dtype).requires_grad_()
        else: # no reduce
            # compute avg violation of PD condition for each sample in batch
            positive_index = zTAz.detach().cpu().numpy() > 0
            negative_index = zTAz.cpu().numpy() < 0
            # no loss contrib from pos so zero them
            zTAz[positive_index] = 0.
            # next average neg values
            num_neg_each_row = negative_index.sum(-1) # if this is zero for any data point, don't want to divide by zero
            num_neg_each_row[num_neg_each_row == 0.0] = 1.0 # 0/1 = 0
            neg_mean = zTAz.sum(dim=-1) / num_neg_each_row 
            print("loss dim: ", neg_mean.shape)
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
        reduce=True
    ):
        # print("W_func.device: ", W_func.model_W[0].weight.device)
        # print("x.device: ", x.device)
        # Otherwise, just train the tanh networks
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

        if reduce:
            # compute a scalar loss
            loss = 0
        else:
            # compute a loss value for every data point in batch
            loss = torch.zeros(x.shape[0])
        loss += loss_pos_matrix_random_sampling(
            -Contraction
            - epsilon * torch.eye(Contraction.shape[-1]).unsqueeze(0).type(x.dtype),
            reduce=reduce
        )
        loss += loss_pos_matrix_random_sampling(
            -C1_LHS_1 - epsilon * torch.eye(C1_LHS_1.shape[-1]).unsqueeze(0).type(x.dtype),
            reduce=reduce
        )
        loss += loss_pos_matrix_random_sampling(
            args.w_ub * torch.eye(W.shape[-1]).unsqueeze(0).type(x.dtype) - W,
            reduce=reduce
        )
        # loss += loss_pos_matrix_random_sampling(W - args.w_lb * torch.eye(W.shape[-1]).unsqueeze(0).type(x.dtype))  # Make sure W is positive definite
        if reduce:
            loss += 1.0 * sum([1.0 * (C2**2).reshape(bs, -1).sum(dim=1).mean() for C2 in C2s])
        else:
            assert(len(sum([1.0 * (C2**2).reshape(bs, -1).sum(dim=1) for C2 in C2s])) == bs)
            loss += 1.0 * sum([1.0 * (C2**2).reshape(bs, -1).sum(dim=1) for C2 in C2s])

        if clone:
            overall_clone_loss = u_error.mean() + M_error.mean() + W_error.mean()
            loss = overall_clone_loss

        if verbose:
            print(
                torch.linalg.eigh(Contraction)[0].min(dim=1)[0].mean(),
                torch.linalg.eigh(Contraction)[0].max(dim=1)[0].mean(),
                torch.linalg.eigh(Contraction)[0].mean(),
            )
        if acc:
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
            )
        else:
            return loss, None, None, None


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

    def attack_pgd(X,
                   robust_eps,
                   ranges,
                   alpha, 
                   attack_iters,
                   restarts,
                   norm,
                   train,
                   acc,
                   detach,
                   clone):
        """
        This function calculates one batch of adversarial inputs using a PGD attack.
        """
        #  keep track of most adversarial input found so far for each training example
        max_loss = torch.zeros(X.shape[0])
        max_delta = torch.zeros_like(X)
        if args.use_cuda:
            max_loss.cuda()
            max_delta.cuda()
        # ranges is a matrix that looks like [[-1,1], [1,2],...] Containing the available ranges for each state variable
        low = ranges[:,0].reshape(1, -1)
        high = ranges[:,1].reshape(1, -1)
        Xrange = high - low
        for _ in range(restarts):
            # Initialize deltas to uniform random in 2*eps*range of each variable centered at center of range
            delta = torch.zeros_like(X)
            delta.uniform_(-robust_eps, robust_eps)
            delta = delta*(Xrange) + (low+high)/2
            assert(delta > low and delta < high) #  sanity check instead of clamp
            delta.requires_grad = True
            # optimize the deltas
            for _ in range(attack_iters):
                # compute the perturbed inputs
                ptbd_X = torch.clamp(X + delta, min=low, max=high)
                x_ptb = ptbd_X[:,0:num_dim_x]
                xref_ptb = ptbd_X[:, num_dim_x:(2*num_dim_x)]
                uref_ptb = ptbd_X[:, (2*num_dim_x):]
                # compute the loss
                loss, p1, p2, l3 = forward( #  this computes a scaler loss
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
                # compute gradients
                loss.backward()
                grad = delta.grad.detach()
                d = delta
                g = grad
                if norm == "l_inf":
                    # Do gradient ascent on the disturbance delta (d)
                    d = d + alpha * torch.sign(g)
                    # clamp to limited disturbance range
                    d = torch.clamp(d, min=-robust_eps*Xrange, max=robust_eps*Xrange)
                elif norm == "l_2":
                    raise NotImplementedError
                    # g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                    # scaled_g = g/(g_norm + 1e-10)
                    # d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
                d = torch.clamp(d, low, high) # probably unecessary?
                delta = d 
                # reset gradient before next iter
                delta.grad.zero_()
            #  here we compute a loss value for each example in the batch
            # compute the perturbed inputs
            ptbd_X = torch.clamp(X + delta, min=low, max=high)
            x_ptb = ptbd_X[:,0:num_dim_x]
            xref_ptb = ptbd_X[:, num_dim_x:(2*num_dim_x)]
            uref_ptb = ptbd_X[:, (2*num_dim_x):]
            # compute the loss
            batch_loss, p1, p2, l3 = forward( #  this computes a scaler loss
                x_ptb,
                xref_ptb,
                uref_ptb,
                _lambda=_lambda,
                verbose=False if not train else False,
                acc=acc,
                detach=detach,
                clone=clone,
                zero_order=False,
                reduce=False # key to computing loss value for every example in the batch
            )
            # store largest delta values
            max_delta[batch_loss >= max_loss] = delta.detach()[batch_loss >= max_loss]
            max_loss = torch.max(max_loss, batch_loss)
        return max_delta

    def robust_trainval(
        X,
        bs=args.bs,
        train=True,
        _lambda=args._lambda,
        acc=False,
        detach=False,
        clone=False,
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

        if train:
            # print("len(X):", len(X), ", bs: ", bs)
            _iter = tqdm(range(len(X) // bs))
        else:
            _iter = range(len(X) // bs)
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

            delta = attack_pgd(torch.stack([x, xref, uref], dim=1),
                               args.robust_eps,
                               ranges,
                               args.robust_alpha, 
                               args.robust_attack_iters,
                               args.robust_restarts,
                               args.robust_norm,
                                train,
                                acc,
                                detach,
                                clone)
            
            print("delta: ",delta.shape)
            assert(1==0)
            
            x_ptb = x + delta[:,0:num_dim_x]
            xref_ptb = xref + delta[:, num_dim_x:(2*num_dim_x)]
            uref_ptb = uref + delta[:, (2*num_dim_x):]

            loss, p1, p2, l3 = forward(
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
        return total_loss / len(X), total_p1 / len(X), total_p2 / len(X), total_l3 / len(X)


    def trainval(
        X,
        bs=args.bs,
        train=True,
        _lambda=args._lambda,
        acc=False,
        detach=False,
        clone=False,
    ):  # trainval a set of x
        # torch.autograd.set_detect_anomaly(True)

        if train:
            indices = np.random.permutation(len(X))
        else:
            indices = np.array(list(range(len(X))))

        total_loss = 0
        total_p1 = 0
        total_p2 = 0
        total_l3 = 0

        if train:
            # print("len(X):", len(X), ", bs: ", bs)
            _iter = tqdm(range(len(X) // bs))
        else:
            _iter = range(len(X) // bs)
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

            loss, p1, p2, l3 = forward(
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
        return total_loss / len(X), total_p1 / len(X), total_p2 / len(X), total_l3 / len(X)


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


    # Start training a tanh network
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        if args.robust_train:
            print("Using adversarial training.")
            loss, _, _, _ = robust_trainval(X_tr,
                            train=True,
                            _lambda=args._lambda,
                            acc=False,
                            detach=True if epoch < args.lr_step else False
                        )
        else:
            print("Using vanilla training.")
            loss, _, _, _ = trainval(X_tr,
                                    train=True,
                                    _lambda=args._lambda,
                                    acc=False,
                                    detach=True if epoch < args.lr_step else False,
                                )
        print("Training loss: ", loss)
        loss, p1, p2, l3 = trainval(X_te, train=False, _lambda=0.0, acc=True, detach=False)
        print("Epoch %d: Testing loss/p1/p2/l3: " % epoch, loss, p1, p2, l3)

        writer.add_scalar("Loss", loss, epoch)
        writer.add_scalar("% of pts with max eig Q < 0", p1, epoch)
        writer.add_scalar("% of pts with max eig C1_LHS < 0", p2, epoch)
        writer.add_scalar("mean C2^2", l3, epoch)

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