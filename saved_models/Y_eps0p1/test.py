import torch


gradient_bundle_size = 16
gradient_bundle_variance = 0.1


def f(x):
    return (x ** 2).sum(dim=-1).unsqueeze(-1)


def f_matrix(x):
    M = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).type(x.type())
    M[:, 0, :] = 2 * x
    M[:, 1, :] = -3 * x
    return M


x = torch.randn(5, 2)


def zero_order_jacobian_estimate(f, x):
    """
    Compute the zero-order estimate of the gradient of f w.r.t. x.

    args:
        f: callable
        x: bs x n tensor
    """
    bs = x.shape[0]
    n = x.shape[1]

    # Get the function value at x
    f_x = f(x)

    # Expand the size of x to match the size of the bundle
    x = torch.repeat_interleave(x.unsqueeze(-1), gradient_bundle_size, dim=-1)

    # Make somewhere to store the Jacobian
    J = torch.zeros(*f_x.shape, n).type(x.type())

    # Estimate the gradient in each direction of x
    for i in range(n):
        # Get the perturbations in this dimension of x
        dx_i = gradient_bundle_variance * torch.randn(gradient_bundle_size)
        x_plus_dx_i = x.clone()
        x_plus_dx_i[:, i, :] += dx_i

        # Get the function value at x + dx (iterate through each sample)
        for j in range(gradient_bundle_size):
            f_x_plus_dx_i = f(x_plus_dx_i[:, :, j])

            # Accumulate it into a Jacobian estimator
            J[:, :, i] += (f_x_plus_dx_i - f_x) / (
                dx_i[j] * gradient_bundle_size
            )

    return J


# print((zero_order_jacobian_estimate(f, x) - 2 * x).mean())
print(zero_order_jacobian_estimate(f_matrix, x))
print(zero_order_jacobian_estimate(f_matrix, x).shape)
