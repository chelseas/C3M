import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=1)
parser.add_argument('-e', '--eps', type=float, default=1e-3)
args = parser.parse_args()

num_dim_x = 4
num_dim_control = 2
num_input = num_dim_x * 2 + num_dim_control
num_output = 4

# x: x, xref, uref
x_L = torch.tensor([-5, -5, -torch.pi, 1,
                    -5, -5, -torch.pi, 1,
                    -1, 0])
x_U = torch.tensor([5, 5, torch.pi, 2,
                    5, 5, torch.pi, 2,
                    1, 0])


for t in range(args.n):
    with open(f'specs/rand_{t}.vnnlib', 'w') as file:
        for i in range(num_input):
            file.write(f'(declare-const X_{i} Real)\n')
        for i in range(num_output + 1):  # An additional output for the xerr constraint
            file.write(f'(declare-const Y_{i} Real)\n')

        # Random
        delta = torch.rand_like(x_L)
        x_center = x_L + (x_U - x_L) * delta
        x_center[4:8] = x_center[0:4] + (torch.rand(4) - 0.5) * 2 * 0.05
        x_center = torch.max(torch.min(x_U, x_center), x_L)
        x_L_ = torch.max(x_L, x_center - args.eps)
        x_U_ = torch.min(x_U, x_center + args.eps)

        for i in range(num_input):
            file.write(f'(assert (>= X_{i} {x_L_[i]:.12f}))\n')
            file.write(f'(assert (<= X_{i} {x_U_[i]:.12f}))\n')

        file.write('(assert (or\n')
        for i in range(num_output):
            # a slight tolerance for the verification
            file.write(f'  (and (>= Y_{i} 1e-4))\n')
        file.write('))\n')
        # a slightly larger region needs to be certified
        file.write(f'(assert (<= Y_{num_output} 1e-4))\n')

with open('specs/rand_instances.csv', 'w') as file:
    for t in range(args.n):
        file.write(f'specs/rand_{t}.vnnlib\n')
