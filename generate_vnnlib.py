import torch

num_dim_x = 4
num_dim_control = 2
num_input = num_dim_x * 2 + num_dim_control
num_output = 4
for i in range(num_input):
    print(f'(declare-const X_{i} Real)')
print()
for i in range(num_output + 1):  # An additional output for the xerr constraint
    print(f'(declare-const Y_{i} Real)')
print()

# x: x, xref, uref
x_L = torch.tensor([-5, -5, -torch.pi, 1,
                    -5, -5, -torch.pi, 1,
                    -1, 0])
x_U = torch.tensor([5, 5, torch.pi, 2,
                    5, 5, torch.pi, 2,
                    1, 0])
eps = 0.01 # certify a small region around the center first

x_center = (x_L + x_U) / 2
x_diff = (x_U - x_L) / 2
x_L = x_center - x_diff * eps
x_U = x_center + x_diff * eps

for i in range(num_input):
    print(f'(assert (>= X_{i} {x_L[i]:.12f}))')
    print(f'(assert (<= X_{i} {x_U[i]:.12f}))')
print()

print('(assert (or')
for i in range(num_output):
    # a slight tolerance for the verification
    print(f'  (and (>= Y_{i} 1e-4))')
print('))')
# a slightly larger region needs to be certified
print(f'(assert (<= Y_{num_output} 1e-4))')
