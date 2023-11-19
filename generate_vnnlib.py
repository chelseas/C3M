import torch

num_dim_x = 4
num_dim_control = 2
num_input = num_dim_x * 2 + num_dim_control
num_output = 4
for i in range(num_input):
    print(f'(declare-const X_{i} Real)')
print()
for i in range(num_output):
    print(f'(declare-const Y_{i} Real)')
print()

x = torch.rand(num_dim_x)
xref = torch.rand(num_dim_x)
uref = torch.rand(num_dim_control)
xall = torch.concat((x, xref, uref), dim=-1)
eps = 0.3

for i in range(num_input):
    print(f'(assert (<= X_{i} {xall[i]+eps:.12f}))')
    print(f'(assert (>= X_{i} {xall[i]-eps:.12f}))')
print()

print('(assert (or')
for i in range(num_output):
    print(f'  (and (>= Y_{i} 0))')
print('))')
