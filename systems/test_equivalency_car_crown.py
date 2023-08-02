import colored_traceback
colored_traceback.add_hook(always=True)

import torch
import systems.system_CAR as orig
import systems.system_CARcrown as crwn
from using_crown_utils import Jacobian
from systems.system_CAR import num_dim_x, num_dim_control

bs=5 # arbitrary
# test the equivalency of f
x= torch.rand((bs,4,1)).requires_grad_()
assert(torch.all(orig.f_func(x) == crwn.f_func(x)))

# test the equivalency DfDx
assert(torch.all(Jacobian(orig.f_func(x), x) == crwn.DfDx_func(x)))

# test equivalency of dbdx
DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())
for i in range(num_dim_control):
    DBDx[:, :, :, i] = Jacobian(orig.B_func(x)[:, :, i].unsqueeze(-1), x)
#
DBDx_new = crwn.DBDx_func(x)
assert(torch.all(DBDx == DBDx_new))

# test equivalency of DBDx * u
u = torch.rand(bs, num_dim_control,1)
DBDx = torch.rand(DBDx.shape)
DBDx_t_u = sum(
    [
        u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
        for i in range(num_dim_control)
    ])
DBDx_t_u_new = (u.reshape(bs, 1, 1, num_dim_control)*DBDx).sum(dim=-1)
assert(torch.all(DBDx_t_u == DBDx_t_u_new))