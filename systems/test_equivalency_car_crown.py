import torch
import system_CAR as orig
import system_CARcrown as crwn
x= torch.rand((5,4,1))
assert(torch.all(orig.f_func(x) == crwn.f_func(x)))
