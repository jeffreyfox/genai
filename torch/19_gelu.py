import torch
import math


def my_gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

# 🧪 Debug
x = torch.tensor([-2., -1., 0., 1., 2.])
print('Output:', my_gelu(x))
print('Ref:   ', torch.nn.functional.gelu(x))
