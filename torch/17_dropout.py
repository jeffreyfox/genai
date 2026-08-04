import torch
import torch.nn as nn


class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
          return x
        mask = torch.rand_like(x) > self.p
        # return torch.where(mask, x / (1 - self.p), 0.0)
        return mask * x / (1-self.p)

# 🧪 Debug
d = MyDropout(p=0.5)
d.train()
x = torch.ones(10)
print('Train:', d(x))
d.eval()
print('Eval: ', d(x))
