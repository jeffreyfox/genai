import torch
import math


def kaiming_init(weight):
  fan_in = weight.shape[1]
  std = math.sqrt(2.0 / fan_in)
  with torch.no_grad():
    weight.normal_(0.0, std)
  return weight

# 🧪 Debug
import math
w = torch.empty(256, 512)
kaiming_init(w)
print(f'Mean: {w.mean():.4f} (expect ~0)')
print(f'Std:  {w.std():.4f} (expect {math.sqrt(2/512):.4f})')
