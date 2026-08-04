import torch


def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight


# 🧪 Debug
x = torch.randn(2, 8)
w = torch.ones(8)
out = rms_norm(x, w)
print("Output shape:", out.shape)
print("RMS of output:", out.pow(2).mean(dim=-1).sqrt())  # should be ~1
