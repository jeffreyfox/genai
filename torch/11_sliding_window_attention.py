import torch
import math


def sliding_window_attention(Q, K, V, window_size):
    S, d_k = Q.shape[1], K.shape[-1]
    idx = torch.arange(S, device=Q.device)
    mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() > window_size
    # mask = (idx[:, None] - idx[None, :]).abs() > window_size
    
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
    scores = scores.masked_fill(mask, float('-inf'))
    weights = scores.softmax(dim=-1)

    return weights @ V


# 🧪 Debug
Q = torch.randn(1, 6, 8)
K = torch.randn(1, 6, 8)
V = torch.randn(1, 6, 8)

out = sliding_window_attention(Q, K, V, window_size=1)
print("Output shape:", out.shape)  # (1, 6, 8)

# window=0 should return V
out0 = sliding_window_attention(Q, K, V, window_size=0)
print("window=0 == V?", torch.allclose(out0, V, atol=1e-5))
