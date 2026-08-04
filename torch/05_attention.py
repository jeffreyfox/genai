import torch
import math


def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

torch.manual_seed(42)
Q = torch.randn(2, 4, 8)
K = torch.randn(2, 4, 8)
V = torch.randn(2, 4, 8)

out = scaled_dot_product_attention(Q, K, V)
print("Output shape:", out.shape)          # should be (2, 4, 8)
print("Has NaN?    ", torch.isnan(out).any().item())  # should be False
print("Has Inf?    ", torch.isinf(out).any().item())  # should be False

# Cross-attention: seq_q != seq_k
Q2 = torch.randn(1, 3, 16)
K2 = torch.randn(1, 5, 16)
V2 = torch.randn(1, 5, 32)
out2 = scaled_dot_product_attention(Q2, K2, V2)
print("Cross-attn shape:", out2.shape)     # should be (1, 3, 32)
