import torch
import torch.nn.functional as F


def linear_attention(Q, K, V):
    Q_prime = F.elu(Q) + 1 # (B, S, d_k)
    K_prime = F.elu(K) + 1 # (B, S, d_k)
    KV = K_prime.transpose(-2, -1) @ V # (B, d_k, d_k)
    # Sum over the sequence dimension
    K_sum = K_prime.sum(dim=1).unsqueeze(-1) # (B, d_k, 1)
    numerator = Q_prime @ KV # (B, S, d_k)
    denominator = Q_prime @ K_sum # (B, S, 1)
    return numerator / (denominator + 1e-6)

# 🧪 Debug
Q = torch.randn(1, 8, 16)
K = torch.randn(1, 8, 16)
V = torch.randn(1, 8, 32)
out = linear_attention(Q, K, V)
print("Output shape:", out.shape)   # (1, 8, 32)
print("Has NaN?", torch.isnan(out).any().item())
