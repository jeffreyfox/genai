import torch
import math

def causal_attention(Q, K, V):
    _, seq, d_k = Q.shape
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k) # (B, seq_len, seq_len)
    mask = torch.tril(torch.ones(seq, seq, device=scores.device, dtype=scores.dtype))
    # scores = torch.where(mask > 0, scores, float('-inf'))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

def causal_attention_2(Q, K, V):
    _, seq, d_k = Q.shape
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k) # (B, seq_len, seq_len)
    mask = torch.triu(torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

# 🧪 Debug
torch.manual_seed(0)
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = causal_attention(Q, K, V)
print("Output shape:", out.shape)          # (1, 4, 8)
print("Pos 0 == V[0]?", torch.allclose(out[:, 0], V[:, 0], atol=1e-5))  # should be True
