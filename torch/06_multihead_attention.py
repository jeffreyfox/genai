import torch
import torch.nn as nn
import math


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # embedding dimension of each head

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        # Q, K, V: (B, seq_q, d_model), (B, seq_k, d_model), (B, seq_k, d_model)
        B, seq_q, seq_k = Q.shape[0], Q.shape[1], K.shape[1]
        q = self.W_q(Q).view(B, seq_q, self.num_heads, self.d_k).transpose(1, 2) # (B, nH, seq_q, d_k)
        k = self.W_k(K).view(B, seq_k, self.num_heads, self.d_k).transpose(1, 2) # (B, nH, seq_k, d_k)
        v = self.W_v(V).view(B, seq_k, self.num_heads, self.d_k).transpose(1, 2) # (B, nH, seq_k, d_k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k) # (B, nH, seq_q, seq_k)
        weights = torch.softmax(scores, dim=-1) # (B, nH, seq_q, seq_k)
        attn = weights @ v  # (B, nH, seq_q, d_k)
        out = attn.transpose(1, 2).contiguous().view(B, seq_q, -1) # (B, seq_q, d_model)
        return self.W_o(out) # (B, seq_q, d_model)


# 🧪 Debug
torch.manual_seed(0)
mha = MultiHeadAttention(d_model=32, num_heads=4)
print("W_q type:", type(mha.W_q))          # should be nn.Linear
print("W_q.weight shape:", mha.W_q.weight.shape)  # (32, 32)

x = torch.randn(2, 6, 32)
out = mha.forward(x, x, x)
print("Output shape:", out.shape)          # (2, 6, 32)

# Cross-attention
Q = torch.randn(1, 3, 32)
K = torch.randn(1, 7, 32)
V = torch.randn(1, 7, 32)
out2 = mha.forward(Q, K, V)
print("Cross-attn shape:", out2.shape)     # (1, 3, 32)
