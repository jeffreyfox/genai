import torch
import torch.nn as nn
import math


class GroupQueryAttention:
    def __init__(self, d_model, num_heads, num_kv_heads):
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads


    def forward(self, x):
        B, S, _ = x.shape  # (B, S, d_model)

        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2) # (B, n_h, S, d_k)
        k = self.W_k(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)  # (B, n_kv_h, S, d_k)
        v = self.W_v(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)  # (B, n_kv_h, S, d_k)

        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats=repeats, dim=1) # (B, n_h, S, d_k)
        v = v.repeat_interleave(repeats=repeats, dim=1) # (B, n_h, S, d_k)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k) # (B, n_h, S, S)
        weights = torch.softmax(scores, dim=-1) # (B, n_h, S, S)
        attn = weights @ v # (B, n_h, S, d_k)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1) # (B, S, d_model)
  
        return self.W_o(attn) # (B, S, d_model)

# 🧪 Debug
torch.manual_seed(0)
gqa = GroupQueryAttention(d_model=32, num_heads=8, num_kv_heads=2)
print("W_q shape:", gqa.W_q.weight.shape)  # (32, 32)
print("W_k shape:", gqa.W_k.weight.shape)  # (8, 32)  — only 2 KV heads * d_k=4

x = torch.randn(2, 6, 32)
out = gqa.forward(x)
print("Output shape:", out.shape)           # (2, 6, 32)
