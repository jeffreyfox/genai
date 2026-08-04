import math

import torch
from torch import nn

from config import ModelConfig

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # weights, no bias
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, 0.0, std, -3*std, 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, 0.0, 1.0, -3.0, 3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.weight) # initializes as all ones

    # Process an input tensor of shape (batch_size, sequence_length, d_model)
    # and return a tensor of the same shape.
    # Upcasts input to torch.float32 before squaring the input to avoid overflow
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = ((x ** 2).sum(dim=-1, keepdims=True) / x.shape[-1] + self.eps) ** 0.5
        result = x / rms * self.weight
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = round(d_model * 8 / 3 / 64) * 64   # round to nearest multiple of 64
        else:
            self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    # x: [..., d_model]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        silu = w1x * torch.sigmoid(w1x)
        w3x = self.w3(x) # [..., d_ff]
        return self.w2(silu * w3x)  # [..., d_model]


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        # indices = [0, 0, 2, 2, 4, 4, ...]
        indices = torch.arange(d_k, device=device) // 2 * 2 # (d_k, )
        angles = torch.outer(torch.arange(max_seq_len, device=device), theta ** (-indices / d_k)) # (max_seq_len, d_k)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # Break x into [x1, x2] and rotate to to [-x2, x1]
    def rotate_half(self, x: torch.Tensor):
        d_k = x.shape[-1]
        x1, x2 = x[..., :d_k:2], x[..., 1:d_k:2]
        # interleave -x2 with x1. Note we use torch.stack which creates a new dimension.
        # torch.cat doesn't create a new dimension.
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

    # Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape
    # The token positions are a tensor of shape (..., seq_len) specifying the token positions of
    # x along the sequence dimension (i.e. [0, 1, 2, ...]). But it doesn't have to start from zero.
    # For example if KV cache is used, it can be [10, 11, 12, ...].
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if token_positions is None:
            raise ValueError("In RotaryPositionalEmbedding.forward, token_positions is None")
        cos = self.cos[token_positions] # use indexing
        sin = self.sin[token_positions]
        return cos * x + sin * self.rotate_half(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.max(x, dim, keepdims=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim, keepdims=True)


# Q: (..., seq_len, d_k)
# K: (..., seq_len, d_k)
# V: (..., seq_len, d_v)
# mask: (seq_len, seq_len)
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    d_k = Q.shape[-1]
    logits = Q @ K.transpose(-1, -2) * (d_k ** -0.5)
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    weights = softmax(logits, dim=-1) # (..., seq_len, seq_len)
    out = weights @ V # (..., seq_len, d_v)
    return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError("d_model must be a multiple of num_heads")

        if theta is None != max_seq_len is None:
            raise ValueError("Only one of theta and max_seq_len is set")

        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device)
        self.k_proj = Linear(d_model, d_model, device)
        self.v_proj = Linear(d_model, d_model, device)
        self.output_proj = Linear(d_model, d_model, device)

        if theta and max_seq_len:
            d_k = d_model // num_heads
            self.rotary_emb = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device)
        else:
            self.rotary_emb = None

    # input of dimension [..., seq_len, d_model]
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Invalid last dimension of input, expected {self.d_model}, got {x.shape[-1]}")
        seq_len = x.shape[-2]
        Q = self.q_proj(x).view(*x.shape[:-1], self.num_heads, -1).transpose(-2, -3) # (..., nH, seq_len, d_k)
        K = self.k_proj(x).view(*x.shape[:-1], self.num_heads, -1).transpose(-2, -3) # (..., nH, seq_len, d_k)
        if self.rotary_emb:
            if token_positions is None:
                raise ValueError("When using RoPE, token_positions cannot be None")
            Q, K = self.rotary_emb(Q, token_positions), self.rotary_emb(K, token_positions)
        V = self.v_proj(x).view(*x.shape[:-1], self.num_heads, -1).transpose(-2, -3) # (..., nH, seq_len, d_v)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attn = scaled_dot_product_attention(Q, K, V, mask) # (..., nH, seq_len, d_v)
        attn = attn.transpose(-2, -3).contiguous().flatten(start_dim=-2) # (..., seq_len, d_model)
        return self.output_proj(attn) # (..., seq_len, d_model)

class TransformerBlock(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device)
        self.ffn = SwiGLU(d_model, d_ff, device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embeddings(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x, token_positions)
        x = self.ln_final(x) # Norm
        x = self.lm_head(x) # linear
        return x


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    shifted = inputs - torch.max(inputs, dim=-1, keepdims=True).values
    picked = shifted.gather(-1, targets.unsqueeze(-1)).unsqueeze(-1) # dim reduced
    lse = torch.logsumexp(shifted, dim=-1) # keepdims=False, dim reduced
    return (lse - picked).mean()