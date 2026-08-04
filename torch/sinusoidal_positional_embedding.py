import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

# Synthetic data
torch.manual_seed(42)
batch_size = 3
seq_len = 4
d_model = 8
num_heads = 2

q = torch.rand(batch_size, seq_len, d_model)
k = torch.rand(batch_size, seq_len, d_model)
v = torch.rand(batch_size, seq_len, d_model)
print(q.shape)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        """
        Initializes the sinusoidal positional embedding.
        
        Args:
            max_seq_len (int): Maximum sequence length.
            d_model (int): Embedding dimension.
        """
        super().__init__()

        if d_model %2 != 0:
            raise ValueError("d_model must be even")

        position = torch.arange(
            max_seq_len, dtype=torch.float32).unsqueeze(dim=1) # (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        ) # (d_model // 2,)
        prod = position * div_term
        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(prod)
        pe[:, 1::2] = torch.cos(prod)

        self.register_buffer("pe", pe.unsqueeze(0)) # [1, max_seq_len, d_model]

    def forward(self, positions):
        """
        Returns the positional embedding for a given input tensor.
        
        Args:
            positions (Tensor): Tensor of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Positional encodings of shape (1, seq_len, d_model).
        """
        seq_len = positions.shape[1]

        if seq_len > self.pe.shape[1]:
            raise ValueError("Sequence length exceeds max_seq_len")

        return self.pe[:, :seq_len, :]

max_seq_len = 100
d_model = 64
seq_len = 50
positions = torch.arange(seq_len).unsqueeze(0)  # Shape: (1, seq_len)
print(positions.shape)
custom_pos_emb = SinusoidalPositionalEmbedding(d_model, max_seq_len)

positional_encoding_custom = custom_pos_emb(positions)

print(positional_encoding_custom.shape)  # (1, 50, 64)
