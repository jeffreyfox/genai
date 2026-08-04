import torch
import torch.nn as nn


class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices):
        return self.weight[indices]


# 🧪 Debug
emb = MyEmbedding(10, 4)
idx = torch.tensor([0, 3, 7])
print('Output shape:', emb(idx).shape)
print('Matches manual:', torch.equal(emb(idx)[0], emb.weight[0]))
