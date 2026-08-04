import torch

def my_layer_norm(x, gamma, beta, eps=1e-5):
  mu = x.mean(dim=-1, keepdim=True)
  var = x.var(dim=-1, keepdim=True, correction=0)
  return gamma * (x - mu) / torch.sqrt(var + eps) + beta

x = torch.randn(2, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)

out = my_layer_norm(x, gamma, beta)
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)

print("Your output mean:", out.mean(dim=-1))   # should be ~0
print("Your output std: ", out.std(dim=-1))     # should be ~1
print("Match ref?      ", torch.allclose(out, ref, atol=1e-4))
