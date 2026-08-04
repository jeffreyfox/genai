import torch

def cross_entropy_loss(logits, targets):
  logits = logits - torch.max(logits, dim=-1, keepdim=True).values
  logprobs = logits - torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
  loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).mean()
  return loss

def cross_entropy_loss_2(logits, targets):
  logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
  loss = -logprobs[torch.arange(logits.shape[0]), targets].mean()
  return loss

torch.manual_seed(42)
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))

print('Loss :', cross_entropy_loss(logits, targets))
print('Loss2:', cross_entropy_loss_2(logits, targets))
print('Ref  :', torch.nn.functional.cross_entropy(logits, targets))

