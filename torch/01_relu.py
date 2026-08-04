import torch

# zeros_like guarantees:
# same dtype (float16, bfloat16, etc.)
# same device (cuda, cpu)
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, x, torch.zeros_like(x))