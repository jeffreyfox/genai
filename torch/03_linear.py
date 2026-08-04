import torch

# Need to use requires_grad. Instead we can use torch.nn.parameter which sets requires_grad by default.

class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(out_features, in_features) * (in_features ** -0.5)
        self.weight.requires_grad_(True)
        self.bias = torch.zeros(out_features, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias
