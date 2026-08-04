import math
from collections.abc import Iterable

import torch
from torch import nn


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = {
            "alpha": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            alpha = group["alpha"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                
                state["t"] += 1
                t = state["t"]
                m = beta1 * state["m"] + (1 - beta1) * p.grad
                v = beta2 * state["v"] + (1 - beta2) * (p.grad ** 2)
                alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p -= alpha * weight_decay * p + alpha_t * m / (v ** 0.5 + eps)
                state["m"] = m
                state["v"] = v

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        factor = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        coeff = 0.5 * (1 + math.cos(factor * math.pi))
        return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def clip_gradient(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    params = [p for p in parameters if p.grad is not None]

    if not params:
        return

    l2_norm = math.sqrt(
        sum(p.grad.detach().pow(2).sum() for p in params)
    )

    if l2_norm > max_l2_norm:
        for p in params:
            p.grad.mul_(max_l2_norm / (l2_norm + 1e-6))
