import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    obj = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    if "model_state" not in obj:
        raise ValueError("Can't load model state from checkpoint file")
    if "optimizer_state" not in obj:
        raise ValueError("Can't load optimizer state from checkpoint file")
    if "iteration" not in obj:
        raise ValueError("Can't load iteration number from checkpoint file")

    model.load_state_dict(obj["model_state"])
    optimizer.load_state_dict(obj["optimizer_state"])
    iteration = obj["iteration"]
    return iteration