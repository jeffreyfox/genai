import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    start = np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size, 1),
    )
    indices = start + np.arange(context_length) # broadcasting

    inputs = torch.from_numpy(dataset[indices]).long().to(device)
    labels = torch.from_numpy(dataset[indices + 1]).long().to(device)

    return inputs, labels
