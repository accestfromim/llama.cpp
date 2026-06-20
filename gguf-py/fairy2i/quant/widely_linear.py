from __future__ import annotations

import torch


def undo_llama_permute(weight: torch.Tensor, n_head: int) -> torch.Tensor:
    return (
        weight.reshape(n_head, 2, weight.shape[0] // n_head // 2, *weight.shape[1:])
        .swapaxes(1, 2)
        .reshape(weight.shape)
    )
