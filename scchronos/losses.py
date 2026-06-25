from __future__ import annotations

import torch

try:
    from geomloss import SamplesLoss
except Exception:
    SamplesLoss = None


def sinkhorn_ot(pred: torch.Tensor, target: torch.Tensor, blur: float = 0.05) -> torch.Tensor:
    if SamplesLoss is None:
        return torch.cdist(pred, target).min(dim=1).values.mean()
    return SamplesLoss("sinkhorn", p=2, blur=blur)(pred, target)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)

