from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from geomloss import SamplesLoss
except Exception:
    SamplesLoss = None


def weighted_ot(pred: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor, blur: float = 0.05) -> torch.Tensor:
    pred_weight = torch.full((pred.shape[0],), 1.0 / pred.shape[0], dtype=pred.dtype, device=pred.device)
    target_weight = target_weight / target_weight.sum().clamp_min(1e-8)
    if SamplesLoss is None:
        return torch.cdist(pred, target).min(dim=1).values.mean()
    solver = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=0.5, debias=True, backend="tensorized")
    return solver(pred_weight.double(), pred.double(), target_weight.double(), target.double()).float()


def weighted_snapshot_stat_loss(pred: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor) -> torch.Tensor:
    target_weight = target_weight / target_weight.sum().clamp_min(1e-8)
    pred_mean = pred.mean(dim=0)
    target_mean = (target * target_weight[:, None]).sum(dim=0)
    pred_var = (pred - pred_mean[None, :]).square().mean(dim=0)
    target_var = ((target - target_mean[None, :]).square() * target_weight[:, None]).sum(dim=0)
    mean_loss = F.mse_loss(pred_mean, target_mean)
    std_loss = F.mse_loss(torch.sqrt(pred_var + 1e-4), torch.sqrt(target_var + 1e-4))
    return (mean_loss + std_loss) * pred.shape[1]
