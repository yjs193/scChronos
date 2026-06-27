from __future__ import annotations

import torch
import torch.nn.functional as F

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


def weighted_ot(pred: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor, blur: float = 0.05) -> torch.Tensor:
    pred_weight = torch.full((pred.shape[0],), 1.0 / pred.shape[0], dtype=pred.dtype, device=pred.device)
    target_weight = target_weight / target_weight.sum().clamp_min(1e-8)
    if SamplesLoss is None:
        return torch.cdist(pred, target).min(dim=1).values.mean()
    solver = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=0.5, debias=True, backend="tensorized")
    return solver(pred_weight.double(), pred.double(), target_weight.double(), target.double()).float()


def weighted_moment_losses(pred: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_weight = target_weight / target_weight.sum().clamp_min(1e-8)
    pred_mean = pred.mean(dim=0)
    target_mean = (target * target_weight[:, None]).sum(dim=0)
    mean_loss = F.mse_loss(pred_mean, target_mean) * pred.shape[1]
    pred_var = (pred - pred_mean[None, :]).square().mean(dim=0)
    target_var = ((target - target_mean[None, :]).square() * target_weight[:, None]).sum(dim=0)
    std_loss = F.mse_loss(torch.sqrt(pred_var + 1e-4), torch.sqrt(target_var + 1e-4)) * pred.shape[1]
    return mean_loss, std_loss


def temporal_loss(z: torch.Tensor, day_hidden: torch.Tensor, context_days: torch.Tensor, target_day: torch.Tensor) -> torch.Tensor:
    losses = []
    for b in range(z.shape[0]):
        days = context_days[b]
        target = target_day[b]
        before = torch.where(days < target)[0]
        after = torch.where(days > target)[0]
        if before.numel() > 0 and after.numel() > 0:
            left = before[torch.argmax(days[before])]
            right = after[torch.argmin(days[after])]
            alpha = ((target - days[left]) / (days[right] - days[left]).clamp_min(1e-3)).clamp(0.0, 1.0)
            prior = (1.0 - alpha) * day_hidden[b, left] + alpha * day_hidden[b, right]
            losses.append(F.mse_loss(z[b], prior))
        elif before.numel() >= 2:
            order = before[torch.argsort(days[before])]
            prev, last = order[-2], order[-1]
            velocity = (day_hidden[b, last] - day_hidden[b, prev]) / (days[last] - days[prev]).clamp_min(1e-3)
            losses.append(F.mse_loss(z[b], day_hidden[b, last] + velocity * (target - days[last])))
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()
