from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import context_days_for, day_index, sample_cells, valid_training_days


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sample_training_batch(
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    values: np.ndarray,
    days: np.ndarray,
    task: str,
    train_days: list[int],
    mode: str,
    context_len: int,
    context_cells: int,
    target_cells: int,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    day_to_idx = day_index(days)
    candidate_targets = valid_training_days(task, train_days, mode, context_len)
    gene_idx, gene_val, totals = tokens
    batch_context_idx = []
    batch_context_val = []
    batch_context_total = []
    batch_context_days = []
    batch_target = []
    batch_target_day = []
    for _ in range(batch_size):
        target_day = int(rng.choice(candidate_targets))
        context_days = context_days_for(task, target_day, train_days, mode, context_len, True)
        ctx_idx = []
        ctx_val = []
        ctx_total = []
        for day in context_days:
            chosen = sample_cells(day_to_idx[day], context_cells, rng)
            if len(chosen) < context_cells:
                chosen = rng.choice(chosen, size=context_cells, replace=True)
            ctx_idx.append(gene_idx[chosen])
            ctx_val.append(gene_val[chosen])
            ctx_total.append(totals[chosen])
        target_idx = sample_cells(day_to_idx[target_day], target_cells, rng)
        if len(target_idx) < target_cells:
            target_idx = rng.choice(target_idx, size=target_cells, replace=True)
        batch_context_idx.append(torch.stack(ctx_idx))
        batch_context_val.append(torch.stack(ctx_val))
        batch_context_total.append(torch.stack(ctx_total))
        batch_context_days.append(torch.tensor(context_days, dtype=torch.float32))
        batch_target.append(torch.from_numpy(values[target_idx].astype(np.float32)))
        batch_target_day.append(float(target_day))
    return {
        "context_idx": torch.stack(batch_context_idx).to(device),
        "context_val": torch.stack(batch_context_val).to(device),
        "context_total": torch.stack(batch_context_total).to(device),
        "context_days": torch.stack(batch_context_days).to(device),
        "target": torch.stack(batch_target).to(device),
        "target_day": torch.tensor(batch_target_day, dtype=torch.float32, device=device),
    }


def build_eval_batch(
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    days: np.ndarray,
    task: str,
    train_days: list[int],
    target_day: int,
    mode: str,
    context_len: int,
    context_cells: int,
    repeats: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    day_to_idx = day_index(days)
    gene_idx, gene_val, totals = tokens
    context_days = context_days_for(task, target_day, train_days, mode, context_len, False)
    batch_context_idx = []
    batch_context_val = []
    batch_context_total = []
    for _ in range(repeats):
        ctx_idx = []
        ctx_val = []
        ctx_total = []
        for day in context_days:
            chosen = sample_cells(day_to_idx[day], context_cells, rng)
            if len(chosen) < context_cells:
                chosen = rng.choice(chosen, size=context_cells, replace=True)
            ctx_idx.append(gene_idx[chosen])
            ctx_val.append(gene_val[chosen])
            ctx_total.append(totals[chosen])
        batch_context_idx.append(torch.stack(ctx_idx))
        batch_context_val.append(torch.stack(ctx_val))
        batch_context_total.append(torch.stack(ctx_total))
    return {
        "context_idx": torch.stack(batch_context_idx).to(device),
        "context_val": torch.stack(batch_context_val).to(device),
        "context_total": torch.stack(batch_context_total).to(device),
        "context_days": torch.tensor([context_days] * repeats, dtype=torch.float32, device=device),
        "target_day": torch.full((repeats,), float(target_day), dtype=torch.float32, device=device),
    }

