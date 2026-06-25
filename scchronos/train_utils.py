from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import context_days_for, day_index, sample_cells, valid_training_days

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import pairwise_distances_argmin_min
except Exception:
    MiniBatchKMeans = None
    pairwise_distances_argmin_min = None


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


def build_day_prototypes(values: np.ndarray, days: np.ndarray, n_prototypes: int, seed: int = 0) -> dict[int, np.ndarray]:
    day_to_idx = day_index(days)
    prototypes = {}
    for day, indices in day_to_idx.items():
        if n_prototypes <= 0 or len(indices) <= n_prototypes or MiniBatchKMeans is None:
            prototypes[day] = indices
            continue
        x = values[indices].astype(np.float32, copy=False)
        kmeans = MiniBatchKMeans(n_clusters=n_prototypes, random_state=seed, batch_size=min(2048, max(256, n_prototypes * 8)), n_init=3)
        kmeans.fit(x)
        nearest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, x)
        prototypes[day] = np.unique(indices[nearest])
    return prototypes


def target_probabilities(targets: list[int], mode: str) -> np.ndarray:
    weights = np.ones(len(targets), dtype=np.float64)
    if mode == "bridge_late":
        order = np.linspace(0.0, 1.0, len(targets), dtype=np.float64)
        weights = 0.5 + order
        if len(weights) > 2:
            weights[1:-1] *= 1.25
    elif mode == "late":
        weights = np.linspace(0.5, 1.5, len(targets), dtype=np.float64)
    return weights / weights.sum()


def choose_source_indices(
    day: int,
    day_to_idx: dict[int, np.ndarray],
    prototypes: dict[int, np.ndarray] | None,
    source: str,
    n_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if source == "prototypes" and prototypes is not None and day in prototypes:
        candidates = prototypes[day]
    elif source == "mixed" and prototypes is not None and day in prototypes:
        n_proto = max(1, n_cells // 2)
        n_cells_part = max(1, n_cells - n_proto)
        first = sample_cells(prototypes[day], n_proto, rng)
        second = sample_cells(day_to_idx[day], n_cells_part, rng)
        candidates = np.concatenate([first, second])
    else:
        candidates = day_to_idx[day]
    chosen = sample_cells(candidates, n_cells, rng)
    if len(chosen) < n_cells:
        chosen = rng.choice(chosen, size=n_cells, replace=True)
    return chosen


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
    prototypes: dict[int, np.ndarray] | None = None,
    context_source: str = "cells",
    target_sampling: str = "uniform",
) -> dict[str, torch.Tensor]:
    day_to_idx = day_index(days)
    candidate_targets = valid_training_days(task, train_days, mode, context_len)
    probabilities = target_probabilities(candidate_targets, target_sampling)
    gene_idx, gene_val, totals = tokens
    batch_context_idx = []
    batch_context_val = []
    batch_context_total = []
    batch_context_days = []
    batch_context_expr = []
    batch_target = []
    batch_target_day = []
    for _ in range(batch_size):
        target_day = int(rng.choice(candidate_targets, p=probabilities))
        context_days = context_days_for(task, target_day, train_days, mode, context_len, True)
        ctx_idx = []
        ctx_val = []
        ctx_total = []
        ctx_expr = []
        for day in context_days:
            chosen = choose_source_indices(day, day_to_idx, prototypes, context_source, context_cells, rng)
            ctx_idx.append(gene_idx[chosen])
            ctx_val.append(gene_val[chosen])
            ctx_total.append(totals[chosen])
            ctx_expr.append(torch.from_numpy(values[chosen].astype(np.float32)))
        target_idx = sample_cells(day_to_idx[target_day], target_cells, rng)
        if len(target_idx) < target_cells:
            target_idx = rng.choice(target_idx, size=target_cells, replace=True)
        batch_context_idx.append(torch.stack(ctx_idx))
        batch_context_val.append(torch.stack(ctx_val))
        batch_context_total.append(torch.stack(ctx_total))
        batch_context_expr.append(torch.stack(ctx_expr))
        batch_context_days.append(torch.tensor(context_days, dtype=torch.float32))
        batch_target.append(torch.from_numpy(values[target_idx].astype(np.float32)))
        batch_target_day.append(float(target_day))
    return {
        "context_idx": torch.stack(batch_context_idx).to(device),
        "context_val": torch.stack(batch_context_val).to(device),
        "context_total": torch.stack(batch_context_total).to(device),
        "context_expr": torch.stack(batch_context_expr).to(device),
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
    prototypes: dict[int, np.ndarray] | None = None,
    context_source: str = "cells",
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
            chosen = choose_source_indices(day, day_to_idx, prototypes, context_source, context_cells, rng)
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
