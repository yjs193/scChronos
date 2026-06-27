from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

from .data import context_days_for, day_index, sample_cells


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


def build_day_prototypes(values: np.ndarray, days: np.ndarray, n_prototypes: int, seed: int = 0) -> dict[int, dict[str, np.ndarray]]:
    day_to_idx = day_index(days)
    n_comp = min(50, values.shape[1], values.shape[0] - 1)
    projected = PCA(n_components=n_comp, random_state=seed).fit_transform(values) if n_comp > 1 else values
    prototypes = {}
    for day, indices in day_to_idx.items():
        k = min(int(n_prototypes), len(indices))
        if k <= 1:
            nearest = indices[:1]
            weights = np.ones(1, dtype=np.float32)
            centers = values[nearest].astype(np.float32)
        elif len(indices) <= k:
            nearest = indices
            weights = np.ones(len(indices), dtype=np.float32) / float(len(indices))
            centers = values[indices].astype(np.float32)
        else:
            xp = projected[indices]
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed + int(day), batch_size=max(256, k * 4), n_init=3, max_iter=100)
            labels = kmeans.fit_predict(xp)
            nearest_local, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, xp)
            nearest = indices[nearest_local]
            weights = []
            centers = []
            for cluster in range(k):
                members = np.where(labels == cluster)[0]
                if members.size == 0:
                    continue
                weights.append(float(members.size) / float(len(indices)))
                centers.append(values[indices[members]].mean(axis=0))
            weights = np.asarray(weights, dtype=np.float32)
            weights = weights / weights.sum()
            centers = np.asarray(centers, dtype=np.float32)
        prototypes[int(day)] = {"indices": nearest.astype(np.int64), "weights": weights.astype(np.float32), "expr": centers.astype(np.float32)}
    return prototypes


def build_vocab_to_column(genes: list[str], vocab: dict[str, int]) -> torch.Tensor:
    size = max(vocab.values()) + 1
    lookup = torch.full((size,), -1, dtype=torch.long)
    for col, gene in enumerate(genes):
        if gene in vocab:
            lookup[int(vocab[gene])] = int(col)
    return lookup


def target_probabilities(targets: list[int], mode: str) -> np.ndarray | None:
    if mode == "uniform":
        return None
    lo, hi = min(targets), max(targets)
    progress = np.asarray([(float(day) - lo) / max(float(hi - lo), 1.0) for day in targets], dtype=np.float64)
    if mode == "late":
        weights = 0.1 + progress**2
    elif mode == "bridge_late":
        focus = np.asarray([6, 8, 10, 12], dtype=np.float64)
        dist = np.min(np.abs(np.asarray(targets, dtype=np.float64)[:, None] - focus[None, :]), axis=1)
        weights = 0.1 + 0.7 * np.exp(-(dist**2) / 4.0) + 0.5 * progress**2
    elif mode == "tail_strong":
        focus = np.asarray([10, 12, 13, 14], dtype=np.float64)
        dist = np.min(np.abs(np.asarray(targets, dtype=np.float64)[:, None] - focus[None, :]), axis=1)
        weights = 0.03 + 1.8 * np.exp(-(dist**2) / 1.6) + 0.7 * progress**3
    else:
        weights = np.ones(len(targets), dtype=np.float64)
    return weights / weights.sum()


def choose_training_targets(task: str, train_days: list[int], mode: str, context_len: int) -> list[int]:
    return [day for day in train_days if context_days_for(task, day, train_days, mode, context_len, True)]


def choose_source_indices(day: int, day_to_idx: dict[int, np.ndarray], prototypes: dict[int, dict[str, np.ndarray]] | None, source: str, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    if source == "prototypes" and prototypes is not None:
        pool = prototypes[int(day)]["indices"]
    elif source == "mixed" and prototypes is not None:
        n_proto = n_cells // 2
        n_cell = n_cells - n_proto
        first = sample_cells(prototypes[int(day)]["indices"], n_proto, rng)
        second = sample_cells(day_to_idx[int(day)], n_cell, rng)
        pool = np.concatenate([first, second])
    else:
        pool = day_to_idx[int(day)]
    replace = len(pool) < n_cells
    return rng.choice(pool, size=n_cells, replace=replace)


def pad_weighted_proto(proto: dict[str, np.ndarray], n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    expr = proto["expr"]
    weights = proto["weights"]
    if expr.shape[0] >= n_cells:
        out_expr = expr[:n_cells]
        out_weight = weights[:n_cells]
        return out_expr, out_weight / out_weight.sum()
    pad_n = n_cells - expr.shape[0]
    out_expr = np.concatenate([expr, np.repeat(expr[-1:], pad_n, axis=0)], axis=0)
    out_weight = np.concatenate([weights, np.zeros(pad_n, dtype=np.float32)], axis=0)
    return out_expr, out_weight / out_weight.sum()


def sample_episode(
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    values: np.ndarray,
    days: np.ndarray,
    task: str,
    train_days: list[int],
    mode: str,
    context_len: int,
    cells_per_day: int,
    mask_count: int,
    pred_cells: int,
    target_cells: int,
    rng: np.random.Generator,
    device: torch.device,
    prototypes: dict[int, dict[str, np.ndarray]] | None,
    target_sampling: str = "uniform",
    target_source: str = "prototypes",
    context_source: str = "prototypes",
) -> dict[str, torch.Tensor]:
    day_to_idx = day_index(days)
    possible = choose_training_targets(task, train_days, mode, context_len)
    probabilities = target_probabilities(possible, target_sampling)
    target_days = sorted(rng.choice(possible, size=min(int(mask_count), len(possible)), replace=False, p=probabilities).astype(int).tolist())
    context_days = context_days_for(task, target_days[0] if len(target_days) == 1 else int(np.mean(target_days)), train_days, mode, context_len, True)
    gene_idx, gene_val, total = tokens
    ctx_indices = []
    ctx_expr = []
    for day in context_days:
        chosen = choose_source_indices(day, day_to_idx, prototypes, context_source, cells_per_day, rng)
        ctx_indices.append(chosen)
        ctx_expr.append(values[chosen].astype(np.float32))
    ctx_indices = np.asarray(ctx_indices, dtype=np.int64)
    target_expr = []
    target_weight = []
    n_target = int(target_cells) if int(target_cells) > 0 else int(pred_cells)
    for day in target_days:
        if target_source == "prototypes" and prototypes is not None:
            expr, weight = pad_weighted_proto(prototypes[int(day)], n_target)
        else:
            pool = day_to_idx[int(day)]
            chosen = rng.choice(pool, size=n_target, replace=len(pool) < n_target)
            expr = values[chosen].astype(np.float32)
            weight = np.full(n_target, 1.0 / float(n_target), dtype=np.float32)
        target_expr.append(expr)
        target_weight.append(weight.astype(np.float32))
    flat = ctx_indices.reshape(-1)
    return {
        "context_idx": gene_idx[flat].reshape(1, len(context_days), cells_per_day, -1).to(device),
        "context_val": gene_val[flat].reshape(1, len(context_days), cells_per_day, -1).to(device),
        "context_total": total[flat].reshape(1, len(context_days), cells_per_day).to(device),
        "context_expr": torch.from_numpy(np.stack(ctx_expr, axis=0)).float().unsqueeze(0).to(device),
        "context_days": torch.tensor(context_days, dtype=torch.float32, device=device).unsqueeze(0),
        "target_day": torch.tensor(target_days, dtype=torch.float32, device=device),
        "target_expr": torch.from_numpy(np.stack(target_expr, axis=0)).float().to(device),
        "target_weight": torch.from_numpy(np.stack(target_weight, axis=0)).float().to(device),
    }


def build_eval_context(
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    values: np.ndarray,
    days: np.ndarray,
    task: str,
    train_days: list[int],
    target_day: int,
    mode: str,
    context_len: int,
    cells_per_day: int,
    rng: np.random.Generator,
    device: torch.device,
    prototypes: dict[int, dict[str, np.ndarray]] | None,
    context_source: str = "prototypes",
) -> dict[str, torch.Tensor]:
    day_to_idx = day_index(days)
    context_days = context_days_for(task, target_day, train_days, mode, context_len, False)
    gene_idx, gene_val, total = tokens
    ctx_indices = []
    for day in context_days:
        ctx_indices.append(choose_source_indices(day, day_to_idx, prototypes, context_source, cells_per_day, rng))
    ctx_indices = np.asarray(ctx_indices, dtype=np.int64)
    flat = ctx_indices.reshape(-1)
    return {
        "context_idx": gene_idx[flat].reshape(1, len(context_days), cells_per_day, -1).to(device),
        "context_val": gene_val[flat].reshape(1, len(context_days), cells_per_day, -1).to(device),
        "context_total": total[flat].reshape(1, len(context_days), cells_per_day).to(device),
        "context_expr": torch.from_numpy(values[ctx_indices].astype(np.float32)).float().unsqueeze(0).to(device),
        "context_days": torch.tensor(context_days, dtype=torch.float32, device=device).unsqueeze(0),
        "target_day": torch.tensor([float(target_day)], dtype=torch.float32, device=device),
    }
