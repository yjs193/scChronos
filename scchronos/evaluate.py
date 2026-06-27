from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

from .train_utils import build_eval_context


def ot_distance(pred: np.ndarray, target: np.ndarray, max_cells: int = 0, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    if max_cells and max_cells > 0:
        if pred.shape[0] > max_cells:
            pred = pred[rng.choice(pred.shape[0], max_cells, replace=False)]
        if target.shape[0] > max_cells:
            target = target[rng.choice(target.shape[0], max_cells, replace=False)]
    cost = pairwise_distances(pred.astype(np.float32), target.astype(np.float32), metric="euclidean")
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


@torch.no_grad()
def predict_day(model, batch: dict[str, torch.Tensor], mean_correction_scale: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred, attention = model.predict(batch["context_idx"], batch["context_val"], batch["context_total"], batch["context_days"], batch["target_day"], mean_correction_scale)
    return pred.squeeze(0).float().cpu().numpy(), attention.float().cpu().numpy()


@torch.no_grad()
def evaluate_model(
    model,
    values: np.ndarray,
    days: np.ndarray,
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    prototypes,
    day_to_idx: dict[int, np.ndarray],
    train_days: list[int],
    target_days: list[int],
    task: str,
    context_mode: str,
    context_len: int,
    context_cells: int,
    pred_cells: int,
    rng: np.random.Generator,
    device: torch.device,
    eval_repeats: int = 1,
    eval_mode: str = "all_cells",
    eval_max_cells: int = 0,
    context_source: str = "prototypes",
    mean_correction_scale: float = 0.0,
    save_predictions: bool = False,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    rows = []
    pred_by_day = {}
    generated = max(1, int(getattr(model, "pred_cells", pred_cells)))
    for target_day in target_days:
        target_pool = day_to_idx[int(target_day)]
        if eval_mode == "all_cells":
            n_eval = len(target_pool)
            target_np = values[target_pool].astype(np.float32, copy=False)
            repeats = max(1, int(np.ceil(n_eval / generated)))
        elif eval_mode == "scmix":
            n_eval = max(pred_cells, (len(target_pool) // pred_cells) * pred_cells)
            target_idx = rng.choice(target_pool, size=n_eval, replace=len(target_pool) < n_eval)
            target_np = values[target_idx].astype(np.float32, copy=False)
            repeats = max(1, int(np.ceil(n_eval / generated)))
        else:
            repeats = max(1, int(eval_repeats))
            target_np = values[target_pool].astype(np.float32, copy=False)
            n_eval = repeats * generated
        pred_parts = []
        attention_parts = []
        for _ in range(repeats):
            batch = build_eval_context(tokens, values, days, task, train_days, int(target_day), context_mode, context_len, context_cells, rng, device, prototypes, context_source)
            pred, attention = predict_day(model, batch, mean_correction_scale)
            pred_parts.append(pred)
            attention_parts.append(torch.from_numpy(attention))
        pred_pool = np.concatenate(pred_parts, axis=0).astype(np.float32)
        if pred_pool.shape[0] > n_eval:
            pred_pool = pred_pool[rng.choice(pred_pool.shape[0], size=n_eval, replace=False)]
        if save_predictions:
            pred_by_day[int(target_day)] = pred_pool
        metric = ot_distance(pred_pool, target_np, max_cells=int(eval_max_cells), seed=int(rng.integers(0, 2**31 - 1)))
        attention_mean = torch.stack(attention_parts, dim=0).mean(dim=0).squeeze(0).tolist()
        rows.append(
            {
                "target_day": int(target_day),
                "ot": metric,
                "sctime_ot": metric,
                "eval_mode": eval_mode,
                "eval_repeats": int(repeats),
                "n_pred": int(pred_pool.shape[0]),
                "n_true_eval": int(target_np.shape[0]),
                "n_true_total": int(len(target_pool)),
                "attention_mean": attention_mean,
            }
        )
    return rows, pred_by_day
