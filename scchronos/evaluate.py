from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances


def ot_distance(pred: np.ndarray, target: np.ndarray, max_cells: int = 512, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    if pred.shape[0] > max_cells:
        pred = pred[rng.choice(pred.shape[0], max_cells, replace=False)]
    if target.shape[0] > max_cells:
        target = target[rng.choice(target.shape[0], max_cells, replace=False)]
    cost = pairwise_distances(pred, target, metric="euclidean")
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


@torch.no_grad()
def predict_day(model, batch: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output = model(batch["context_idx"], batch["context_val"], batch["context_total"], batch["context_days"], batch["target_day"])
    pred = output["pred"].reshape(-1, output["pred"].shape[-1]).float().cpu().numpy()
    attention = output["attention"].float().cpu().numpy()
    return pred, attention

