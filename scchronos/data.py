from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class TemporalDataset:
    expression: pd.DataFrame
    metadata: pd.DataFrame
    genes: list[str]
    days: np.ndarray
    train_days: list[int]
    target_days: list[int]


def load_temporal_dataset(data_dir: str | Path, task: str) -> TemporalDataset:
    data_dir = Path(data_dir)
    expression = pd.read_csv(data_dir / f"{task}-norm_data-hvg.csv", index_col=0)
    metadata = pd.read_csv(data_dir / f"{task}-meta_data.csv", index_col=0).loc[expression.index]
    var_path = data_dir / f"{task}-var_genes_list.csv"
    genes = pd.read_csv(var_path, header=None).iloc[:, 0].astype(str).tolist() if var_path.exists() else list(map(str, expression.columns))
    days = metadata["day"].to_numpy(np.int64)
    train_days, target_days = split_days(days, task)
    return TemporalDataset(expression, metadata, genes, days, train_days, target_days)


def split_days(days: np.ndarray, task: str) -> tuple[list[int], list[int]]:
    unique_days = [int(day) for day in sorted(np.unique(days))]
    if unique_days == list(range(19)):
        if task == "remove_recovery":
            target_days = [5, 7, 9, 11, 15, 16, 17, 18]
        elif task == "three_forecasting":
            target_days = [16, 17, 18]
        elif task == "two_forecasting":
            target_days = [17, 18]
        else:
            raise ValueError(f"Unsupported task: {task}")
        return [day for day in unique_days if day not in target_days], target_days
    if task in {"three_forecasting", "two_forecasting"}:
        horizon = 3 if task == "three_forecasting" else 2
        if len(unique_days) <= horizon:
            raise ValueError(f"{task} requires more than {horizon} time points.")
        return unique_days[:-horizon], unique_days[-horizon:]
    if task == "remove_recovery":
        if len(unique_days) == 8:
            positions = [2, 4, 6, 7]
        else:
            positions = [5, 7, 9, 11, 15, 16, 17, 18]
        target_days = [unique_days[pos] for pos in positions if pos < len(unique_days)]
        return [day for day in unique_days if day not in target_days], target_days
    raise ValueError(f"Unsupported task: {task}")


def context_days_for(task: str, target_day: int, train_days: list[int], mode: str, context_len: int, is_training: bool) -> list[int]:
    if task in {"three_forecasting", "two_forecasting"} or mode == "past":
        candidates = [day for day in train_days if day < target_day]
        return candidates[-context_len:] if context_len > 0 else candidates
    candidates = [day for day in train_days if is_training and day != target_day] if is_training else list(train_days)
    if context_len <= 0 or len(candidates) <= context_len:
        return candidates
    before = [day for day in candidates if day < target_day]
    after = [day for day in candidates if day > target_day]
    left = context_len // 2
    selected = before[-left:] + after[: context_len - left]
    if len(selected) < context_len:
        rest = sorted([day for day in candidates if day not in selected], key=lambda day: abs(day - target_day))
        selected += rest[: context_len - len(selected)]
    return sorted(selected)


def valid_training_days(task: str, train_days: list[int], mode: str, context_len: int) -> list[int]:
    return [day for day in train_days if context_days_for(task, day, train_days, mode, context_len, True)]


def make_gene_tokens(
    values: np.ndarray,
    genes: list[str],
    vocab: dict[str, int],
    pad_id: int,
    unk_id: int,
    max_genes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    mapper = np.asarray([vocab.get(str(gene), unk_id) for gene in genes], dtype=np.int64)
    known = mapper != unk_id
    token_ids = torch.full((values.shape[0], max_genes), pad_id, dtype=torch.long)
    token_values = torch.zeros((values.shape[0], max_genes), dtype=torch.float32)
    totals = torch.from_numpy(values.sum(axis=1).astype(np.float32))
    for row in range(values.shape[0]):
        nonzero = np.where((values[row] > 0) & known)[0]
        if nonzero.shape[0] > max_genes:
            row_values = values[row, nonzero]
            nonzero = nonzero[np.argpartition(-row_values, max_genes - 1)[:max_genes]]
        if nonzero.shape[0] == 0:
            continue
        row_values = values[row, nonzero].astype(np.float32)
        order = np.argsort(-row_values)
        nonzero = nonzero[order]
        row_values = row_values[order]
        n_tokens = min(nonzero.shape[0], max_genes)
        token_ids[row, :n_tokens] = torch.from_numpy(mapper[nonzero[:n_tokens]])
        token_values[row, :n_tokens] = torch.from_numpy(row_values[:n_tokens])
    return token_ids, token_values, totals, float(known.mean())


def day_index(days: np.ndarray) -> dict[int, np.ndarray]:
    return {int(day): np.where(days == int(day))[0] for day in sorted(np.unique(days))}


def sample_cells(indices: np.ndarray, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= n_cells:
        return indices
    return rng.choice(indices, size=n_cells, replace=False)

