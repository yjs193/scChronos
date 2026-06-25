from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scchronos.data import day_index, load_temporal_dataset, make_gene_tokens
from scchronos.evaluate import ot_distance, predict_day
from scchronos.model import ScChronos
from scchronos.train_utils import build_day_prototypes, build_eval_batch, save_json, seed_everything
from scchronos.vocab import extend_vocab_with_genes, load_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    seed_everything(int(cfg.get("seed", 42)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    data = load_temporal_dataset(cfg["data_dir"], cfg["task"])
    values = data.expression.to_numpy(np.float32)
    vocab, pad_id, unk_id = load_vocab(cfg["vocab_path"])
    if bool(cfg.get("extend_vocab", True)):
        vocab = extend_vocab_with_genes(vocab, data.genes)
        pad_id = vocab["<pad>"]
        unk_id = vocab["<unk>"]
    tokens = make_gene_tokens(values, data.genes, vocab, pad_id, unk_id, int(cfg.get("max_genes", 600)))[:3]
    prototypes = None
    if str(cfg.get("context_source", "cells")) in {"prototypes", "mixed"}:
        prototypes = build_day_prototypes(values, data.days, int(cfg.get("prototype_count", 256)), int(cfg.get("seed", 42)))
    model = ScChronos(
        vocab_size=len(vocab),
        pad_id=pad_id,
        n_genes=len(data.genes),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        pred_cells=int(cfg.get("pred_cells", 128)),
        query_groups=int(cfg.get("query_groups", 1)),
        day_slots=int(cfg.get("day_slots", 1)),
        local_context_k=int(cfg.get("local_context_k", 2)),
        local_strategy=str(cfg.get("local_strategy", "nearest")),
        local_gate_init=float(cfg.get("local_gate_init", 0.5)),
        fusion_mode=str(cfg.get("fusion_mode", "local_global_residual")),
        local_residual_scale=float(cfg.get("local_residual_scale", 1.2)),
        local_residual_max=float(cfg.get("local_residual_max", 2.5)),
        local_residual_interp_only=bool(cfg.get("local_residual_interp_only", True)),
        local_residual_input=str(cfg.get("local_residual_input", "local")),
        encode_chunk_size=int(cfg.get("encode_chunk_size", 8)),
        output_activation=str(cfg.get("output_activation", "softplus")),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    day_to_idx = day_index(data.days)
    results = []
    arrays = {}
    for target_day in data.target_days:
        batch = build_eval_batch(tokens, data.days, cfg["task"], data.train_days, int(target_day), str(cfg.get("context_mode", "bidirectional")), int(cfg.get("context_len", 0)), int(cfg.get("context_cells", 8)), int(cfg.get("eval_repeats", 8)), rng, device, prototypes=prototypes, context_source=str(cfg.get("context_source", "cells")))
        pred, attention = predict_day(model, batch)
        target = values[day_to_idx[int(target_day)]]
        metric = ot_distance(pred, target, max_cells=int(cfg.get("eval_max_cells", 512)), seed=int(cfg.get("seed", 42)) + int(target_day))
        arrays[f"pred_day_{target_day}"] = pred.astype(np.float32)
        arrays[f"attention_day_{target_day}"] = attention.astype(np.float32)
        results.append({"target_day": int(target_day), "ot": metric})
    np.savez_compressed(out_dir / "predictions_by_day.npz", **arrays)
    save_json({"per_day": results, "mean_ot": float(np.mean([row["ot"] for row in results]))}, out_dir / "run_summary.json")


if __name__ == "__main__":
    main()
