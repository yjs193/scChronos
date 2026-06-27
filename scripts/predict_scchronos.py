from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scchronos.data import day_index, load_temporal_dataset, make_gene_tokens
from scchronos.evaluate import evaluate_model
from scchronos.model import ScChronos, load_model_state_flexible
from scchronos.train_utils import build_day_prototypes, save_json, seed_everything
from scchronos.vocab import extend_vocab_with_genes, load_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_model(cfg: dict, vocab_size: int, pad_id: int, n_genes: int) -> ScChronos:
    return ScChronos(
        vocab_size=vocab_size,
        pad_id=pad_id,
        n_genes=n_genes,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        pred_cells=int(cfg.get("pred_cells", 128)),
        query_groups=int(cfg.get("query_groups", 1)),
        day_slots=int(cfg.get("day_slots", 1)),
        aggregation_mode=str(cfg.get("aggregation_mode", "day_slots")),
        decoder_mode=str(cfg.get("decoder_mode", "global_query")),
        fusion_mode=str(cfg.get("fusion_mode", "single")),
        local_context_k=int(cfg.get("local_context_k", 2)),
        local_strategy=str(cfg.get("local_strategy", cfg.get("local_context_strategy", "nearest"))),
        local_gate_init=float(cfg.get("local_gate_init", 0.5)),
        local_residual_scale=float(cfg.get("local_residual_scale", 1.0)),
        local_residual_max=float(cfg.get("local_residual_max", 2.0)),
        local_residual_interp_only=bool(cfg.get("local_residual_interp_only", False)),
        local_residual_input=str(cfg.get("local_residual_input", "local")),
        encode_chunk_size=int(cfg.get("encode_chunk_size", 8)),
        mean_correction_max=float(cfg.get("mean_correction_max", 3.0)),
        distance_penalty=float(cfg.get("distance_penalty", 0.5)),
        output_activation=str(cfg.get("output_activation", "softplus")),
        sparse_activation_threshold=float(cfg.get("sparse_activation_threshold", -2.0)),
        sparse_activation_temp=float(cfg.get("sparse_activation_temp", 0.5)),
    )


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
    prototypes = build_day_prototypes(values, data.days, int(cfg.get("prototype_count", 256)), int(cfg.get("seed", 42)))
    model = build_model(cfg, len(vocab), pad_id, len(data.genes)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_model_state_flexible(model, checkpoint["model"], strict=False)
    rows, pred_by_day = evaluate_model(
        model,
        values,
        data.days,
        tokens,
        prototypes,
        day_index(data.days),
        data.train_days,
        data.target_days,
        cfg["task"],
        str(cfg.get("context_mode", "bidirectional")),
        int(cfg.get("context_len", 0)),
        int(cfg.get("context_cells", cfg.get("cells_per_day", 32))),
        int(cfg.get("pred_cells", 128)),
        np.random.default_rng(int(cfg.get("seed", 42)) + 500000),
        device,
        eval_repeats=int(cfg.get("eval_repeats", 1)),
        eval_mode=str(cfg.get("eval_mode", "all_cells")),
        eval_max_cells=int(cfg.get("eval_max_cells", 0)),
        context_source=str(cfg.get("context_source", "prototypes")),
        mean_correction_scale=float(cfg.get("mean_correction_scale", 0.0)),
        save_predictions=True,
    )
    pd.DataFrame(rows).to_csv(out_dir / "metrics_by_time.csv", index=False)
    np.savez_compressed(out_dir / "predictions_by_day.npz", **{f"pred_E{day}": arr for day, arr in pred_by_day.items()})
    save_json({"mean_ot": float(np.mean([row["sctime_ot"] for row in rows])), "per_day": rows}, out_dir / "run_summary.json")


if __name__ == "__main__":
    main()
