from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

try:
    import wandb
except Exception:
    wandb = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scchronos.data import day_index, load_temporal_dataset, make_gene_tokens
from scchronos.evaluate import ot_distance, predict_day
from scchronos.foundation import load_pretrained_encoder
from scchronos.losses import reconstruction_loss, sinkhorn_ot
from scchronos.model import ScChronos
from scchronos.train_utils import build_eval_batch, sample_training_batch, save_json, seed_everything
from scchronos.vocab import extend_vocab_with_genes, load_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    if args.task is not None:
        cfg["task"] = args.task
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    seed_everything(int(cfg.get("seed", 42)))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    data = load_temporal_dataset(cfg["data_dir"], cfg["task"])
    values = data.expression.to_numpy(np.float32)
    vocab, pad_id, unk_id = load_vocab(cfg["vocab_path"])
    if bool(cfg.get("extend_vocab", True)):
        vocab = extend_vocab_with_genes(vocab, data.genes)
        pad_id = vocab["<pad>"]
        unk_id = vocab["<unk>"]
    tokens = make_gene_tokens(values, data.genes, vocab, pad_id, unk_id, int(cfg.get("max_genes", 600)))
    token_tensors = tokens[:3]
    save_json({"gene_coverage": tokens[3], "train_days": data.train_days, "target_days": data.target_days, "vocab_size": len(vocab), "pad_id": pad_id, "unk_id": unk_id}, out_dir / "run_metadata.json")

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
        encode_chunk_size=int(cfg.get("encode_chunk_size", 8)),
        output_activation=str(cfg.get("output_activation", "softplus")),
    ).to(device)
    if cfg.get("pretrained_path"):
        info = load_pretrained_encoder(model.encoder, cfg["pretrained_path"])
        save_json(info, out_dir / "pretrained_load.json")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    day_to_idx = day_index(data.days)

    run = None
    if args.wandb and wandb is not None:
        run = wandb.init(project=cfg.get("wandb_project", "scChronos"), name=cfg.get("run_name"), config=cfg)

    best = float("inf")
    patience = int(cfg.get("patience", 20))
    bad_epochs = 0
    history = []
    for epoch in range(1, int(cfg.get("epochs", 50)) + 1):
        model.train()
        losses = []
        for _ in range(int(cfg.get("steps_per_epoch", 100))):
            batch = sample_training_batch(
                token_tensors,
                values,
                data.days,
                cfg["task"],
                data.train_days,
                str(cfg.get("context_mode", "bidirectional")),
                int(cfg.get("context_len", 0)),
                int(cfg.get("context_cells", 8)),
                int(cfg.get("target_cells", 128)),
                int(cfg.get("batch_size", 1)),
                rng,
                device,
            )
            output = model(batch["context_idx"], batch["context_val"], batch["context_total"], batch["context_days"], batch["target_day"])
            pred = output["pred"]
            target = batch["target"]
            ot = sum(sinkhorn_ot(pred[i], target[i], blur=float(cfg.get("sinkhorn_blur", 0.05))) for i in range(pred.shape[0])) / pred.shape[0]
            recon_weight = float(cfg.get("cell_recon_weight", 0.5))
            cell_recon = output["cell_recon"].reshape(-1, output["cell_recon"].shape[-1])
            context_expr = []
            for row in range(batch["context_days"].shape[0]):
                for day in batch["context_days"][row].long().cpu().numpy().tolist():
                    idx = day_to_idx[int(day)]
                    selected = idx[: min(len(idx), int(cfg.get("context_cells", 8)))]
                    context_expr.append(torch.from_numpy(values[selected].astype(np.float32)).to(device))
            recon_target = torch.cat(context_expr, dim=0)
            recon_target = recon_target[: cell_recon.shape[0]]
            loss = ot + recon_weight * reconstruction_loss(cell_recon[: recon_target.shape[0]], recon_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        eval_rows = []
        for target_day in data.target_days:
            eval_batch = build_eval_batch(token_tensors, data.days, cfg["task"], data.train_days, int(target_day), str(cfg.get("context_mode", "bidirectional")), int(cfg.get("context_len", 0)), int(cfg.get("context_cells", 8)), int(cfg.get("eval_repeats", 4)), rng, device)
            pred, attention = predict_day(model, eval_batch)
            target = values[day_to_idx[int(target_day)]]
            metric = ot_distance(pred, target, max_cells=int(cfg.get("eval_max_cells", 512)), seed=int(cfg.get("seed", 42)) + epoch + int(target_day))
            eval_rows.append({"epoch": epoch, "target_day": int(target_day), "ot": metric, "attention_mean": attention.mean(axis=0).tolist()})
        mean_ot = float(np.mean([row["ot"] for row in eval_rows]))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "mean_ot": mean_ot}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
        pd.DataFrame(eval_rows).to_json(out_dir / f"eval_epoch_{epoch:03d}.json", orient="records", indent=2)
        if run is not None:
            wandb.log(row, step=epoch)
        if mean_ot < best:
            best = mean_ot
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "config": cfg, "genes": data.genes, "vocab": vocab}, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    torch.save({"model": model.state_dict(), "config": cfg, "genes": data.genes, "vocab": vocab}, out_dir / "last.pt")
    save_json({"best_mean_ot": best, "epochs_ran": len(history)}, out_dir / "summary.json")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
