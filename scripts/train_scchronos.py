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
from scchronos.evaluate import evaluate_model
from scchronos.foundation import load_pretrained_encoder
from scchronos.losses import weighted_ot, weighted_snapshot_stat_loss
from scchronos.model import ScChronos, load_model_state_flexible
from scchronos.train_utils import build_day_prototypes, build_vocab_to_column, pad_weighted_proto, sample_episode, save_json, seed_everything
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


def cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return min_lr + (base_lr - min_lr) * float(step) / float(warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))))
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def init_decoder_bias(model: ScChronos, values: np.ndarray) -> None:
    mean = torch.from_numpy(values.mean(axis=0).astype(np.float32)).clamp_min(1e-4)
    bias = torch.log(torch.expm1(mean).clamp_min(1e-6))
    with torch.no_grad():
        model.decoder[-1].bias.copy_(bias)
        model.cell_decoder[-1].bias.copy_(bias)
        model.recon_decoder[-1].bias.copy_(bias)


def run_eval(model, data, values, token_tensors, prototypes, cfg, rng, device, save_predictions=False):
    day_to_idx = day_index(data.days)
    return evaluate_model(
        model,
        values,
        data.days,
        token_tensors,
        prototypes,
        day_to_idx,
        data.train_days,
        data.target_days,
        cfg["task"],
        str(cfg.get("context_mode", "bidirectional")),
        int(cfg.get("context_len", 0)),
        int(cfg.get("context_cells", cfg.get("cells_per_day", 32))),
        int(cfg.get("pred_cells", 128)),
        rng,
        device,
        eval_repeats=int(cfg.get("eval_repeats", 1)),
        eval_mode=str(cfg.get("eval_mode", "all_cells")),
        eval_max_cells=int(cfg.get("eval_max_cells", 0)),
        context_source=str(cfg.get("context_source", "prototypes")),
        mean_correction_scale=float(cfg.get("mean_correction_scale", 0.0)),
        save_predictions=save_predictions,
    )


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
    if cfg["task"] in {"three_forecasting", "two_forecasting"}:
        cfg["context_mode"] = "past"

    seed_everything(int(cfg.get("seed", 42)))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    data = load_temporal_dataset(cfg["data_dir"], cfg["task"])
    values = data.expression.to_numpy(np.float32)
    vocab, pad_id, unk_id = load_vocab(cfg["vocab_path"])
    if bool(cfg.get("extend_vocab", True)):
        vocab = extend_vocab_with_genes(vocab, data.genes)
        pad_id = vocab["<pad>"]
        unk_id = vocab["<unk>"]
    tokens = make_gene_tokens(values, data.genes, vocab, pad_id, unk_id, int(cfg.get("max_genes", 600)))
    token_tensors = tokens[:3]
    prototypes = build_day_prototypes(values, data.days, int(cfg.get("prototype_count", 256)), int(cfg.get("seed", 42)))
    vocab_to_column = build_vocab_to_column(data.genes, vocab).to(device)
    save_json({"gene_coverage": tokens[3], "train_days": data.train_days, "target_days": data.target_days, "vocab_size": len(vocab), "pad_id": pad_id, "unk_id": unk_id}, out_dir / "run_metadata.json")

    model = ScChronos(
        vocab_size=len(vocab),
        pad_id=pad_id,
        n_genes=len(data.genes),
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
    ).to(device)
    init_decoder_bias(model, values)
    if cfg.get("pretrained_path"):
        info = load_pretrained_encoder(model.encoder, cfg["pretrained_path"])
        save_json(info, out_dir / "pretrained_load.json")
    if cfg.get("resume"):
        checkpoint = torch.load(cfg["resume"], map_location=device, weights_only=False)
        load_model_state_flexible(model, checkpoint["model"], strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-5)), weight_decay=float(cfg.get("weight_decay", 0.03)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    run = None
    if args.wandb and wandb is not None:
        run = wandb.init(project=cfg.get("wandb_project", "scChronos"), name=cfg.get("run_name"), config=cfg)

    steps = int(cfg.get("steps", int(cfg.get("epochs", 20)) * int(cfg.get("steps_per_epoch", 100))))
    steps_per_epoch = int(cfg.get("steps_per_epoch", max(1, len(data.train_days))))
    warmup_steps = int(round(float(cfg.get("warmup_epochs", 0.0)) * float(steps_per_epoch)))
    min_lr = float(cfg.get("min_lr", cfg.get("lr", 2e-5)))
    eval_every = int(cfg.get("eval_every", steps_per_epoch))
    patience = int(cfg.get("early_stop_patience", cfg.get("patience", 0)))
    early_min_step = int(cfg.get("early_stop_min_step", 0))
    best = float("inf")
    bad = 0
    history = []
    eval_history = []
    top_checkpoints = []

    for step in range(1, steps + 1):
        model.train()
        set_lr(optimizer, cosine_lr(step, steps, float(cfg.get("lr", 2e-5)), min_lr, warmup_steps))
        batch = sample_episode(
            token_tensors,
            values,
            data.days,
            cfg["task"],
            data.train_days,
            str(cfg.get("context_mode", "bidirectional")),
            int(cfg.get("context_len", 0)),
            int(cfg.get("context_cells", cfg.get("cells_per_day", 32))),
            int(cfg.get("mask_count", 1)),
            int(cfg.get("pred_cells", 128)),
            int(cfg.get("train_target_cells", cfg.get("target_cells", 0))),
            rng,
            device,
            prototypes,
            target_sampling=str(cfg.get("target_sampling", "uniform")),
            target_source=str(cfg.get("target_source", "prototypes")),
            context_source=str(cfg.get("context_source", "prototypes")),
        )
        context_val = batch["context_val"]
        valid = (batch["context_idx"] != pad_id) & (context_val > 0)
        col_idx = vocab_to_column[batch["context_idx"].clamp(0, vocab_to_column.numel() - 1)]
        valid = valid & (col_idx >= 0)
        if float(cfg.get("cell_recon_weight", 0.0)) > 0 and float(cfg.get("cell_mask_ratio", 0.0)) > 0:
            cell_mask = valid & (torch.rand(context_val.shape, device=device) < float(cfg.get("cell_mask_ratio", 0.6)))
            context_val_model = context_val.masked_fill(cell_mask, 0.0)
        else:
            cell_mask = torch.zeros_like(context_val, dtype=torch.bool)
            context_val_model = context_val

        optimizer.zero_grad(set_to_none=True)
        target_losses = []
        recon_ref = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for i, target_day in enumerate(batch["target_day"]):
                pred, recon, _z, _day_hidden, _day_weights, _cell_weights, cell_latent = model(
                    batch["context_idx"],
                    context_val_model,
                    batch["context_total"],
                    batch["context_days"],
                    target_day.reshape(1),
                    mean_correction_scale=float(cfg.get("mean_correction_scale", 0.0)),
                )
                target_losses.append(weighted_ot(pred[0], batch["target_expr"][i], batch["target_weight"][i], float(cfg.get("sinkhorn_blur", 0.05))))
                if recon_ref is None:
                    recon_ref = recon
            loss_target = torch.stack(target_losses).mean()
            recon_terms = []
            if float(cfg.get("recon_weight", 0.0)) != 0:
                day_to_idx = day_index(data.days)
                for j, context_day in enumerate(batch["context_days"][0].detach().cpu().numpy().astype(int).tolist()):
                    if str(cfg.get("recon_source", "prototypes")) == "prototypes":
                        expr_np, weight_np = pad_weighted_proto(prototypes[int(context_day)], int(cfg.get("pred_cells", 128)))
                    else:
                        pool = day_to_idx[int(context_day)]
                        chosen = rng.choice(pool, size=int(cfg.get("pred_cells", 128)), replace=len(pool) < int(cfg.get("pred_cells", 128)))
                        expr_np = values[chosen].astype(np.float32)
                        weight_np = np.full(expr_np.shape[0], 1.0 / expr_np.shape[0], dtype=np.float32)
                    expr_t = torch.from_numpy(expr_np).float().to(device)
                    weight_t = torch.from_numpy(weight_np).float().to(device)
                    recon_terms.append(weighted_snapshot_stat_loss(recon_ref[0, j], expr_t, weight_t))
            loss_recon = torch.stack(recon_terms).mean() if recon_terms else loss_target * 0.0
            loss_cell_recon, cell_mask_fraction = model.reconstruct_masked_cells_from_latent(cell_latent, batch["context_idx"], batch["context_val"], vocab_to_column, pad_id, cell_mask)
            loss = (
                float(cfg.get("target_weight", 1.0)) * loss_target
                + float(cfg.get("recon_weight", 0.0)) * loss_recon
                + float(cfg.get("cell_recon_weight", 0.0)) * loss_cell_recon
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 50 == 0:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "target_ot": float(loss_target.detach().cpu()),
                "snap_stat": float(loss_recon.detach().cpu()),
                "cell_recon": float(loss_cell_recon.detach().cpu()),
                "cell_mask_fraction": float(cell_mask_fraction.detach().cpu()),
            }
            history.append(row)
            if run is not None:
                wandb.log({f"train/{key}": value for key, value in row.items() if key != "step"} | {"step": step})

        if eval_every > 0 and step % eval_every == 0:
            rows, _ = run_eval(model, data, values, token_tensors, prototypes, cfg, np.random.default_rng(int(cfg.get("seed", 42)) + 400000 + step), device, False)
            mean_ot = float(np.mean([row["sctime_ot"] for row in rows]))
            eval_row = {"step": step, "eval_mean_ot": mean_ot, "eval_mode": str(cfg.get("eval_mode", "all_cells"))}
            eval_history.append(eval_row)
            if run is not None:
                wandb.log({"eval/mean_ot": mean_ot, "step": step})
            pd.DataFrame(rows).to_csv(out_dir / f"metrics_by_time_step{step}.csv", index=False)
            top_path = out_dir / f"top_step{step}_eval{mean_ot:.6f}.pt"
            torch.save({"model": model.state_dict(), "config": cfg, "step": step, "eval_mean_ot": mean_ot, "genes": data.genes, "vocab": vocab}, top_path)
            top_checkpoints.append({"step": step, "eval_mean_ot": mean_ot, "path": str(top_path)})
            top_checkpoints = sorted(top_checkpoints, key=lambda item: (item["eval_mean_ot"], item["step"]))[: int(cfg.get("top_k_checkpoints", 1))]
            if mean_ot < best:
                best = mean_ot
                bad = 0
                torch.save({"model": model.state_dict(), "config": cfg, "best_mean_ot": best, "genes": data.genes, "vocab": vocab}, out_dir / "best.pt")
            elif step >= early_min_step:
                bad += 1
                if patience > 0 and bad >= patience:
                    break

    torch.save({"model": model.state_dict(), "config": cfg, "genes": data.genes, "vocab": vocab}, out_dir / "last.pt")
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    pd.DataFrame(eval_history).to_csv(out_dir / "eval_history.csv", index=False)
    pd.DataFrame(top_checkpoints).to_csv(out_dir / "top_checkpoints_by_train_eval.csv", index=False)

    best_path = top_checkpoints[0]["path"] if top_checkpoints else str(out_dir / "best.pt")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    load_model_state_flexible(model, checkpoint["model"], strict=False)
    rows, pred_by_day = run_eval(model, data, values, token_tensors, prototypes, cfg, np.random.default_rng(int(cfg.get("seed", 42)) + 500000), device, True)
    pd.DataFrame(rows).to_csv(out_dir / "metrics_by_time.csv", index=False)
    np.savez_compressed(out_dir / "predictions_by_day.npz", **{f"pred_E{day}": arr for day, arr in pred_by_day.items()})
    save_json({"best_mean_ot": float(np.mean([row["sctime_ot"] for row in rows])), "checkpoint_path": best_path, "n_steps": step}, out_dir / "summary.json")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
