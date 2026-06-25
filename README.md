# scChronos

scChronos is a reference-conditioned Transformer framework for temporal single-cell transcriptome recovery and forecasting. It adapts a pretrained gene-token single-cell encoder to predict missing or future transcriptomic snapshots from observed reference time points.

## Overview

The model has three main components:

1. A trainable gene-token foundation encoder that consumes gene IDs and expression values for each cell.
2. A temporal context memory that summarizes observed reference snapshots and assigns target-specific reference attention.
3. A local-global fusion decoder that predicts the target snapshot expression distribution.

The repository contains code only. Processed datasets and pretrained checkpoints should be placed under `Data/` following the layout below.

```text
Data/
  mouse/
    Schiebinger2019/
      hvg1000/
        remove_recovery-norm_data-hvg.csv
        remove_recovery-meta_data.csv
        remove_recovery-var_genes_list.csv
        three_forecasting-norm_data-hvg.csv
        three_forecasting-meta_data.csv
        three_forecasting-var_genes_list.csv
    ECED_Kidney/
      hvg1000/
        ...
  human/
    Veres2019/
      hvg1000/
        ...
    Cao2020/
      hvg1000/
        ...
  pretrained/
    mouse/
      checkpoint-132.pth
      mouse_gene_vocab.json
    human/
      scPISR-pretrain-checkpoint-99.pth
      updated_gene_vocab.json
```

Each task directory uses the scMix-style file names:

```text
<task>-norm_data-hvg.csv
<task>-meta_data.csv
<task>-var_genes_list.csv
```

Supported tasks are `remove_recovery`, `two_forecasting`, and `three_forecasting`.

## Installation

```bash
conda create -n scchronos python=3.11 -y
conda activate scchronos
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA driver before running large jobs if needed.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_scchronos.py \
  --config configs/schiebinger2019_recovery_hvg1000.yaml \
  --wandb
```

The output directory stores:

```text
best.pt
last.pt
run_metadata.json
train_history.csv
summary.json
eval_epoch_*.json
```

## Predict

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict_scchronos.py \
  --config configs/schiebinger2019_recovery_hvg1000.yaml \
  --checkpoint runs/schiebinger2019_recovery_hvg1000/best.pt \
  --output-dir outputs/schiebinger2019_recovery_hvg1000
```

The prediction script writes per-day predictions and attention weights to `predictions_by_day.npz` and a summary file to `run_summary.json`.

## Data package check

```bash
python scripts/check_data_package.py --data-root Data
```

This creates `Data/data_package_check.csv` with dataset, task, cell count and feature-list checks.

## Notes

The vocabulary loader reads the actual `<pad>` and `<unk>` IDs from the JSON file. If either token is absent, it appends a missing special token without assuming that `<unk>` is ID 1. When `extend_vocab: true`, dataset genes absent from the pretrained vocabulary are appended and initialized from the pretrained gene embedding mean during checkpoint loading.
