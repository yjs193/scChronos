# Data package

The release data package should be stored outside the Git repository and copied or uploaded separately.

Required datasets:

| Dataset | Species | Directory |
| --- | --- | --- |
| Schiebinger2019 | mouse | `Data/mouse/Schiebinger2019` |
| ECED Kidney | mouse | `Data/mouse/ECED_Kidney` |
| Veres2019 | human | `Data/human/Veres2019` |
| Cao2020 | human | `Data/human/Cao2020` |

Each dataset may contain `hvg500`, `hvg1000`, and `hvg2000` subdirectories. Each HVG directory should contain one recovery task and one forecasting task.

Recovery files:

```text
remove_recovery-norm_data-hvg.csv
remove_recovery-meta_data.csv
remove_recovery-var_genes_list.csv
remove_recovery-day_counts.csv
```

Forecasting files:

```text
two_forecasting-norm_data-hvg.csv
two_forecasting-meta_data.csv
two_forecasting-var_genes_list.csv
two_forecasting-day_counts.csv
```

or

```text
three_forecasting-norm_data-hvg.csv
three_forecasting-meta_data.csv
three_forecasting-var_genes_list.csv
three_forecasting-day_counts.csv
```

The pretrained directory should contain:

```text
Data/pretrained/mouse/checkpoint-132.pth
Data/pretrained/mouse/mouse_gene_vocab.json
Data/pretrained/human/scPISR-pretrain-checkpoint-99.pth
Data/pretrained/human/updated_gene_vocab.json
```

Use `scripts/check_data_package.py` to verify that expected files are visible.

