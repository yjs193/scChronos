from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    return parser.parse_args()


def main() -> None:
    root = Path(parse_args().data_root)
    rows = []
    for data_dir in sorted(root.glob("*/*/hvg*/")):
        for task in ["remove_recovery", "two_forecasting", "three_forecasting"]:
            expr = data_dir / f"{task}-norm_data-hvg.csv"
            meta = data_dir / f"{task}-meta_data.csv"
            genes = data_dir / f"{task}-var_genes_list.csv"
            if expr.exists() and meta.exists():
                shape = pd.read_csv(expr, index_col=0, nrows=5).shape
                full_meta = pd.read_csv(meta, index_col=0)
                rows.append({"dataset": data_dir.parts[-2], "hvg": data_dir.name, "task": task, "cells": len(full_meta), "genes_preview": shape[1], "has_gene_list": genes.exists()})
    pd.DataFrame(rows).to_csv(root / "data_package_check.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
