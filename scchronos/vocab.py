from __future__ import annotations

import json
from pathlib import Path


def load_vocab(path: str | Path, add_missing_special: bool = True) -> tuple[dict[str, int], int, int]:
    with open(path, "r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    vocab = {str(key): int(value) for key, value in vocab.items()}
    if add_missing_special:
        vocab = ensure_special_tokens(vocab)
    pad_id = int(vocab["<pad>"])
    unk_id = int(vocab["<unk>"])
    return vocab, pad_id, unk_id


def ensure_special_tokens(vocab: dict[str, int]) -> dict[str, int]:
    vocab = dict(vocab)
    used = set(vocab.values())
    next_id = max(used) + 1 if used else 0
    if "<pad>" not in vocab:
        if 0 not in used:
            vocab["<pad>"] = 0
            used.add(0)
        else:
            vocab["<pad>"] = next_id
            used.add(next_id)
            next_id += 1
    if "<unk>" not in vocab:
        while next_id in used:
            next_id += 1
        vocab["<unk>"] = next_id
    return vocab


def save_vocab(vocab: dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(sorted(vocab.items(), key=lambda item: item[1])), handle, indent=2)


def extend_vocab_with_genes(vocab: dict[str, int], genes: list[str]) -> dict[str, int]:
    vocab = ensure_special_tokens(vocab)
    next_id = max(vocab.values()) + 1
    for gene in genes:
        gene = str(gene)
        if gene not in vocab:
            vocab[gene] = next_id
            next_id += 1
    return vocab

