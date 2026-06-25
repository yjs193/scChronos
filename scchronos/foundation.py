from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def time_features(source_day: torch.Tensor, target_day: torch.Tensor, bands: int = 8) -> torch.Tensor:
    values = torch.stack([source_day, target_day, target_day - source_day], dim=-1)
    freqs = 2.0 ** torch.arange(bands, dtype=values.dtype, device=values.device)
    angles = values.unsqueeze(-1) * freqs * math.pi
    return torch.cat([values.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2)


class SDPABlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, length, dim = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(bsz, length, 3, self.heads, dim // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).contiguous().unbind(0)
        mask = key_padding_mask[:, None, None, :].to(torch.bool) if key_padding_mask is not None else None
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=self.drop.p if self.training else 0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, length, dim)
        x = x + self.drop(self.proj(out))
        return x + self.mlp(self.norm2(x))


class TemporalAdapter(nn.Module):
    def __init__(self, dim: int, cond_dim: int, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.gate = nn.Linear(cond_dim, dim)
        self.shift = nn.Linear(cond_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        update = self.up(F.gelu(self.down(x)))
        gate = torch.sigmoid(self.gate(cond)).unsqueeze(1)
        shift = self.shift(cond).unsqueeze(1)
        return x + gate * update + shift


class TemporalFoundationEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        embed_dim: int = 128,
        depth: int = 12,
        heads: int = 8,
        cond_dim: int = 128,
        prompts: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.prompts = int(prompts)
        self.gene_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.value_embed = nn.Linear(1, embed_dim)
        self.total_embed = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([SDPABlock(embed_dim, heads, dropout) for _ in range(depth)])
        self.adapters = nn.ModuleList([TemporalAdapter(embed_dim, cond_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_dim = embed_dim * 2
        time_dim = 3 * (2 * 8 + 1)
        self.cond = nn.Sequential(nn.Linear(time_dim, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim), nn.LayerNorm(cond_dim))
        self.time_token = nn.Sequential(nn.Linear(time_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.prompt = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.GELU(), nn.Linear(cond_dim, prompts * embed_dim)) if prompts > 0 else None

    def forward(self, gene_idx: torch.Tensor, gene_val: torch.Tensor, total: torch.Tensor, source_day: torch.Tensor, target_day: torch.Tensor) -> torch.Tensor:
        if total.ndim == 2:
            total = total.squeeze(-1)
        tf = time_features(source_day, target_day)
        cond = self.cond(tf)
        gene_tokens = self.gene_embed(gene_idx) + self.value_embed(gene_val.unsqueeze(-1))
        cls = self.cls_token + self.total_embed(total.unsqueeze(-1)).unsqueeze(1)
        cls = cls.expand(gene_idx.shape[0], -1, -1)
        time_token = self.time_token(tf).unsqueeze(1)
        parts = [cls, time_token]
        if self.prompt is not None:
            parts.append(self.prompt(cond).reshape(gene_idx.shape[0], self.prompts, gene_tokens.shape[-1]))
        parts.append(gene_tokens)
        x = torch.cat(parts, dim=1)
        valid = gene_idx != self.padding_idx
        prefix = torch.ones(gene_idx.shape[0], x.shape[1] - gene_idx.shape[1], dtype=torch.bool, device=gene_idx.device)
        key_padding_mask = ~torch.cat([prefix, valid], dim=1)
        for block, adapter in zip(self.blocks, self.adapters):
            x = adapter(block(x, key_padding_mask), cond)
        x = self.norm(x)
        cls_feature = x[:, 0]
        gene_out = x[:, -gene_idx.shape[1] :]
        weights = valid.float().unsqueeze(-1)
        gene_feature = (gene_out * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return torch.cat([cls_feature, gene_feature], dim=-1)


def load_pretrained_encoder(model: nn.Module, checkpoint_path: str) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    target = model.state_dict()
    copied = {}
    skipped = 0
    for key, value in source.items():
        if key in target and target[key].shape == value.shape:
            copied[key] = value
        elif key == "gene_embed.weight" and key in target and target[key].shape[1:] == value.shape[1:]:
            merged = target[key].clone()
            n_copy = min(merged.shape[0], value.shape[0])
            merged[:n_copy] = value[:n_copy]
            if merged.shape[0] > n_copy:
                start = 1 if value.shape[0] > 1 else 0
                merged[n_copy:] = value[start:].mean(dim=0, keepdim=True)
            copied[key] = merged
        else:
            skipped += 1
    target.update(copied)
    model.load_state_dict(target, strict=False)
    return {"copied": len(copied), "skipped": skipped}

