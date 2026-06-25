from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .foundation import TemporalFoundationEncoder, time_features


class ScChronos(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        n_genes: int,
        hidden_dim: int = 256,
        pred_cells: int = 128,
        query_groups: int = 1,
        day_slots: int = 1,
        local_context_k: int = 2,
        local_strategy: str = "nearest",
        local_gate_init: float = 0.5,
        fusion_mode: str = "local_global_residual",
        local_residual_scale: float = 1.2,
        local_residual_max: float = 2.5,
        local_residual_interp_only: bool = True,
        local_residual_input: str = "local",
        dropout: float = 0.05,
        encode_chunk_size: int = 8,
        output_activation: str = "softplus",
    ):
        super().__init__()
        self.encoder = TemporalFoundationEncoder(vocab_size=vocab_size, padding_idx=pad_id)
        self.encode_chunk_size = int(encode_chunk_size)
        self.hidden_dim = int(hidden_dim)
        self.base_pred_cells = int(pred_cells)
        self.query_groups = int(max(1, query_groups))
        self.pred_cells = self.base_pred_cells * self.query_groups
        self.day_slots = int(max(1, day_slots))
        self.local_context_k = int(max(1, local_context_k))
        self.local_strategy = str(local_strategy)
        self.fusion_mode = str(fusion_mode)
        self.local_residual_scale = float(local_residual_scale)
        self.local_residual_max = float(local_residual_max)
        self.local_residual_interp_only = bool(local_residual_interp_only)
        self.local_residual_input = str(local_residual_input)
        self.output_activation = str(output_activation)
        enc_dim = self.encoder.output_dim
        time_dim = 3 * (2 * 8 + 1)
        self.cell_projector = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, hidden_dim), nn.GELU())
        self.cell_score = nn.Sequential(nn.LayerNorm(hidden_dim + time_dim), nn.Linear(hidden_dim + time_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.day_slot_queries = nn.Parameter(torch.randn(self.day_slots, hidden_dim) * 0.02)
        self.day_slot_key = nn.Sequential(nn.LayerNorm(hidden_dim + time_dim), nn.Linear(hidden_dim + time_dim, hidden_dim), nn.GELU())
        self.target_query = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.day_key = nn.Sequential(nn.LayerNorm(hidden_dim + time_dim), nn.Linear(hidden_dim + time_dim, hidden_dim), nn.GELU())
        self.day_value = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.target_fuse = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.local_global_gate = nn.Sequential(nn.LayerNorm(hidden_dim * 2 + time_dim), nn.Linear(hidden_dim * 2 + time_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.cell_queries = nn.Parameter(torch.randn(self.query_groups, self.base_pred_cells, hidden_dim) * 0.02)
        self.query_group_embed = nn.Parameter(torch.randn(self.query_groups, hidden_dim) * 0.02)
        self.memory_key = nn.Sequential(nn.LayerNorm(hidden_dim + time_dim), nn.Linear(hidden_dim + time_dim, hidden_dim), nn.GELU())
        self.memory_value = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.memory_fuse = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.decoder = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, n_genes))
        self.local_residual = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, n_genes))
        self.cell_decoder = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_genes))
        self.recon_decoder = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, n_genes))
        self.mean_corrector = nn.Sequential(nn.LayerNorm(hidden_dim + time_dim), nn.Linear(hidden_dim + time_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_genes))
        for module in [self.decoder, self.local_residual, self.cell_decoder, self.recon_decoder]:
            nn.init.zeros_(module[-1].weight)
            nn.init.constant_(module[-1].bias, -4.0)
        nn.init.zeros_(self.mean_corrector[-1].weight)
        nn.init.zeros_(self.mean_corrector[-1].bias)
        nn.init.zeros_(self.local_global_gate[-1].weight)
        gate = float(np.clip(local_gate_init, 1e-4, 1.0 - 1e-4))
        nn.init.constant_(self.local_global_gate[-1].bias, float(np.log(gate / (1.0 - gate))))

    def encode_conditioned(self, idx: torch.Tensor, val: torch.Tensor, total: torch.Tensor, source_day: torch.Tensor, target_day: torch.Tensor) -> torch.Tensor:
        if idx.shape[0] <= self.encode_chunk_size:
            return self.encoder(idx, val, total, source_day, target_day)
        chunks = []
        for start in range(0, idx.shape[0], self.encode_chunk_size):
            stop = min(start + self.encode_chunk_size, idx.shape[0])
            chunks.append(self.encoder(idx[start:stop], val[start:stop], total[start:stop], source_day[start:stop], target_day[start:stop]))
        return torch.cat(chunks, dim=0)

    def aggregate_days(self, idx: torch.Tensor, val: torch.Tensor, total: torch.Tensor, context_days: torch.Tensor, target_day: torch.Tensor):
        bsz, n_days, n_cells, seq_len = idx.shape
        flat_idx = idx.reshape(bsz * n_days * n_cells, seq_len)
        flat_val = val.reshape(bsz * n_days * n_cells, seq_len)
        flat_total = total.reshape(bsz * n_days * n_cells)
        source = context_days[:, :, None].expand(bsz, n_days, n_cells).reshape(-1)
        target = target_day[:, None, None].expand(bsz, n_days, n_cells).reshape(-1)
        latent = self.encode_conditioned(flat_idx, flat_val, flat_total, source, target)
        cell_latent = latent.reshape(bsz, n_days, n_cells, -1)
        cell_hidden = self.cell_projector(latent).reshape(bsz, n_days, n_cells, self.hidden_dim)
        tf = time_features(context_days[:, :, None].expand(bsz, n_days, n_cells), target_day[:, None, None].expand(bsz, n_days, n_cells))
        if self.day_slots == 1:
            scores = self.cell_score(torch.cat([cell_hidden, tf], dim=-1)).squeeze(-1)
            weights = torch.softmax(scores, dim=2)
            day_hidden = (cell_hidden * weights.unsqueeze(-1)).sum(dim=2)
            return day_hidden, day_hidden.unsqueeze(2), cell_hidden, weights.unsqueeze(2), cell_latent
        keys = self.day_slot_key(torch.cat([cell_hidden, tf], dim=-1))
        slot_scores = torch.einsum("sd,btcd->btsc", self.day_slot_queries, keys) / (self.hidden_dim ** 0.5)
        base_scores = self.cell_score(torch.cat([cell_hidden, tf], dim=-1)).squeeze(-1).unsqueeze(2)
        weights = torch.softmax(slot_scores + base_scores, dim=-1)
        slots = torch.einsum("btsc,btcd->btsd", weights, cell_hidden)
        return slots.mean(dim=2), slots, cell_hidden, weights, cell_latent

    def local_mask(self, context_days: torch.Tensor, target_day: torch.Tensor) -> torch.Tensor:
        distance = (context_days - target_day[:, None]).abs()
        mask = torch.zeros_like(context_days, dtype=torch.bool)
        for row in range(context_days.shape[0]):
            candidates = torch.arange(context_days.shape[1], device=context_days.device)
            if self.local_strategy == "past":
                candidates = candidates[context_days[row] < target_day[row]]
            if candidates.numel() == 0:
                candidates = torch.arange(context_days.shape[1], device=context_days.device)
            selected = candidates[torch.topk(-distance[row, candidates], k=min(self.local_context_k, candidates.numel())).indices]
            mask[row, selected] = True
        return mask

    def interpolation_mask(self, context_days: torch.Tensor, target_day: torch.Tensor) -> torch.Tensor:
        left = (context_days < target_day[:, None]).any(dim=1)
        right = (context_days > target_day[:, None]).any(dim=1)
        return (left & right).float()

    def decode_activation(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "relu":
            return F.relu(logits)
        if self.output_activation == "exp":
            return torch.exp(torch.clamp(logits, max=8.0))
        return F.softplus(logits)

    def forward(self, idx: torch.Tensor, val: torch.Tensor, total: torch.Tensor, context_days: torch.Tensor, target_day: torch.Tensor):
        bsz, n_days, _, _ = idx.shape
        day_hidden, day_slots, _, _, cell_latent = self.aggregate_days(idx, val, total, context_days, target_day)
        tf_day = time_features(context_days, target_day[:, None].expand_as(context_days))
        query = self.target_query(time_features(target_day, target_day))
        keys = self.day_key(torch.cat([day_hidden, tf_day], dim=-1))
        values = self.day_value(day_hidden)
        attn = torch.softmax(torch.einsum("bd,btd->bt", query, keys) / (self.hidden_dim ** 0.5), dim=-1)
        global_context = torch.einsum("bt,btd->bd", attn, values)
        local_mask = self.local_mask(context_days, target_day)
        masked_attn = torch.where(local_mask, attn, torch.zeros_like(attn))
        masked_attn = masked_attn / masked_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        local_context = torch.einsum("bt,btd->bd", masked_attn, values)
        gate_input = torch.cat([global_context, local_context, time_features(target_day, target_day)], dim=-1)
        gate = torch.sigmoid(self.local_global_gate(gate_input))
        if self.fusion_mode == "global":
            context = global_context
        elif self.fusion_mode == "local":
            context = local_context
        else:
            context = gate * local_context + (1.0 - gate) * global_context
        target_hidden = self.target_fuse(torch.cat([context, query], dim=-1))
        outputs = []
        for group in range(self.query_groups):
            q = self.cell_queries[group].unsqueeze(0).expand(bsz, -1, -1) + self.query_group_embed[group].view(1, 1, -1)
            mem_keys = self.memory_key(torch.cat([day_slots.reshape(bsz, n_days * self.day_slots, -1), tf_day[:, :, None, :].expand(-1, -1, self.day_slots, -1).reshape(bsz, n_days * self.day_slots, -1)], dim=-1))
            mem_values = self.memory_value(day_slots.reshape(bsz, n_days * self.day_slots, -1))
            mem_attn = torch.softmax(torch.einsum("bqd,bkd->bqk", q, mem_keys) / (self.hidden_dim ** 0.5), dim=-1)
            mem = torch.einsum("bqk,bkd->bqd", mem_attn, mem_values)
            fused = self.memory_fuse(torch.cat([q, mem, target_hidden.unsqueeze(1).expand_as(q)], dim=-1))
            outputs.append(fused)
        cell_hidden = torch.cat(outputs, dim=1)
        global_pred = self.decoder(torch.cat([cell_hidden, target_hidden.unsqueeze(1).expand_as(cell_hidden)], dim=-1))
        if self.fusion_mode == "local_global_residual":
            residual_context = local_context if self.local_residual_input == "local" else context
            local_pred = self.local_residual(torch.cat([cell_hidden, residual_context.unsqueeze(1).expand_as(cell_hidden)], dim=-1))
            if self.local_residual_max > 0:
                local_pred = self.local_residual_max * torch.tanh(local_pred / self.local_residual_max)
            if self.local_residual_interp_only:
                active = self.interpolation_mask(context_days, target_day).view(bsz, 1, 1)
                local_pred = local_pred * active
            pred_logits = global_pred + self.local_residual_scale * local_pred
        else:
            pred_logits = global_pred
        pred = self.decode_activation(pred_logits)
        context_pred = self.decode_activation(self.recon_decoder(torch.cat([day_hidden, context.unsqueeze(1).expand_as(day_hidden)], dim=-1)))
        cell_recon = self.decode_activation(self.cell_decoder(cell_latent))
        mean_delta = self.mean_corrector(torch.cat([target_hidden, time_features(target_day, target_day)], dim=-1))
        pred = pred + torch.tanh(mean_delta).unsqueeze(1)
        return {"pred": pred.clamp_min(0.0), "context_pred": context_pred, "cell_recon": cell_recon, "attention": attn, "local_gate": gate.squeeze(-1)}
