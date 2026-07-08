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
        aggregation_mode: str = "day_slots",
        decoder_mode: str = "global_query",
        fusion_mode: str = "single",
        local_context_k: int = 2,
        local_strategy: str = "nearest",
        local_gate_init: float = 0.5,
        dropout: float = 0.05,
        encode_chunk_size: int = 8,
        mean_correction_max: float = 3.0,
        distance_penalty: float = 0.5,
        output_activation: str = "softplus",
        sparse_activation_threshold: float = -2.0,
        sparse_activation_temp: float = 0.5,
    ):
        super().__init__()
        self.encoder = TemporalFoundationEncoder(vocab_size=vocab_size, padding_idx=pad_id)
        self.encode_chunk_size = int(encode_chunk_size)
        self.hidden_dim = int(hidden_dim)
        self.base_pred_cells = int(pred_cells)
        self.query_groups = max(1, int(query_groups))
        self.pred_cells = self.base_pred_cells * self.query_groups
        self.day_slots = int(day_slots)
        self.aggregation_mode = str(aggregation_mode)
        self.decoder_mode = str(decoder_mode)
        self.fusion_mode = str(fusion_mode)
        self.local_context_k = max(1, int(local_context_k))
        self.local_context_strategy = str(local_strategy)
        self.local_gate_init = float(local_gate_init)
        self.mean_correction_max = float(mean_correction_max)
        self.distance_penalty = float(distance_penalty)
        self.output_activation = str(output_activation)
        self.sparse_activation_threshold = float(sparse_activation_threshold)
        self.sparse_activation_temp = max(float(sparse_activation_temp), 1e-4)
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
        self.output_query = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.memory_key = nn.Sequential(nn.LayerNorm(hidden_dim + time_dim), nn.Linear(hidden_dim + time_dim, hidden_dim), nn.GELU())
        self.memory_value = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.memory_fuse = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.decoder = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, n_genes))
        self.cell_decoder = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_genes))
        self.snapshot_stat_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, n_genes * 2))
        self.mean_corrector = nn.Sequential(nn.LayerNorm(hidden_dim + time_dim), nn.Linear(hidden_dim + time_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_genes))
        for module, bias in [(self.decoder, -4.0), (self.cell_decoder, -4.0)]:
            nn.init.zeros_(module[-1].weight)
            nn.init.constant_(module[-1].bias, bias)
        nn.init.zeros_(self.mean_corrector[-1].weight)
        nn.init.zeros_(self.mean_corrector[-1].bias)
        nn.init.zeros_(self.local_global_gate[-1].weight)
        gate_init = float(np.clip(self.local_gate_init, 1e-4, 1.0 - 1e-4))
        nn.init.constant_(self.local_global_gate[-1].bias, float(np.log(gate_init / (1.0 - gate_init))))

    def encode_conditioned(self, idx, val, total, source_day, target_day):
        if idx.shape[0] <= self.encode_chunk_size:
            return self.encoder(idx, val, total, source_day, target_day)
        chunks = []
        for start in range(0, idx.shape[0], self.encode_chunk_size):
            stop = min(start + self.encode_chunk_size, idx.shape[0])
            chunks.append(self.encoder(idx[start:stop], val[start:stop], total[start:stop], source_day[start:stop], target_day[start:stop]))
        return torch.cat(chunks, dim=0)

    def aggregate_days(self, idx, val, total, context_days, target_day):
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
            cell_weights = torch.softmax(scores, dim=2)
            day_hidden = (cell_hidden * cell_weights.unsqueeze(-1)).sum(dim=2)
            return day_hidden, day_hidden.unsqueeze(2), cell_hidden, cell_weights.unsqueeze(2), cell_latent
        slot_keys = self.day_slot_key(torch.cat([cell_hidden, tf], dim=-1))
        slot_scores = torch.einsum("sd,btcd->btsc", self.day_slot_queries, slot_keys) / (self.hidden_dim**0.5)
        base_scores = self.cell_score(torch.cat([cell_hidden, tf], dim=-1)).squeeze(-1).unsqueeze(2)
        cell_weights = torch.softmax(slot_scores + base_scores, dim=-1)
        day_slots = torch.einsum("btsc,btcd->btsd", cell_weights, cell_hidden)
        return day_slots.mean(dim=2), day_slots, cell_hidden, cell_weights, cell_latent

    def local_context_mask(self, context_days: torch.Tensor, target_day: torch.Tensor) -> torch.Tensor:
        dist = (context_days - target_day[:, None]).abs()
        bsz, n_days = context_days.shape
        mask = torch.zeros_like(context_days, dtype=torch.bool)
        if self.local_context_strategy == "balanced":
            for b in range(bsz):
                selected_parts = []
                left = torch.nonzero(context_days[b] < target_day[b], as_tuple=False).flatten()
                right = torch.nonzero(context_days[b] > target_day[b], as_tuple=False).flatten()
                for part in (left, right):
                    if part.numel() > 0:
                        part_dist = (context_days[b, part] - target_day[b]).abs()
                        selected_parts.append(part[torch.topk(-part_dist, k=min(self.local_context_k, part.numel())).indices])
                selected = torch.cat(selected_parts).unique() if selected_parts else torch.empty(0, dtype=torch.long, device=context_days.device)
                if selected.numel() == 0:
                    selected = torch.topk(-dist[b], k=min(self.local_context_k, n_days)).indices
                mask[b, selected] = True
            return mask
        for b in range(bsz):
            candidates = torch.arange(n_days, device=context_days.device)
            if self.local_context_strategy == "past":
                past = candidates[context_days[b] < target_day[b]]
                if past.numel() > 0:
                    candidates = past
            selected = candidates[torch.topk(-dist[b, candidates], k=min(self.local_context_k, candidates.numel())).indices]
            mask[b, selected] = True
        return mask

    def target_representation(self, day_hidden, day_slots, cell_hidden, context_days, target_day, day_mask=None):
        if self.aggregation_mode == "cell_cross":
            bsz, n_days, n_cells, _ = cell_hidden.shape
            query = self.target_query(time_features(target_day, target_day)).unsqueeze(1)
            hidden = cell_hidden.reshape(bsz, n_days * n_cells, self.hidden_dim)
            flat_days = context_days[:, :, None].expand(bsz, n_days, n_cells).reshape(bsz, n_days * n_cells)
            tf = time_features(flat_days, target_day[:, None].expand_as(flat_days))
            keys = self.day_key(torch.cat([hidden, tf], dim=-1))
            values = self.day_value(hidden)
            scores = (query * keys).sum(-1) / (self.hidden_dim**0.5)
            scores = scores - (flat_days - target_day[:, None]).abs() * self.distance_penalty
            if day_mask is not None:
                flat_mask = day_mask[:, :, None].expand(bsz, n_days, n_cells).reshape(bsz, n_days * n_cells)
                scores = scores.masked_fill(~flat_mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
            pooled = (values * weights.unsqueeze(-1)).sum(dim=1)
            z = self.target_fuse(torch.cat([pooled, query.squeeze(1)], dim=-1))
            return z, weights.reshape(bsz, n_days, n_cells).sum(dim=-1)
        bsz, n_days, n_slots, _ = day_slots.shape
        query = self.target_query(time_features(target_day, target_day)).unsqueeze(1)
        hidden = day_slots.reshape(bsz, n_days * n_slots, self.hidden_dim)
        slot_days = context_days[:, :, None].expand(bsz, n_days, n_slots).reshape(bsz, n_days * n_slots)
        tf = time_features(slot_days, target_day[:, None].expand_as(slot_days))
        keys = self.day_key(torch.cat([hidden, tf], dim=-1))
        values = self.day_value(hidden)
        scores = (query * keys).sum(-1) / (self.hidden_dim**0.5)
        scores = scores - (slot_days - target_day[:, None]).abs() * self.distance_penalty
        if day_mask is not None:
            slot_mask = day_mask[:, :, None].expand(bsz, n_days, n_slots).reshape(bsz, n_days * n_slots)
            scores = scores.masked_fill(~slot_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = (values * weights.unsqueeze(-1)).sum(dim=1)
        z = self.target_fuse(torch.cat([pooled, query.squeeze(1)], dim=-1))
        return z, weights.reshape(bsz, n_days, n_slots).sum(dim=-1)

    def target_representation_fused(self, day_hidden, day_slots, cell_hidden, context_days, target_day):
        z_global, weights_global = self.target_representation(day_hidden, day_slots, cell_hidden, context_days, target_day)
        if self.fusion_mode not in {"local_global_gate", "local_global_sum"} or context_days.shape[1] <= 1:
            return z_global, weights_global
        local_mask = self.local_context_mask(context_days, target_day)
        z_local, weights_local = self.target_representation(day_hidden, day_slots, cell_hidden, context_days, target_day, day_mask=local_mask)
        if self.fusion_mode == "local_global_sum":
            return z_global + z_local, weights_global
        gate = torch.sigmoid(self.local_global_gate(torch.cat([z_global, z_local, time_features(target_day, target_day)], dim=-1)))
        return (1.0 - gate) * z_global + gate * z_local, (1.0 - gate) * weights_global + gate * weights_local

    def expanded_cell_queries(self, bsz: int) -> torch.Tensor:
        query = self.cell_queries + self.query_group_embed[:, None, :]
        return query.reshape(self.pred_cells, self.hidden_dim).unsqueeze(0).expand(bsz, -1, -1)

    def activate_output(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "relu":
            return F.relu(logits)
        if self.output_activation == "sparse_softplus":
            gate = torch.sigmoid((logits - self.sparse_activation_threshold) / self.sparse_activation_temp)
            return F.softplus(logits) * gate
        return F.softplus(logits)

    def decode_cells(self, z, decoder):
        query = self.expanded_cell_queries(z.shape[0])
        z_expand = z[:, None, :].expand(-1, self.pred_cells, -1)
        return self.activate_output(decoder(torch.cat([z_expand, query], dim=-1)))

    def predict_snapshot_stats(self, day_hidden):
        out = self.snapshot_stat_head(day_hidden)
        mean, raw_std = out.chunk(2, dim=-1)
        return self.activate_output(mean), F.softplus(raw_std)

    def decode_cells_from_memory(self, z, cell_hidden, context_days, target_day, decoder):
        bsz, n_days, n_cells, _ = cell_hidden.shape
        base_query = self.expanded_cell_queries(bsz)
        z_expand = z[:, None, :].expand(-1, self.pred_cells, -1)
        query = self.output_query(torch.cat([z_expand, base_query], dim=-1))
        hidden = cell_hidden.reshape(bsz, n_days * n_cells, self.hidden_dim)
        flat_days = context_days[:, :, None].expand(bsz, n_days, n_cells).reshape(bsz, n_days * n_cells)
        tf = time_features(flat_days, target_day[:, None].expand_as(flat_days))
        keys = self.memory_key(torch.cat([hidden, tf], dim=-1))
        values = self.memory_value(hidden)
        scores = torch.einsum("bph,bmh->bpm", query, keys) / (self.hidden_dim**0.5)
        scores = scores - (flat_days[:, None, :] - target_day[:, None, None]).abs() * self.distance_penalty
        weights = torch.softmax(scores, dim=-1)
        memory = torch.einsum("bpm,bmh->bph", weights, values)
        out_hidden = self.memory_fuse(torch.cat([z_expand, base_query, memory], dim=-1))
        return self.activate_output(decoder(torch.cat([out_hidden, base_query], dim=-1)))

    def apply_mean_correction(self, pred, z, target_day, scale: float):
        if scale == 0.0:
            return pred, pred.new_zeros((pred.shape[0], pred.shape[-1]))
        correction = self.mean_corrector(torch.cat([z, time_features(target_day, target_day)], dim=-1))
        if self.mean_correction_max > 0:
            correction = self.mean_correction_max * torch.tanh(correction / self.mean_correction_max)
        return (pred + correction[:, None, :] * float(scale)).clamp_min(0.0), correction

    def forward(self, idx, val, total, context_days, target_day, mean_correction_scale: float = 0.0, return_raw: bool = False):
        day_hidden, day_slots, cell_hidden, cell_weights, cell_latent = self.aggregate_days(idx, val, total, context_days, target_day)
        z, day_weights = self.target_representation_fused(day_hidden, day_slots, cell_hidden, context_days, target_day)
        if self.decoder_mode == "cell_memory":
            pred_raw = self.decode_cells_from_memory(z, cell_hidden, context_days, target_day, self.decoder)
        else:
            pred_raw = self.decode_cells(z, self.decoder)
        pred, mean_correction = self.apply_mean_correction(pred_raw, z, target_day, mean_correction_scale)
        recon_stack = None
        if return_raw:
            return pred, recon_stack, z, day_hidden, day_weights, cell_weights, cell_latent, pred_raw, mean_correction
        return pred, recon_stack, z, day_hidden, day_weights, cell_weights, cell_latent

    def reconstruct_masked_cells_from_latent(self, cell_latent, gene_idx, gene_val, vocab_to_column, pad_id, mask):
        flat_latent = cell_latent.reshape(-1, cell_latent.shape[-1])
        flat_idx = gene_idx.reshape(-1, gene_idx.shape[-1])
        flat_val = gene_val.reshape(-1, gene_val.shape[-1])
        flat_mask = mask.reshape(-1, mask.shape[-1])
        if not flat_mask.any():
            return flat_val.sum() * 0.0, flat_mask.float().mean()
        col_idx = vocab_to_column[flat_idx.clamp(0, vocab_to_column.numel() - 1)]
        pred = self.activate_output(self.cell_decoder(flat_latent))
        pred_values = pred.gather(1, col_idx.clamp_min(0))
        return F.smooth_l1_loss(pred_values[flat_mask], flat_val[flat_mask]), flat_mask.float().mean()

    @torch.no_grad()
    def predict(self, idx, val, total, context_days, target_day, mean_correction_scale: float = 0.0):
        pred, _recon, _z, _day_hidden, day_weights, _cell_weights, _cell_latent = self.forward(idx, val, total, context_days, target_day, mean_correction_scale)
        return pred, day_weights


def load_model_state_flexible(model: nn.Module, state: dict, strict: bool = False):
    state = dict(state)
    if "cell_queries" in state:
        source = state["cell_queries"]
        target = model.state_dict().get("cell_queries")
        if target is not None and source.ndim == 2 and target.ndim == 3:
            state["cell_queries"] = source.unsqueeze(0)
    if "query_group_embed" not in state and hasattr(model, "query_group_embed"):
        state["query_group_embed"] = torch.zeros_like(model.query_group_embed)
    return model.load_state_dict(state, strict=strict)
