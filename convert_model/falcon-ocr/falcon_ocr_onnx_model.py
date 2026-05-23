from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FalconOCRExportConfig:
    dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    n_kv_heads: int
    vocab_size: int
    ffn_dim: int
    norm_eps: float
    max_seq_len: int
    rope_theta: float
    channel_size: int
    spatial_patch_size: int
    temporal_patch_size: int
    img_id: int

    @property
    def image_patch_dim(self) -> int:
        return (
            self.temporal_patch_size
            * self.spatial_patch_size
            * self.spatial_patch_size
            * self.channel_size
        )


def _rms_norm_no_weight(x: torch.Tensor, eps: float = 1.1920928955078125e-07) -> torch.Tensor:
    scale = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * scale


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, seq, heads, dim = x.shape
    return (
        x.unsqueeze(3)
        .expand(batch, seq, heads, n_rep, dim)
        .reshape(batch, seq, heads * n_rep, dim)
    )


def _build_temporal_rope_tables(dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class ExportRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class ExportAttention(nn.Module):
    def __init__(self, config: FalconOCRExportConfig, layer_id: int) -> None:
        super().__init__()
        del layer_id
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.head_dim
        self.q_dim = config.n_heads * config.head_dim
        self.kv_dim = config.n_kv_heads * config.head_dim
        self.scale = 1.0 / math.sqrt(config.head_dim)

        self.wqkv = nn.Linear(config.dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.sinks = nn.Parameter(torch.empty((config.n_heads,)))

    def _pre_attention_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.wqkv(_rms_norm_no_weight(x))
        xq, xk, xv = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        batch, seq, _ = x.shape
        xq = xq.reshape(batch, seq, self.n_heads, self.head_dim)
        xk = xk.reshape(batch, seq, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(batch, seq, self.n_kv_heads, self.head_dim)
        xq = _rms_norm_no_weight(xq)
        xk = _rms_norm_no_weight(xk)
        xk = _repeat_kv(xk, self.n_rep)
        xv = _repeat_kv(xv, self.n_rep)
        return xq, xk, xv

    def _apply_rope(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        spatial_cos: torch.Tensor,
        spatial_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_t, q_hw = xq.chunk(2, dim=-1)
        k_t, k_hw = xk.chunk(2, dim=-1)

        rope_cos = rope_cos.unsqueeze(2)
        rope_sin = rope_sin.unsqueeze(2)
        q_t = _rotate_pairs(q_t, rope_cos, rope_sin)
        k_t = _rotate_pairs(k_t, rope_cos, rope_sin)

        q_hw = _rotate_pairs(q_hw, spatial_cos, spatial_sin)
        k_hw = _rotate_pairs(k_hw, spatial_cos, spatial_sin)
        return torch.cat((q_t, q_hw), dim=-1), torch.cat((k_t, k_hw), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        spatial_cos: torch.Tensor,
        spatial_sin: torch.Tensor,
    ) -> torch.Tensor:
        xq, xk, xv = self._pre_attention_qkv(x)
        xq, xk = self._apply_rope(xq, xk, rope_cos, rope_sin, spatial_cos, spatial_sin)

        q = xq.permute(0, 2, 1, 3)
        k = xk.permute(0, 2, 1, 3)
        v = xv.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask_value = torch.tensor(-3.4028234663852886e38, dtype=scores.dtype, device=scores.device)
        scores = torch.where(attention_mask.unsqueeze(1), scores, mask_value)

        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v)
        lse = torch.logsumexp(scores, dim=-1)
        sink_scale = torch.sigmoid(lse - self.sinks.reshape(1, -1, 1))
        output = output * sink_scale.unsqueeze(-1)

        output = output.permute(0, 2, 1, 3).contiguous().flatten(2)
        return self.wo(output)


class ExportFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _rms_norm_no_weight(x)
        packed = self.w13(x)
        gate = packed[..., 0::2]
        up = packed[..., 1::2]
        return self.w2(torch.relu(gate) * torch.relu(gate) * up)


class ExportTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: FalconOCRExportConfig) -> None:
        super().__init__()
        self.attention = ExportAttention(config, layer_id)
        self.feed_forward = ExportFeedForward(config.dim, config.ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        spatial_cos: torch.Tensor,
        spatial_sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(x, attention_mask, rope_cos, rope_sin, spatial_cos, spatial_sin)
        return x + self.feed_forward(x)


class FalconOCROnnxModel(nn.Module):
    def __init__(self, config: FalconOCRExportConfig) -> None:
        super().__init__()
        self.config = config
        self.img_projector = nn.Linear(config.image_patch_dim, config.dim, bias=False)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [ExportTransformerBlock(layer_id, config) for layer_id in range(config.n_layers)]
        )
        self.norm = ExportRMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        temporal_cos, temporal_sin = _build_temporal_rope_tables(
            config.head_dim // 2,
            config.max_seq_len,
            config.rope_theta,
        )
        self.register_buffer("temporal_cos", temporal_cos, persistent=False)
        self.register_buffer("temporal_sin", temporal_sin, persistent=False)
        self.register_buffer(
            "freqs_cis_golden",
            torch.empty((config.n_heads, config.head_dim // 4, 2), dtype=torch.float32),
            persistent=True,
        )

    def _rope_inputs(
        self,
        pos_t: torch.Tensor,
        pos_hw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rope_cos = self.temporal_cos[pos_t]
        rope_sin = self.temporal_sin[pos_t]

        clean_pos_hw = torch.where(torch.isnan(pos_hw), torch.zeros_like(pos_hw), pos_hw)
        theta = torch.einsum("bsp,hfp->bshf", clean_pos_hw, self.freqs_cis_golden)
        spatial_cos = torch.cos(theta)
        spatial_sin = torch.sin(theta)
        return rope_cos, rope_sin, spatial_cos, spatial_sin

    def forward(
        self,
        tokens: torch.Tensor,
        image_patches: torch.Tensor,
        pos_t: torch.Tensor,
        pos_hw: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        image_features = self.img_projector(image_patches)
        image_token_mask = (tokens == self.config.img_id).unsqueeze(-1)
        h = torch.where(image_token_mask, image_features, h)

        rope_cos, rope_sin, spatial_cos, spatial_sin = self._rope_inputs(pos_t, pos_hw)
        for layer in self.layers:
            h = layer(h, attention_mask, rope_cos, rope_sin, spatial_cos, spatial_sin)

        h = self.norm(h)
        return self.output(h)


class FalconOCRPrefillModel(nn.Module):
    def __init__(self, base: FalconOCROnnxModel) -> None:
        super().__init__()
        self.base = base

    def forward(
        self,
        tokens: torch.Tensor,
        image_patches: torch.Tensor,
        pos_t: torch.Tensor,
        pos_hw: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.base.tok_embeddings(tokens)
        image_features = self.base.img_projector(image_patches)
        image_token_mask = (tokens == self.base.config.img_id).unsqueeze(-1)
        h = torch.where(image_token_mask, image_features, h)

        rope_cos, rope_sin, spatial_cos, spatial_sin = self.base._rope_inputs(pos_t, pos_hw)
        layer_caches = []
        for layer in self.base.layers:
            attention = layer.attention
            xq, xk, xv = attention._pre_attention_qkv(h)
            xq, xk = attention._apply_rope(xq, xk, rope_cos, rope_sin, spatial_cos, spatial_sin)

            q = xq.permute(0, 2, 1, 3)
            k = xk.permute(0, 2, 1, 3).contiguous()
            v = xv.permute(0, 2, 1, 3).contiguous()

            scores = torch.matmul(q, k.transpose(-2, -1)) * attention.scale
            mask_value = torch.tensor(
                -3.4028234663852886e38,
                dtype=scores.dtype,
                device=scores.device,
            )
            scores = torch.where(attention_mask.unsqueeze(1), scores, mask_value)
            probs = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, v)
            lse = torch.logsumexp(scores, dim=-1)
            sink_scale = torch.sigmoid(lse - attention.sinks.reshape(1, -1, 1))
            attn_out = attn_out * sink_scale.unsqueeze(-1)
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().flatten(2)

            h = h + attention.wo(attn_out)
            h = h + layer.feed_forward(h)
            layer_caches.append(torch.stack((k, v), dim=0))

        logits = self.base.output(self.base.norm(h[:, -1, :]))
        present_key_values = torch.stack(layer_caches, dim=0)
        return logits, present_key_values


class FalconOCRDecodeModel(nn.Module):
    def __init__(self, base: FalconOCROnnxModel) -> None:
        super().__init__()
        self.base = base

    def forward(
        self,
        token: torch.Tensor,
        pos_t: torch.Tensor,
        pos_hw: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.base.tok_embeddings(token)
        rope_cos, rope_sin, spatial_cos, spatial_sin = self.base._rope_inputs(pos_t, pos_hw)

        layer_caches = []
        for layer_id, layer in enumerate(self.base.layers):
            attention = layer.attention
            xq, xk, xv = attention._pre_attention_qkv(h)
            xq, xk = attention._apply_rope(xq, xk, rope_cos, rope_sin, spatial_cos, spatial_sin)

            q = xq.permute(0, 2, 1, 3)
            new_k = xk.permute(0, 2, 1, 3).contiguous()
            new_v = xv.permute(0, 2, 1, 3).contiguous()
            k = torch.cat((past_key_values[layer_id, 0], new_k), dim=2)
            v = torch.cat((past_key_values[layer_id, 1], new_v), dim=2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * attention.scale
            mask_value = torch.tensor(
                -3.4028234663852886e38,
                dtype=scores.dtype,
                device=scores.device,
            )
            scores = torch.where(attention_mask.unsqueeze(1), scores, mask_value)
            probs = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, v)
            lse = torch.logsumexp(scores, dim=-1)
            sink_scale = torch.sigmoid(lse - attention.sinks.reshape(1, -1, 1))
            attn_out = attn_out * sink_scale.unsqueeze(-1)
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().flatten(2)

            h = h + attention.wo(attn_out)
            h = h + layer.feed_forward(h)
            layer_caches.append(torch.stack((k, v), dim=0))

        logits = self.base.output(self.base.norm(h[:, 0, :]))
        present_key_values = torch.stack(layer_caches, dim=0)
        return logits, present_key_values


class FalconOCRUnifiedKVModel(nn.Module):
    def __init__(self, base: FalconOCROnnxModel) -> None:
        super().__init__()
        self.base = base

    def forward(
        self,
        tokens: torch.Tensor,
        image_patches: torch.Tensor,
        pos_t: torch.Tensor,
        pos_hw: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.base.tok_embeddings(tokens)
        image_features = self.base.img_projector(image_patches)
        image_token_mask = (tokens == self.base.config.img_id).unsqueeze(-1)
        h = torch.where(image_token_mask, image_features, h)

        rope_cos, rope_sin, spatial_cos, spatial_sin = self.base._rope_inputs(pos_t, pos_hw)
        layer_caches = []
        for layer_id, layer in enumerate(self.base.layers):
            attention = layer.attention
            xq, xk, xv = attention._pre_attention_qkv(h)
            xq, xk = attention._apply_rope(xq, xk, rope_cos, rope_sin, spatial_cos, spatial_sin)

            q = xq.permute(0, 2, 1, 3)
            new_k = xk.permute(0, 2, 1, 3).contiguous()
            new_v = xv.permute(0, 2, 1, 3).contiguous()
            k = torch.cat((past_key_values[layer_id, 0], new_k), dim=2)
            v = torch.cat((past_key_values[layer_id, 1], new_v), dim=2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * attention.scale
            mask_value = torch.tensor(
                -3.4028234663852886e38,
                dtype=scores.dtype,
                device=scores.device,
            )
            scores = torch.where(attention_mask.unsqueeze(1), scores, mask_value)
            probs = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, v)
            lse = torch.logsumexp(scores, dim=-1)
            sink_scale = torch.sigmoid(lse - attention.sinks.reshape(1, -1, 1))
            attn_out = attn_out * sink_scale.unsqueeze(-1)
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().flatten(2)

            h = h + attention.wo(attn_out)
            h = h + layer.feed_forward(h)
            layer_caches.append(torch.stack((k, v), dim=0))

        logits = self.base.output(self.base.norm(h[:, -1, :]))
        present_key_values = torch.stack(layer_caches, dim=0)
        return logits, present_key_values


class FalconOCRUnifiedKVTokenModel(FalconOCRUnifiedKVModel):
    def forward(
        self,
        tokens: torch.Tensor,
        image_patches: torch.Tensor,
        pos_t: torch.Tensor,
        pos_hw: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, present_key_values = super().forward(
            tokens,
            image_patches,
            pos_t,
            pos_hw,
            attention_mask,
            past_key_values,
        )
        next_token = torch.argmax(logits, dim=-1)
        return next_token, present_key_values
