from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from model import ACT2FN, Gemma4RMSNorm, Gemma4VisionConfig, apply_rotary_pos_emb, repeat_kv


class Gemma4ClippableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.register_buffer("input_min", torch.tensor(0.0))
        self.register_buffer("input_max", torch.tensor(0.0))
        self.register_buffer("output_min", torch.tensor(0.0))
        self.register_buffer("output_max", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = nn.Linear(3 * self.patch_size**2, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(torch.ones(2, self.position_embedding_size, config.hidden_size))

    def _position_embeddings(self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor) -> torch.Tensor:
        clamped_positions = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(clamped_positions, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        position_embeddings = one_hot @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(dim=1)
        return torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)

    def forward(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        return hidden_states + self._position_embeddings(pixel_position_ids, padding_positions)


class Gemma4VisionPooler(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.root_hidden_size = config.hidden_size**0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // output_length) ** 0.5)
        k_squared = k**2
        if k_squared * output_length != input_seq_len:
            raise ValueError(f"cannot pool {input_seq_len} positions to {output_length}")

        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(hidden_states, pixel_position_ids, output_length)
        return hidden_states * self.root_hidden_size, padding_positions


class Gemma4VisionMLP(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.gate_proj = Gemma4ClippableLinear(config.hidden_size, config.intermediate_size)
        self.up_proj = Gemma4ClippableLinear(config.hidden_size, config.intermediate_size)
        self.down_proj = Gemma4ClippableLinear(config.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        base = config.rope_parameters["rope_theta"]
        spatial_dim = config.head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        all_cos, all_sin = [], []
        for dim in range(2):
            dim_position_ids = position_ids[:, :, dim][:, None, :].float()
            freqs = (inv_freq_expanded @ dim_position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        return torch.cat(all_cos, dim=-1).to(dtype=x.dtype), torch.cat(all_sin, dim=-1).to(dtype=x.dtype)


def apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        apply_rotary_pos_emb(x=x_part, cos=cos_part, sin=sin_part, unsqueeze_dim=unsqueeze_dim)
        for x_part, cos_part, sin_part in zip(x_parts, cos_parts, sin_parts, strict=True)
    ]
    return torch.cat(y_parts, dim=-1)


def build_bidirectional_attention_mask(attention_mask: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    valid_keys = attention_mask[:, None, None, :].to(device=device)
    mask = torch.zeros(valid_keys.shape, device=device, dtype=dtype)
    return mask.masked_fill(~valid_keys, torch.finfo(dtype).min)


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.q_proj = Gemma4ClippableLinear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = Gemma4ClippableLinear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.v_proj = Gemma4ClippableLinear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.o_proj = Gemma4ClippableLinear(config.num_attention_heads * self.head_dim, config.hidden_size)
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_multidimensional_rope(query_states, cos, sin, position_ids).transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states)
        key_states = apply_multidimensional_rope(key_states, cos, sin, position_ids).transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape)
        value_states = self.v_norm(value_states).transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o_proj(attn_output)


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(config)
        self.mlp = Gemma4VisionMLP(config)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask, position_ids)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList([Gemma4VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, pixel_position_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        mask = build_bidirectional_attention_mask(attention_mask, inputs_embeds.device, inputs_embeds.dtype)
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, mask, pixel_position_ids)
        return hidden_states


class Gemma4VisionModel(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)

    def forward(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor) -> torch.Tensor:
        output_length = pixel_values.shape[-2] // (self.config.pooling_kernel_size**2)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        hidden_states = self.encoder(inputs_embeds, ~padding_positions, pixel_position_ids)
        hidden_states, pooler_mask = self.pooler(hidden_states, pixel_position_ids, padding_positions, output_length)
        return hidden_states[pooler_mask]
