from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from model import (
    ACT2FN,
    Gemma4Cache,
    Gemma4RMSNorm,
    Gemma4TextConfig,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    apply_rotary_pos_emb,
    build_attention_mask,
    build_causal_mask,
    build_sliding_causal_mask,
    repeat_kv,
)


class Gemma4TextMLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
        intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4TextAttention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            self.store_full_length_kv = layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        num_key_value_heads = (
            config.num_global_key_value_heads if config.attention_k_eq_v and not self.is_sliding else config.num_key_value_heads
        )
        if num_key_value_heads is None:
            num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
        self.v_proj = nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Gemma4Cache | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)

        if self.is_kv_shared_layer and past_key_values is not None:
            key_states, value_states = past_key_values.get_shared(self.kv_shared_layer_index, query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape)
            value_states = self.v_norm(value_states).transpose(1, 2)

        if past_key_values is not None:
            if not self.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(self.layer_idx, key_states, value_states, self.sliding_window)
            if self.store_full_length_kv:
                past_key_values.set_shared(self.layer_idx, key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        return self.o_proj(attn_output)


class Gemma4TextDecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4TextAttention(config, layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))
        self.per_layer_input_gate = nn.Linear(config.hidden_size, config.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(config.hidden_size_per_layer_input, config.hidden_size, bias=False)
        self.post_per_layer_input_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Gemma4Cache | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.per_layer_input_gate(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states * self.layer_scalar


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList([Gemma4TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            config.pad_token_id,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )
        self.per_layer_input_scale = 2.0**-0.5
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_model_projection_scale = config.hidden_size**-0.5
        self.per_layer_projection_norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> Gemma4TextScaledWordEmbedding:
        return self.embed_tokens

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(self, inputs_embeds: torch.Tensor, per_layer_inputs: torch.Tensor) -> torch.Tensor:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        past_key_values: Gemma4Cache | None = None,
        use_cache: bool | None = None,
    ) -> tuple[torch.Tensor, Gemma4Cache | None]:
        use_cache = False if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = Gemma4Cache()

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids is required when inputs_embeds is not provided")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        if per_layer_inputs is None:
            if input_ids is None:
                raise ValueError("input_ids is required when per_layer_inputs is not provided")
            per_layer_inputs = self.project_per_layer_inputs(hidden_states, self.get_per_layer_inputs(input_ids))

        batch, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = (torch.arange(seq_len, device=device) + past_seen_tokens).unsqueeze(0).expand(batch, -1)
        q_offset = past_seen_tokens
        if past_key_values is None:
            masks = {
                "full_attention": build_causal_mask(seq_len, device, hidden_states.dtype),
                "sliding_attention": build_sliding_causal_mask(seq_len, self.config.sliding_window, device, hidden_states.dtype),
            }
        else:
            masks = {
                "full_attention": build_attention_mask(
                    seq_len,
                    q_offset,
                    past_seen_tokens + seq_len,
                    0,
                    None,
                    device,
                    hidden_states.dtype,
                ),
                "sliding_attention": build_attention_mask(
                    seq_len,
                    q_offset,
                    min(past_seen_tokens, self.config.sliding_window - 1) + seq_len,
                    max(past_seen_tokens - self.config.sliding_window + 1, 0),
                    self.config.sliding_window,
                    device,
                    hidden_states.dtype,
                ),
            }
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }

        for idx, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[idx]
            hidden_states = layer(
                hidden_states=hidden_states,
                per_layer_input=per_layer_inputs[:, :, idx, :],
                position_embeddings=position_embeddings[layer_type],
                attention_mask=masks[layer_type],
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        if past_key_values is not None:
            past_key_values.advance(seq_len)
        return hidden_states, past_key_values
