from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.json"
WEIGHTS_PATH = ROOT / "model.safetensors"


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


ACT2FN = {
    "gelu_pytorch_tanh": gelu_pytorch_tanh,
    "silu": F.silu,
}


@dataclass
class Gemma4TextConfig:
    attention_bias: bool
    attention_dropout: float
    attention_k_eq_v: bool
    final_logit_softcapping: float | None
    global_head_dim: int
    head_dim: int
    hidden_activation: str
    hidden_size: int
    hidden_size_per_layer_input: int
    intermediate_size: int
    layer_types: list[str]
    max_position_embeddings: int
    num_attention_heads: int
    num_global_key_value_heads: int | None
    num_hidden_layers: int
    num_key_value_heads: int
    num_kv_shared_layers: int
    pad_token_id: int
    rms_norm_eps: float
    rope_parameters: dict
    sliding_window: int
    tie_word_embeddings: bool
    use_cache: bool
    use_double_wide_mlp: bool
    vocab_size: int
    vocab_size_per_layer_input: int
    enable_moe_block: bool = False

    @classmethod
    def from_file(cls, path: str | Path = CONFIG_PATH) -> "Gemma4TextConfig":
        data = json.loads(Path(path).read_text())
        text = data["text_config"]
        return cls(
            attention_bias=text["attention_bias"],
            attention_dropout=text["attention_dropout"],
            attention_k_eq_v=text["attention_k_eq_v"],
            final_logit_softcapping=text.get("final_logit_softcapping"),
            global_head_dim=text["global_head_dim"],
            head_dim=text["head_dim"],
            hidden_activation=text["hidden_activation"],
            hidden_size=text["hidden_size"],
            hidden_size_per_layer_input=text["hidden_size_per_layer_input"],
            intermediate_size=text["intermediate_size"],
            layer_types=text["layer_types"],
            max_position_embeddings=text["max_position_embeddings"],
            num_attention_heads=text["num_attention_heads"],
            num_global_key_value_heads=text.get("num_global_key_value_heads"),
            num_hidden_layers=text["num_hidden_layers"],
            num_key_value_heads=text["num_key_value_heads"],
            num_kv_shared_layers=text["num_kv_shared_layers"],
            pad_token_id=text["pad_token_id"],
            rms_norm_eps=text["rms_norm_eps"],
            rope_parameters=text["rope_parameters"],
            sliding_window=text["sliding_window"],
            tie_word_embeddings=text["tie_word_embeddings"],
            use_cache=text["use_cache"],
            use_double_wide_mlp=text["use_double_wide_mlp"],
            vocab_size=text["vocab_size"],
            vocab_size_per_layer_input=text["vocab_size_per_layer_input"],
            enable_moe_block=text.get("enable_moe_block", False),
        )


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        hidden_states = hidden_states * torch.pow(mean_squared, -0.5)
        if self.with_scale:
            hidden_states = hidden_states * self.weight.float()
        return hidden_states.to(dtype=input_dtype)


class Gemma4TextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class Gemma4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.layer_types = set(config.layer_types)
        for layer_type in self.layer_types:
            inv_freq = self._build_inv_freq(layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)

    def _build_inv_freq(self, layer_type: str) -> torch.Tensor:
        rope = self.config.rope_parameters[layer_type]
        rope_type = rope["rope_type"]
        if layer_type == "full_attention":
            head_dim = self.config.global_head_dim
        else:
            head_dim = self.config.head_dim

        base = rope["rope_theta"]
        if rope_type == "proportional":
            rope_proportion = rope.get("partial_rotary_factor", 1.0)
            rope_angles = int(rope_proportion * head_dim // 2)
            inv_freq_rotated = 1.0 / (
                base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
            )
            nope_angles = head_dim // 2 - rope_angles
            if nope_angles > 0:
                inv_freq = torch.cat((inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)), dim=0)
            else:
                inv_freq = inv_freq_rotated
            return inv_freq

        return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        inv_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


def build_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return build_attention_mask(
        q_len=seq_len,
        q_offset=0,
        kv_len=seq_len,
        kv_offset=0,
        sliding_window=None,
        device=device,
        dtype=dtype,
    )


def build_sliding_causal_mask(seq_len: int, sliding_window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return build_attention_mask(
        q_len=seq_len,
        q_offset=0,
        kv_len=seq_len,
        kv_offset=0,
        sliding_window=sliding_window,
        device=device,
        dtype=dtype,
    )


def build_attention_mask(
    q_len: int,
    q_offset: int,
    kv_len: int,
    kv_offset: int,
    sliding_window: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    q_idx = torch.arange(q_len, device=device) + q_offset
    kv_idx = torch.arange(kv_len, device=device) + kv_offset
    allowed = kv_idx[None, :] <= q_idx[:, None]
    if sliding_window is not None:
        allowed = allowed & (kv_idx[None, :] > (q_idx[:, None] - sliding_window))
    mask = torch.full((q_len, kv_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.where(allowed, torch.zeros((), device=device, dtype=dtype), mask)
    return mask.unsqueeze(0).unsqueeze(0)


@dataclass
class Gemma4LayerCache:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    seq_length: int = 0
    sliding_window: int | None = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None or self.values is None:
            full_key_states = key_states
            full_value_states = value_states
        else:
            full_key_states = torch.cat([self.keys, key_states], dim=-2)
            full_value_states = torch.cat([self.values, value_states], dim=-2)

        self.seq_length += key_states.shape[-2]
        if self.sliding_window is None:
            self.keys = full_key_states
            self.values = full_value_states
        else:
            keep = max(self.sliding_window - 1, 0)
            if keep == 0:
                self.keys = full_key_states[:, :, :0, :]
                self.values = full_value_states[:, :, :0, :]
            else:
                self.keys = full_key_states[:, :, -keep:, :]
                self.values = full_value_states[:, :, -keep:, :]
        return full_key_states, full_value_states


class Gemma4Cache:
    def __init__(self):
        self.seq_length = 0
        self.layers: dict[int, Gemma4LayerCache] = {}
        self.shared_layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get_seq_length(self) -> int:
        return self.seq_length

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        sliding_window: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers.get(layer_idx)
        if layer is None:
            layer = Gemma4LayerCache(sliding_window=sliding_window)
            self.layers[layer_idx] = layer
        return layer.update(key_states, value_states)

    def set_shared(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.shared_layers[layer_idx] = (key_states, value_states)

    def get_shared(self, layer_idx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = self.shared_layers[layer_idx]
        return key_states.to(device), value_states.to(device)

    def advance(self, tokens: int) -> None:
        self.seq_length += tokens


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
                key_states, value_states = past_key_values.update(
                    self.layer_idx,
                    key_states,
                    value_states,
                    self.sliding_window,
                )
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
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class Gemma4LanguageModel(nn.Module):
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
        input_ids: torch.Tensor,
        past_key_values: Gemma4Cache | None = None,
        use_cache: bool | None = None,
    ) -> tuple[torch.Tensor, Gemma4Cache | None]:
        use_cache = False if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = Gemma4Cache()

        hidden_states = self.embed_tokens(input_ids)
        per_layer_inputs = self.get_per_layer_inputs(input_ids)
        per_layer_inputs = self.project_per_layer_inputs(hidden_states, per_layer_inputs)

        batch, seq_len = input_ids.shape
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = (torch.arange(seq_len, device=input_ids.device) + past_seen_tokens).unsqueeze(0).expand(batch, -1)
        q_offset = past_seen_tokens
        if past_key_values is None:
            masks = {
                "full_attention": build_causal_mask(seq_len, input_ids.device, hidden_states.dtype),
                "sliding_attention": build_sliding_causal_mask(seq_len, self.config.sliding_window, input_ids.device, hidden_states.dtype),
            }
        else:
            masks = {
                "full_attention": build_attention_mask(
                    q_len=seq_len,
                    q_offset=q_offset,
                    kv_len=past_seen_tokens + seq_len,
                    kv_offset=0,
                    sliding_window=None,
                    device=input_ids.device,
                    dtype=hidden_states.dtype,
                ),
                "sliding_attention": build_attention_mask(
                    q_len=seq_len,
                    q_offset=q_offset,
                    kv_len=min(past_seen_tokens, self.config.sliding_window - 1) + seq_len,
                    kv_offset=max(past_seen_tokens - self.config.sliding_window + 1, 0),
                    sliding_window=self.config.sliding_window,
                    device=input_ids.device,
                    dtype=hidden_states.dtype,
                ),
            }
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type) for layer_type in set(self.config.layer_types)
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


class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.language_model = Gemma4LanguageModel(config)


class Gemma4TextOnly(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Gemma4Cache | None = None,
        use_cache: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Gemma4Cache]:
        hidden_states, past_key_values = self.model.language_model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = F.linear(hidden_states, self.model.language_model.embed_tokens.weight)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        if use_cache:
            return logits, past_key_values
        return logits


def list_text_keys(path: str | Path = WEIGHTS_PATH) -> list[str]:
    with safe_open(str(path), framework="pt") as f:
        return [k for k in f.keys() if k.startswith("model.language_model.")]


def load_text_state_dict(path: str | Path = WEIGHTS_PATH) -> dict[str, torch.Tensor]:
    state_dict = load_file(str(path))
    return {k: v for k, v in state_dict.items() if k.startswith("model.language_model.")}


def stream_load_text_weights(model: nn.Module, path: str | Path = WEIGHTS_PATH) -> None:
    model_state = model.state_dict()
    expected = set(model_state.keys())
    with safe_open(str(path), framework="pt") as f:
        available = {k for k in f.keys() if k.startswith("model.language_model.")}
        missing = sorted(expected - available)
        unexpected = sorted(available - expected)
        if missing or unexpected:
            raise RuntimeError(f"missing={missing[:10]} unexpected={unexpected[:10]}")

        for key in model_state:
            tensor = f.get_tensor(key)
            target = model_state[key]
            if tensor.shape != target.shape:
                raise RuntimeError(f"shape mismatch for {key}: checkpoint={tuple(tensor.shape)} model={tuple(target.shape)}")
            target.copy_(tensor.to(dtype=target.dtype))


def validate_text_checkpoint(path: str | Path = WEIGHTS_PATH) -> tuple[int, int]:
    config = Gemma4TextConfig.from_file()
    with torch.device("meta"):
        model = Gemma4TextOnly(config)

    expected_items = model.state_dict()
    expected = set(expected_items.keys())

    with safe_open(str(path), framework="pt") as f:
        available = {k for k in f.keys() if k.startswith("model.language_model.")}
        missing = sorted(expected - available)
        unexpected = sorted(available - expected)
        if missing or unexpected:
            raise RuntimeError(f"missing={missing[:10]} unexpected={unexpected[:10]}")

        for key, value in expected_items.items():
            shape = tuple(f.get_slice(key).get_shape())
            if shape != tuple(value.shape):
                raise RuntimeError(f"shape mismatch for {key}: checkpoint={shape} model={tuple(value.shape)}")

    return len(expected), len(available)


def build_model_naive(device: str | torch.device = "cpu", dtype: torch.dtype = torch.bfloat16) -> Gemma4TextOnly:
    config = Gemma4TextConfig.from_file()
    model = Gemma4TextOnly(config).to(device=device, dtype=dtype)
    text_state = load_text_state_dict()
    missing, unexpected = model.load_state_dict(text_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"missing={missing} unexpected={unexpected}")
    return model.eval()


def build_model(device: str | torch.device = "cpu", dtype: torch.dtype = torch.bfloat16) -> Gemma4TextOnly:
    config = Gemma4TextConfig.from_file()
    model = Gemma4TextOnly(config).to(device=device, dtype=dtype)
    stream_load_text_weights(model)
    return model.to(device=device, dtype=dtype).eval()
