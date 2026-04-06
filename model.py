from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.json"
PROCESSOR_CONFIG_PATH = ROOT / "config" / "processor_config.json"
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
    use_bidirectional_attention: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Gemma4TextConfig":
        return cls(
            attention_bias=data["attention_bias"],
            attention_dropout=data["attention_dropout"],
            attention_k_eq_v=data["attention_k_eq_v"],
            final_logit_softcapping=data.get("final_logit_softcapping"),
            global_head_dim=data["global_head_dim"],
            head_dim=data["head_dim"],
            hidden_activation=data["hidden_activation"],
            hidden_size=data["hidden_size"],
            hidden_size_per_layer_input=data["hidden_size_per_layer_input"],
            intermediate_size=data["intermediate_size"],
            layer_types=data["layer_types"],
            max_position_embeddings=data["max_position_embeddings"],
            num_attention_heads=data["num_attention_heads"],
            num_global_key_value_heads=data.get("num_global_key_value_heads"),
            num_hidden_layers=data["num_hidden_layers"],
            num_key_value_heads=data["num_key_value_heads"],
            num_kv_shared_layers=data["num_kv_shared_layers"],
            pad_token_id=data["pad_token_id"],
            rms_norm_eps=data["rms_norm_eps"],
            rope_parameters=data["rope_parameters"],
            sliding_window=data["sliding_window"],
            tie_word_embeddings=data["tie_word_embeddings"],
            use_cache=data["use_cache"],
            use_double_wide_mlp=data["use_double_wide_mlp"],
            vocab_size=data["vocab_size"],
            vocab_size_per_layer_input=data["vocab_size_per_layer_input"],
            enable_moe_block=data.get("enable_moe_block", False),
            use_bidirectional_attention=data.get("use_bidirectional_attention"),
        )

    @classmethod
    def from_file(cls, path: str | Path = CONFIG_PATH) -> "Gemma4TextConfig":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data["text_config"])


@dataclass
class Gemma4VisionConfig:
    attention_bias: bool
    attention_dropout: float
    global_head_dim: int
    head_dim: int
    hidden_activation: str
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    patch_size: int
    pooling_kernel_size: int
    position_embedding_size: int
    rms_norm_eps: float
    rope_parameters: dict
    standardize: bool
    use_clipped_linears: bool

    @classmethod
    def from_dict(cls, data: dict) -> "Gemma4VisionConfig":
        return cls(
            attention_bias=data["attention_bias"],
            attention_dropout=data["attention_dropout"],
            global_head_dim=data["global_head_dim"],
            head_dim=data["head_dim"],
            hidden_activation=data["hidden_activation"],
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            max_position_embeddings=data["max_position_embeddings"],
            num_attention_heads=data["num_attention_heads"],
            num_hidden_layers=data["num_hidden_layers"],
            num_key_value_heads=data["num_key_value_heads"],
            patch_size=data["patch_size"],
            pooling_kernel_size=data["pooling_kernel_size"],
            position_embedding_size=data["position_embedding_size"],
            rms_norm_eps=data["rms_norm_eps"],
            rope_parameters=data["rope_parameters"],
            standardize=data["standardize"],
            use_clipped_linears=data["use_clipped_linears"],
        )


@dataclass
class Gemma4AudioConfig:
    hidden_size: int
    output_proj_dims: int
    rms_norm_eps: float

    @classmethod
    def from_dict(cls, data: dict) -> "Gemma4AudioConfig":
        return cls(
            hidden_size=data["hidden_size"],
            output_proj_dims=data["output_proj_dims"],
            rms_norm_eps=data["rms_norm_eps"],
        )


@dataclass
class Gemma4Config:
    text_config: Gemma4TextConfig
    vision_config: Gemma4VisionConfig | None
    audio_config: Gemma4AudioConfig | None
    image_token_id: int | None
    video_token_id: int | None
    audio_token_id: int | None
    boi_token_id: int | None
    eoi_token_id: int | None
    boa_token_id: int | None
    eoa_token_id: int | None
    tie_word_embeddings: bool

    @classmethod
    def from_file(cls, path: str | Path = CONFIG_PATH) -> "Gemma4Config":
        data = json.loads(Path(path).read_text())
        return cls(
            text_config=Gemma4TextConfig.from_dict(data["text_config"]),
            vision_config=Gemma4VisionConfig.from_dict(data["vision_config"]) if data.get("vision_config") else None,
            audio_config=Gemma4AudioConfig.from_dict(data["audio_config"]) if data.get("audio_config") else None,
            image_token_id=data.get("image_token_id"),
            video_token_id=data.get("video_token_id"),
            audio_token_id=data.get("audio_token_id"),
            boi_token_id=data.get("boi_token_id"),
            eoi_token_id=data.get("eoi_token_id"),
            boa_token_id=data.get("boa_token_id"),
            eoa_token_id=data.get("eoa_token_index"),
            tie_word_embeddings=data["tie_word_embeddings"],
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
        self.layer_types = set(config.layer_types)
        self.config = config
        for layer_type in self.layer_types:
            self.register_buffer(f"{layer_type}_inv_freq", self._build_inv_freq(layer_type), persistent=False)

    def _build_inv_freq(self, layer_type: str) -> torch.Tensor:
        rope = self.config.rope_parameters[layer_type]
        rope_type = rope["rope_type"]
        head_dim = self.config.global_head_dim if layer_type == "full_attention" else self.config.head_dim
        base = rope["rope_theta"]
        if rope_type == "proportional":
            rope_proportion = rope.get("partial_rotary_factor", 1.0)
            rope_angles = int(rope_proportion * head_dim // 2)
            inv_freq_rotated = 1.0 / (
                base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
            )
            nope_angles = head_dim // 2 - rope_angles
            if nope_angles > 0:
                return torch.cat((inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)), dim=0)
            return inv_freq_rotated
        return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        inv_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


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


def build_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return build_attention_mask(seq_len, 0, seq_len, 0, None, device, dtype)


def build_sliding_causal_mask(seq_len: int, sliding_window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return build_attention_mask(seq_len, 0, seq_len, 0, sliding_window, device, dtype)


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


from audio_model import Gemma4AudioModel
from text_model import Gemma4TextModel
from vision_model import Gemma4VisionModel


class Gemma4MultimodalEmbedder(nn.Module):
    def __init__(self, multimodal_hidden_size: int, text_hidden_size: int, eps: float):
        super().__init__()
        self.embedding_projection = nn.Linear(multimodal_hidden_size, text_hidden_size, bias=False)
        self.embedding_pre_projection_norm = Gemma4RMSNorm(multimodal_hidden_size, eps=eps, with_scale=False)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))


class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4Config, include_vision: bool = False, include_audio: bool = False):
        super().__init__()
        self.config = config
        self.language_model = Gemma4TextModel(config.text_config)
        self.vision_tower = Gemma4VisionModel(config.vision_config) if include_vision and config.vision_config is not None else None
        self.embed_vision = (
            Gemma4MultimodalEmbedder(config.vision_config.hidden_size, config.text_config.hidden_size, config.vision_config.rms_norm_eps)
            if include_vision and config.vision_config is not None
            else None
        )
        self.audio_tower = Gemma4AudioModel(config.audio_config) if include_audio and config.audio_config is not None else None
        self.embed_audio = (
            Gemma4MultimodalEmbedder(config.audio_config.output_proj_dims, config.text_config.hidden_size, config.audio_config.rms_norm_eps)
            if include_audio and config.audio_config is not None
            else None
        )

    def get_input_embeddings(self) -> Gemma4TextScaledWordEmbedding:
        return self.language_model.get_input_embeddings()

    def get_image_features(self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> torch.Tensor:
        if self.vision_tower is None or self.embed_vision is None:
            raise ValueError("vision support is not enabled in this model build")
        return self.embed_vision(self.vision_tower(pixel_values, image_position_ids))

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        past_key_values: Gemma4Cache | None = None,
        use_cache: bool | None = None,
    ) -> tuple[torch.Tensor, Gemma4Cache | None]:
        image_mask = (input_ids == self.config.image_token_id) if self.config.image_token_id is not None else torch.zeros_like(input_ids, dtype=torch.bool)
        multimodal_mask = image_mask

        llm_input_ids = input_ids.clone()
        llm_input_ids[multimodal_mask] = self.config.text_config.pad_token_id
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)
        per_layer_inputs = self.language_model.project_per_layer_inputs(
            inputs_embeds, self.language_model.get_per_layer_inputs(llm_input_ids)
        )

        if pixel_values is not None or image_position_ids is not None:
            if pixel_values is None or image_position_ids is None:
                raise ValueError("pixel_values and image_position_ids must be provided together")
            image_features = self.get_image_features(pixel_values, image_position_ids)
            image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            if int(image_mask.sum().item()) != image_features.shape[0]:
                raise RuntimeError(
                    f"image features ({image_features.shape[0]}) do not match image placeholder count ({int(image_mask.sum().item())})"
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_features.reshape(-1)
            )

        return self.language_model(
            input_ids=llm_input_ids,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )


class Gemma4ForConditionalGeneration(nn.Module):
    def __init__(self, config: Gemma4Config, include_vision: bool = False, include_audio: bool = False):
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config, include_vision=include_vision, include_audio=include_audio)

    def get_input_embeddings(self) -> Gemma4TextScaledWordEmbedding:
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        past_key_values: Gemma4Cache | None = None,
        use_cache: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Gemma4Cache]:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = F.linear(hidden_states, self.model.language_model.embed_tokens.weight)
        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.text_config.final_logit_softcapping
        if use_cache:
            return logits, past_key_values
        return logits


def _filter_checkpoint_keys(keys: list[str], include_vision: bool, include_audio: bool) -> list[str]:
    prefixes = ["model.language_model."]
    if include_vision:
        prefixes.extend(["model.vision_tower.", "model.embed_vision."])
    if include_audio:
        prefixes.extend(["model.audio_tower.", "model.embed_audio."])
    return [key for key in keys if any(key.startswith(prefix) for prefix in prefixes)]


def list_text_keys(path: str | Path = WEIGHTS_PATH) -> list[str]:
    with safe_open(str(path), framework="pt") as f:
        return [key for key in f.keys() if key.startswith("model.language_model.")]


def load_model_state_dict(
    path: str | Path = WEIGHTS_PATH,
    include_vision: bool = False,
    include_audio: bool = False,
    show_progress: bool = False,
) -> dict[str, torch.Tensor]:
    state_dict = load_file(str(path))
    keys = _filter_checkpoint_keys(list(state_dict.keys()), include_vision=include_vision, include_audio=include_audio)
    key_iter = tqdm(keys, total=len(keys), desc="Loading weights", leave=False) if show_progress else keys
    return {key: state_dict[key] for key in key_iter}


def stream_load_model_weights(
    model: nn.Module,
    path: str | Path = WEIGHTS_PATH,
    show_progress: bool = False,
) -> None:
    model_state = model.state_dict()
    expected = set(model_state.keys())
    with safe_open(str(path), framework="pt") as f:
        available = {key for key in f.keys() if key in expected}
        missing = sorted(expected - available)
        unexpected = sorted(available - expected)
        if missing or unexpected:
            raise RuntimeError(f"missing={missing[:10]} unexpected={unexpected[:10]}")

        key_iter = tqdm(model_state, total=len(model_state), desc="Loading weights", leave=False) if show_progress else model_state
        for key in key_iter:
            tensor = f.get_tensor(key)
            target = model_state[key]
            if tensor.shape != target.shape:
                raise RuntimeError(f"shape mismatch for {key}: checkpoint={tuple(tensor.shape)} model={tuple(target.shape)}")
            target.copy_(tensor.to(dtype=target.dtype))


def validate_checkpoint(
    path: str | Path = WEIGHTS_PATH,
    include_vision: bool = False,
    include_audio: bool = False,
) -> tuple[int, int]:
    config = Gemma4Config.from_file()
    with torch.device("meta"):
        model = Gemma4ForConditionalGeneration(config, include_vision=include_vision, include_audio=include_audio)
    expected_items = model.state_dict()
    expected = set(expected_items.keys())
    with safe_open(str(path), framework="pt") as f:
        available = {key for key in f.keys() if key in expected}
        missing = sorted(expected - available)
        unexpected = sorted(available - expected)
        if missing or unexpected:
            raise RuntimeError(f"missing={missing[:10]} unexpected={unexpected[:10]}")
        for key, value in expected_items.items():
            shape = tuple(f.get_slice(key).get_shape())
            if shape != tuple(value.shape):
                raise RuntimeError(f"shape mismatch for {key}: checkpoint={shape} model={tuple(value.shape)}")
    return len(expected), len(available)


def validate_text_checkpoint(path: str | Path = WEIGHTS_PATH) -> tuple[int, int]:
    return validate_checkpoint(path=path, include_vision=False, include_audio=False)


def build_model_naive(
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    show_progress: bool | None = None,
    include_vision: bool = False,
    include_audio: bool = False,
) -> Gemma4ForConditionalGeneration:
    if show_progress is None:
        show_progress = sys.stderr.isatty()
    config = Gemma4Config.from_file()
    model = Gemma4ForConditionalGeneration(config, include_vision=include_vision, include_audio=include_audio).to(device=device, dtype=dtype)
    state_dict = load_model_state_dict(
        include_vision=include_vision,
        include_audio=include_audio,
        show_progress=show_progress,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"missing={missing} unexpected={unexpected}")
    return model.eval()


def build_model(
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    show_progress: bool | None = None,
    include_vision: bool = False,
    include_audio: bool = False,
) -> Gemma4ForConditionalGeneration:
    if show_progress is None:
        show_progress = sys.stderr.isatty()
    config = Gemma4Config.from_file()
    model = Gemma4ForConditionalGeneration(config, include_vision=include_vision, include_audio=include_audio).to(device=device, dtype=dtype)
    stream_load_model_weights(model, show_progress=show_progress)
    return model.eval()


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size
    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult
    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError("resize would collapse to 0x0")
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)
    return target_height, target_width


def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.transpose(1, 3, 2, 4, 0)
    return patched_image.reshape(num_patches_height * num_patches_width, -1)


def pad_along_first_dim(image: np.ndarray, positions: np.ndarray, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    current_length = image.shape[0]
    padding_length = target_length - current_length
    if padding_length > 0:
        paddings = [(0, padding_length)] + [(0, 0)] * (image.ndim - 1)
        pos_paddings = [(0, padding_length), (0, 0)]
        image = np.pad(image, paddings, mode="constant", constant_values=0)
        positions = np.pad(positions, pos_paddings, mode="constant", constant_values=-1)
    return image, positions


def prepare_image_inputs(
    image_path: str | Path,
    processor_config_path: str | Path = PROCESSOR_CONFIG_PATH,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image multimodality. Install it with `uv sync`.") from exc

    processor_config = json.loads(Path(processor_config_path).read_text())["image_processor"]
    patch_size = processor_config["patch_size"]
    pooling_kernel_size = processor_config["pooling_kernel_size"]
    max_soft_tokens = processor_config["max_soft_tokens"]
    max_patches = max_soft_tokens * pooling_kernel_size**2
    rescale_factor = processor_config["rescale_factor"]

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        target_height, target_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size,
            max_patches=max_patches,
            pooling_kernel_size=pooling_kernel_size,
        )
        if (target_width, target_height) != image.size:
            image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)

    array *= rescale_factor
    patch_height = array.shape[-2] // patch_size
    patch_width = array.shape[-1] // patch_size
    patches = convert_image_to_patches(array, patch_size)
    num_soft_tokens = patches.shape[0] // pooling_kernel_size**2
    patch_grid = np.meshgrid(np.arange(patch_width), np.arange(patch_height), indexing="xy")
    positions = np.stack(patch_grid, axis=-1).reshape(patches.shape[0], 2)
    patches, positions = pad_along_first_dim(patches, positions, max_patches)
    pixel_values = torch.from_numpy(patches).unsqueeze(0)
    image_position_ids = torch.from_numpy(positions).unsqueeze(0).long()
    return pixel_values, image_position_ids, num_soft_tokens
