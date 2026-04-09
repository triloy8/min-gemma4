from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import (
    build_causal_mask,
    build_model,
    build_model_naive,
    build_sliding_causal_mask,
)


TOKENIZER_PATH = ROOT / "config" / "tokenizer.json"
TOKENIZER_CONFIG_PATH = ROOT / "config" / "tokenizer_config.json"
WEIGHTS_PATH = ROOT / "model.safetensors"
CONFIG_DIR = ROOT / "config"


class LocalTokenizer:
    def __init__(self) -> None:
        self.backend = Tokenizer.from_file(str(TOKENIZER_PATH))
        self.config = json.loads(TOKENIZER_CONFIG_PATH.read_text())
        self.bos_token = self.config["bos_token"]
        self.eos_token = self.config["eos_token"]
        self.sot_token = self.config["sot_token"]
        self.eot_token = self.config["eot_token"]
        self.eos_token_id = self.backend.token_to_id(self.eos_token)

    def format_user_prompt(self, prompt: str) -> str:
        prompt = prompt.rstrip()
        return (
            f"{self.bos_token}"
            f"{self.sot_token}user\n"
            f"{prompt}"
            f"{self.eot_token}\n"
            f"{self.sot_token}model\n"
        )

    def encode_chat_prompt(self, prompt: str) -> torch.Tensor:
        encoding = self.backend.encode(self.format_user_prompt(prompt))
        return torch.tensor([encoding.ids], dtype=torch.long)

    def decode_token(self, token_id: int) -> str:
        return self.backend.decode([token_id], skip_special_tokens=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare peeled Gemma 4 logits with Hugging Face transformers.")
    parser.add_argument("--prompt", default="Hello.", help="Prompt to compare.")
    parser.add_argument("--device", default="cpu", help="Torch device for both models.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--loader", choices=("streamed", "naive"), default="naive")
    parser.add_argument("--use-cache", action="store_true", help="Compare the cached decoding path instead of the no-cache prefill path.")
    parser.add_argument("--decode-steps", type=int, default=0, help="After cached prefill, decode this many greedy steps token-by-token and compare logits at each step.")
    parser.add_argument("--top-k", type=int, default=5, help="How many top tokens to print.")
    parser.add_argument("--layerwise", action="store_true", help="Also compare embeddings, projected per-layer inputs, and each decoder layer output.")
    parser.add_argument("--blockwise", action="store_true", help="Compare intermediate activations inside each decoder block.")
    parser.add_argument("--blockwise-from", type=int, default=None, help="First decoder layer index to include in --blockwise output.")
    parser.add_argument("--blockwise-to", type=int, default=None, help="Last decoder layer index to include in --blockwise output.")
    return parser.parse_args()


def choose_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_hf_model(device: str, dtype: torch.dtype) -> torch.nn.Module:
    try:
        from transformers import Gemma4Config, Gemma4ForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "transformers is required for this sanity check. Install it in the env first, for example: "
            "`uv pip install transformers`"
        ) from exc

    full_config = Gemma4Config.from_pretrained(str(CONFIG_DIR))
    hf_model = Gemma4ForCausalLM(full_config.text_config)
    hf_model.config._attn_implementation = "eager"
    hf_model.model.config._attn_implementation = "eager"

    full_state = load_file(str(WEIGHTS_PATH))
    text_state = {
        f"model.{key.removeprefix('model.language_model.')}": value
        for key, value in full_state.items()
        if key.startswith("model.language_model.")
    }
    missing, unexpected = hf_model.load_state_dict(text_state, strict=False)
    hf_model.tie_weights()

    if unexpected:
        raise RuntimeError(f"unexpected HF keys: {unexpected[:10]}")

    allowed_missing = {"lm_head.weight"}
    if set(missing) - allowed_missing:
        raise RuntimeError(f"missing HF keys: {missing[:10]}")

    return hf_model.to(device=device, dtype=dtype).eval()


def top_tokens(logits: torch.Tensor, tokenizer: LocalTokenizer, k: int) -> list[tuple[int, float, str]]:
    values, indices = torch.topk(logits[0, -1], k=min(k, logits.shape[-1]))
    result = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        result.append((index, value, tokenizer.decode_token(index)))
    return result


def tensor_stats(name: str, ours: torch.Tensor, hf: torch.Tensor) -> None:
    diff = (ours.float() - hf.float()).abs()
    print(name, "shape", tuple(ours.shape), "max_abs_diff", diff.max().item(), "mean_abs_diff", diff.mean().item())


def to_cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [to_cpu_tree(item) for item in value]
    return value


def release_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_ours_stages(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    lm = model.model.language_model
    hidden_states = lm.embed_tokens(input_ids)
    raw_per_layer = lm.get_per_layer_inputs(input_ids)
    projected_per_layer = lm.project_per_layer_inputs(hidden_states, raw_per_layer)

    batch, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
    masks = {
        "full_attention": build_causal_mask(seq_len, input_ids.device, hidden_states.dtype),
        "sliding_attention": build_sliding_causal_mask(seq_len, lm.config.sliding_window, input_ids.device, hidden_states.dtype),
    }
    position_embeddings = {
        layer_type: lm.rotary_emb(hidden_states, position_ids, layer_type) for layer_type in set(lm.config.layer_types)
    }

    out: dict[str, torch.Tensor] = {
        "embed": hidden_states.detach(),
        "per_layer_projected": projected_per_layer.detach(),
    }

    num_layers = lm.config.num_hidden_layers
    for idx, layer in enumerate(lm.layers):
        layer_type = lm.config.layer_types[idx]
        hidden_states = layer(
            hidden_states=hidden_states,
            per_layer_input=projected_per_layer[:, :, idx, :],
            position_embeddings=position_embeddings[layer_type],
            attention_mask=masks[layer_type],
        )
        # Keep layerwise reporting aligned with the hidden states that HF exposes directly:
        # raw decoder outputs up to the penultimate layer, then final_norm separately.
        if idx < num_layers - 1:
            out[f"layer_{idx}"] = hidden_states.detach()

    out["final_norm"] = lm.norm(hidden_states).detach()
    out["logits"] = model(input_ids).detach()
    return out


def collect_hf_stages(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    lm = model.model
    inputs_embeds = lm.embed_tokens(input_ids)
    raw_per_layer = lm.get_per_layer_inputs(input_ids, inputs_embeds)
    projected_per_layer = lm.project_per_layer_inputs(inputs_embeds, raw_per_layer)
    lm_outputs = lm(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    logits_outputs = model(
        input_ids=input_ids,
        use_cache=False,
        return_dict=True,
    )

    out: dict[str, torch.Tensor] = {
        "embed": inputs_embeds.detach(),
        "per_layer_projected": projected_per_layer.detach(),
    }

    hidden_states = lm_outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("HF model did not return hidden_states with output_hidden_states=True")

    num_layers = lm.config.num_hidden_layers
    exposed_hidden_states = hidden_states[1:]
    if len(exposed_hidden_states) < num_layers:
        raise RuntimeError(
            f"HF hidden_states too short: got {len(hidden_states)} entries for {num_layers} decoder layers"
        )

    # Empirically the last exposed hidden state corresponds to the final post-norm output,
    # not a separate raw `layer_{num_layers - 1}` tensor.
    for idx, hidden in enumerate(exposed_hidden_states[:-1]):
        out[f"layer_{idx}"] = hidden.detach()
    out["final_norm"] = exposed_hidden_states[-1].detach()
    out["logits"] = logits_outputs.logits.detach()
    return out


def print_layerwise_report(ours: dict[str, torch.Tensor], hf: dict[str, torch.Tensor]) -> None:
    print("--- layerwise ---")
    our_layer_keys = {k for k in ours if k.startswith("layer_")}
    hf_layer_keys = {k for k in hf if k.startswith("layer_")}
    if our_layer_keys != hf_layer_keys:
        missing_in_hf = sorted(our_layer_keys - hf_layer_keys, key=lambda key: int(key.split("_")[1]))
        extra_in_hf = sorted(hf_layer_keys - our_layer_keys, key=lambda key: int(key.split("_")[1]))
        raise RuntimeError(f"layer key mismatch: missing_in_hf={missing_in_hf} extra_in_hf={extra_in_hf}")

    tensor_stats("embed", ours["embed"], hf["embed"])
    tensor_stats("per_layer_projected", ours["per_layer_projected"], hf["per_layer_projected"])

    first_bad = None
    layer_keys = sorted((k for k in ours if k.startswith("layer_")), key=lambda key: int(key.split("_")[1]))
    for key in layer_keys:
        diff = (ours[key].float() - hf[key].float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(key, "max_abs_diff", max_diff, "mean_abs_diff", mean_diff)
        if first_bad is None and max_diff > 1e-2:
            first_bad = key

    tensor_stats("final_norm", ours["final_norm"], hf["final_norm"])
    tensor_stats("logits", ours["logits"], hf["logits"])
    if first_bad is not None:
        print("first_large_divergence", first_bad)
    else:
        print("first_large_divergence", "none")


def build_common_context(lm: torch.nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    hidden_states = lm.embed_tokens(input_ids)
    batch, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
    masks = {
        "full_attention": build_causal_mask(seq_len, input_ids.device, hidden_states.dtype),
        "sliding_attention": build_sliding_causal_mask(seq_len, lm.config.sliding_window, input_ids.device, hidden_states.dtype),
    }
    position_embeddings = {
        layer_type: lm.rotary_emb(hidden_states, position_ids, layer_type) for layer_type in set(lm.config.layer_types)
    }
    return hidden_states, position_ids, masks, position_embeddings


def run_prefix_layers_ours(
    ours_model: torch.nn.Module,
    input_ids: torch.Tensor,
    stop_before_layer: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    ours_lm = ours_model.model.language_model
    ours_hidden, _, ours_masks, ours_pos = build_common_context(ours_lm, input_ids)
    ours_per_layer = ours_lm.project_per_layer_inputs(ours_hidden, ours_lm.get_per_layer_inputs(input_ids))
    for idx in range(stop_before_layer):
        layer_type = ours_lm.config.layer_types[idx]
        ours_hidden = ours_lm.layers[idx](
            hidden_states=ours_hidden,
            per_layer_input=ours_per_layer[:, :, idx, :],
            position_embeddings=ours_pos[layer_type],
            attention_mask=ours_masks[layer_type],
        )
    return ours_hidden, ours_masks, ours_pos, ours_per_layer


def run_prefix_layers_hf(
    hf_model: torch.nn.Module,
    input_ids: torch.Tensor,
    stop_before_layer: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    hf_lm = hf_model.model
    hf_hidden, _, hf_masks, hf_pos = build_common_context(hf_lm, input_ids)
    hf_per_layer = hf_lm.project_per_layer_inputs(hf_hidden, hf_lm.get_per_layer_inputs(input_ids, hf_hidden))
    for idx in range(stop_before_layer):
        layer_type = hf_lm.config.layer_types[idx]
        hf_hidden = hf_lm.layers[idx](
            hf_hidden,
            hf_per_layer[:, :, idx, :],
            position_embeddings=hf_pos[layer_type],
            attention_mask=hf_masks[layer_type],
            position_ids=None,
            past_key_values=None,
        )
    return hf_hidden, hf_masks, hf_pos, hf_per_layer


def collect_block_activations_ours(
    ours_model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    ours_lm = ours_model.model.language_model
    ours_hidden, ours_masks, ours_pos, ours_per_layer = run_prefix_layers_ours(ours_model, input_ids, stop_before_layer=layer_idx)
    layer_type = ours_lm.config.layer_types[layer_idx]
    ours_layer = ours_lm.layers[layer_idx]
    ours_pli = ours_per_layer[:, :, layer_idx, :]

    ours_residual_0 = ours_hidden
    ours_input_ln = ours_layer.input_layernorm(ours_hidden)
    ours_attn = ours_layer.self_attn(ours_input_ln, ours_pos[layer_type], ours_masks[layer_type])
    ours_post_attn_ln = ours_layer.post_attention_layernorm(ours_attn)
    ours_after_attn = ours_residual_0 + ours_post_attn_ln
    ours_residual_1 = ours_after_attn
    ours_pre_ffn_ln = ours_layer.pre_feedforward_layernorm(ours_after_attn)
    ours_mlp = ours_layer.mlp(ours_pre_ffn_ln)
    ours_post_ffn_ln = ours_layer.post_feedforward_layernorm(ours_mlp)
    ours_after_ffn = ours_residual_1 + ours_post_ffn_ln
    ours_residual_2 = ours_after_ffn
    ours_pli_gate = ours_layer.per_layer_input_gate(ours_after_ffn)
    ours_pli_act = ours_layer.act_fn(ours_pli_gate)
    ours_pli_mul = ours_pli_act * ours_pli
    ours_pli_proj = ours_layer.per_layer_projection(ours_pli_mul)
    ours_post_pli_ln = ours_layer.post_per_layer_input_norm(ours_pli_proj)
    ours_after_pli = ours_residual_2 + ours_post_pli_ln
    ours_final = ours_after_pli * ours_layer.layer_scalar

    return {
        "residual_in": ours_residual_0,
        "input_ln": ours_input_ln,
        "attn": ours_attn,
        "post_attn_ln": ours_post_attn_ln,
        "after_attn": ours_after_attn,
        "pre_ffn_ln": ours_pre_ffn_ln,
        "mlp": ours_mlp,
        "post_ffn_ln": ours_post_ffn_ln,
        "after_ffn": ours_after_ffn,
        "pli_gate": ours_pli_gate,
        "pli_act": ours_pli_act,
        "pli_mul": ours_pli_mul,
        "pli_proj": ours_pli_proj,
        "post_pli_ln": ours_post_pli_ln,
        "after_pli": ours_after_pli,
        "final": ours_final,
    }


def collect_block_activations_hf(
    hf_model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    hf_lm = hf_model.model
    hf_hidden, hf_masks, hf_pos, hf_per_layer = run_prefix_layers_hf(hf_model, input_ids, stop_before_layer=layer_idx)
    layer_type = hf_lm.config.layer_types[layer_idx]
    hf_layer = hf_lm.layers[layer_idx]
    hf_pli = hf_per_layer[:, :, layer_idx, :]

    hf_residual_0 = hf_hidden
    hf_input_ln = hf_layer.input_layernorm(hf_hidden)
    hf_attn, _ = hf_layer.self_attn(
        hidden_states=hf_input_ln,
        position_embeddings=hf_pos[layer_type],
        attention_mask=hf_masks[layer_type],
        position_ids=None,
        past_key_values=None,
    )
    hf_post_attn_ln = hf_layer.post_attention_layernorm(hf_attn)
    hf_after_attn = hf_residual_0 + hf_post_attn_ln
    hf_residual_1 = hf_after_attn
    hf_pre_ffn_ln = hf_layer.pre_feedforward_layernorm(hf_after_attn)
    hf_mlp = hf_layer.mlp(hf_pre_ffn_ln)
    hf_post_ffn_ln = hf_layer.post_feedforward_layernorm(hf_mlp)
    hf_after_ffn = hf_residual_1 + hf_post_ffn_ln
    hf_residual_2 = hf_after_ffn
    hf_pli_gate = hf_layer.per_layer_input_gate(hf_after_ffn)
    hf_pli_act = hf_layer.act_fn(hf_pli_gate)
    hf_pli_mul = hf_pli_act * hf_pli
    hf_pli_proj = hf_layer.per_layer_projection(hf_pli_mul)
    hf_post_pli_ln = hf_layer.post_per_layer_input_norm(hf_pli_proj)
    hf_after_pli = hf_residual_2 + hf_post_pli_ln
    hf_final = hf_after_pli * hf_layer.layer_scalar

    return {
        "residual_in": hf_residual_0,
        "input_ln": hf_input_ln,
        "attn": hf_attn,
        "post_attn_ln": hf_post_attn_ln,
        "after_attn": hf_after_attn,
        "pre_ffn_ln": hf_pre_ffn_ln,
        "mlp": hf_mlp,
        "post_ffn_ln": hf_post_ffn_ln,
        "after_ffn": hf_after_ffn,
        "pli_gate": hf_pli_gate,
        "pli_act": hf_pli_act,
        "pli_mul": hf_pli_mul,
        "pli_proj": hf_pli_proj,
        "post_pli_ln": hf_post_pli_ln,
        "after_pli": hf_after_pli,
        "final": hf_final,
    }


def print_block_activations(
    ours_activations: dict[str, torch.Tensor],
    hf_activations: dict[str, torch.Tensor],
    layer_idx: int,
) -> None:
    print(f"--- layer_{layer_idx} block ---")
    for name in ours_activations:
        tensor_stats(f"layer_{layer_idx}_{name}", ours_activations[name], hf_activations[name])


def print_blockwise_report(
    ours_activations_by_layer: dict[int, dict[str, torch.Tensor]],
    hf_activations_by_layer: dict[int, dict[str, torch.Tensor]],
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> None:
    layer_indices = sorted(ours_activations_by_layer)
    if not layer_indices:
        raise ValueError("no block activations collected")

    num_layers = max(layer_indices) + 1
    start = 0 if layer_start is None else max(0, layer_start)
    end = num_layers - 1 if layer_end is None else min(num_layers - 1, layer_end)

    if start > end:
        raise ValueError(f"invalid blockwise range: start={start} end={end}")

    print("--- blockwise ---")
    for layer_idx in range(start, end + 1):
        print_block_activations(ours_activations_by_layer[layer_idx], hf_activations_by_layer[layer_idx], layer_idx)


def collect_blockwise_range_ours(
    ours_model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    num_layers = ours_model.model.language_model.config.num_hidden_layers
    start = 0 if layer_start is None else max(0, layer_start)
    end = num_layers - 1 if layer_end is None else min(num_layers - 1, layer_end)
    if start > end:
        raise ValueError(f"invalid blockwise range: start={start} end={end}")
    return {
        layer_idx: to_cpu_tree(collect_block_activations_ours(ours_model, input_ids, layer_idx))
        for layer_idx in range(start, end + 1)
    }


def collect_blockwise_range_hf(
    hf_model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    num_layers = hf_model.model.config.num_hidden_layers
    start = 0 if layer_start is None else max(0, layer_start)
    end = num_layers - 1 if layer_end is None else min(num_layers - 1, layer_end)
    if start > end:
        raise ValueError(f"invalid blockwise range: start={start} end={end}")
    return {
        layer_idx: to_cpu_tree(collect_block_activations_hf(hf_model, input_ids, layer_idx))
        for layer_idx in range(start, end + 1)
    }


def collect_rope_tensors_ours(ours_model: torch.nn.Module, input_ids: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    ours_lm = ours_model.model.language_model
    _, _, _, ours_pos = build_common_context(ours_lm, input_ids)
    return ours_pos


def collect_rope_tensors_hf(hf_model: torch.nn.Module, input_ids: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    hf_lm = hf_model.model
    _, _, _, hf_pos = build_common_context(hf_lm, input_ids)
    return hf_pos


def compare_rope_tensors(ours_pos: dict[str, tuple[torch.Tensor, torch.Tensor]], hf_pos: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
    print("--- rope ---")
    for layer_type in sorted(ours_pos):
        ours_cos, ours_sin = ours_pos[layer_type]
        hf_cos, hf_sin = hf_pos[layer_type]
        tensor_stats(f"{layer_type}_cos", ours_cos, hf_cos)
        tensor_stats(f"{layer_type}_sin", ours_sin, hf_sin)


def compare_last_layer_attention(ours_activations: dict[str, torch.Tensor], hf_activations: dict[str, torch.Tensor], last_idx: int) -> None:
    ours_normed = ours_activations["input_ln"]
    hf_normed = hf_activations["input_ln"]
    ours_attn = ours_activations["attn"]
    hf_attn = hf_activations["attn"]

    print("--- last layer attention ---")
    tensor_stats(f"layer_{last_idx}_input_normed", ours_normed, hf_normed)
    tensor_stats(f"layer_{last_idx}_attn_output", ours_attn, hf_attn)


def compare_last_layer_block(ours_activations: dict[str, torch.Tensor], hf_activations: dict[str, torch.Tensor], last_idx: int) -> None:
    print("--- last layer block ---")
    for name in ours_activations:
        tensor_stats(f"layer_{last_idx}_{name}", ours_activations[name], hf_activations[name])


def print_cached_decode_report(
    ours_model: torch.nn.Module,
    hf_model: torch.nn.Module,
    tokenizer: LocalTokenizer,
    input_ids: torch.Tensor,
    decode_steps: int,
) -> None:
    print("--- cached decode ---")
    with torch.inference_mode():
        hf_outputs = hf_model(input_ids=input_ids, use_cache=True)
        hf_cache = hf_outputs.past_key_values
        hf_steps = []

        for step in range(decode_steps):
            next_token = torch.argmax(hf_outputs.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = next_token.item()
            token_text = tokenizer.decode_token(token_id)
            hf_outputs = hf_model(input_ids=next_token, past_key_values=hf_cache, use_cache=True)
            hf_cache = hf_outputs.past_key_values
            hf_steps.append(
                {
                    "token_id": token_id,
                    "token_text": token_text,
                    "hf_logits": hf_outputs.logits.detach().cpu(),
                }
            )

        our_logits, our_cache = ours_model(input_ids, use_cache=True)
        for step, hf_step in enumerate(hf_steps, start=1):
            next_token = torch.tensor([[hf_step["token_id"]]], device=input_ids.device, dtype=torch.long)
            our_logits, our_cache = ours_model(next_token, past_key_values=our_cache, use_cache=True)
            hf_logits = hf_step["hf_logits"]
            diff = (our_logits.detach().cpu().float() - hf_logits.float()).abs()

            print(
                "step",
                step,
                "token",
                hf_step["token_id"],
                repr(hf_step["token_text"]),
                "max_abs_diff",
                diff.max().item(),
                "mean_abs_diff",
                diff.mean().item(),
                "next_token_match", int(torch.argmax(our_logits[0, -1]).item() == torch.argmax(hf_logits[0, -1]).item()),
            )


def collect_hf_cached_decode_steps(
    hf_model: torch.nn.Module,
    tokenizer: LocalTokenizer,
    input_ids: torch.Tensor,
    decode_steps: int,
) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    with torch.inference_mode():
        hf_outputs = hf_model(input_ids=input_ids, use_cache=True)
        hf_cache = hf_outputs.past_key_values
        for _ in range(decode_steps):
            next_token = torch.argmax(hf_outputs.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = next_token.item()
            hf_outputs = hf_model(input_ids=next_token, past_key_values=hf_cache, use_cache=True)
            hf_cache = hf_outputs.past_key_values
            steps.append(
                {
                    "token_id": token_id,
                    "token_text": tokenizer.decode_token(token_id),
                    "hf_logits": hf_outputs.logits.detach().cpu(),
                }
            )
    return steps


def print_cached_decode_report_from_steps(
    ours_model: torch.nn.Module,
    tokenizer: LocalTokenizer,
    input_ids: torch.Tensor,
    hf_steps: list[dict[str, object]],
) -> None:
    print("--- cached decode ---")
    with torch.inference_mode():
        _, our_cache = ours_model(input_ids, use_cache=True)
        for step, hf_step in enumerate(hf_steps, start=1):
            next_token = torch.tensor([[hf_step["token_id"]]], device=input_ids.device, dtype=torch.long)
            our_logits, our_cache = ours_model(next_token, past_key_values=our_cache, use_cache=True)
            hf_logits = hf_step["hf_logits"]
            diff = (our_logits.detach().cpu().float() - hf_logits.float()).abs()

            print(
                "step",
                step,
                "token",
                hf_step["token_id"],
                repr(hf_step["token_text"]),
                "max_abs_diff",
                diff.max().item(),
                "mean_abs_diff",
                diff.mean().item(),
                "next_token_match",
                int(torch.argmax(our_logits[0, -1]).item() == torch.argmax(hf_logits[0, -1]).item()),
            )


def main() -> None:
    args = parse_args()
    if args.decode_steps and not args.use_cache:
        raise SystemExit("--decode-steps requires --use-cache.")
    if args.use_cache and (args.layerwise or args.blockwise):
        raise SystemExit("--use-cache currently supports headline logits only; layerwise/blockwise tracing remains no-cache only.")

    dtype = choose_dtype(args.dtype)
    tokenizer = LocalTokenizer()
    input_ids = tokenizer.encode_chat_prompt(args.prompt)
    input_ids_device = input_ids.to(args.device)

    builder = build_model if args.loader == "streamed" else build_model_naive

    ours_logits = None
    ours_stages = None
    ours_rope = None
    ours_last_layer = None
    hf_logits = None
    hf_stages = None
    hf_rope = None
    hf_last_layer = None
    hf_decode_steps = None

    ours = builder(device=args.device, dtype=dtype)
    with torch.inference_mode():
        if args.use_cache:
            ours_logits, _ = ours(input_ids_device, use_cache=True)
        else:
            ours_logits = ours(input_ids_device, use_cache=False)
        ours_logits = ours_logits.detach().cpu()

        if args.layerwise:
            ours_stages = to_cpu_tree(collect_ours_stages(ours, input_ids_device))
            ours_rope = to_cpu_tree(collect_rope_tensors_ours(ours, input_ids_device))
            last_idx = ours.model.language_model.config.num_hidden_layers - 1
            ours_last_layer = to_cpu_tree(collect_block_activations_ours(ours, input_ids_device, last_idx))
    del ours
    release_cuda_memory()

    hf = load_hf_model(device=args.device, dtype=dtype)
    with torch.inference_mode():
        if args.use_cache:
            hf_logits = hf(input_ids=input_ids_device, use_cache=True).logits
            if args.decode_steps:
                hf_decode_steps = collect_hf_cached_decode_steps(hf, tokenizer, input_ids_device, args.decode_steps)
        else:
            hf_logits = hf(input_ids=input_ids_device, use_cache=False).logits
        hf_logits = hf_logits.detach().cpu()

        if args.layerwise:
            hf_stages = to_cpu_tree(collect_hf_stages(hf, input_ids_device))
            hf_rope = to_cpu_tree(collect_rope_tensors_hf(hf, input_ids_device))
            last_idx = hf.model.config.num_hidden_layers - 1
            hf_last_layer = to_cpu_tree(collect_block_activations_hf(hf, input_ids_device, last_idx))

    diff = (ours_logits.float() - hf_logits.float()).abs()
    print("prompt")
    print(tokenizer.format_user_prompt(args.prompt))
    print("---")
    print("use_cache", int(args.use_cache))
    print("shape", tuple(ours_logits.shape), tuple(hf_logits.shape))
    print("max_abs_diff", diff.max().item())
    print("mean_abs_diff", diff.mean().item())
    print("next_token_match", int(torch.argmax(ours_logits[0, -1]).item() == torch.argmax(hf_logits[0, -1]).item()))
    print("--- ours top tokens ---")
    for token_id, score, text in top_tokens(ours_logits, tokenizer, args.top_k):
        print(token_id, score, repr(text))
    print("--- hf top tokens ---")
    for token_id, score, text in top_tokens(hf_logits, tokenizer, args.top_k):
        print(token_id, score, repr(text))

    if args.decode_steps:
        del hf
        release_cuda_memory()
        ours = builder(device=args.device, dtype=dtype)
        print_cached_decode_report_from_steps(ours, tokenizer, input_ids_device, hf_decode_steps or [])
        del ours
        release_cuda_memory()
    else:
        del hf
        release_cuda_memory()

    if args.layerwise:
        if ours_stages is None or hf_stages is None or ours_rope is None or hf_rope is None or ours_last_layer is None or hf_last_layer is None:
            raise RuntimeError("layerwise artifacts were not collected")
        print_layerwise_report(ours_stages, hf_stages)
        compare_rope_tensors(ours_rope, hf_rope)
        last_idx = len([key for key in ours_stages if key.startswith("layer_")])
        compare_last_layer_attention(ours_last_layer, hf_last_layer, last_idx)
        compare_last_layer_block(ours_last_layer, hf_last_layer, last_idx)
    if args.blockwise:
        ours = builder(device=args.device, dtype=dtype)
        with torch.inference_mode():
            ours_blockwise = collect_blockwise_range_ours(
                ours,
                input_ids_device,
                layer_start=args.blockwise_from,
                layer_end=args.blockwise_to,
            )
        del ours
        release_cuda_memory()
        hf = load_hf_model(device=args.device, dtype=dtype)
        with torch.inference_mode():
            hf_blockwise = collect_blockwise_range_hf(
                hf,
                input_ids_device,
                layer_start=args.blockwise_from,
                layer_end=args.blockwise_to,
            )
        del hf
        release_cuda_memory()
        print_blockwise_report(
            ours_blockwise,
            hf_blockwise,
            layer_start=args.blockwise_from,
            layer_end=args.blockwise_to,
        )


if __name__ == "__main__":
    main()
