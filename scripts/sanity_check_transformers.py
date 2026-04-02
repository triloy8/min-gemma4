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
    parser.add_argument("--top-k", type=int, default=5, help="How many top tokens to print.")
    parser.add_argument("--layerwise", action="store_true", help="Also compare embeddings, projected per-layer inputs, and each decoder layer output.")
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

    for idx, layer in enumerate(lm.layers):
        layer_type = lm.config.layer_types[idx]
        hidden_states = layer(
            hidden_states=hidden_states,
            per_layer_input=projected_per_layer[:, :, idx, :],
            position_embeddings=position_embeddings[layer_type],
            attention_mask=masks[layer_type],
        )
        out[f"layer_{idx}"] = hidden_states.detach()

    out["final_norm"] = lm.norm(hidden_states).detach()
    out["logits"] = model(input_ids).detach()
    return out


def collect_hf_stages(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    lm = model.model
    inputs_embeds = lm.embed_tokens(input_ids)
    raw_per_layer = lm.get_per_layer_inputs(input_ids, inputs_embeds)
    projected_per_layer = lm.project_per_layer_inputs(inputs_embeds, raw_per_layer)
    outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)

    out: dict[str, torch.Tensor] = {
        "embed": inputs_embeds.detach(),
        "per_layer_projected": projected_per_layer.detach(),
    }
    for idx, hidden in enumerate(outputs.hidden_states[1:]):
        out[f"layer_{idx}"] = hidden.detach()
    out["final_norm"] = outputs.hidden_states[-1].detach()
    out["logits"] = outputs.logits.detach()
    return out


def print_layerwise_report(ours: dict[str, torch.Tensor], hf: dict[str, torch.Tensor]) -> None:
    print("--- layerwise ---")
    tensor_stats("embed", ours["embed"], hf["embed"])
    tensor_stats("per_layer_projected", ours["per_layer_projected"], hf["per_layer_projected"])

    first_bad = None
    layer_keys = sorted(k for k in ours if k.startswith("layer_"))
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


def compare_rope_tensors(ours_model: torch.nn.Module, hf_model: torch.nn.Module, input_ids: torch.Tensor) -> None:
    ours_lm = ours_model.model.language_model
    hf_lm = hf_model.model
    _, _, _, ours_pos = build_common_context(ours_lm, input_ids)
    _, _, _, hf_pos = build_common_context(hf_lm, input_ids)

    print("--- rope ---")
    for layer_type in sorted(ours_pos):
        ours_cos, ours_sin = ours_pos[layer_type]
        hf_cos, hf_sin = hf_pos[layer_type]
        tensor_stats(f"{layer_type}_cos", ours_cos, hf_cos)
        tensor_stats(f"{layer_type}_sin", ours_sin, hf_sin)


def compare_last_layer_attention(ours_model: torch.nn.Module, hf_model: torch.nn.Module, input_ids: torch.Tensor) -> None:
    ours_lm = ours_model.model.language_model
    hf_lm = hf_model.model

    ours_hidden, _, ours_masks, ours_pos = build_common_context(ours_lm, input_ids)
    hf_hidden, _, hf_masks, hf_pos = build_common_context(hf_lm, input_ids)

    ours_per_layer = ours_lm.project_per_layer_inputs(ours_hidden, ours_lm.get_per_layer_inputs(input_ids))
    hf_per_layer = hf_lm.project_per_layer_inputs(hf_hidden, hf_lm.get_per_layer_inputs(input_ids, hf_hidden))

    for idx in range(ours_lm.config.num_hidden_layers - 1):
        layer_type = ours_lm.config.layer_types[idx]
        ours_hidden = ours_lm.layers[idx](
            hidden_states=ours_hidden,
            per_layer_input=ours_per_layer[:, :, idx, :],
            position_embeddings=ours_pos[layer_type],
            attention_mask=ours_masks[layer_type],
        )
        hf_hidden = hf_lm.layers[idx](
            hf_hidden,
            hf_per_layer[:, :, idx, :],
            position_embeddings=hf_pos[layer_type],
            attention_mask=hf_masks[layer_type],
            position_ids=None,
            past_key_values=None,
        )

    last_idx = ours_lm.config.num_hidden_layers - 1
    layer_type = ours_lm.config.layer_types[last_idx]
    ours_layer = ours_lm.layers[last_idx]
    hf_layer = hf_lm.layers[last_idx]

    ours_normed = ours_layer.input_layernorm(ours_hidden)
    hf_normed = hf_layer.input_layernorm(hf_hidden)
    ours_attn = ours_layer.self_attn(ours_normed, ours_pos[layer_type], ours_masks[layer_type])
    hf_attn, _ = hf_layer.self_attn(
        hidden_states=hf_normed,
        position_embeddings=hf_pos[layer_type],
        attention_mask=hf_masks[layer_type],
        position_ids=None,
        past_key_values=None,
    )

    print("--- last layer attention ---")
    tensor_stats("layer_41_input_normed", ours_normed, hf_normed)
    tensor_stats("layer_41_attn_output", ours_attn, hf_attn)


def main() -> None:
    args = parse_args()
    dtype = choose_dtype(args.dtype)
    tokenizer = LocalTokenizer()
    input_ids = tokenizer.encode_chat_prompt(args.prompt)

    builder = build_model if args.loader == "streamed" else build_model_naive
    ours = builder(device=args.device, dtype=dtype)
    hf = load_hf_model(device=args.device, dtype=dtype)

    with torch.inference_mode():
        our_logits = ours(input_ids.to(args.device))
        hf_logits = hf(input_ids=input_ids.to(args.device)).logits

    diff = (our_logits.float() - hf_logits.float()).abs()
    print("prompt")
    print(tokenizer.format_user_prompt(args.prompt))
    print("---")
    print("shape", tuple(our_logits.shape), tuple(hf_logits.shape))
    print("max_abs_diff", diff.max().item())
    print("mean_abs_diff", diff.mean().item())
    print("next_token_match", int(torch.argmax(our_logits[0, -1]).item() == torch.argmax(hf_logits[0, -1]).item()))
    print("--- ours top tokens ---")
    for token_id, score, text in top_tokens(our_logits, tokenizer, args.top_k):
        print(token_id, score, repr(text))
    print("--- hf top tokens ---")
    for token_id, score, text in top_tokens(hf_logits, tokenizer, args.top_k):
        print(token_id, score, repr(text))

    if args.layerwise:
        with torch.inference_mode():
            ours_stages = collect_ours_stages(ours, input_ids.to(args.device))
            hf_stages = collect_hf_stages(hf, input_ids.to(args.device))
        print_layerwise_report(ours_stages, hf_stages)
        with torch.inference_mode():
            compare_rope_tensors(ours, hf, input_ids.to(args.device))
            compare_last_layer_attention(ours, hf, input_ids.to(args.device))


if __name__ == "__main__":
    main()
