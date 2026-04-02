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

from model import build_model, build_model_naive


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


if __name__ == "__main__":
    main()
