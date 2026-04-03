from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import Gemma4TextConfig, build_model, build_model_naive, list_text_keys, validate_text_checkpoint


TOKENIZER_PATH = ROOT / "config" / "tokenizer.json"
TOKENIZER_CONFIG_PATH = ROOT / "config" / "tokenizer_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal text-only Gemma 4 inference entrypoint.")
    parser.add_argument("--prompt", help="User message to send through the chat template.")
    parser.add_argument("--prompt-file", help="Read the user prompt from a file.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 means greedy decoding.")
    parser.add_argument("--top-k", type=int, default=0, help="Optional top-k sampling cutoff.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed. Used when temperature > 0.")
    parser.add_argument(
        "--loader",
        choices=("streamed", "naive"),
        default="naive",
        help="Weight loading strategy.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model dtype used during loading and inference.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate checkpoint names and shapes against the extracted text model.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip the checkpoint key and shape validation before inference.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the fully formatted prompt string before inference.",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Drop special tokens from the decoded output.",
    )
    return parser.parse_args()


def choose_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


class LocalTokenizer:
    def __init__(self, tokenizer_path: Path = TOKENIZER_PATH, config_path: Path = TOKENIZER_CONFIG_PATH):
        self.backend = Tokenizer.from_file(str(tokenizer_path))
        self.config = json.loads(config_path.read_text())
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
        formatted = self.format_user_prompt(prompt)
        encoding = self.backend.encode(formatted)
        return torch.tensor([encoding.ids], dtype=torch.long)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.backend.decode(token_ids, skip_special_tokens=skip_special_tokens)


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    logits = logits[:, -1, :]
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        cutoff = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model: torch.nn.Module,
    tokenizer: LocalTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    skip_special_tokens: bool,
) -> tuple[torch.Tensor, str]:
    generated = input_ids.to(device)
    past_key_values = None
    with torch.no_grad():
        logits, past_key_values = model(generated, use_cache=True)
        for step in range(max_new_tokens):
            if step > 0:
                logits, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
            next_token = sample_next_token(logits, temperature=temperature, top_k=top_k)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=skip_special_tokens)
    return generated, text


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text().rstrip()
    if args.prompt is not None:
        return args.prompt
    return "Hello."


def main() -> None:
    args = parse_args()
    config = Gemma4TextConfig.from_file()
    prompt = resolve_prompt(args)

    if not args.skip_validate or args.validate_only:
        keys = list_text_keys()
        expected, available = validate_text_checkpoint()
        print("validated")
        print("text_layers", config.num_hidden_layers)
        print("text_tensors", len(keys))
        print("expected", expected)
        print("available", available)
        if args.validate_only:
            return

    tokenizer = LocalTokenizer()
    formatted_prompt = tokenizer.format_user_prompt(prompt)
    input_ids = tokenizer.encode_chat_prompt(prompt)

    if args.show_prompt:
        print("--- prompt ---")
        print(formatted_prompt)
        print("--------------")

    builder = build_model if args.loader == "streamed" else build_model_naive
    model = builder(device=args.device, dtype=choose_dtype(args.dtype))

    if args.temperature > 0.0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    generated_ids, generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        skip_special_tokens=args.skip_special_tokens,
    )

    print("loader", args.loader)
    print("device", args.device)
    print("dtype", args.dtype)
    print("prompt_tokens", input_ids.shape[-1])
    print("total_tokens", generated_ids.shape[-1])
    print("---")
    print(generated_text)


if __name__ == "__main__":
    main()
