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

from model import build_model, build_model_naive, prepare_image_inputs


TOKENIZER_PATH = ROOT / "config" / "tokenizer.json"
TOKENIZER_CONFIG_PATH = ROOT / "config" / "tokenizer_config.json"
PROCESSOR_CONFIG_PATH = ROOT / "config" / "processor_config.json"
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
        self.image_token = self.config["image_token"]
        self.boi_token = self.config["boi_token"]
        self.eoi_token = self.config["eoi_token"]
        self.eos_token_id = self.backend.token_to_id(self.eos_token)

    def format_user_prompt(self, prompt: str, num_image_tokens: int) -> str:
        prompt = prompt.rstrip()
        if self.image_token not in prompt:
            prompt = f"{self.image_token}\n{prompt}"
        if prompt.count(self.image_token) != 1:
            raise ValueError("vision sanity check currently supports exactly one image placeholder")
        replacement = f"{self.boi_token}{self.image_token * num_image_tokens}{self.eoi_token}"
        prompt = prompt.replace(self.image_token, replacement, 1)
        return (
            f"{self.bos_token}"
            f"{self.sot_token}user\n"
            f"{prompt}"
            f"{self.eot_token}\n"
            f"{self.sot_token}model\n"
        )

    def encode_chat_prompt(self, prompt: str, num_image_tokens: int) -> torch.Tensor:
        encoding = self.backend.encode(self.format_user_prompt(prompt, num_image_tokens))
        return torch.tensor([encoding.ids], dtype=torch.long)

    def decode_token(self, token_id: int) -> str:
        return self.backend.decode([token_id], skip_special_tokens=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local Gemma 4 vision path with Hugging Face transformers.")
    parser.add_argument("--image", required=True, help="Image path to use for parity checks.")
    parser.add_argument("--prompt", default="Describe the image briefly.", help="Text prompt to pair with the image.")
    parser.add_argument("--device", default="cpu", help="Torch device for both models.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--loader", choices=("streamed", "naive"), default="naive")
    parser.add_argument("--top-k", type=int, default=5, help="How many top multimodal logits to print.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing parity and only compare model-side outputs.")
    parser.add_argument("--skip-layerwise", action="store_true", help="Skip per-layer vision tower comparison.")
    return parser.parse_args()


def choose_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_hf_model(device: str, dtype: torch.dtype) -> torch.nn.Module:
    try:
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "transformers is required for this vision sanity check. Install it in the env first."
        ) from exc

    full_config = Gemma4Config.from_pretrained(str(CONFIG_DIR))
    hf_model = Gemma4ForConditionalGeneration(full_config)
    hf_model.model.config._attn_implementation = "eager"
    hf_model.model.language_model.config._attn_implementation = "eager"

    full_state = load_file(str(WEIGHTS_PATH))
    missing, unexpected = hf_model.load_state_dict(full_state, strict=False)
    hf_model.tie_weights()

    if unexpected:
        raise RuntimeError(f"unexpected HF keys: {unexpected[:10]}")
    allowed_missing = {"lm_head.weight"}
    if set(missing) - allowed_missing:
        raise RuntimeError(f"missing HF keys: {missing[:10]}")
    return hf_model.to(device=device, dtype=dtype).eval()


def load_hf_image_processor():
    try:
        from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
    except ImportError as exc:
        raise SystemExit(
            "transformers vision image processor support is required for preprocessing parity."
        ) from exc

    image_processor_config = json.loads(PROCESSOR_CONFIG_PATH.read_text())["image_processor"]
    return Gemma4ImageProcessorPil(**image_processor_config)


def tensor_stats(name: str, ours: torch.Tensor, hf: torch.Tensor) -> None:
    diff = (ours.float() - hf.float()).abs()
    print(name, "shape", tuple(ours.shape), "max_abs_diff", diff.max().item(), "mean_abs_diff", diff.mean().item())


def exact_tensor_report(name: str, ours: torch.Tensor, hf: torch.Tensor) -> None:
    equal = torch.equal(ours, hf)
    print(name, "shape", tuple(ours.shape), "exact_match", equal)
    if not equal:
        diff = (ours.float() - hf.float()).abs()
        print(name, "max_abs_diff", diff.max().item(), "mean_abs_diff", diff.mean().item())


def register_tensor_hooks(module: torch.nn.Module, mapping: dict[str, torch.nn.Module]) -> tuple[dict[str, torch.Tensor], list]:
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if isinstance(value, torch.Tensor):
                captured[name] = value.detach()

        return hook

    for name, child in mapping.items():
        handles.append(child.register_forward_hook(make_hook(name)))
    return captured, handles


def compare_preprocessing(image_path: str | Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    local_pixel_values, local_image_position_ids, local_num_soft_tokens = prepare_image_inputs(image_path)

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required for preprocessing parity.") from exc

    hf_processor = load_hf_image_processor()
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        hf_inputs = hf_processor.preprocess(images=image, return_tensors="pt")

    hf_pixel_values = hf_inputs["pixel_values"]
    hf_image_position_ids = hf_inputs["image_position_ids"]
    hf_num_soft_tokens = int(hf_inputs["num_soft_tokens_per_image"][0])

    print("--- preprocess ---")
    exact_tensor_report("pixel_values", local_pixel_values, hf_pixel_values)
    exact_tensor_report("image_position_ids", local_image_position_ids, hf_image_position_ids)
    print("num_soft_tokens", "local", local_num_soft_tokens, "hf", hf_num_soft_tokens, "exact_match", local_num_soft_tokens == hf_num_soft_tokens)

    return local_pixel_values, local_image_position_ids, local_num_soft_tokens


def collect_local_vision_stages(model: torch.nn.Module, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    vision = model.model.vision_tower
    if vision is None:
        raise RuntimeError("local model was built without vision support")

    mapping = {"patch_embedder": vision.patch_embedder}
    for idx, layer in enumerate(vision.encoder.layers):
        mapping[f"layer_{idx}"] = layer
    captured, handles = register_tensor_hooks(vision, mapping)
    try:
        tower_output = vision(pixel_values, image_position_ids)
    finally:
        for handle in handles:
            handle.remove()

    projected = model.model.get_image_features(pixel_values, image_position_ids)
    captured["vision_output"] = tower_output.detach()
    captured["projected"] = projected.detach()
    return captured


def collect_hf_vision_stages(model: torch.nn.Module, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    vision = model.model.vision_tower
    mapping = {"patch_embedder": vision.patch_embedder}
    for idx, layer in enumerate(vision.encoder.layers):
        mapping[f"layer_{idx}"] = layer
    captured, handles = register_tensor_hooks(vision, mapping)
    try:
        tower_output = vision(pixel_values=pixel_values, pixel_position_ids=image_position_ids).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()

    projected = model.model.get_image_features(pixel_values=pixel_values, image_position_ids=image_position_ids).pooler_output
    captured["vision_output"] = tower_output.detach()
    captured["projected"] = projected.detach()
    return captured


def print_layerwise_report(ours: dict[str, torch.Tensor], hf: dict[str, torch.Tensor], skip_layerwise: bool) -> None:
    print("--- vision_tower ---")
    tensor_stats("patch_embedder", ours["patch_embedder"], hf["patch_embedder"])
    if not skip_layerwise:
        layer_keys = sorted((key for key in ours if key.startswith("layer_")), key=lambda key: int(key.split("_")[1]))
        first_bad = None
        for key in layer_keys:
            diff = (ours[key].float() - hf[key].float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(key, "shape", tuple(ours[key].shape), "max_abs_diff", max_diff, "mean_abs_diff", mean_diff)
            if first_bad is None and max_diff > 1e-2:
                first_bad = key
        print("first_large_divergence", first_bad or "none")
    tensor_stats("vision_output", ours["vision_output"], hf["vision_output"])
    tensor_stats("projected_image_features", ours["projected"], hf["projected"])


def top_tokens(logits: torch.Tensor, tokenizer: LocalTokenizer, k: int) -> list[tuple[int, float, str]]:
    values, indices = torch.topk(logits[0, -1], k=min(k, logits.shape[-1]))
    return [(index.item(), value.item(), tokenizer.decode_token(index.item())) for value, index in zip(values, indices, strict=True)]


def print_multimodal_logits_report(
    ours_model: torch.nn.Module,
    hf_model: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_position_ids: torch.Tensor,
    tokenizer: LocalTokenizer,
    top_k: int,
) -> None:
    ours_logits = ours_model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_position_ids=image_position_ids,
        use_cache=False,
    ).detach()
    hf_outputs = hf_model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_position_ids=image_position_ids,
        use_cache=False,
        return_dict=True,
    )
    hf_logits = hf_outputs.logits.detach()

    print("--- multimodal_logits ---")
    tensor_stats("logits", ours_logits, hf_logits)
    print("ours_top_tokens", top_tokens(ours_logits, tokenizer, top_k))
    print("hf_top_tokens", top_tokens(hf_logits, tokenizer, top_k))


def main() -> None:
    args = parse_args()
    dtype = choose_dtype(args.dtype)

    if args.skip_preprocess:
        pixel_values, image_position_ids, num_soft_tokens = prepare_image_inputs(args.image)
    else:
        pixel_values, image_position_ids, num_soft_tokens = compare_preprocessing(args.image)

    tokenizer = LocalTokenizer()
    input_ids = tokenizer.encode_chat_prompt(args.prompt, num_soft_tokens)

    builder = build_model if args.loader == "streamed" else build_model_naive
    ours_model = builder(device=args.device, dtype=dtype, include_vision=True)
    hf_model = load_hf_model(device=args.device, dtype=dtype)

    pixel_values = pixel_values.to(device=args.device, dtype=dtype)
    image_position_ids = image_position_ids.to(device=args.device)
    input_ids = input_ids.to(device=args.device)

    ours_stages = collect_local_vision_stages(ours_model, pixel_values, image_position_ids)
    hf_stages = collect_hf_vision_stages(hf_model, pixel_values, image_position_ids)

    print_layerwise_report(ours_stages, hf_stages, skip_layerwise=args.skip_layerwise)
    print_multimodal_logits_report(
        ours_model=ours_model,
        hf_model=hf_model,
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_position_ids=image_position_ids,
        tokenizer=tokenizer,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
