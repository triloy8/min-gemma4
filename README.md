# min-gemma4

Minimal Gemma 4 text peeling workbench.

The centerpiece is [`model.py`](/home/tadhiel/min-gemma4/model.py): a plain PyTorch text-side extraction of Gemma 4 built to load the original checkpoint names directly from `model.safetensors`.

## Current scope

- current target is `Gemma 4 4B`
- current runtime scope is `text only`
- the peeled model implementation lives in [`model.py`](/home/tadhiel/min-gemma4/model.py)
- tokenizer assets and chat formatting come from the dumped `config/`
- model weights come from the original `model.safetensors`

Multimodal support is not implemented yet.

That means:

- no image path
- no audio path
- no video path
- no multimodal placeholder handling at inference time

## Current repo shape

```text
.
├── config/
├── model.py
├── model.safetensors
├── scripts/
│   ├── download_weights.sh
│   ├── download_weights.py
│   ├── infer.sh
│   └── infer.py
├── pyproject.toml
└── uv.lock
```

## Setup

Install the environment:

```bash
uv sync
```

## Validation

Validate that the extracted text model matches the checkpoint key names and shapes:

```bash
./scripts/infer.sh --validate-only
```

This does not fully hydrate the model weights into RAM.

## Text-only inference

Run a small chat-style text inference:

```bash
./scripts/infer.sh --prompt "Write one short sentence about Paris."
```

Useful flags:

```bash
./scripts/infer.sh --prompt "Hello" --loader naive --max-new-tokens 16
./scripts/infer.sh --prompt "Hello" --loader streamed --device cuda
./scripts/infer.sh --prompt "Hello" --temperature 0.8 --top-k 20
```

## Weights download

Download the current 4B weights file into the repo root:

```bash
./scripts/download_weights.sh
```

Choose a named variant:

```bash
./scripts/download_weights.sh --variant e4b-it
```

The script keeps a small internal variant registry so more Gemma 4 safetensors URLs can be added as new variants land.

The `.sh` files are the intended entrypoints.
They are `uv`-based wrappers over the Python implementation files.

Notes:

- `naive` loads the full text state dict in one shot
- `streamed` copies tensors from the checkpoint incrementally
- both paths are text-only
- the prompt is formatted locally from the tokenizer assets in `config/`
- this is still an experimental extraction, not a polished inference package
