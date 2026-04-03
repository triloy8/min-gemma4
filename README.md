# min-gemma4

Minimal Gemma 4 extraction work focused on the `4B` line.

The core implementation is [`model.py`](/home/tadhiel/min-gemma4/model.py): a plain PyTorch text-side extraction that loads the original `model.language_model.*` checkpoint keys directly from `model.safetensors`.

## Current status

There is now a working text version in this repo.

What is implemented today:

- text-only forward pass and greedy/sampled generation
- direct loading from the original safetensors checkpoint
- both `naive` and streamed weight-loading paths
- local prompt formatting from the dumped tokenizer/config assets
- Hugging Face parity/debug scripts for checking logits and intermediate activations

What is not implemented yet:

- image input path
- audio input path
- video input path
- multimodal placeholder handling in the local inference path
- a proper 1:1 parity test suite

## Active work

The scope has narrowed to two concrete tracks:

1. add multimodal support to the first `4B` model path
2. replace the current transformer sanity scripts with a real 1:1 test suite

The existing parity scripts are still useful for manual investigation, but they are transitional tooling rather than the final validation interface.

## Repository shape

```text
.
├── config/
│   ├── config.json
│   ├── generation_config.json
│   ├── processor_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── model.py
├── model.safetensors
├── scripts/
│   ├── download_weights.py
│   ├── download_weights.sh
│   ├── infer.py
│   ├── infer.sh
│   ├── sanity_check_transformers.py
│   └── sanity_check_transformers.sh
├── pyproject.toml
└── uv.lock
```

## Setup

```bash
uv sync
```

## Working text path

Validate that the extracted text model matches the checkpoint key names and tensor shapes:

```bash
./scripts/infer.sh --validate-only
```

Run a small text generation example:

```bash
./scripts/infer.sh --prompt "Write one short sentence about Paris."
```

Useful variants:

```bash
./scripts/infer.sh --prompt "Hello" --loader naive --max-new-tokens 16
./scripts/infer.sh --prompt "Hello" --loader streamed --device cuda
./scripts/infer.sh --prompt "Hello" --temperature 0.8 --top-k 20
```

Notes:

- `naive` loads the full text state dict in one shot
- `streamed` copies tensors incrementally from the checkpoint
- both runtime paths are text-only today
- prompt formatting currently comes from the local tokenizer/config assets

## Transitional parity scripts

[`scripts/sanity_check_transformers.py`](/home/tadhiel/min-gemma4/scripts/sanity_check_transformers.py) compares the peeled text model against Hugging Face `transformers`.

Examples:

```bash
./scripts/sanity_check_transformers.sh --prompt "Hello."
./scripts/sanity_check_transformers.sh --use-cache --decode-steps 4
./scripts/sanity_check_transformers.sh --layerwise --blockwise
```

This is the current manual validation path. It is expected to be replaced by a 1:1 test suite rather than expanded indefinitely as a script.

## Weights

Download the current `4B` weights into the repo root:

```bash
./scripts/download_weights.sh
```

Choose a named variant:

```bash
./scripts/download_weights.sh --variant e4b-it
```

The shell scripts are thin `uv run` wrappers over the Python entrypoints.

## Intention

This repo is not trying to be a polished inference package. It is a minimal extraction and parity workbench for understanding Gemma 4 internals, getting the text path correct, and then extending that work toward multimodal support with tighter test coverage.
