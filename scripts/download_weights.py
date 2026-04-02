from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

VARIANT_URLS = {
    "e4b-it": "https://huggingface.co/google/gemma-4-E4B-it/resolve/main/model.safetensors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Gemma 4 weights file into the repo root.")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_URLS),
        default="e4b-it",
        help="Model variant to download.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "model.safetensors"),
        help="Destination path for the downloaded safetensors file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = VARIANT_URLS[args.variant]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"variant {args.variant}")
    print(f"url {url}")
    print(f"output {output}")

    with urllib.request.urlopen(url) as response, output.open("wb") as destination:
        shutil.copyfileobj(response, destination)

    print("downloaded")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
