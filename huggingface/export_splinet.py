#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import torch

from splinet.builders import base_splinet_config
from splinet.checkpoint import load_splinet_checkpoint
from splinet.modeling_splinet import SpliNetForMaskedLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--tokenizer-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    model = SpliNetForMaskedLM(base_splinet_config())
    payload, incompatible = load_splinet_checkpoint(model, args.checkpoint, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(incompatible)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    model.config.save_pretrained(args.output_dir)

    for name in ("spiece.model", "spiece.vocab", "tokenizer_metadata.json"):
        src = args.tokenizer_dir / name
        if src.exists():
            shutil.copy2(src, args.output_dir / name)

    metadata = {k: v for k, v in payload.items() if k != "model_state"}
    metadata["public_model_name"] = "SpliNet"
    metadata["spline_order"] = 2
    metadata["sidedness"] = "single"
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
