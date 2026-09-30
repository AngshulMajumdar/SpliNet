#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from baselines.fnet import build_fnet
from baselines.transformer import build_transformer
from splinet.builders import build_splinet
from splinet.checkpoint import _extract_state_dict, remap_legacy_state_dict
from splinet.mlm import make_mlm_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["splinet", "fnet", "transformer"], required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = _extract_state_dict(payload)

    if args.model == "splinet":
        model = build_splinet(matched_to_fnet=False)
        state = remap_legacy_state_dict(state)
    elif args.model == "fnet":
        model = build_fnet()
    else:
        model = build_transformer()

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    valid = np.load(args.validation, mmap_mode="r")

    losses = []
    with torch.no_grad():
        for i in range(0, len(valid), args.batch):
            clean = valid[i:i + args.batch]
            ids, labels = make_mlm_batch(clean, i, validation=True)
            ids, labels = ids.to(device), labels.to(device)
            if args.model == "transformer":
                loss = model(ids, labels)[0]
            else:
                loss = model(input_ids=ids, labels=labels).loss
            losses.append(float(loss.cpu()))

    loss = sum(losses) / len(losses)
    result = {"validation_loss": loss, "validation_perplexity": math.exp(min(loss, 20.0))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
