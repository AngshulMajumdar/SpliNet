#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from baselines.fnet import build_fnet
from baselines.transformer import build_transformer
from splinet.builders import build_splinet
from splinet.mlm import make_mlm_batch

GLOBAL_SEED = 2026
DATA_ORDER_SEED = 7301
SEQ_LENGTH = 512
LEARNING_RATE = 1.0e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1.0e-8


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(kind):
    if kind == "splinet":
        return build_splinet(GLOBAL_SEED, matched_to_fnet=True)
    if kind == "fnet":
        return build_fnet(GLOBAL_SEED)
    if kind == "transformer":
        return build_transformer(GLOBAL_SEED)
    raise ValueError(kind)


def forward_model(model, kind, ids, labels):
    if kind == "transformer":
        return model(ids, labels)[0]
    return model(input_ids=ids, labels=labels).loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["splinet", "fnet", "transformer"], required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--effective-batch", type=int, required=True)
    ap.add_argument("--micro-batch", type=int, required=True)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.effective_batch % args.micro_batch:
        raise ValueError("effective batch must be divisible by micro batch")
    grad_accum = args.effective_batch // args.micro_batch
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train = np.load(args.data_dir / "train_blocks.npy", mmap_mode="r")
    valid = np.load(args.data_dir / "validation_blocks.npy", mmap_mode="r")
    if train.shape[1] != SEQ_LENGTH or valid.shape[1] != SEQ_LENGTH:
        raise RuntimeError("Expected sequence length 512")

    usable_rows = (len(train) // args.effective_batch) * args.effective_batch
    total_updates = usable_rows // args.effective_batch
    actual_train_tokens = total_updates * args.effective_batch * SEQ_LENGTH

    rng = np.random.default_rng(DATA_ORDER_SEED)
    permutation = rng.permutation(len(train))[:usable_rows]
    groups = permutation.reshape(total_updates, args.effective_batch)

    set_seeds(GLOBAL_SEED)
    model = build_model(args.model).to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=0.0,
    )
    warmup = max(1, round(0.01 * total_updates))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: float(step + 1) / warmup if step < warmup else 1.0,
    )

    use_amp = device.type == "cuda"
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    else:
        amp_dtype = torch.float32
        scaler = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training_log.jsonl"
    start = time.perf_counter()

    for update in range(total_updates):
        optimizer.zero_grad(set_to_none=True)
        group = groups[update]
        for j in range(grad_accum):
            idx = group[j * args.micro_batch:(j + 1) * args.micro_batch]
            ids, labels = make_mlm_batch(train[idx], update * grad_accum + j, validation=False)
            ids = ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = forward_model(model, args.model, ids, labels) / grad_accum
                scaler.scale(loss).backward()
            else:
                loss = forward_model(model, args.model, ids, labels) / grad_accum
                loss.backward()

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        if (update + 1) % 25 == 0 or update == 0:
            elapsed = time.perf_counter() - start
            tps = (update + 1) * args.effective_batch * SEQ_LENGTH / max(elapsed, 1e-9)
            rec = {
                "update": update + 1,
                "loss": float(loss.detach().cpu()) * grad_accum,
                "tokens_per_second": tps,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for start_idx in range(0, len(valid), args.effective_batch):
            batch = valid[start_idx:start_idx + args.effective_batch]
            if len(batch) == 0:
                continue
            for j in range(0, len(batch), args.micro_batch):
                chunk = batch[j:j + args.micro_batch]
                ids, labels = make_mlm_batch(chunk, start_idx + j, validation=True)
                ids = ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        l = forward_model(model, args.model, ids, labels)
                else:
                    l = forward_model(model, args.model, ids, labels)
                losses.append(float(l.detach().cpu()))
        return sum(losses) / len(losses)

    final_val = evaluate()
    final_ppl = math.exp(min(final_val, 20.0))
    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**3
    elapsed = time.perf_counter() - start
    tps = actual_train_tokens / max(elapsed, 1e-9)

    payload = {
        "model_name": args.model,
        "dataset": args.dataset_name,
        "model_state": model.state_dict(),
        "trained_from_scratch": True,
        "pretrained_weights": False,
        "training_tokens": actual_train_tokens,
        "optimizer_updates": total_updates,
        "micro_batch": args.micro_batch,
        "gradient_accumulation": grad_accum,
        "effective_batch": args.effective_batch,
        "validation_loss": final_val,
        "validation_perplexity": final_ppl,
        "training_tokens_per_second": tps,
        "peak_training_vram_gb": peak,
    }
    torch.save(payload, args.output_dir / "trained_model_final.pt")
    (args.output_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "model_state"}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "model_state"}, indent=2))


if __name__ == "__main__":
    main()
