#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
from datasets import load_dataset

SEQ_LENGTH = 512
CONTENT_LENGTH = 510
TRAIN_BLOCKS = 193_359
VALID_BLOCKS = 1_953
CLS_ID = 4
SEP_ID = 5

DATASETS = {
    "fineweb_edu": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "revision": "fc9850dff5e2d0f8f776efe41b24a1c49556cfc5",
        "text_field": "text",
        "train_split": "train",
        "valid_split": None,
        "continuation_validation": True,
        "display": "FineWeb-Edu",
    },
    "arxiv": {
        "repo": "ccdv/arxiv-summarization",
        "config": "document",
        "revision": None,
        "text_field": "article",
        "train_split": "train",
        "valid_split": "validation",
        "continuation_validation": False,
        "display": "arXiv",
    },
    "pg19": {
        "repo": "emozilla/pg19",
        "config": None,
        "revision": None,
        "text_field": "text",
        "train_split": "train",
        "valid_split": "validation",
        "continuation_validation": False,
        "display": "PG-19",
    },
    "wikitext103": {
        "repo": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "revision": "main",
        "text_field": "text",
        "train_split": "train",
        "valid_split": None,
        "continuation_validation": True,
        "display": "WikiText-103",
    },
}


def tokenizer_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_stream(spec, split):
    kwargs = dict(split=split, streaming=True)
    if spec["revision"]:
        kwargs["revision"] = spec["revision"]
    if spec["config"] is None:
        return load_dataset(spec["repo"], **kwargs)
    return load_dataset(spec["repo"], spec["config"], **kwargs)


def encode_text(sp, text):
    text = " ".join((text or "").lower().split())
    if not text:
        return []
    return sp.encode(text, out_type=int)


def blocks_from_stream(stream, spec, sp, target_blocks, carry=None):
    buffer = [] if carry is None else list(carry)
    rows = []
    docs = 0
    for ex in stream:
        docs += 1
        ids = encode_text(sp, ex.get(spec["text_field"], ""))
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(SEP_ID)
        while len(buffer) >= CONTENT_LENGTH and len(rows) < target_blocks:
            content = buffer[:CONTENT_LENGTH]
            del buffer[:CONTENT_LENGTH]
            rows.append([CLS_ID, *content, SEP_ID])
        if len(rows) >= target_blocks:
            return rows, buffer, docs, stream
    raise RuntimeError(f"Dataset stream ended at {len(rows)}/{target_blocks} blocks")


def continuation_blocks(stream, spec, sp):
    train_rows = []
    valid_rows = []
    buffer = []
    phase = "train"
    docs = 0
    for ex in stream:
        docs += 1
        ids = encode_text(sp, ex.get(spec["text_field"], ""))
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(SEP_ID)

        target = TRAIN_BLOCKS if phase == "train" else VALID_BLOCKS
        rows = train_rows if phase == "train" else valid_rows
        while len(buffer) >= CONTENT_LENGTH and len(rows) < target:
            content = buffer[:CONTENT_LENGTH]
            del buffer[:CONTENT_LENGTH]
            rows.append([CLS_ID, *content, SEP_ID])

        if phase == "train" and len(train_rows) >= TRAIN_BLOCKS:
            # Match the recorded FineWeb/WikiText protocol: the boundary
            # document remainder is discarded before validation starts.
            buffer = []
            phase = "validation"
            continue

        if phase == "validation" and len(valid_rows) >= VALID_BLOCKS:
            return train_rows, valid_rows, docs

    raise RuntimeError(
        f"Dataset stream ended early: train={len(train_rows)}, valid={len(valid_rows)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if sp.get_piece_size() != 32000:
        raise RuntimeError("Expected 32k tokenizer")

    if spec["continuation_validation"]:
        stream = load_stream(spec, spec["train_split"])
        train_rows, valid_rows, docs = continuation_blocks(stream, spec, sp)
        protocol = "continuation after train target; boundary-document remainder discarded"
    else:
        train_stream = load_stream(spec, spec["train_split"])
        valid_stream = load_stream(spec, spec["valid_split"])
        train_rows, _, train_docs, _ = blocks_from_stream(
            train_stream, spec, sp, TRAIN_BLOCKS
        )
        valid_rows, _, valid_docs, _ = blocks_from_stream(
            valid_stream, spec, sp, VALID_BLOCKS
        )
        docs = {"train": train_docs, "validation": valid_docs}
        protocol = "dataset-provided train and validation splits"

    train = np.asarray(train_rows, dtype=np.uint16)
    valid = np.asarray(valid_rows, dtype=np.uint16)
    np.save(args.output_dir / "train_blocks.npy", train, allow_pickle=False)
    np.save(args.output_dir / "validation_blocks.npy", valid, allow_pickle=False)

    meta = {
        "source": spec["repo"],
        "config": spec["config"],
        "revision": spec["revision"],
        "text_field": spec["text_field"],
        "tokenizer_training_corpus": "C4",
        "model_training_corpus": spec["display"],
        "tokenizer_sha256": tokenizer_sha(args.tokenizer),
        "sequence_length": SEQ_LENGTH,
        "train_blocks": int(train.shape[0]),
        "train_tokens": int(train.size),
        "validation_blocks": int(valid.shape[0]),
        "validation_tokens": int(valid.size),
        "validation_protocol": protocol,
        "documents_consumed": docs,
    }
    (args.output_dir / "COMPLETE.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
